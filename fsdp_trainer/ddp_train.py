"""DDP training loop for running comparison experiments."""
from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .config import TrainConfig
from .data import build_dataloader
from .metrics import MetricsLogger
from .model import TransformerLanguageModel
from .train import _average_across_ranks, _maybe_autocast_dtype, _prepare_device
from .utils import (
    cleanup_distributed,
    ensure_dir,
    get_rank,
    get_world_size,
    init_distributed_process_group,
    is_rank_zero,
    log,
    seed_everything,
)


def _save_checkpoint(
    cfg: TrainConfig,
    model: DDP,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    step: int,
    metrics: Optional[MetricsLogger] = None,
) -> None:
    ckpt_dir = ensure_dir(cfg.runtime.ckpt_dir)
    ckpt_path = Path(ckpt_dir) / f"ddp_epoch{epoch}_step{step}.pt"
    if not is_rank_zero():
        return
    payload = {
        "model": model.module.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "config": cfg.to_dict(),
    }
    log(f"[checkpoint-ddp] Saving to {ckpt_path}")
    torch.save(payload, ckpt_path)
    if metrics is not None:
        metrics.log(
            {
                "event": "checkpoint",
                "mode": "ddp",
                "experiment": cfg.runtime.experiment_name,
                "epoch": epoch,
                "step": step,
                "path": str(ckpt_path),
            }
        )


def _load_checkpoint(
    cfg: TrainConfig,
    model: DDP,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> tuple[int, int]:
    if not cfg.runtime.resume_path:
        return 0, 0
    path = Path(cfg.runtime.resume_path)
    if not path.exists():
        log(f"Checkpoint {path} not found, starting DDP training fresh")
        return 0, 0
    map_location = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(path, map_location=map_location)
    model.module.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if scaler is not None and checkpoint.get("scaler"):
        scaler.load_state_dict(checkpoint["scaler"])

    epoch_step = [int(checkpoint.get("epoch", 0)), int(checkpoint.get("step", 0))]
    if dist.is_available() and dist.is_initialized():
        dist.broadcast_object_list(epoch_step, src=0)
    return epoch_step[0], epoch_step[1]


def train(cfg: TrainConfig) -> None:
    init_distributed_process_group(cfg.runtime)
    rank = get_rank()
    world_size = get_world_size()
    seed_everything(cfg.runtime.seed + rank)
    device = _prepare_device()
    metrics = MetricsLogger(cfg.runtime.metrics_path)
    metrics.log(
        {
            "event": "init",
            "mode": "ddp",
            "experiment": cfg.runtime.experiment_name,
            "rank": rank,
            "world_size": world_size,
            "device": str(device),
        }
    )

    try:
        model = TransformerLanguageModel(cfg.model).to(device)
        if cfg.runtime.use_compile and hasattr(torch, "compile"):
            model = torch.compile(model)  # type: ignore[assignment]
        ddp_model = DDP(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            output_device=device if device.type == "cuda" else None,
            broadcast_buffers=False,
        )

        dataloader = build_dataloader(
            cfg.data,
            batch_size=cfg.batch_size_per_device,
            rank=rank,
            world_size=world_size,
        )

        optimizer = torch.optim.AdamW(
            ddp_model.parameters(),
            lr=cfg.optim.lr,
            betas=cfg.optim.betas,
            weight_decay=cfg.optim.weight_decay,
            eps=cfg.optim.eps,
        )
        autocast_dtype = _maybe_autocast_dtype(cfg.runtime.mixed_precision)
        use_autocast = device.type == "cuda" and autocast_dtype is not None
        scaler = torch.cuda.amp.GradScaler(
            enabled=device.type == "cuda" and cfg.runtime.mixed_precision.lower() == "fp16"
        )

        current_epoch, current_step = _load_checkpoint(cfg, ddp_model, optimizer, scaler)
        log(f"[DDP] Starting training from epoch {current_epoch}, step {current_step}")
        metrics.log(
            {
                "event": "resume",
                "mode": "ddp",
                "experiment": cfg.runtime.experiment_name,
                "epoch": current_epoch,
                "step": current_step,
            }
        )
        optimizer.zero_grad(set_to_none=True)
        log_interval_start = time.time()

        for epoch in range(current_epoch, cfg.epochs):
            sampler = getattr(dataloader, "sampler", None)
            if sampler and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            epoch_start = time.time()
            for batch_idx, (tokens, targets) in enumerate(dataloader, start=1):
                tokens = tokens.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                autocast_ctx = (
                    torch.cuda.amp.autocast(dtype=autocast_dtype)
                    if use_autocast
                    else nullcontext()
                )
                with autocast_ctx:
                    logits = ddp_model(tokens)
                    batch_loss = F.cross_entropy(
                        logits.view(-1, logits.size(-1)),
                        targets.view(-1),
                    )
                    loss = batch_loss / cfg.grad_accum_steps

                if scaler.is_enabled():
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                if batch_idx % cfg.grad_accum_steps == 0:
                    if cfg.optim.clip_grad_norm > 0:
                        if scaler.is_enabled():
                            scaler.unscale_(optimizer)
                        grad_norm = float(
                            torch.nn.utils.clip_grad_norm_(
                                ddp_model.parameters(), cfg.optim.clip_grad_norm
                            )
                        )
                    else:
                        grad_norm = None

                    if scaler.is_enabled():
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    current_step += 1

                    if current_step % cfg.runtime.log_every == 0:
                        reduced_loss = _average_across_ranks(batch_loss.detach()).item()
                        elapsed = time.time() - log_interval_start
                        tokens_per_step = (
                            cfg.batch_size_per_device
                            * cfg.grad_accum_steps
                            * cfg.data.seq_len
                            * world_size
                        )
                        toks_per_sec = tokens_per_step * (
                            cfg.runtime.log_every / max(elapsed, 1e-6)
                        )
                        log_interval_start = time.time()
                        log(
                            f"[DDP epoch {epoch} step {current_step}] loss={reduced_loss:.4f} tok/s={toks_per_sec:.0f}"
                        )
                        metrics.log(
                            {
                                "event": "progress",
                                "mode": "ddp",
                                "experiment": cfg.runtime.experiment_name,
                                "epoch": epoch,
                                "step": current_step,
                                "loss": reduced_loss,
                                "tokens_per_sec": toks_per_sec,
                                "grad_norm": grad_norm,
                            }
                        )

                    if cfg.runtime.ckpt_every > 0 and current_step % cfg.runtime.ckpt_every == 0:
                        _save_checkpoint(cfg, ddp_model, optimizer, scaler, epoch, current_step, metrics)

            epoch_duration = time.time() - epoch_start
            log(f"[DDP] Epoch {epoch} finished on rank {rank}")
            metrics.log(
                {
                    "event": "epoch_complete",
                    "mode": "ddp",
                    "experiment": cfg.runtime.experiment_name,
                    "epoch": epoch,
                    "duration_sec": epoch_duration,
                    "step": current_step,
                }
            )

        metrics.log(
            {
                "event": "complete",
                "mode": "ddp",
                "experiment": cfg.runtime.experiment_name,
                "final_step": current_step,
                "epochs": cfg.epochs,
            }
        )
    finally:
        metrics.close()
        cleanup_distributed()

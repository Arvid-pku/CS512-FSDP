"""Single-process training loop for baseline comparisons."""
from __future__ import annotations

import time
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from .config import TrainConfig
from .data import build_dataloader
from .metrics import MetricsLogger
from .model import TransformerLanguageModel
from .train import (
    _enable_activation_checkpointing,
    _maybe_autocast_dtype,
    _prepare_device,
    _start_profiler,
)
from .utils import ensure_dir, log, seed_everything


def train(cfg: TrainConfig) -> None:
    seed_everything(cfg.runtime.seed)
    device = _prepare_device()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    metrics = MetricsLogger(cfg.runtime.metrics_path)
    metrics.log(
        {
            "event": "init",
            "mode": "single",
            "experiment": cfg.runtime.experiment_name,
            "rank": 0,
            "world_size": 1,
            "device": str(device),
        }
    )

    model = TransformerLanguageModel(cfg.model).to(device)
    if cfg.fsdp.activation_checkpointing:
        _enable_activation_checkpointing(model)
    if cfg.runtime.use_compile and hasattr(torch, "compile"):
        model = torch.compile(model)  # type: ignore[assignment]

    dataloader = build_dataloader(
        cfg.data,
        batch_size=cfg.batch_size_per_device,
        rank=0,
        world_size=1,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
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

    profiler, trace_path = _start_profiler(cfg, "single", device)
    profiled_steps = 0
    log_interval_start = time.time()
    data_wait_accum = 0.0

    for epoch in range(cfg.epochs):
        epoch_start = time.time()
        for batch_idx, (tokens, targets) in enumerate(dataloader, start=1):
            load_start = time.time()
            tokens = tokens.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            data_wait_accum += time.time() - load_start

            autocast_ctx = (
                torch.cuda.amp.autocast(dtype=autocast_dtype)
                if use_autocast
                else nullcontext()
            )
            with autocast_ctx:
                logits = model(tokens)
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
                            model.parameters(), cfg.optim.clip_grad_norm
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

                if profiler is not None:
                    profiler.step()
                    profiled_steps += 1
                    if profiled_steps >= cfg.runtime.profile_steps:
                        profiler.export_chrome_trace(str(trace_path))
                        profiler.__exit__(None, None, None)
                        profiler = None
                        log(f"[profile] Trace saved to {trace_path}")

                if (batch_idx // cfg.grad_accum_steps) % cfg.runtime.log_every == 0:
                    elapsed = time.time() - log_interval_start
                    tokens_per_step = (
                        cfg.batch_size_per_device
                        * cfg.grad_accum_steps
                        * cfg.data.seq_len
                    )
                    toks_per_sec = tokens_per_step * (
                        cfg.runtime.log_every / max(elapsed, 1e-6)
                    )
                    log_interval_start = time.time()
                    avg_data_wait = data_wait_accum / max(cfg.runtime.log_every, 1)
                    data_wait_accum = 0.0
                    mem_msg = ""
                    mem_gb = None
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                        mem_gb = torch.cuda.max_memory_allocated(device) / (1024**3)
                        torch.cuda.reset_peak_memory_stats(device)
                        mem_msg = f", peak_mem={mem_gb:.2f}GB"
                    log(
                        f"[single epoch {epoch}] loss={batch_loss.item():.4f} tok/s={toks_per_sec:.0f}, data_wait={avg_data_wait*1000:.1f}ms{mem_msg}"
                    )
                    metrics.log(
                        {
                            "event": "progress",
                            "mode": "single",
                            "experiment": cfg.runtime.experiment_name,
                            "epoch": epoch,
                            "step": batch_idx,
                            "loss": batch_loss.item(),
                            "tokens_per_sec": toks_per_sec,
                            "grad_norm": grad_norm,
                            "avg_collective_time_sec": 0.0,
                            "peak_memory_gb": mem_gb,
                            "avg_data_wait_sec": avg_data_wait,
                        }
                    )

        metrics.log(
            {
                "event": "epoch_complete",
                "mode": "single",
                "experiment": cfg.runtime.experiment_name,
                "epoch": epoch,
                "duration_sec": time.time() - epoch_start,
                "step": len(dataloader),
            }
        )

    metrics.log(
        {
            "event": "complete",
            "mode": "single",
            "experiment": cfg.runtime.experiment_name,
            "final_step": len(dataloader),
            "epochs": cfg.epochs,
        }
    )

    if profiler is not None:
        profiler.export_chrome_trace(str(trace_path))
        profiler.__exit__(None, None, None)
        log(f"[profile] Trace saved to {trace_path}")
    metrics.close()


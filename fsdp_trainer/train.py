"""End-to-end training loop that wires the pieces together under FSDP."""
from __future__ import annotations

import time
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    CheckpointImpl,
    apply_activation_checkpointing,
    checkpoint_wrapper,
)
from torch.distributed.fsdp import (
    CPUOffload,
    FullyShardedDataParallel as FSDP,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    MixedPrecision,
    ShardingStrategy,
    StateDictType,
)
from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy, transformer_auto_wrap_policy

from .config import TrainConfig
from .data import build_dataloader
from .manual_fsdp import ManualFSDP
from .model import TransformerLanguageModel
from .metrics import MetricsLogger
from .utils import (
    cleanup_distributed,
    ensure_dir,
    get_local_rank,
    get_rank,
    get_world_size,
    init_distributed_process_group,
    is_rank_zero,
    log,
    seed_everything,
)


_SHARDING_MAP = {
    "FULL_SHARD": ShardingStrategy.FULL_SHARD,
    "SHARD_GRAD_OP": ShardingStrategy.SHARD_GRAD_OP,
    "NO_SHARD": ShardingStrategy.NO_SHARD,
    "HYBRID_SHARD": ShardingStrategy.HYBRID_SHARD,
}


def _resolve_sharding(strategy: str) -> ShardingStrategy:
    try:
        return _SHARDING_MAP[strategy.upper()]
    except KeyError as exc:
        raise ValueError(f"Unknown FSDP sharding_strategy={strategy}") from exc


def _maybe_autocast_dtype(precision: str) -> Optional[torch.dtype]:
    precision = precision.lower()
    if precision == "bf16":
        return torch.bfloat16
    if precision in {"fp16", "half"}:
        return torch.float16
    return None


def _build_mixed_precision(precision: str) -> Optional[MixedPrecision]:
    dtype = _maybe_autocast_dtype(precision)
    if dtype is None:
        return None
    return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)


def _build_auto_wrap_policy(cfg: TrainConfig):
    if cfg.fsdp.auto_wrap_policy == "transformer":
        return partial(transformer_auto_wrap_policy, transformer_layer_cls={nn.TransformerEncoderLayer})
    if cfg.fsdp.auto_wrap_policy == "size":
        return partial(size_based_auto_wrap_policy, min_num_params=cfg.fsdp.min_params_to_wrap)
    return None


def _enable_activation_checkpointing(model: nn.Module) -> None:
    wrapper = partial(checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT)
    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=wrapper,
        check_fn=lambda module: isinstance(module, nn.TransformerEncoderLayer),
    )


def _prepare_device() -> torch.device:
    local_rank = get_local_rank()
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank)
    return torch.device("cpu")


def _wrap_with_official_fsdp(model: nn.Module, cfg: TrainConfig, device: torch.device) -> FSDP:
    auto_wrap_policy = _build_auto_wrap_policy(cfg)
    mixed_precision = _build_mixed_precision(cfg.runtime.mixed_precision)
    cpu_offload = CPUOffload(offload_params=True) if cfg.fsdp.cpu_offload else None

    return FSDP(
        model.to(device),
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=_resolve_sharding(cfg.fsdp.sharding_strategy),
        device_id=device if device.type == "cuda" else None,
        sync_module_states=cfg.fsdp.sync_module_states,
        mixed_precision=mixed_precision,
        cpu_offload=cpu_offload,
        limit_all_gathers=cfg.fsdp.limit_all_gathers,
        use_orig_params=cfg.fsdp.use_orig_params,
    )


def _average_across_ranks(value: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(value, op=dist.ReduceOp.AVG)
    return value


def _save_checkpoint(
    cfg: TrainConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    epoch: int,
    step: int,
    metrics: Optional[MetricsLogger] = None,
) -> None:
    ckpt_dir = ensure_dir(cfg.runtime.ckpt_dir)
    ckpt_path = Path(ckpt_dir) / f"fsdp_epoch{epoch}_step{step}.pt"

    rank_zero = is_rank_zero()
    if isinstance(model, ManualFSDP):
        if not rank_zero:
            return
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()
    else:
        state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_cfg):
            model_state = model.state_dict()
        optim_state = FSDP.optim_state_dict(model, optimizer, optim_state_dict_cfg=optim_cfg)

    if not rank_zero:
        return
    payload = {
        "model": model_state,
        "optimizer": optim_state,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "epoch": epoch,
        "step": step,
        "config": cfg.to_dict(),
    }
    log(f"[checkpoint] Saving to {ckpt_path}")
    torch.save(payload, ckpt_path)
    if metrics is not None:
        metrics.log(
            {
                "event": "checkpoint",
                "mode": f"fsdp-{cfg.fsdp.implementation.lower()}",
                "experiment": cfg.runtime.experiment_name,
                "epoch": epoch,
                "step": step,
                "path": str(ckpt_path),
            }
        )


def _load_checkpoint(
    cfg: TrainConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
) -> tuple[int, int]:
    if not cfg.runtime.resume_path:
        return 0, 0
    path = Path(cfg.runtime.resume_path)
    if not path.exists():
        log(f"Checkpoint {path} not found, starting fresh")
        return 0, 0

    if isinstance(model, ManualFSDP):
        checkpoint = torch.load(path, map_location="cpu")
        log(f"[checkpoint] Loaded {path}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        epoch_step = [int(checkpoint.get("epoch", 0)), int(checkpoint.get("step", 0))]
    else:
        if is_rank_zero():
            checkpoint = torch.load(path, map_location="cpu")
            log(f"[checkpoint] Loaded {path}")
        else:
            checkpoint = None

        state_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, state_cfg):
            FSDP.load_state_dict(model, checkpoint["model"] if checkpoint is not None else None)

        optim_state = checkpoint["optimizer"] if checkpoint is not None else None
        optim_state = FSDP.optim_state_dict_to_load(optim_state, model, optimizer)
        optimizer.load_state_dict(optim_state)

        if scaler is not None and checkpoint is not None and checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])

        epoch_step = [0, 0]
        if checkpoint is not None:
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
    impl = cfg.fsdp.implementation.lower()
    manual_impl = impl == "manual"
    mode_label = f"fsdp-{impl}"
    metrics = MetricsLogger(cfg.runtime.metrics_path)
    metrics.log(
        {
            "event": "init",
            "mode": mode_label,
            "experiment": cfg.runtime.experiment_name,
            "rank": rank,
            "world_size": world_size,
            "device": str(device),
        }
    )

    try:
        base_model = TransformerLanguageModel(cfg.model).to(device)
        if cfg.fsdp.activation_checkpointing:
            _enable_activation_checkpointing(base_model)

        if cfg.runtime.use_compile and hasattr(torch, "compile"):
            base_model = torch.compile(base_model)  # type: ignore[assignment]

        if manual_impl:
            fsdp_model: nn.Module = ManualFSDP(base_model)
        else:
            fsdp_model = _wrap_with_official_fsdp(base_model, cfg, device)

        dataloader = build_dataloader(
            cfg.data,
            batch_size=cfg.batch_size_per_device,
            rank=rank,
            world_size=world_size,
        )

        optim_params = (
            list(fsdp_model.sharded_parameters()) if manual_impl else fsdp_model.parameters()
        )
        optimizer = torch.optim.AdamW(
            optim_params,
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

        current_epoch, current_step = _load_checkpoint(cfg, fsdp_model, optimizer, scaler)
        log(f"Starting training from epoch {current_epoch}, step {current_step}")
        metrics.log(
            {
                "event": "resume",
                "mode": mode_label,
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
                    logits = fsdp_model(tokens)
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
                    if manual_impl:
                        fsdp_model.sync_gradients()  # type: ignore[attr-defined]

                    if cfg.optim.clip_grad_norm > 0:
                        if scaler.is_enabled():
                            scaler.unscale_(optimizer)
                        if manual_impl:
                            grad_norm = float(
                                torch.nn.utils.clip_grad_norm_(  # type: ignore[attr-defined]
                                    list(fsdp_model.sharded_parameters()),  # type: ignore[attr-defined]
                                    cfg.optim.clip_grad_norm,
                                )
                            )
                        else:
                            grad_norm = float(
                                fsdp_model.clip_grad_norm_(cfg.optim.clip_grad_norm)  # type: ignore[attr-defined]
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
                        grad_norm_msg = (
                            f", grad_norm={grad_norm:.2f}" if grad_norm is not None else ""
                        )
                        log(
                            f"[epoch {epoch} step {current_step}] loss={reduced_loss:.4f} tok/s={toks_per_sec:.0f}{grad_norm_msg}"
                        )
                        metrics.log(
                            {
                                "event": "progress",
                                "mode": mode_label,
                                "experiment": cfg.runtime.experiment_name,
                                "epoch": epoch,
                                "step": current_step,
                                "loss": reduced_loss,
                                "tokens_per_sec": toks_per_sec,
                                "grad_norm": grad_norm,
                            }
                        )

                    if cfg.runtime.ckpt_every > 0 and current_step % cfg.runtime.ckpt_every == 0:
                        _save_checkpoint(
                            cfg, fsdp_model, optimizer, scaler, epoch, current_step, metrics
                        )

            epoch_duration = time.time() - epoch_start
            log(f"Epoch {epoch} finished on rank {rank}")
            metrics.log(
                {
                    "event": "epoch_complete",
                    "mode": mode_label,
                    "experiment": cfg.runtime.experiment_name,
                    "epoch": epoch,
                    "duration_sec": epoch_duration,
                    "step": current_step,
                }
            )

        metrics.log(
            {
                "event": "complete",
                "mode": mode_label,
                "experiment": cfg.runtime.experiment_name,
                "final_step": current_step,
                "epochs": cfg.epochs,
            }
        )
    finally:
        metrics.close()
        cleanup_distributed()

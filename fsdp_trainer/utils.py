"""Utility helpers for logging, seeding, and distributed setup."""
from __future__ import annotations

import os
import random
from pathlib import Path

import torch
import torch.distributed as dist

from .config import RuntimeConfig


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_rank() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return int(os.environ.get("RANK", 0))


def get_world_size() -> int:
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return int(os.environ.get("WORLD_SIZE", 1))


def is_rank_zero() -> bool:
    return get_rank() == 0


def log(message: str) -> None:
    if is_rank_zero():
        print(message, flush=True)


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def init_distributed_process_group(cfg: RuntimeConfig) -> None:
    if dist.is_available() and dist.is_initialized():
        return
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        raise RuntimeError(
            "Distributed environment variables not set. Launch via torchrun or mpirun."
        )
    dist.init_process_group(backend=cfg.backend)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def ensure_dir(path: str | Path) -> Path:
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj

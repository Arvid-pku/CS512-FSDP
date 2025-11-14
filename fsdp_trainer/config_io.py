"""Helpers for loading and overriding TrainConfig objects."""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None

from .config import DataConfig, FsdpConfig, ModelConfig, OptimizerConfig, RuntimeConfig, TrainConfig


def load_config_file(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file {path} does not exist")
    text = path.read_text()
    if path.suffix in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("pyyaml is required to read YAML configs")
        return dict(yaml.safe_load(text) or {})
    return json.loads(text)


def build_config_from_dict(raw: Dict[str, Any]) -> TrainConfig:
    base = TrainConfig()
    return TrainConfig(
        model=ModelConfig(**raw.get("model", {})) if "model" in raw else deepcopy(base.model),
        data=DataConfig(**raw.get("data", {})) if "data" in raw else deepcopy(base.data),
        optim=OptimizerConfig(**raw.get("optim", {})) if "optim" in raw else deepcopy(base.optim),
        runtime=RuntimeConfig(**raw.get("runtime", {})) if "runtime" in raw else deepcopy(base.runtime),
        fsdp=FsdpConfig(**raw.get("fsdp", {})) if "fsdp" in raw else deepcopy(base.fsdp),
        epochs=raw.get("epochs", base.epochs),
        batch_size_per_device=raw.get("batch_size_per_device", base.batch_size_per_device),
        grad_accum_steps=raw.get("grad_accum_steps", base.grad_accum_steps),
    )


def apply_overrides_from_args(cfg: TrainConfig, args) -> TrainConfig:
    if getattr(args, "epochs", None) is not None:
        cfg.epochs = args.epochs
    if getattr(args, "batch_size", None) is not None:
        cfg.batch_size_per_device = args.batch_size
    if getattr(args, "grad_accum", None) is not None:
        cfg.grad_accum_steps = args.grad_accum
    if getattr(args, "precision", None) is not None:
        cfg.runtime.mixed_precision = args.precision
    if getattr(args, "sharding", None) is not None:
        cfg.fsdp.sharding_strategy = args.sharding
    if getattr(args, "auto_wrap", None) is not None:
        cfg.fsdp.auto_wrap_policy = args.auto_wrap
    if getattr(args, "fsdp_impl", None) is not None:
        cfg.fsdp.implementation = args.fsdp_impl
    if getattr(args, "resume", None) is not None:
        cfg.runtime.resume_path = str(args.resume)
    if getattr(args, "ckpt_dir", None) is not None:
        cfg.runtime.ckpt_dir = str(args.ckpt_dir)
    if getattr(args, "compile", False):
        cfg.runtime.use_compile = True
    return cfg


__all__ = [
    "load_config_file",
    "build_config_from_dict",
    "apply_overrides_from_args",
]

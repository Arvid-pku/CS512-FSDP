"""CLI entrypoint to launch DDP training runs for comparison experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

from fsdp_trainer import TrainConfig
from fsdp_trainer.config_io import (
    apply_overrides_from_args,
    apply_simple_task_overrides,
    build_config_from_dict,
    load_config_file,
)
from fsdp_trainer.ddp_train import train


_SHARDING_CHOICES = ["FULL_SHARD", "SHARD_GRAD_OP", "NO_SHARD", "HYBRID_SHARD"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the demo transformer with DDP")
    parser.add_argument("--config", type=Path, help="Optional JSON/YAML config file", default=None)
    parser.add_argument("--epochs", type=int, help="Override number of epochs", default=None)
    parser.add_argument("--batch-size", type=int, help="Micro-batch size per device", default=None)
    parser.add_argument("--grad-accum", type=int, help="Gradient accumulation steps", default=None)
    parser.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default=None)
    parser.add_argument("--sharding", choices=_SHARDING_CHOICES, default=None)
    parser.add_argument("--auto-wrap", choices=["transformer", "size"], default=None)
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume")
    parser.add_argument("--ckpt-dir", type=Path, default=None, help="Directory to store checkpoints")
    parser.add_argument("--compile", action="store_true", help="Enable torch.compile before wrapping with DDP")
    parser.add_argument(
        "--activation-checkpoint",
        dest="activation_checkpoint",
        action="store_true",
        help="Enable activation checkpointing for the baseline model",
    )
    parser.add_argument(
        "--no-activation-checkpoint",
        dest="activation_checkpoint",
        action="store_false",
        help="Disable activation checkpointing",
    )
    parser.set_defaults(activation_checkpoint=None)
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=None,
        help="Number of optimizer steps to profile (0 to disable)",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=None,
        help="Where profiler traces should be written",
    )
    parser.add_argument(
        "--simple-task",
        action="store_true",
        help="Use the tiny, low-noise synthetic task for fast loss drops",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = TrainConfig()
    if args.config:
        cfg = build_config_from_dict(load_config_file(args.config))
    cfg = apply_overrides_from_args(cfg, args)
    if args.simple_task:
        cfg = apply_simple_task_overrides(cfg)
    train(cfg)


if __name__ == "__main__":
    main()

"""Run quick DDP/FSDP sanity checks to guard against regressions."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


COMMANDS = [
    (
        "single_gpu_fp32",
        "python run_single.py --epochs 2 --batch-size 16 --grad-accum 1 "
        "--precision fp32 --ckpt-dir artifacts/smoke/single",
    ),
    (
        "ddp_fp32",
        "torchrun --standalone --nproc_per_node=1 "
        "run_ddp.py --epochs 2 --batch-size 16 --grad-accum 1 "
        "--precision fp32 --ckpt-dir artifacts/smoke/ddp",
    ),
    (
        "fsdp_manual_bf16",
        "torchrun --standalone --nproc_per_node=1 "
        "run_fsdp.py --epochs 2 --batch-size 16 --grad-accum 1 "
        "--precision bf16 --fsdp-impl manual "
        "--ckpt-dir artifacts/smoke/fsdp_manual --activation-checkpoint "
        "--limit-all-gathers --use-orig-params",
    ),
    (
        "fsdp_torch_fp16",
        "torchrun --standalone --nproc_per_node=1 "
        "run_fsdp.py --epochs 2 --batch-size 16 --grad-accum 1 "
        "--precision fp16 --fsdp-impl torch "
        "--ckpt-dir artifacts/smoke/fsdp_torch --activation-checkpoint "
        "--limit-all-gathers --use-orig-params",
    ),
]


def run(command: str) -> None:
    print(f"\n[smoke] Executing: {command}", flush=True)
    subprocess.run(command, shell=True, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run quick DDP/FSDP smoke tests")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/smoke"),
        help="Where temporary checkpoints/metrics should live",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for subdir in ["single", "ddp", "fsdp_manual", "fsdp_torch"]:
        (args.output_dir / subdir).mkdir(parents=True, exist_ok=True)
    for name, command in COMMANDS:
        try:
            run(command)
            print(f"[smoke] {name} passed")
        except subprocess.CalledProcessError as exc:
            raise SystemExit(f"[smoke] {name} FAILED with exit code {exc.returncode}") from exc


if __name__ == "__main__":
    main()

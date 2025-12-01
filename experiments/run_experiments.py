"""Utility to materialize and (optionally) launch comparison experiments."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fsdp_trainer import TrainConfig
from fsdp_trainer.config_io import build_config_from_dict, load_config_file

from experiments.specs import ExperimentSpec, default_experiments

SIZE_VARIANTS = {
    "small": {
        "model": {"embed_dim": 512, "num_layers": 6, "num_heads": 8, "ff_hidden_dim": 2048},
        "data": {"seq_len": 128},
    },
    "medium": {
        "model": {"embed_dim": 768, "num_layers": 12, "num_heads": 12, "ff_hidden_dim": 3072},
        "data": {"seq_len": 128},
    },
    "large": {
        "model": {"embed_dim": 1024, "num_layers": 16, "num_heads": 16, "ff_hidden_dim": 4096},
        "data": {"seq_len": 128},
    },
}


def deep_update(target: Dict[str, object], patch: Dict[str, object]) -> Dict[str, object]:
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            deep_update(target[key], value)  # type: ignore[index]
        else:
            target[key] = value
    return target


def materialize_config(
    spec: ExperimentSpec,
    output_dir: Path,
    base_cfg: TrainConfig,
    size_variant: str | None = None,
) -> Path:
    cfg_dict = json.loads(json.dumps(base_cfg.to_dict()))
    deep_update(cfg_dict, spec.overrides)
    if size_variant:
        deep_update(cfg_dict, SIZE_VARIANTS[size_variant])
    runtime_block = cfg_dict.setdefault("runtime", {})
    exp_name = spec.name if size_variant is None else f"{spec.name}_{size_variant}"
    runtime_block["experiment_name"] = exp_name
    metrics_path = output_dir / f"{exp_name}_metrics.jsonl"
    runtime_block["metrics_path"] = str(metrics_path)
    cfg_path = output_dir / f"{exp_name}.json"
    cfg_path.write_text(json.dumps(cfg_dict, indent=2))
    return cfg_path


def build_command(
    default_launcher: str,
    entrypoint: str,
    config_path: Path,
    tags: tuple[str, ...],
) -> str:
    if "single" in tags:
        runner = shlex.quote(sys.executable)
    else:
        runner = default_launcher.strip()
    if not runner:
        return f"{entrypoint} --config {shlex.quote(str(config_path))}"
    return f"{runner} {entrypoint} --config {shlex.quote(str(config_path))}"


def filter_specs(specs: Iterable[ExperimentSpec], only: List[str] | None) -> List[ExperimentSpec]:
    if not only:
        return list(specs)
    only_set = set(only)
    return [spec for spec in specs if spec.name in only_set]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a suite of DDP/FSDP comparison experiments")
    parser.add_argument(
        "--launcher",
        default="torchrun --standalone --nproc_per_node=2",
        help="Command prefix used to launch distributed jobs",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/experiments"),
        help="Where to write generated config + metrics files",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        help="Subset of experiment names to generate/run",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually launch experiments instead of only printing commands",
    )
    parser.add_argument(
        "--base-config",
        type=Path,
        default=None,
        help="Optional JSON/YAML config file that provides shared defaults",
    )
    parser.add_argument(
        "--size-variants",
        nargs="*",
        choices=list(SIZE_VARIANTS.keys()),
        help="Optional model size variants (small/medium/large) to expand each experiment",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip launching runs whose metrics file already exists and is non-empty",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    specs = filter_specs(default_experiments(), args.only)
    if not specs:
        raise SystemExit("No experiments selected. Check --only filter.")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = TrainConfig()
    if args.base_config:
        base_cfg = build_config_from_dict(load_config_file(args.base_config))

    size_variants = args.size_variants or [None]
    commands: List[tuple[str, str, str, bool]] = []
    for spec in specs:
        for variant in size_variants:
            exp_name = spec.name if variant is None else f"{spec.name}_{variant}"
            cfg_path = materialize_config(spec, args.output_dir, base_cfg, variant)
            command = build_command(args.launcher, spec.entrypoint, cfg_path, spec.tags)
            metrics_path = args.output_dir / f"{exp_name}_metrics.jsonl"
            metrics_exists = metrics_path.exists() and metrics_path.stat().st_size > 0
            skip_run = args.skip_existing and metrics_exists
            skipped = False
            if args.execute and skip_run:
                skipped = True
                print(f"\n[skip] {exp_name}: found existing metrics at {metrics_path}")
            elif args.execute:
                print(f"\n[exec] {exp_name}: {command}")
                subprocess.run(command, shell=True, check=True)
            commands.append((exp_name, command, spec.description, skipped))

    print("\nGenerated experiments:")
    for name, command, desc, skipped in commands:
        print(f"- {name}: {command}")
        print(f"    {desc}")
        if skipped:
            print("    (skipped execution; metrics already present)")


if __name__ == "__main__":
    main()

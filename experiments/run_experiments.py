"""Utility to materialize and (optionally) launch comparison experiments."""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List

from fsdp_trainer import TrainConfig
from fsdp_trainer.config_io import build_config_from_dict, load_config_file

from .specs import ExperimentSpec, default_experiments


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
) -> Path:
    cfg_dict = json.loads(json.dumps(base_cfg.to_dict()))  # deep copy via json round-trip
    deep_update(cfg_dict, spec.overrides)
    runtime_block = cfg_dict.setdefault("runtime", {})
    runtime_block["experiment_name"] = spec.name
    metrics_path = output_dir / f"{spec.name}_metrics.jsonl"
    runtime_block["metrics_path"] = str(metrics_path)
    cfg_path = output_dir / f"{spec.name}.json"
    cfg_path.write_text(json.dumps(cfg_dict, indent=2))
    return cfg_path


def build_command(launcher: str, entrypoint: str, config_path: Path) -> str:
    return f"{launcher} {entrypoint} --config {shlex.quote(str(config_path))}"


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

    commands: List[tuple[str, str, str]] = []
    for spec in specs:
        cfg_path = materialize_config(spec, args.output_dir, base_cfg)
        command = build_command(args.launcher, spec.entrypoint, cfg_path)
        commands.append((spec.name, command, spec.description))
        if args.execute:
            print(f"\n[exec] {spec.name}: {command}")
            subprocess.run(command, shell=True, check=True)

    print("\nGenerated experiments:")
    for name, command, desc in commands:
        print(f"- {name}: {command}\n    {desc}")


if __name__ == "__main__":
    main()

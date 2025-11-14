"""Parse metrics JSONL files and report DDP/FSDP experiment comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import median
from typing import Dict, List


def load_metrics(path: Path) -> List[Dict[str, object]]:
    entries: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def summarize(path: Path) -> Dict[str, object]:
    entries = load_metrics(path)
    progress = [e for e in entries if e.get("event") == "progress"]
    throughput = [float(e.get("tokens_per_sec", 0.0)) for e in progress if e.get("tokens_per_sec")]
    losses = [float(e.get("loss", 0.0)) for e in progress if e.get("loss") is not None]
    name = ""
    mode = ""
    if entries:
        name = str(entries[0].get("experiment"))
        mode = str(entries[0].get("mode"))
    if not name:
        name = path.stem.replace("_metrics", "")
    summary = {
        "name": name,
        "mode": mode or "unknown",
        "path": str(path),
        "final_loss": losses[-1] if losses else None,
        "best_loss": min(losses) if losses else None,
        "median_tokens_per_sec": median(throughput) if throughput else None,
        "max_tokens_per_sec": max(throughput) if throughput else None,
        "num_points": len(progress),
    }
    return summary


def gather_paths(args: argparse.Namespace) -> List[Path]:
    paths: List[Path] = []
    if args.metrics:
        paths.extend(Path(p) for p in args.metrics)
    if args.experiment_dir:
        paths.extend(sorted(Path(args.experiment_dir).glob("*_metrics.jsonl")))
    if not paths:
        raise SystemExit("No metrics files provided. Use --metrics or --experiment-dir.")
    return paths


def attach_speedup(summaries: List[Dict[str, object]], baseline_name: str | None) -> None:
    baseline = None
    if baseline_name:
        for summary in summaries:
            if summary["name"] == baseline_name:
                baseline = summary
                break
    if baseline is None and summaries:
        baseline = summaries[0]
    if not baseline:
        return
    base_tps = baseline.get("median_tokens_per_sec") or 0.0
    for summary in summaries:
        if base_tps:
            summary["speedup_vs_baseline"] = (
                (summary.get("median_tokens_per_sec") or 0.0) / base_tps
            )
        else:
            summary["speedup_vs_baseline"] = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize experiment metrics JSONL files")
    parser.add_argument("--metrics", nargs="*", help="Explicit list of metrics JSONL files")
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        default=None,
        help="Directory produced by experiments/run_experiments.py",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="Experiment name that should be treated as the baseline for speedup calculations",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path to write the aggregated summary as JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = gather_paths(args)
    summaries = [summarize(path) for path in paths]
    attach_speedup(summaries, args.baseline)

    print("Experiment summaries:")
    for summary in summaries:
        speedup = summary.get("speedup_vs_baseline")
        speedup_str = f" x{speedup:.2f}" if speedup else ""
        print(
            f"- {summary['name']} ({summary['mode']}): loss={summary['final_loss']} "
            f"best={summary['best_loss']} median_tok/s={summary['median_tokens_per_sec']}{speedup_str}"
        )

    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(summaries, indent=2))
        print(f"\nWrote summary JSON to {args.json_out}")


if __name__ == "__main__":
    main()

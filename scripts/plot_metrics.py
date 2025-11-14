"""Generate charts from experiment summary JSON."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt


def load_summary(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text())
    if isinstance(data, dict):
        return data.get("experiments", [])
    return data


def plot_bar(summary: List[Dict[str, Any]], metric: str, title: str, output: Path, ylabel: str) -> None:
    names = [item["name"] for item in summary]
    values = [item.get(metric, 0.0) or 0.0 for item in summary]
    plt.figure(figsize=(10, 4))
    bars = plt.bar(names, values, color="#4C72B0")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=200)
    plt.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot throughput/memory charts from summary JSON")
    parser.add_argument("--summary", type=Path, default=Path("artifacts/experiments/summary.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/experiments/plots"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = load_summary(args.summary)
    if not summary:
        raise SystemExit("Summary is empty; run experiments/analyze first.")

    plot_bar(summary, "median_tokens_per_sec", "Median Throughput", args.output_dir / "throughput.png", "tokens/sec")
    plot_bar(summary, "peak_memory_gb", "Peak GPU Memory", args.output_dir / "memory.png", "GB")
    plot_bar(summary, "final_loss", "Final Loss", args.output_dir / "loss.png", "cross-entropy")


if __name__ == "__main__":
    main()


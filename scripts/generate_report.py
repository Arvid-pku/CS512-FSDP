"""Generate a Markdown report summarizing experiment metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Markdown report from experiment summary")
    parser.add_argument("--summary", type=Path, default=Path("artifacts/experiments/summary.json"))
    parser.add_argument("--plots-dir", type=Path, default=Path("artifacts/experiments/plots"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments/report.md"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.summary.read_text())
    experiments: List[Dict[str, Any]] = data if isinstance(data, list) else data.get("experiments", [])
    if not experiments:
        raise SystemExit("Summary JSON has no experiment entries.")
    lines = [
        "# Experiment Report",
        "",
        "## Summary Table",
        "",
        "| Experiment | Mode | Final Loss | Median tok/s | Peak Mem (GB) | Speedup vs Baseline |",
        "|-----------|------|------------|---------------|---------------|---------------------|",
    ]
    for item in experiments:
        lines.append(
            f"| {item['name']} | {item.get('mode','')} | "
            f"{item.get('final_loss','-')} | {item.get('median_tokens_per_sec','-')} | "
            f"{item.get('peak_memory_gb','-')} | {item.get('speedup_vs_baseline','-')} |"
        )

    lines.extend(
        [
            "",
            "## Visualizations",
            "",
        ]
    )
    for plot_name in ["throughput.png", "memory.png", "loss.png"]:
        plot_path = args.plots_dir / plot_name
        if plot_path.exists():
            lines.extend(
                [
                    f"![{plot_name}]({plot_path})",
                    "",
                ]
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))
    print(f"Wrote report to {args.output}")


if __name__ == "__main__":
    main()


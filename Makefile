.PHONY: smoke sweep analyze plots report

SMOKE?=python scripts/smoke_test.py
SWEEP?=python experiments/run_experiments.py --launcher "torchrun --standalone --nproc_per_node=2" --output-dir artifacts/experiments --execute
ANALYZE?=python experiments/analyze_metrics.py --experiment-dir artifacts/experiments --baseline ddp_fp32 --json-out artifacts/experiments/summary.json
PLOTS?=python scripts/plot_metrics.py --summary artifacts/experiments/summary.json --output-dir artifacts/experiments/plots
REPORT?=python scripts/generate_report.py --summary artifacts/experiments/summary.json --plots-dir artifacts/experiments/plots --output artifacts/experiments/report.md

smoke:
	$(SMOKE)

sweep:
	$(SWEEP)

analyze:
	$(ANALYZE)

plots:
	$(PLOTS)

report: analyze plots
	$(REPORT)


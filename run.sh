#!/usr/bin/env bash
set -euo pipefail

# Launcher can be overridden (e.g., LAUNCHER="torchrun --nnodes=2 --nproc_per_node=4")
LAUNCHER=${LAUNCHER:-"torchrun --standalone --nproc_per_node=2"}
BASE_CONFIG=${BASE_CONFIG:-""}

# echo "[run] Smoke-testing entrypoints..."
# python scripts/smoke_test.py

VARIANTS=("small" "medium" "large")
# VARIANTS=("medium")

for variant in "${VARIANTS[@]}"; do
  OUT_DIR="artifacts/experiments/${variant}"
  echo "[run] Launching ${variant} experiments -> ${OUT_DIR}"
  CMD=(python experiments/run_experiments.py
       --launcher "${LAUNCHER}"
       --output-dir "${OUT_DIR}"
       --size-variants "${variant}"
       --skip-existing
       --execute)
  if [[ -n "${BASE_CONFIG}" ]]; then
    CMD+=(--base-config "${BASE_CONFIG}")
  fi
  "${CMD[@]}"

  echo "[run] Analyzing ${variant} results"
  # python experiments/analyze_metrics.py \
  #   --experiment-dir "${OUT_DIR}" \
  #   --baseline "single_gpu_fp32_${variant}" \
  #   --json-out "${OUT_DIR}/summary.json"

  echo "[run] Plotting ${variant} metrics"
  python scripts/plot_metrics.py \
    --summary "${OUT_DIR}/summary.json" \
    --output-dir "${OUT_DIR}/plots"

  echo "[run] Generating ${variant} report"
  python scripts/generate_report.py \
    --summary "${OUT_DIR}/summary.json" \
    --plots-dir "${OUT_DIR}/plots" \
    --output "${OUT_DIR}/report.md"
done

echo "[run] Completed all experiments. Reports live under artifacts/experiments/{variant}/report.md"

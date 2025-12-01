# Experiment Plan: Manual FSDP vs DDP vs torch.distributed.fsdp

## Goals
- Quantify memory (via peak GPU telemetry) and throughput differences between our manual FSDP implementation, the official torch FSDP, and DDP baselines.
- Evaluate the effect of activation checkpointing and `limit_all_gathers` on both implementations, using the new CLI toggles for rapid iteration.
- Produce reproducible metrics (loss curves, tokens/sec, comm time, memory) that support the final analysis section.
- Provide single-variable ablations (precision, activation checkpointing, `limit_all_gathers`) so manual vs official FSDP comparisons have causal backing.

## Hardware / Environment
- Nodes: 1x server with 2-4 identical GPUs (A100 preferred) and NVLink or PCIe Gen4 interconnect.
- Software: PyTorch 2.1+, CUDA 12.x, NCCL backend. This repo with dependencies installed per `requirements.txt`.
- Launcher: `torchrun --standalone --nproc_per_node={num_gpus}`.

## Data / Model
- Synthetic language-model dataset (`fsdp_trainer/data.py`) now uses a learnable modular-addition pattern: targets are `(token + pattern_period) % vocab_size` with configurable noise, so all methods should see decreasing loss.
- Transformer config: embed_dim 768, 6 layers, 12 heads, FFN 3072 (configurable via JSON/YAML).
- Training length: 2 epochs, `batch_size_per_device=8`, `grad_accum_steps=2`.

## Experiments
| ID | Description | Command Stub | Metrics File |
|----|-------------|--------------|--------------|
| `single_gpu_fp32` | Single-process throughput reference | `python run_single.py --config artifacts/.../single_gpu_fp32.json` | `.../single_gpu_fp32_metrics.jsonl` |
| `ddp_fp32` | Vanilla DDP FP32 without checkpointing (baseline) | `torchrun ... run_ddp.py --config artifacts/.../ddp_fp32.json` | `.../ddp_fp32_metrics.jsonl` |
| `ddp_checkpointed_fp32` | DDP FP32 + activation checkpointing only (isolates checkpoint cost) | `torchrun ... run_ddp.py --config artifacts/.../ddp_checkpointed_fp32.json` | `.../ddp_checkpointed_fp32_metrics.jsonl` |
| `ddp_fp16` | DDP FP16 without checkpointing (isolates precision gain) | `torchrun ... run_ddp.py --config artifacts/.../ddp_fp16.json` | `.../ddp_fp16_metrics.jsonl` |
| `ddp_checkpointed_fp16` | DDP + activation checkpointing + FP16 (ZeRO-lite combo) | `torchrun ... run_ddp.py --config artifacts/.../ddp_checkpointed_fp16.json` | `.../ddp_checkpointed_fp16_metrics.jsonl` |
| `fsdp_manual_bf16` | Manual FSDP bf16 baseline (checkpoint + `limit_all_gathers` on) | `torchrun ... run_fsdp.py --config artifacts/.../fsdp_manual_bf16.json` | `.../fsdp_manual_bf16_metrics.jsonl` |
| `fsdp_manual_fp32` | Manual FSDP fp32 (precision ablation) | `torchrun ... run_fsdp.py --config artifacts/.../fsdp_manual_fp32.json` | `.../fsdp_manual_fp32_metrics.jsonl` |
| `fsdp_manual_bf16_no_checkpoint` | Manual FSDP bf16 without checkpointing | `torchrun ... run_fsdp.py --config artifacts/.../fsdp_manual_bf16_no_checkpoint.json` | `.../fsdp_manual_bf16_no_checkpoint_metrics.jsonl` |
| `fsdp_manual_bf16_no_limit_all_gathers` | Manual FSDP bf16 toggling `limit_all_gathers` off | `torchrun ... run_fsdp.py --config artifacts/.../fsdp_manual_bf16_no_limit_all_gathers.json` | `.../fsdp_manual_bf16_no_limit_all_gathers_metrics.jsonl` |
| `fsdp_official` | Official torch FSDP fp32 baseline (size auto-wrap, checkpointing on) | `torchrun ... run_fsdp.py --config artifacts/.../fsdp_official.json` | `.../fsdp_official_metrics.jsonl` |
| `fsdp_official_bf16` | Official torch FSDP bf16 (precision ablation) | `torchrun ... run_fsdp.py --config artifacts/.../fsdp_official_bf16.json` | `.../fsdp_official_bf16_metrics.jsonl` |
| `fsdp_official_no_checkpoint` | Official FSDP w/o activation checkpointing | `torchrun ... run_fsdp.py --config artifacts/.../fsdp_official_no_checkpoint.json` | respective metrics |
| `fsdp_official_no_limit_all_gathers` | Official FSDP w/o `limit_all_gathers` | `torchrun ... run_fsdp.py --config artifacts/.../fsdp_official_no_limit_all_gathers.json` | metrics file |

All configs generated via:
```bash
python experiments/run_experiments.py \
  --launcher "torchrun --standalone --nproc_per_node=2" \
  --output-dir artifacts/experiments \
  --skip-existing \
  --execute
```
(omit `--execute` to preview commands without running.)
You can append `--size-variants small medium large` to the command above to materialize each experiment at multiple model sizes in one sweep.
Use `single_gpu_fp32` as the “no parallel technique” baseline when analyzing metrics and computing speedups.

- From JSONL logs: `loss`, `tokens_per_sec`, `grad_norm`, `avg_collective_time_sec`, `peak_memory_gb`, `avg_data_wait_sec`, and timestamps per progress interval.
- Derived metrics:
  - Median tokens/sec (per experiment).
  - Final / best loss comparison.
  - Speedup relative to `ddp_fp32`.
- Optional: enable `--profile-steps N --profile-dir artifacts/profiler` to capture Chrome traces for representative steps (keep N<=5 to limit overhead).

## Analysis Procedure
1. Run `make smoke` to verify single/DDP/FSDP entrypoints.
2. Launch the sweep via `make sweep` (or the `experiments/run_experiments.py` command above). Re-running `run.sh` is idempotent because it now forwards `--skip-existing`, so completed experiments are skipped while new configs still run.
3. Aggregate metrics: `make analyze` (writes `artifacts/experiments/summary.json`).
4. Visualization + report:
   ```bash
   make plots
   make report
   ```
   This produces PNG charts under `artifacts/experiments/plots` and a Markdown summary at `artifacts/experiments/report.md`.
5. Highlight in the write-up:
   - Manual vs official FSDP throughput/loss/memory trade-offs.
   - Impact of activation checkpointing and `limit_all_gathers`.
   - DDP (with/without checkpointing) versus sharded approaches.
   - Any profiler insights (attach Chrome traces as supplementary material).

## Risks / Mitigations
- **OOM on manual FSDP**: adjust `batch_size_per_device` downward or increase grad accumulation.
- **Inconsistent seeds**: ensure `runtime.seed` identical across configs for fair loss comparisons.
- **Profiler overhead**: keep `--profile-steps` small (≤5) and disable for final throughput measurements.
- **Clock skew**: use the same host for all runs; avoid concurrent workloads or throttle background jobs.

## Deliverables
- Metrics JSONL files + summarized JSON (`artifacts/experiments/summary.json`).
- Plots (`artifacts/experiments/plots/*.png`) and Markdown report (`artifacts/experiments/report.md`).
- Reference profiler traces (if captured) under `artifacts/profiler`.
- Final write-up citing manual vs official FSDP trade-offs and a recommended configuration for the class project demo.

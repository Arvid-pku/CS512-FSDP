# Experiment Plan: Manual FSDP vs DDP vs torch.distributed.fsdp

## Goals
- Quantify memory and throughput differences between our manual FSDP implementation, the official torch FSDP, and DDP baselines.
- Evaluate the effect of activation checkpointing and `limit_all_gathers` on both implementations.
- Produce reproducible metrics (loss curves, tokens/sec) that support a final analysis section for the class project report.

## Hardware / Environment
- Nodes: 1x server with 2–4 identical GPUs (A100 preferred) and NVLink or PCIe Gen4 interconnect.
- Software: PyTorch 2.1+, CUDA 12.x, NCCL backend. This repo with dependencies installed per `requirements.txt`.
- Launcher: `torchrun --standalone --nproc_per_node={num_gpus}`.

## Data / Model
- Synthetic language-model dataset (`fsdp_trainer/data.py`) with sequence length 256 and vocab size 32k.
- Transformer config: embed_dim 768, 6 layers, 12 heads, FFN 3072 (configurable via JSON/YAML).
- Training length: 2 epochs, `batch_size_per_device=8`, `grad_accum_steps=2`.

## Experiments
| ID | Description | Command Stub | Metrics File |
|----|-------------|--------------|--------------|
| `ddp_fp32` | Baseline throughput/accuracy using vanilla DDP FP32 | `torchrun ... run_ddp.py --config artifacts/experiments/ddp_fp32.json` | `artifacts/experiments/ddp_fp32_metrics.jsonl` |
| `fsdp_manual_bf16` | Manual FSDP (explicit gather/scatter) with bf16 | `torchrun ... run_fsdp.py --fsdp-impl manual --precision bf16 ...` | `.../fsdp_manual_bf16_metrics.jsonl` |
| `fsdp_official` | Official torch FSDP (size auto-wrap, bf16 or fp32) | `torchrun ... run_fsdp.py --fsdp-impl torch ...` | `.../fsdp_official_metrics.jsonl` |
| `fsdp_official_no_checkpoint` | Torch FSDP ablation w/o activation checkpointing | same as above + `--config artifacts/.../fsdp_official_no_checkpoint.json` | respective metrics |
| `fsdp_official_no_limit_all_gathers` | Torch FSDP ablation toggling all-gather behavior | similar config override | metrics file |

All configs generated via:
```bash
python experiments/run_experiments.py \
  --launcher "torchrun --standalone --nproc_per_node=2" \
  --output-dir artifacts/experiments \
  --execute
```
(omit `--execute` to preview commands without running.)

## Measurements
- From JSONL logs: `loss`, `tokens_per_sec`, `grad_norm`, and timestamps per progress interval.
- Derived metrics:
  - Median tokens/sec (per experiment).
  - Final / best loss comparison.
  - Speedup relative to `ddp_fp32`.
- Memory observations: capture `nvidia-smi` snapshots during each run (manual step).

## Analysis Procedure
1. Run all experiments sequentially (ensure clean GPU memory between runs).
2. Summarize metrics:
   ```bash
   python experiments/analyze_metrics.py \
     --experiment-dir artifacts/experiments \
     --baseline ddp_fp32 \
     --json-out artifacts/experiments/summary.json
   ```
3. Plot (optional): load `summary.json` in a notebook to produce throughput bar charts and loss curves.
4. Document findings:
   - Manual FSDP vs official FSDP throughput/loss differences.
   - Effect of activation checkpointing and all-gather throttling.
   - Comparison to DDP baseline.

## Risks / Mitigations
- **OOM on manual FSDP**: adjust `batch_size_per_device` downward or increase grad accumulation.
- **Inconsistent seeds**: ensure `runtime.seed` identical across configs for fair loss comparisons.
- **Clock skew**: use the same host for all runs; avoid concurrent workloads.

## Deliverables
- Metrics JSONL files + summarized JSON.
- Short write-up with plots/tables describing the observed trade-offs.
- Recommendation on which FSDP variant to use for the final project demo.


# FSDP Demo Trainer

This repository provides a clean PyTorch training stack that showcases how to train a Transformer-like model with [`torch.distributed.fsdp`](https://pytorch.org/docs/stable/fsdp.html). The code is designed to be easy to read and easy to modify for a final project or experimentation.

## Features
- Modular project layout (`fsdp_trainer/`) with clear separation of config, data, model, and training logic.
- Manual FSDP implementation (explicit all-gather/reduce-scatter sharding) plus the option to fall back to `torch.distributed.fsdp` for apples-to-apples comparisons.
- Activation checkpointing, optional `torch.compile`, and CLI toggles for `limit_all_gathers` and `use_orig_params`.
- Synthetic dataset defaults to a learnable modular-addition pattern (inputs are random tokens, targets are `(token + pattern_period) % vocab_size` with mild noise) so loss decreases over time even without real data.
- Rich telemetry per log event: loss, tokens/sec, gradient norm, average data-loader wait, collective time, and peak GPU memory.
- Optional torch.profiler capture via `--profile-steps/--profile-dir`.
- Multiple baselines: single-process (`run_single.py`), DDP (`run_ddp.py`), manual FSDP, and official FSDP accessible via CLI + curated experiment specs.
- Visualization/report pipeline (`scripts/plot_metrics.py`, `scripts/generate_report.py`) and a `Makefile` for repeatable workflows (`smoke`, `sweep`, `analyze`, `report`).

## Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
> PyTorch itself is not pinned here because you should pick the wheel that matches your CUDA runtime. Install `torch`/`torchvision`/`torchaudio` per the [official instructions](https://pytorch.org/get-started/locally/).

## Launching Training
Always launch distributed jobs with `torchrun` (or `torch.distributed.run`). Example for 2 GPUs on a single node:
```bash
torchrun --standalone --nproc_per_node=2 run_fsdp.py \
  --epochs 1 \
  --batch-size 4 \
  --grad-accum 2 \
  --precision bf16
```
Manual FSDP is the default (`--fsdp-impl manual`). To run the official PyTorch wrapper for comparison add `--fsdp-impl torch`.
- `--config`: path to a JSON/YAML file that mirrors the `TrainConfig` layout (`model`, `data`, `optim`, `runtime`, `fsdp`, `epochs`, etc.).
- `--batch-size`, `--grad-accum`, `--precision`, `--sharding`, `--auto-wrap`: quick overrides without editing files.
- `--activation-checkpoint/--no-activation-checkpoint`, `--limit-all-gathers/--no-limit-all-gathers`, `--use-orig-params/--no-use-orig-params`: flip the core FSDP toggles without editing configs.
- `--profile-steps/--profile-dir`: capture Chrome-trace profiler files for the first N optimizer steps.
- `--resume` / `--ckpt-dir`: control checkpoint IO locations.
- `--compile`: compile the model before wrapping it with FSDP (requires PyTorch 2.0+).

### Example Config File (YAML)
```yaml
model:
  embed_dim: 768
  num_layers: 6
  num_heads: 12
  ff_hidden_dim: 3072
  dropout: 0.1
runtime:
  mixed_precision: bf16
  log_every: 5
fsdp:
  sharding_strategy: FULL_SHARD
  auto_wrap_policy: transformer
  activation_checkpointing: true
optim:
  lr: 3.0e-4
  warmup_steps: 200
```
Then run:
```bash
torchrun --nproc_per_node=4 run_fsdp.py --config configs/demo.yaml
```

### Single & DDP Baselines
Single-GPU reference:
```bash
python run_single.py --epochs 1 --precision fp32
```

To run the same model with classic DistributedDataParallel for comparison:
```bash
torchrun --standalone --nproc_per_node=2 run_ddp.py --epochs 1 --precision fp32
```

## Experiments & Analysis
Use the provided harness to materialize configs (and optionally execute them) for the curated experiments in `experiments/specs.py`.
```bash
python experiments/run_experiments.py \
  --launcher "torchrun --standalone --nproc_per_node=2" \
  --output-dir artifacts/experiments \
  --execute
```
This generates JSON configs, metrics files (`*_metrics.jsonl`), and optionally launches:
- `single_gpu_fp32`: single-process reference.
- `ddp_fp32`: pure DDP baseline.
- `ddp_checkpointed_fp16`: DDP + activation checkpointing.
- `fsdp_manual_bf16`: this repo's manual FSDP (explicit gather/scatter) in bf16.
- `fsdp_official` + ablations: official PyTorch FSDP with/without activation checkpointing and `limit_all_gathers` to highlight implementation differences.
- Pass `--size-variants small medium large` to `experiments/run_experiments.py` (or `MAKE_SIZE_VARIANTS="small medium large" make sweep`) to materialize each experiment at multiple model scales.

After runs complete, summarize throughput/loss and relative speedups:
```bash
python experiments/analyze_metrics.py \
  --experiment-dir artifacts/experiments \
  --baseline ddp_fp32
```
Every trainer logs JSON events (loss, tokens/sec, checkpoints) whenever `runtime.metrics_path` is set—`run_experiments.py` wires this up automatically.

## Visualization & Reporting
Generate plots and a Markdown report from the analyzed summary:
```bash
python scripts/plot_metrics.py --summary artifacts/experiments/summary.json \
  --output-dir artifacts/experiments/plots
python scripts/generate_report.py --summary artifacts/experiments/summary.json \
  --plots-dir artifacts/experiments/plots \
  --output artifacts/experiments/report.md
```
or simply run `make report` to execute the full pipeline (`smoke`, `sweep`, `analyze`, `plots`, `report` targets are available in the root Makefile).

## Automation Cheatsheet
| Command | Description |
|---------|-------------|
| `make smoke` | Run single, DDP, manual FSDP, and torch FSDP smoke tests on one GPU. |
| `make sweep` | Materialize configs and launch every experiment defined in `experiments/specs.py`. |
| `make analyze` | Aggregate metrics JSONL files into `summary.json`. |
| `make plots` | Render throughput/memory/loss charts under `artifacts/experiments/plots`. |
| `make report` | Full pipeline: analyze, plot, and emit `artifacts/experiments/report.md`. |

## Smoke Tests
Before larger sweeps, run the quick sanity suite:
```bash
python scripts/smoke_test.py
```
This launches single-GPU DDP, manual FSDP, and torch FSDP jobs to make sure checkpointing, telemetry, and collectives still work after changes.

## Manual FSDP Notes
- `fsdp_trainer/manual_fsdp.py` implements sharding by keeping only the local shard as an optimizer parameter, all-gathering the full parameter vector before each forward, and reduce-scattering gradients after every backward pass.
- Reduce-scatter calls are issued only at gradient-accumulation boundaries, so you pay the collective cost once per optimizer step even though multiple micro-batches may contribute.
- The wrapper automatically uses `dist.group.WORLD` (or a user-supplied process group) and only rank zero materializes checkpoints, mirroring the rank-aware behavior of the official FSDP APIs.
- The optimizer only sees the sharded flat parameter, so memory usage scales similarly to official FSDP while remaining easy to reason about. Use `--fsdp-impl` to switch implementations at runtime without code edits.

## Troubleshooting
- **“Distributed environment variables not set”**: Always launch via `torchrun` (or `python -m torch.distributed.run`). For smoke/local tests you can use `torchrun --standalone --nproc_per_node=1 ...`.
- **CPU-only runs**: The manual FSDP wrapper supports CPU tensors, but throughput will be extremely low. Prefer GPUs and ensure `NCCL_P2P_DISABLE=1` if your hardware lacks NVLink/InfiniBand.
- **OOM during experiments**: Drop `--batch-size`, increase `--grad-accum`, or disable `--activation-checkpoint`. The telemetry log will show peak GPU memory after each logging interval.
- **Inconsistent losses across runs**: Keep `runtime.seed` identical and rerun the smoke tests to verify deterministic launch settings.
- **Profiler traces are empty**: ensure `--profile-steps > 0`, the output directory exists, and beware that profiler capture adds overhead—use small step counts (e.g., 5) for representative traces.

## Project Layout
```
fsdp_trainer/
  __init__.py           # re-exports config dataclasses
  config.py             # configuration objects with sane defaults
  data.py               # synthetic dataset + distributed dataloader helper
  model.py              # TransformerLanguageModel definition
  train.py              # end-to-end FSDP training loop + checkpoint helpers
run_fsdp.py             # CLI entrypoint that wires configs to the trainer
README.md               # this file
requirements.txt        # minimal dependency list (PyTorch not pinned)
```

## Next Steps / Customization
- Replace `SyntheticLanguageModelingDataset` with your real dataset loader; just keep the `(tokens, targets)` contract.
- Adjust `ModelConfig` or drop in a different architecture in `fsdp_trainer/model.py`.
- Tweak the FSDP knobs under `FsdpConfig` (e.g., switch to `size` auto-wrap for non-transformer models).
- Extend `TrainConfig` or the CLI with additional arguments that your project needs.

Feel free to open issues or comments if you plan to evolve this into a larger research project.

# FSDP Demo Trainer

This repository provides a clean PyTorch training stack that showcases how to train a Transformer-like model with [`torch.distributed.fsdp`](https://pytorch.org/docs/stable/fsdp.html). The code is designed to be easy to read and easy to modify for a final project or experimentation.

## Features
- Modular project layout (`fsdp_trainer/`) with clear separation of config, data, model, and training logic.
- Manual FSDP implementation (explicit all-gather/reduce-scatter sharding) plus the option to fall back to `torch.distributed.fsdp` for apples-to-apples comparisons.
- Activation checkpointing and optional `torch.compile` integration for squeezing extra performance.
- Synthetic dataset for quick experiments; swap in a real dataset by editing `fsdp_trainer/data.py`.
- Robust checkpoint save/resume support that uses the recommended FSDP APIs for model and optimizer state dicts.
- A CLI (`run_fsdp.py`) that accepts JSON/YAML configs *or* direct CLI overrides to keep iteration fast.
- Matching `run_ddp.py` baseline and experiment utilities under `experiments/` to compare DDP, tutorial-style FSDP, and optimized FSDP variants with structured metrics.

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
Key CLI flags:
- `--config`: path to a JSON/YAML file that mirrors the `TrainConfig` layout (`model`, `data`, `optim`, `runtime`, `fsdp`, `epochs`, etc.).
- `--batch-size`, `--grad-accum`, `--precision`, `--sharding`, `--auto-wrap`: quick overrides without editing files.
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

### DDP Baseline
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
- `ddp_fp32`: pure DDP baseline.
- `fsdp_manual_bf16`: this repo's manual FSDP (explicit gather/scatter) in bf16.
- `fsdp_official` + ablations: official PyTorch FSDP with/without activation checkpointing and `limit_all_gathers` to highlight implementation differences.

After runs complete, summarize throughput/loss and relative speedups:
```bash
python experiments/analyze_metrics.py \
  --experiment-dir artifacts/experiments \
  --baseline ddp_fp32
```
Every trainer logs JSON events (loss, tokens/sec, checkpoints) whenever `runtime.metrics_path` is set—`run_experiments.py` wires this up automatically.

## Manual FSDP Notes
- `fsdp_trainer/manual_fsdp.py` implements sharding by keeping only the local shard as an optimizer parameter, all-gathering the full parameter vector before each forward, and reduce-scattering gradients after every backward pass.
- The optimizer only sees the sharded flat parameter, so memory usage scales similarly to official FSDP while remaining easy to reason about.
- Checkpoints are written by gathering the full parameter vector (manual) or via the official `state_dict_type` APIs (torch implementation). Use `--fsdp-impl` to switch between them at runtime without changing code.

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

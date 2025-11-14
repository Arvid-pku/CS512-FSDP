# Manual FSDP Implementation Report

## Overview
To support a transparent comparison in the Distributed Systems project, we built a **manual Fully Sharded Data Parallel (FSDP)** wrapper (`fsdp_trainer/manual_fsdp.py`) instead of relying solely on `torch.distributed.fsdp`. The manual path exposes every collective and memory trade-off so we can study how sharding works under the hood and reason about performance characteristics.

Key objectives:
1. **Shard parameters explicitly** so each rank only optimizes a slice of the flattened parameter vector.
2. **All-gather full weights lazily** right before the forward pass, then release them promptly.
3. **Reduce-scatter gradients manually** to keep optimizer state sharded.
4. **Preserve checkpoint compatibility** by reconstructing full parameter vectors on demand.

## Architecture

### Parameter Flattening & Sharding
- On construction, we gather every parameter tensor from the wrapped module and flatten them into a single vector.
- The vector is padded to a multiple of the world size and reshaped into `[world_size, shard_size]`.
- Each rank keeps its shard as a learnable `nn.Parameter` (`self._flat_param`) that the optimizer updates.
- The wrapper binds to `dist.group.WORLD` (or a user-supplied process group) so it never relies on private distributed APIs.
- Original module parameters are detached and zeroed so they no longer own storage; they’re treated as views that get materialized only when needed.

### Forward Path
1. `_materialize_full_params()` uses `all_gather`/`all_gather_into_tensor` to build the full flat vector on every rank.
2. `vector_to_parameters` restores original parameter tensors inside the wrapped module.
3. The user’s forward then proceeds normally.
4. After the forward (and once gradients have been reduce-scattered) we call `_release_full_params()` to zero/detach those tensors, freeing memory.

### Backward & Gradient Sync
- During backward, gradients accumulate on the “materialized” parameters across micro-batches.
- `sync_gradients()` is called only at gradient-accumulation boundaries so reduce-scatter happens once per optimizer step.
- The method flattens grads, pads them, reshapes to `[world_size, shard_size]`, and calls `reduce_scatter`/`reduce_scatter_tensor`.
- The resulting shard is divided by `world_size` (averaging semantics) and assigned to the flat sharded parameter’s `.grad`.
- Local parameter grads are cleared and full params are released again. This keeps optimizer state and step updates sharded without redundant collectives.

### Checkpointing
- `state_dict()` (invoked only on rank zero) gathers the full parameter vector (via `_gather_full_flat`) and populates module parameters before delegating to the wrapped module’s state dict.
- `load_state_dict()` does the inverse: load into the module, flatten, pad, and scatter the shards back into `self._flat_param`.
- The trainer detects whether the manual or official FSDP path is active and saves either the manual state dict (rank-zero `torch.save`) or the official `FULL_STATE_DICT` + `FSDP.optim_state_dict`.

## Integration in Training Loop
- `FsdpConfig.implementation` picks between `ManualFSDP` and the official wrapper at runtime (`run_fsdp.py --fsdp-impl {manual,torch}`).
- When manual FSDP is active:
  - The optimizer receives `list(fsdp_model.sharded_parameters())` instead of `model.parameters()`.
  - After each gradient-accumulation boundary we call `fsdp_model.sync_gradients()` once to reduce network chatter.
  - Grad clipping uses `torch.nn.utils.clip_grad_norm_` over the sharded parameters.
- Metrics label the mode as `fsdp-manual` vs `fsdp-torch`, aiding downstream analysis.
- Telemetry injected into the training loop logs peak memory, average data-loader wait, and manual collective latency so we can quantify the implementation’s overhead.
- Optional `--profile-steps/--profile-dir` flags piggyback on the manual FSDP path too, producing Chrome traces that clearly show the spiky all-gather / reduce-scatter regions.

## Communication Primitives Used
- `dist.all_gather_into_tensor` / `dist.all_gather` to assemble full parameter vectors.
- `dist.reduce_scatter_tensor` / `dist.reduce_scatter` to distribute gradients.
- `dist.broadcast_object_list` handles epoch/step synchronization when resuming checkpoints.
- `torch.profiler` (optional) wraps both CPU and CUDA activities so we can visualize the manual collectives alongside forward/backward compute.

## Trade-offs & Limitations
- **Performance**: without nested auto-wrap policies or per-layer sharding, the manual approach pays extra overhead gathering the entire model each forward pass. This is acceptable for pedagogy but slower than production FSDP.
- **Memory**: still significantly lower than DDP because only the shard is an optimizer parameter, yet temporary full weights exist during the forward/backward window.
- **Features**: does not currently support activation checkpoint integration at the wrapper layer (handled at the model level) nor per-layer shard policies; however, the implementation is concise and easy to extend.

## Validation Plan
- Compare throughput, loss curves (on the modular-addition synthetic dataset), and memory footprints between the single-process baseline (`single_gpu_fp32`), `ddp_fp32`, `fsdp_manual_bf16`, and `fsdp_official` using `experiments/run_experiments.py` + `experiments/analyze_metrics.py`.
- Ensure checkpoint save/resume works on both paths by running short restarts with `--resume`.

## Conclusion
The manual FSDP wrapper gives full visibility into how sharding works—parameter flattening, collective communication, and optimizer interaction—while remaining API-compatible with the existing trainer. This sets the stage for rigorous experiments and an educational narrative in the final report.

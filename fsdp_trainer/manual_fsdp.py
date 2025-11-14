"""Manual FSDP implementation that uses explicit collectives instead of torch FSDP."""
from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Iterable, List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters


class ManualFSDP(nn.Module):
    """A minimal FSDP-style wrapper implemented manually with collectives.

    The wrapper keeps a sharded flat parameter per rank. Before each forward it
    all-gathers the full parameter vector, materializes the original module
    parameters, and after backward it reduce-scatters gradients back onto the
    local shard so the optimizer only steps on the sharded parameter.
    """

    def __init__(self, module: nn.Module, process_group: dist.ProcessGroup | None = None) -> None:
        super().__init__()
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError("ManualFSDP requires torch.distributed to be initialized")
        self.module = module
        pg = process_group or dist.group.WORLD
        if pg is None:
            raise RuntimeError("ManualFSDP could not find a default process group; pass one explicitly.")
        self.process_group = pg
        self.rank = dist.get_rank(self.process_group)
        self.world_size = dist.get_world_size(self.process_group)
        if self.world_size <= 0:
            raise RuntimeError("Invalid world size for ManualFSDP")

        params = [p for p in module.parameters()]
        if not params:
            raise RuntimeError("ManualFSDP expects module with parameters")
        self._managed_params: List[torch.nn.Parameter] = params
        flat = parameters_to_vector([p.detach().clone() for p in params])
        self._total_numel = flat.numel()
        self._shard_size = math.ceil(self._total_numel / self.world_size)
        self._padded_numel = self._shard_size * self.world_size
        if self._padded_numel > self._total_numel:
            flat = F.pad(flat, (0, self._padded_numel - self._total_numel))
        shards = flat.view(self.world_size, self._shard_size)
        local_shard = shards[self.rank].contiguous().to(flat.device)
        self._flat_param = nn.Parameter(local_shard)
        self._materialized = False
        # Break the link so the managed module parameters no longer own storage.
        for p in self._managed_params:
            p.detach_()
            p.zero_()

    def sharded_parameters(self) -> Iterable[nn.Parameter]:
        yield self._flat_param

    def extra_repr(self) -> str:
        return f"world_size={self.world_size}, shard_size={self._shard_size}, total={self._total_numel}"

    def forward(self, *args, **kwargs):  # type: ignore[override]
        self._materialize_full_params()
        output = self.module(*args, **kwargs)
        if not torch.is_grad_enabled():
            self._release_full_params()
        return output

    def sync_gradients(self) -> None:
        """Reduce-scatter gradients from full params onto the local shard."""
        if not self._materialized:
            # Nothing to do if we ran in eval/no-grad mode.
            return
        grads = []
        for param in self._managed_params:
            grad = param.grad
            if grad is None:
                grad = torch.zeros_like(param, device=self._flat_param.device)
            grads.append(grad.reshape(-1).to(self._flat_param.device))
        flat_grad = torch.cat(grads)
        if flat_grad.numel() < self._padded_numel:
            flat_grad = F.pad(flat_grad, (0, self._padded_numel - flat_grad.numel()))
        flat_grad = flat_grad.view(self.world_size, self._shard_size).contiguous()
        local_grad = torch.zeros_like(self._flat_param)
        if hasattr(dist, "reduce_scatter_tensor"):
            dist.reduce_scatter_tensor(
                local_grad,
                flat_grad,
                op=dist.ReduceOp.SUM,
                group=self.process_group,
            )
        else:
            scatter_list = list(flat_grad.unbind(0))
            dist.reduce_scatter(
                local_grad,
                scatter_list,
                op=dist.ReduceOp.SUM,
                group=self.process_group,
            )
        local_grad.div_(self.world_size)
        self._flat_param.grad = local_grad
        for param in self._managed_params:
            param.grad = None
        self._release_full_params()

    def state_dict(self, *args, **kwargs):  # type: ignore[override]
        full_flat = self._gather_full_flat().detach()
        vector_to_parameters(full_flat, self._managed_params)
        state = self.module.state_dict(*args, **kwargs)
        self._release_full_params()
        return state

    def load_state_dict(self, state_dict, strict: bool = True):  # type: ignore[override]
        result = self.module.load_state_dict(state_dict, strict)
        flat = parameters_to_vector([p.detach() for p in self._managed_params])
        if flat.numel() < self._padded_numel:
            flat = F.pad(flat, (0, self._padded_numel - flat.numel()))
        shards = flat.view(self.world_size, self._shard_size)
        with torch.no_grad():
            self._flat_param.copy_(shards[self.rank].to(self._flat_param.device))
        self._release_full_params()
        return result

    @contextmanager
    def summon_full_params(self):
        self._materialize_full_params()
        try:
            yield
        finally:
            self._release_full_params()

    def _gather_full_flat(self) -> torch.Tensor:
        if self.world_size == 1:
            return self._flat_param[: self._total_numel].clone()
        gather_buffer = torch.empty(
            self.world_size,
            self._shard_size,
            device=self._flat_param.device,
            dtype=self._flat_param.dtype,
        )
        if hasattr(dist, "all_gather_into_tensor"):
            dist.all_gather_into_tensor(gather_buffer, self._flat_param, group=self.process_group)
        else:
            gather_list = list(gather_buffer.unbind(0))
            dist.all_gather(gather_list, self._flat_param, group=self.process_group)
        flat = gather_buffer.reshape(-1)
        return flat[: self._total_numel]

    def _materialize_full_params(self) -> None:
        if self._materialized:
            return
        full_flat = self._gather_full_flat()
        vector_to_parameters(full_flat, self._managed_params)
        self._materialized = True

    def _release_full_params(self) -> None:
        if not self._materialized:
            return
        for param in self._managed_params:
            param.detach_()
            param.zero_()
        self._materialized = False

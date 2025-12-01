"""Curated experiment definitions for comparing parallel training strategies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    entrypoint: str
    overrides: Dict[str, object]
    description: str
    tags: tuple[str, ...] = ()


def default_experiments() -> List[ExperimentSpec]:
    return [
        ExperimentSpec(
            name="single_gpu_fp32",
            entrypoint="run_single.py",
            description="Single-process baseline to measure pure compute throughput.",
            tags=("single", "baseline"),
            overrides={
                "runtime": {
                    "mixed_precision": "fp32",
                    "experiment_name": "single_gpu_fp32",
                }
            },
        ),
        ExperimentSpec(
            name="ddp_fp32",
            entrypoint="run_ddp.py",
            description="DDP baseline in full precision for throughput and loss reference.",
            tags=("ddp", "baseline"),
            overrides={
                "runtime": {
                    "mixed_precision": "fp32",
                    "experiment_name": "ddp_fp32",
                },
                "fsdp": {
                    "activation_checkpointing": False,
                },
            },
        ),
        ExperimentSpec(
            name="ddp_checkpointed_fp32",
            entrypoint="run_ddp.py",
            description="DDP + activation checkpointing while staying in fp32 (isolates checkpoint cost).",
            tags=("ddp", "checkpoint", "ablation"),
            overrides={
                "runtime": {
                    "mixed_precision": "fp32",
                    "experiment_name": "ddp_checkpointed_fp32",
                },
                "fsdp": {
                    "activation_checkpointing": True,
                },
            },
        ),
        ExperimentSpec(
            name="ddp_fp16",
            entrypoint="run_ddp.py",
            description="DDP in fp16 without checkpointing to isolate precision-only gains.",
            tags=("ddp", "precision"),
            overrides={
                "runtime": {
                    "mixed_precision": "fp16",
                    "experiment_name": "ddp_fp16",
                },
                "fsdp": {
                    "activation_checkpointing": False,
                },
            },
        ),
        ExperimentSpec(
            name="ddp_checkpointed_fp16",
            entrypoint="run_ddp.py",
            description="DDP baseline with activation checkpointing + FP16 to mimic ZeRO-style savings.",
            tags=("ddp", "checkpoint"),
            overrides={
                "runtime": {
                    "mixed_precision": "fp16",
                    "experiment_name": "ddp_checkpointed_fp16",
                },
                "fsdp": {
                    "activation_checkpointing": True,
                },
            },
        ),
        ExperimentSpec(
            name="fsdp_manual_bf16",
            entrypoint="run_fsdp.py",
            description="Manual FSDP implementation (this repo) running in bf16 for throughput gains.",
            tags=("fsdp", "manual"),
            overrides={
                "runtime": {
                    "mixed_precision": "bf16",
                    "experiment_name": "fsdp_manual_bf16",
                },
                "fsdp": {
                    "implementation": "manual",
                    "activation_checkpointing": True,
                    "limit_all_gathers": True,
                },
            },
        ),
        ExperimentSpec(
            name="fsdp_manual_fp32",
            entrypoint="run_fsdp.py",
            description="Manual FSDP in fp32 with the same checkpoint/all-gather toggles (precision ablation).",
            tags=("fsdp", "manual", "precision"),
            overrides={
                "runtime": {
                    "mixed_precision": "fp32",
                    "experiment_name": "fsdp_manual_fp32",
                },
                "fsdp": {
                    "implementation": "manual",
                    "activation_checkpointing": True,
                    "limit_all_gathers": True,
                },
            },
        ),
        ExperimentSpec(
            name="fsdp_manual_bf16_no_checkpoint",
            entrypoint="run_fsdp.py",
            description="Manual FSDP in bf16 without activation checkpointing to isolate its impact.",
            tags=("fsdp", "manual", "checkpoint"),
            overrides={
                "runtime": {
                    "mixed_precision": "bf16",
                    "experiment_name": "fsdp_manual_bf16_no_checkpoint",
                },
                "fsdp": {
                    "implementation": "manual",
                    "activation_checkpointing": False,
                    "limit_all_gathers": True,
                },
            },
        ),
        ExperimentSpec(
            name="fsdp_manual_bf16_no_limit_all_gathers",
            entrypoint="run_fsdp.py",
            description="Manual FSDP in bf16 disabling limit_all_gathers to measure comm overhead.",
            tags=("fsdp", "manual", "ablation"),
            overrides={
                "runtime": {
                    "mixed_precision": "bf16",
                    "experiment_name": "fsdp_manual_bf16_no_limit_all_gathers",
                },
                "fsdp": {
                    "implementation": "manual",
                    "activation_checkpointing": True,
                    "limit_all_gathers": False,
                },
            },
        ),
        # ExperimentSpec(
        #     name="fsdp_official",
        #     entrypoint="run_fsdp.py",
        #     description="torch.distributed.fsdp configuration mirroring the official tutorial (size wrap, fp32).",
        #     tags=("fsdp", "official"),
        #     overrides={
        #         "runtime": {
        #             "mixed_precision": "fp32",
        #             "experiment_name": "fsdp_official",
        #         },
        #         "fsdp": {
        #             "implementation": "torch",
        #             "auto_wrap_policy": "size",
        #             "activation_checkpointing": True,
        #             "limit_all_gathers": True,
        #             "use_orig_params": True,
        #             "cpu_offload": False,
        #         },
        #     },
        # ),
        # ExperimentSpec(
        #     name="fsdp_official_bf16",
        #     entrypoint="run_fsdp.py",
        #     description="Official torch FSDP running in bf16 (precision ablation relative to fp32 baseline).",
        #     tags=("fsdp", "official", "precision"),
        #     overrides={
        #         "runtime": {
        #             "mixed_precision": "bf16",
        #             "experiment_name": "fsdp_official_bf16",
        #         },
        #         "fsdp": {
        #             "implementation": "torch",
        #             "auto_wrap_policy": "size",
        #             "activation_checkpointing": True,
        #             "limit_all_gathers": True,
        #             "use_orig_params": True,
        #             "cpu_offload": False,
        #         },
        #     },
        # ),
        # ExperimentSpec(
        #     name="fsdp_official_no_checkpoint",
        #     entrypoint="run_fsdp.py",
        #     description="Official FSDP ablation: disable activation checkpointing to inspect memory/perf impact.",
        #     tags=("fsdp", "official", "ablation"),
        #     overrides={
        #         "runtime": {
        #             "mixed_precision": "fp32",
        #             "experiment_name": "fsdp_official_no_checkpoint",
        #         },
        #         "fsdp": {
        #             "implementation": "torch",
        #             "auto_wrap_policy": "size",
        #             "activation_checkpointing": False,
        #             "limit_all_gathers": True,
        #             "use_orig_params": True,
        #         },
        #     },
        # ),
        # ExperimentSpec(
        #     name="fsdp_official_no_limit_all_gathers",
        #     entrypoint="run_fsdp.py",
        #     description="Official FSDP ablation: disable limit_all_gathers to expose comm overhead.",
        #     tags=("fsdp", "official", "ablation"),
        #     overrides={
        #         "runtime": {
        #             "mixed_precision": "fp32",
        #             "experiment_name": "fsdp_official_no_limit_all_gathers",
        #         },
        #         "fsdp": {
        #             "implementation": "torch",
        #             "auto_wrap_policy": "size",
        #             "activation_checkpointing": True,
        #             "limit_all_gathers": False,
        #             "use_orig_params": True,
        #         },
        #     },
        # ),
    ]


__all__ = ["ExperimentSpec", "default_experiments"]

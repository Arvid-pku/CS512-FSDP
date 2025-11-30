"""Configuration dataclasses for the FSDP training stack."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple


@dataclass
class ModelConfig:
    """Hyperparameters that define the transformer language model."""

    vocab_size: int = 16384
    embed_dim: int = 1024
    num_layers: int = 12
    num_heads: int = 16
    ff_hidden_dim: int = 4096
    dropout: float = 0.1
    max_seq_len: int = 128
    tie_embedding_weights: bool = True


@dataclass
class DataConfig:
    """Parameters describing the dataset and data-loading strategy."""

    dataset_impl: str = "synthetic"
    total_samples: int = 81920
    seq_len: int = 128
    vocab_size: int = 16384
    num_workers: int = 2
    seed: int = 42
    pattern_period: int = 5
    noise_prob: float = 0.05


@dataclass
class OptimizerConfig:
    """Optimizer and scheduler hyperparameters."""

    lr: float = 2e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    warmup_steps: int = 200
    max_steps: int = 1000
    clip_grad_norm: float = 1.0


@dataclass
class RuntimeConfig:
    """Runtime toggles that are orthogonal to the core algorithm."""

    seed: int = 1337
    backend: str = "nccl"
    log_every: int = 100
    ckpt_every: int = 4000
    ckpt_dir: str = "artifacts/checkpoints"
    resume_path: str | None = None
    mixed_precision: str = "bf16"  # bf16 | fp16 | fp32
    use_compile: bool = False
    experiment_name: str = "fsdp_demo"
    metrics_path: str | None = None
    profile_steps: int = 0
    profile_dir: str | None = None


@dataclass
class FsdpConfig:
    """FSDP-specific knobs."""

    implementation: str = "manual"  # manual | torch
    sharding_strategy: str = "FULL_SHARD"
    auto_wrap_policy: str = "transformer"
    min_params_to_wrap: int = 1_000_000
    sync_module_states: bool = True
    cpu_offload: bool = False
    limit_all_gathers: bool = True
    activation_checkpointing: bool = True
    use_orig_params: bool = True


@dataclass
class TrainConfig:
    """Top-level configuration object aggregated from the smaller pieces."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    fsdp: FsdpConfig = field(default_factory=FsdpConfig)
    epochs: int = 3
    batch_size_per_device: int = 8
    grad_accum_steps: int = 1

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

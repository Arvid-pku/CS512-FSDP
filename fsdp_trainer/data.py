"""Dataset and dataloader utilities for FSDP training."""
from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from .config import DataConfig


class SyntheticLanguageModelingDataset(Dataset):
    """Generates synthetic token sequences with a learnable repeating pattern.

    Pattern definition:
      - Input tokens are random integers in [0, vocab_size).
      - The target token is (input_token + pattern_period) % vocab_size
        corrupted with probability `noise_prob`.

    Because the mapping is deterministic most of the time, models can reduce loss
    by learning the modular addition rule while remaining robust to injected noise.
    """

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg
        self.offset = cfg.pattern_period % cfg.vocab_size

    def __len__(self) -> int:
        return self.cfg.total_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator()
        generator.manual_seed(self.cfg.seed + index)
        inputs = torch.randint(
            low=0,
            high=self.cfg.vocab_size,
            size=(self.cfg.seq_len,),
            generator=generator,
            dtype=torch.long,
        )
        targets = (inputs + self.offset) % self.cfg.vocab_size
        if self.cfg.noise_prob > 0:
            noise_mask = torch.rand(self.cfg.seq_len, generator=generator) < self.cfg.noise_prob
            noise = torch.randint(
                0,
                self.cfg.vocab_size,
                size=(self.cfg.seq_len,),
                generator=generator,
                dtype=torch.long,
            )
            targets = torch.where(noise_mask, noise, targets)
        return inputs, targets


def build_dataloader(
    cfg: DataConfig,
    batch_size: int,
    rank: int,
    world_size: int,
    shuffle: bool = True,
) -> DataLoader:
    dataset: Dataset
    if cfg.dataset_impl == "synthetic":
        dataset = SyntheticLanguageModelingDataset(cfg)
    else:
        raise ValueError(f"Unsupported dataset_impl={cfg.dataset_impl}")

    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle,
        drop_last=True,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        pin_memory=torch.cuda.is_available(),
        num_workers=cfg.num_workers,
    )

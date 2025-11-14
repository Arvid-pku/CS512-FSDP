"""Dataset and dataloader utilities for FSDP training."""
from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

from .config import DataConfig


class SyntheticLanguageModelingDataset(Dataset):
    """Generates synthetic token sequences on the fly.

    Each sample is a tuple (input_tokens, target_tokens) shifted by one position
    for next-token prediction.
    """

    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg

    def __len__(self) -> int:
        return self.cfg.total_samples

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        generator = torch.Generator()
        generator.manual_seed(self.cfg.seed + index)
        tokens = torch.randint(
            low=0,
            high=self.cfg.vocab_size,
            size=(self.cfg.seq_len + 1,),
            generator=generator,
            dtype=torch.long,
        )
        return tokens[:-1], tokens[1:]


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

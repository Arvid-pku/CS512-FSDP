"""Simple transformer language model used for the FSDP example."""
from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn

from .config import ModelConfig


class PositionalEncoding(nn.Module):
    """Deterministic sinusoidal positional embedding."""

    def __init__(self, embed_dim: int, dropout: float, max_len: int) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embed_dim, 2) * (-(math.log(10000.0) / embed_dim))
        )
        pe = torch.zeros(max_len, embed_dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("positional_encoding", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = x + self.positional_encoding[:, : x.size(1)]
        return self.dropout(x)


class TransformerLanguageModel(nn.Module):
    """A compact GPT-style model that still stresses sharding."""

    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_emb = PositionalEncoding(cfg.embed_dim, cfg.dropout, cfg.max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.embed_dim,
            nhead=cfg.num_heads,
            dim_feedforward=cfg.ff_hidden_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.num_layers)
        self.final_norm = nn.LayerNorm(cfg.embed_dim)
        self.lm_head = nn.Linear(cfg.embed_dim, cfg.vocab_size, bias=False)
        if cfg.tie_embedding_weights:
            self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.token_emb(tokens)
        x = self.pos_emb(x)
        x = self.encoder(x)
        x = self.final_norm(x)
        return self.lm_head(x)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["TransformerLanguageModel"]

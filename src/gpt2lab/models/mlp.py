"""Position-wise feed-forward network."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn

from gpt2lab.config import ModelConfig
from gpt2lab.models.initialization import mark_residual_projection

__all__ = ["MLP"]


class MLP(nn.Module):
    """Two-layer MLP applied independently to every token position."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        hidden_size = config.mlp_hidden_size

        self.c_fc = nn.Linear(config.n_embd, hidden_size, bias=config.bias)
        self.gelu = nn.GELU(approximate="tanh")

        self.c_proj = nn.Linear(hidden_size, config.n_embd, bias=config.bias)
        mark_residual_projection(self.c_proj)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return cast(torch.Tensor, self.dropout(x))

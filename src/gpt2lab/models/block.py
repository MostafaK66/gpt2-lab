"""A single pre-normalization transformer block."""

from __future__ import annotations

import torch
import torch.nn as nn

from gpt2lab.config import ModelConfig
from gpt2lab.models.attention import CausalSelfAttention
from gpt2lab.models.mlp import MLP

__all__ = ["TransformerBlock"]


class TransformerBlock(nn.Module):
    """LayerNorm -> attention -> residual, then LayerNorm -> MLP -> residual.

    Normalizing *inside* each residual branch keeps the residual stream a clean
    additive path, which is what makes deep stacks trainable without warm-up
    tricks on the skip connections.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


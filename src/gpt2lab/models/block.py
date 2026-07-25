"""Transformer block = LayerNorm + Attention + LayerNorm + MLP."""
from __future__ import annotations

import torch.nn as nn

from ..config.sections import ModelConfig
from .attention import CausalSelfAttention
from .mlp import MLP


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.ln_1 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.attn = CausalSelfAttention(cfg)
        self.ln_2 = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.mlp = MLP(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


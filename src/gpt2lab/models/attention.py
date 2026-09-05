"""Multi-head causal self-attention."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt2lab.config import ModelConfig
from gpt2lab.models.initialization import mark_residual_projection

__all__ = ["CausalSelfAttention"]


class CausalSelfAttention(nn.Module):
    """Scaled dot-product attention with a causal mask.

    Input  : ``[B, T, C]``
    Output : ``[B, T, C]``
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_size = config.head_size

        # Query, key and value projections fused into one matmul.
        # Output dimension must be exactly 3 * n_embd so the split produces
        # three equal tensors: query, key, value — each of size n_embd.
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)

        # Recombines the per-head outputs; this is the residual output
        # projection, hence the scaled initialization.
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        mark_residual_projection(self.c_proj)
        self.dropout = config.dropout
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, time, channels = x.shape

        # [B, T, 3C] -> three tensors of [B, T, C]
        query, key, value = self.c_attn(x).split(self.n_embd, dim=2)

        # [B, T, C] -> [B, n_head, T, head_size]
        query = self._split_heads(query, batch, time)
        key = self._split_heads(key, batch, time)
        value = self._split_heads(value, batch, time)

        # is_causal=True lets every token attend to itself and earlier tokens
        # only, without materialising an explicit mask.
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )

        # [B, n_head, T, head_size] -> [B, T, C]
        merged = (
            attended.transpose(1, 2).contiguous().reshape(batch, time, channels)
        )
        return cast(torch.Tensor, self.resid_dropout(self.c_proj(merged)))

    def _split_heads(
        self, tensor: torch.Tensor, batch: int, time: int
    ) -> torch.Tensor:
        return tensor.reshape(batch, time, self.n_head, self.head_size).transpose(1, 2)

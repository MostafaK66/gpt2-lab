"""Decoder-only GPT language model."""

from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from gpt2lab.config import ModelConfig
from gpt2lab.models.block import TransformerBlock
from gpt2lab.models.initialization import WeightInitializer

__all__ = ["GPT"]


class GPT(nn.Module):
    """GPT-2 architecture: learned positions, pre-norm blocks, tied embeddings."""

    def __init__(self, config: ModelConfig, vocab_size: int) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive (received {vocab_size}).")

        self.config = config
        self.vocab_size = vocab_size

        self.transformer = nn.ModuleDict(
            {
                "wte": nn.Embedding(vocab_size, config.n_embd),
                "wpe": nn.Embedding(config.block_size, config.n_embd),
                "h": nn.ModuleList(
                    [TransformerBlock(config) for _ in range(config.n_layer)]
                ),
                "ln_f": nn.LayerNorm(config.n_embd),
            }
        )

        self.lm_head = nn.Linear(config.n_embd, vocab_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)

        self.apply(WeightInitializer(config.init_std, config.n_layer))

        # Weight tying happens *after* initialization so the shared matrix
        # carries exactly one draw (the token-embedding one) instead of being
        # overwritten a second time through the lm_head alias.
        self.lm_head.weight = self._token_embeddings.weight

    @property
    def _token_embeddings(self) -> nn.Embedding:
        return cast(nn.Embedding, self.transformer["wte"])

    @property
    def _position_embeddings(self) -> nn.Embedding:
        return cast(nn.Embedding, self.transformer["wpe"])

    @property
    def _blocks(self) -> nn.ModuleList:
        return cast(nn.ModuleList, self.transformer["h"])

    @property
    def _final_norm(self) -> nn.LayerNorm:
        return cast(nn.LayerNorm, self.transformer["ln_f"])

    # -- introspection ------------------------------------------------------

    @property
    def weights_are_tied(self) -> bool:
        return self.lm_head.weight is self._token_embeddings.weight

    def num_parameters(self, include_embeddings: bool = True) -> int:
        """Count trainable parameters.

        The tied token-embedding / lm_head matrix is counted once, because
        ``parameters()`` de-duplicates shared tensors.
        """
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        if not include_embeddings:
            total -= self._position_embeddings.weight.numel()
            total -= self._token_embeddings.weight.numel()
        return total

    # -- forward ------------------------------------------------------------

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Run the model and optionally compute next-token cross-entropy.

        Args:
            idx: input token ids, shape ``[B, T]``.
            targets: correct next-token ids, shape ``[B, T]``.

        Returns:
            ``(logits, loss)`` where ``logits`` has shape ``[B, T, vocab_size]``
            and ``loss`` is ``None`` when ``targets`` is omitted.
        """
        if idx.ndim != 2:
            raise ValueError(f"idx must have shape [batch, time]; got {tuple(idx.shape)}")
        if idx.dtype != torch.long:
            raise ValueError("idx must have torch.long dtype")
        if targets is not None and targets.shape != idx.shape:
            raise ValueError("targets must have the same shape as idx")
        if targets is not None and targets.dtype != torch.long:
            raise ValueError("targets must have torch.long dtype")

        _, time = idx.shape

        if time == 0:
            raise ValueError("input sequence must not be empty")

        if time > self.config.block_size:
            raise ValueError(
                f"Sequence length {time} exceeds block_size "
                f"{self.config.block_size}."
            )

        positions = torch.arange(time, dtype=torch.long, device=idx.device)

        token_embeddings = self._token_embeddings(idx)  # [B, T, C]
        position_embeddings = self._position_embeddings(positions)  # [T, C]

        # [T, C] broadcasts across the batch dimension.
        x = self.dropout(token_embeddings + position_embeddings)

        for block in self._blocks:
            x = block(x)

        x = self._final_norm(x)
        logits = self.lm_head(x)

        loss: torch.Tensor | None = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
            )

        return logits, loss

    # -- sampling -----------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """Autoregressively extend ``idx`` by ``max_new_tokens`` tokens.

        The training mode of the module is restored before returning.
        """
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive.")
        if temperature <= 0:
            raise ValueError("temperature must be greater than zero.")
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive when provided.")

        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                context = idx[:, -self.config.block_size :]
                logits, _ = self(context)
                logits = logits[:, -1, :] / temperature

                if top_k is not None:
                    k = min(top_k, logits.size(-1))
                    kth_value = torch.topk(logits, k, dim=-1).values[:, -1:]
                    logits = logits.masked_fill(logits < kth_value, float("-inf"))

                probabilities = F.softmax(logits.float(), dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
                idx = torch.cat((idx, next_token), dim=1)
        finally:
            self.train(was_training)

        return idx

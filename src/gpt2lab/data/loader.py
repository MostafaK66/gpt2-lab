"""Deterministic sequential batches for next-token prediction."""

from __future__ import annotations

from collections.abc import Iterator

import torch


class SequentialTokenLoader:
    """Walk a flat token tensor and skip incomplete trailing batches."""

    def __init__(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        *,
        pin_memory: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive")
        if tokens.ndim != 1:
            raise ValueError(f"tokens must be 1-D; got shape {tuple(tokens.shape)}")
        if tokens.dtype != torch.long:
            raise ValueError("tokens must have torch.long dtype")
        required = batch_size * sequence_length + 1
        if tokens.numel() < required:
            raise ValueError(
                f"token stream has {tokens.numel()} values; a batch requires {required}"
            )
        self._tokens = (
            tokens.pin_memory() if pin_memory and torch.cuda.is_available() else tokens
        )
        self._batch_size = batch_size
        self._sequence_length = sequence_length
        self._position = 0

    @property
    def tokens_per_batch(self) -> int:
        return self._batch_size * self._sequence_length

    @property
    def batches_per_epoch(self) -> int:
        return (self._tokens.numel() - 1) // self.tokens_per_batch

    @property
    def position(self) -> int:
        return self._position

    def reset(self) -> None:
        self._position = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        span = self.tokens_per_batch
        buffer = self._tokens[self._position : self._position + span + 1]
        inputs = buffer[:-1].reshape(self._batch_size, self._sequence_length)
        targets = buffer[1:].reshape(self._batch_size, self._sequence_length)
        self._position += span
        if self._position + span + 1 > self._tokens.numel():
            self.reset()
        return inputs, targets

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        while True:
            yield self.next_batch()

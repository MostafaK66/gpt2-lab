"""Sequential mini-batch loader for next-token prediction."""

from __future__ import annotations

from typing import Iterator

import torch

from gpt2lab.utils.logging import get_logger

__all__ = ["SequentialTokenLoader"]

LOGGER = get_logger(__name__)


class SequentialTokenLoader:
    """Walks a flat token tensor and yields ``(inputs, targets)`` pairs.

    ``targets`` is ``inputs`` shifted by one position, so a batch consumes
    ``B * T + 1`` tokens but advances the cursor by ``B * T``.  When the next
    batch would run past the end of the corpus the cursor wraps to zero, which
    means the trailing partial batch is skipped rather than padded.

    The loader deliberately stays on the CPU; moving tensors to the accelerator
    is the trainer's responsibility.
    """

    def __init__(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        sequence_length: int,
        pin_memory: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive (received {batch_size}).")
        if sequence_length <= 0:
            raise ValueError(
                f"sequence_length must be positive (received {sequence_length})."
            )
        if tokens.dim() != 1:
            raise ValueError(
                f"tokens must be a 1-D tensor (received shape {tuple(tokens.shape)})."
            )

        required = batch_size * sequence_length + 1
        if tokens.numel() < required:
            raise ValueError(
                f"The corpus holds {tokens.numel():,} tokens but a single batch "
                f"needs at least {required:,}. Reduce batch_size or "
                f"sequence_length, or use a larger corpus."
            )

        if pin_memory:
            if torch.cuda.is_available():
                tokens = tokens.pin_memory()
            else:
                LOGGER.warning("pin_memory requested but CUDA is unavailable; ignoring.")
                pin_memory = False

        self._tokens = tokens
        self._batch_size = batch_size
        self._sequence_length = sequence_length
        self._pinned = pin_memory
        self._position = 0

        LOGGER.info(
            "Loader ready: %s tokens, batch shape [%d, %d], %s tokens/batch, "
            "~%s batches per pass (pinned=%s)",
            f"{tokens.numel():,}",
            batch_size,
            sequence_length,
            f"{self.tokens_per_batch:,}",
            f"{self.batches_per_epoch:,}",
            pin_memory,
        )

    # -- properties ---------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def sequence_length(self) -> int:
        return self._sequence_length

    @property
    def tokens_per_batch(self) -> int:
        return self._batch_size * self._sequence_length

    @property
    def total_tokens(self) -> int:
        return self._tokens.numel()

    @property
    def batches_per_epoch(self) -> int:
        return (self.total_tokens - 1) // self.tokens_per_batch

    @property
    def position(self) -> int:
        return self._position

    @property
    def is_pinned(self) -> bool:
        return self._pinned

    # -- behaviour ----------------------------------------------------------

    def reset(self) -> None:
        """Rewind the cursor to the start of the corpus."""
        self._position = 0

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        span = self.tokens_per_batch
        start = self._position
        buffer = self._tokens[start : start + span + 1]

        inputs = buffer[:-1].reshape(self._batch_size, self._sequence_length)
        targets = buffer[1:].reshape(self._batch_size, self._sequence_length)

        self._position += span
        if self._position + span + 1 > self.total_tokens:
            self.reset()

        return inputs, targets

    def __iter__(self) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Infinite iterator over batches."""
        while True:
            yield self.next_batch()


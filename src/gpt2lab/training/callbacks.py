"""Optional observers for training progress."""

from __future__ import annotations

import sys
from typing import Protocol, TextIO

from gpt2lab.training.metrics import Metrics


class Callback(Protocol):
    def on_step_end(self, step: int, metrics: Metrics) -> None: ...

    def on_train_end(self, metrics: Metrics) -> None: ...


class PrintCallback:
    """Write concise progress messages to an injected text stream."""

    def __init__(self, every: int, stream: TextIO | None = None) -> None:
        if every <= 0:
            raise ValueError("every must be positive")
        self.every = every
        self.stream = stream or sys.stdout

    def on_step_end(self, step: int, metrics: Metrics) -> None:
        if (step + 1) % self.every == 0 and metrics.steps:
            print(
                f"step {step + 1:5d} | loss {metrics.train_losses[-1]:.4f} | "
                f"lr {metrics.learning_rates[-1]:.2e}",
                file=self.stream,
            )

    def on_train_end(self, metrics: Metrics) -> None:
        print(f"Training complete ({len(metrics.steps)} steps).", file=self.stream)

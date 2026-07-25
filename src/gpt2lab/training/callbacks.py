"""Training-loop callbacks."""
from __future__ import annotations

from typing import Protocol

from .metrics import Metrics


class Callback(Protocol):
    def on_step_end(self, step: int, metrics: Metrics) -> None: ...
    def on_train_end(self, metrics: Metrics) -> None: ...


class PrintCallback:
    def __init__(self, log_interval: int) -> None:
        self.log_interval = log_interval

    def on_step_end(self, step: int, metrics: Metrics) -> None:
        if step % self.log_interval == 0 and metrics.steps:
            lr = metrics.learning_rates[-1]
            loss = metrics.train_losses[-1]
            print(f"step {step:5d} | loss {loss:.4f} | lr {lr:.2e}")

    def on_train_end(self, metrics: Metrics) -> None:
        print("Training complete.")


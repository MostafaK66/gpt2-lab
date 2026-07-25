"""Training metrics accumulation."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Metrics:
    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    val_steps: list[int] = field(default_factory=list)

    def log_train(self, step: int, loss: float, lr: float) -> None:
        self.steps.append(step)
        self.train_losses.append(loss)
        self.learning_rates.append(lr)

    def log_val(self, step: int, loss: float) -> None:
        self.val_steps.append(step)
        self.val_losses.append(loss)


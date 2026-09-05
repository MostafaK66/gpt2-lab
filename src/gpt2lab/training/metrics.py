"""Training metrics and deterministic CSV persistence."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(slots=True)
class Metrics:
    """In-memory history for training and validation measurements."""

    train_losses: list[float] = field(default_factory=list)
    val_losses: list[float] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    val_steps: list[int] = field(default_factory=list)

    def log_train(self, step: int, loss: float, learning_rate: float) -> None:
        self.steps.append(step)
        self.train_losses.append(loss)
        self.learning_rates.append(learning_rate)

    def log_validation(self, step: int, loss: float) -> None:
        self.val_steps.append(step)
        self.val_losses.append(loss)

    def write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        validation = dict(zip(self.val_steps, self.val_losses, strict=True))
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("step", "train_loss", "learning_rate", "validation_loss"))
            for step, loss, rate in zip(
                self.steps, self.train_losses, self.learning_rates, strict=True
            ):
                writer.writerow((step, loss, rate, validation.get(step, "")))

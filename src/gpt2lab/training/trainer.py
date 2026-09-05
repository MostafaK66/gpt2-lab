"""Device-aware, testable GPT training loop."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Protocol

import torch

from gpt2lab.config import ExperimentConfig
from gpt2lab.errors import TrainingError
from gpt2lab.models import GPT
from gpt2lab.training.callbacks import Callback
from gpt2lab.training.checkpoint import CheckpointManager
from gpt2lab.training.metrics import Metrics
from gpt2lab.training.optim import build_optimizer, learning_rate_at
from gpt2lab.training.runtime import autocast_context


class BatchSource(Protocol):
    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]: ...


class Trainer:
    """Coordinate optimization while keeping I/O and device choices injectable."""

    def __init__(
        self,
        config: ExperimentConfig,
        model: GPT,
        train_loader: BatchSource,
        validation_loader: BatchSource,
        device: torch.device,
        *,
        callbacks: list[Callback] | None = None,
        checkpoint_manager: CheckpointManager | None = None,
    ) -> None:
        self.config = config
        self.device = device
        self.model = model.to(device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.optimizer = build_optimizer(self.model, config.optimizer)
        self.callbacks = list(callbacks or [])
        self.checkpoints = checkpoint_manager or CheckpointManager(
            config.training.checkpoint_dir,
            config.training.keep_last_checkpoints,
        )
        self.metrics = Metrics()
        self.scaler = torch.cuda.amp.GradScaler(
            enabled=config.runtime.use_mixed_precision and device.type == "cuda"
        )

    def _move_batch(self, source: BatchSource) -> tuple[torch.Tensor, torch.Tensor]:
        inputs, targets = source.next_batch()
        non_blocking = self.device.type == "cuda"
        return (
            inputs.to(self.device, non_blocking=non_blocking),
            targets.to(self.device, non_blocking=non_blocking),
        )

    @torch.no_grad()
    def estimate_loss(self) -> float:
        was_training = self.model.training
        self.model.eval()
        losses: list[float] = []
        try:
            for _ in range(self.config.training.eval_steps):
                inputs, targets = self._move_batch(self.validation_loader)
                with autocast_context(self.device, self.config.runtime):
                    _, loss = self.model(inputs, targets)
                if loss is None:
                    raise TrainingError("model did not return validation loss")
                losses.append(float(loss.detach()))
        finally:
            self.model.train(was_training)
        return sum(losses) / len(losses)

    def run(self, resume_from: Path | None = None) -> Metrics:
        start_step = 0
        if resume_from is not None:
            completed_step, self.metrics = self.checkpoints.restore(
                resume_from, self.model, self.optimizer, self.device
            )
            start_step = completed_step + 1
            if start_step > self.config.training.num_steps:
                raise TrainingError("checkpoint is beyond configured training length")

        self.model.train()
        for step in range(start_step, self.config.training.num_steps):
            learning_rate = learning_rate_at(
                step, self.config.training.num_steps, self.config.optimizer
            )
            for group in self.optimizer.param_groups:
                group["lr"] = learning_rate

            inputs, targets = self._move_batch(self.train_loader)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast_context(self.device, self.config.runtime):
                _, loss = self.model(inputs, targets)
            if loss is None:
                raise TrainingError("model did not return training loss")
            loss_value = float(loss.detach())
            if self.config.training.stop_on_non_finite_loss and not math.isfinite(
                loss_value
            ):
                raise TrainingError(f"non-finite loss at step {step}: {loss_value}")

            self.scaler.scale(loss).backward()
            if self.config.optimizer.grad_clip_norm is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.optimizer.grad_clip_norm
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.metrics.log_train(step, loss_value, learning_rate)

            if self._is_due(step, self.config.training.eval_every):
                self.metrics.log_validation(step, self.estimate_loss())
            if self._is_due(step, self.config.training.checkpoint_every):
                self._save(f"step_{step + 1:08d}.pt", step)
            for callback in self.callbacks:
                callback.on_step_end(step, self.metrics)

        if self.config.training.metrics_csv is not None:
            self.metrics.write_csv(self.config.training.metrics_csv)
        if self.config.training.save_final_checkpoint:
            final_step = max(self.config.training.num_steps - 1, 0)
            self._save("final.pt", final_step)
        for callback in self.callbacks:
            callback.on_train_end(self.metrics)
        return self.metrics

    @staticmethod
    def _is_due(step: int, every: int | None) -> bool:
        return every is not None and (step + 1) % every == 0

    def _save(self, name: str, step: int) -> None:
        self.checkpoints.save(
            name,
            self.model,
            self.optimizer,
            step,
            self.config,
            self.metrics,
        )

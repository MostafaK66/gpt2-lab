"""Main training loop."""
from __future__ import annotations

import torch

from ..config.experiment import ExperimentConfig
from ..models.gpt import GPT
from ..data.loader import DataLoader
from .optim import build_optimizer, cosine_lr
from .runtime import select_device, get_autocast_ctx
from .metrics import Metrics
from .callbacks import Callback, PrintCallback
from .checkpoint import save_checkpoint, load_checkpoint


class Trainer:
    def __init__(
        self,
        cfg: ExperimentConfig,
        model: GPT,
        train_loader: DataLoader,
        val_loader: DataLoader,
        callbacks: list[Callback] | None = None,
    ) -> None:
        self.cfg = cfg
        self.device = select_device(cfg.training.device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = build_optimizer(model, cfg.training)
        self.metrics = Metrics()
        self.callbacks = callbacks or [PrintCallback(cfg.eval.log_interval)]
        self.scaler = torch.amp.GradScaler(enabled=(self.device.type == "cuda"))

    @torch.no_grad()
    def estimate_loss(self) -> float:
        self.model.eval()
        losses = []
        for _ in range(self.cfg.eval.eval_steps):
            x, y = self.val_loader.get_batch()
            with get_autocast_ctx(self.device):
                _, loss = self.model(x, y)
            losses.append(loss.item())
        self.model.train()
        return sum(losses) / len(losses)

    def run(self, resume_from: str | None = None) -> Metrics:
        start_step = 0
        if resume_from is not None:
            ckpt = load_checkpoint(resume_from, self.model, self.optimizer,
                                   str(self.device))
            start_step = ckpt.get("step", 0)

        self.model.train()
        for step in range(start_step, self.cfg.training.max_steps):
            # LR schedule
            lr = cosine_lr(step, self.cfg.training)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            x, y = self.train_loader.get_batch()
            with get_autocast_ctx(self.device):
                _, loss = self.model(x, y)

            self.scaler.scale(loss).backward()
            if self.cfg.training.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.cfg.training.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            self.metrics.log_train(step, loss.item(), lr)

            # eval
            if step > 0 and step % self.cfg.eval.eval_interval == 0:
                val_loss = self.estimate_loss()
                self.metrics.log_val(step, val_loss)

            # checkpoint
            if step > 0 and step % self.cfg.checkpoint.save_interval == 0:
                path = f"{self.cfg.checkpoint.checkpoint_dir}/step_{step}.pt"
                save_checkpoint(path, self.model, self.optimizer, step,
                                self.cfg.model, self.metrics)

            for cb in self.callbacks:
                cb.on_step_end(step, self.metrics)

        for cb in self.callbacks:
            cb.on_train_end(self.metrics)

        # final checkpoint
        path = f"{self.cfg.checkpoint.checkpoint_dir}/final.pt"
        save_checkpoint(path, self.model, self.optimizer,
                        self.cfg.training.max_steps, self.cfg.model, self.metrics)
        return self.metrics


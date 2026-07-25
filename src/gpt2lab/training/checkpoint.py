"""Checkpoint save / load."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

import torch

from ..config.sections import ModelConfig
from .metrics import Metrics


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    model_cfg: ModelConfig,
    metrics: Metrics,
) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "model_config": asdict(model_cfg),
            "metrics": {
                "train_losses": metrics.train_losses,
                "val_losses": metrics.val_losses,
                "steps": metrics.steps,
                "val_steps": metrics.val_steps,
                "learning_rates": metrics.learning_rates,
            },
        },
        path,
    )


def load_checkpoint(path: str, model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer | None = None,
                    device: str = "cpu") -> dict:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return ckpt


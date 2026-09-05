"""AdamW construction and learning-rate scheduling."""

from __future__ import annotations

import math

import torch

from gpt2lab.config import OptimizerConfig


def build_optimizer(model: torch.nn.Module, config: OptimizerConfig) -> torch.optim.AdamW:
    """Build AdamW with matrix-only weight decay."""
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(
        groups,
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.epsilon,
    )


def learning_rate_at(step: int, total_steps: int, config: OptimizerConfig) -> float:
    """Return the configured constant or warmup-plus-cosine learning rate."""
    if step < 0:
        raise ValueError("step cannot be negative")
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if config.schedule == "constant":
        return config.learning_rate
    if config.warmup_steps and step < config.warmup_steps:
        return config.learning_rate * (step + 1) / config.warmup_steps
    if step >= total_steps:
        return config.min_learning_rate
    decay_steps = total_steps - config.warmup_steps
    progress = (step - config.warmup_steps) / max(decay_steps, 1)
    coefficient = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.min_learning_rate + coefficient * (
        config.learning_rate - config.min_learning_rate
    )

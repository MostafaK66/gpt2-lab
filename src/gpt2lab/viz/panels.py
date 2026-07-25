"""Individual plot panels."""
from __future__ import annotations

from matplotlib.axes import Axes
from ..training.metrics import Metrics
from .axes import style_axis


def loss_panel(ax: Axes, metrics: Metrics) -> None:
    if metrics.steps:
        ax.plot(metrics.steps, metrics.train_losses, label="train", alpha=0.7)
    if metrics.val_steps:
        ax.plot(metrics.val_steps, metrics.val_losses, label="val", linewidth=2)
    style_axis(ax, title="Loss", xlabel="step", ylabel="loss")
    ax.legend(fontsize=7)


def lr_panel(ax: Axes, metrics: Metrics) -> None:
    if metrics.steps:
        ax.plot(metrics.steps, metrics.learning_rates, color="tab:orange")
    style_axis(ax, title="Learning Rate", xlabel="step", ylabel="lr")


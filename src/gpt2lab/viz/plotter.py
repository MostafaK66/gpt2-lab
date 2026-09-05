"""Live training plotter."""

from __future__ import annotations

import matplotlib.pyplot as plt

from ..training.callbacks import Callback
from ..training.metrics import Metrics
from .panels import loss_panel, lr_panel


class LivePlotter(Callback):
    def __init__(self, plot_interval: int = 50) -> None:
        if plot_interval <= 0:
            raise ValueError("plot_interval must be positive")
        self.plot_interval = plot_interval
        self.fig, self.axes = plt.subplots(1, 2, figsize=(12, 4))
        plt.ion()
        plt.tight_layout()

    def on_step_end(self, step: int, metrics: Metrics) -> None:
        if step % self.plot_interval != 0:
            return
        for ax in self.axes:
            ax.clear()
        loss_panel(self.axes[0], metrics)
        lr_panel(self.axes[1], metrics)
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()
        plt.pause(0.001)

    def on_train_end(self, metrics: Metrics) -> None:
        for ax in self.axes:
            ax.clear()
        loss_panel(self.axes[0], metrics)
        lr_panel(self.axes[1], metrics)
        self.fig.canvas.draw_idle()
        plt.ioff()
        plt.show()

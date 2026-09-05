from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pytest

from gpt2lab.training.metrics import Metrics
from gpt2lab.utils import configure_logging, get_logger
from gpt2lab.viz.axes import style_axis
from gpt2lab.viz.panels import loss_panel, lr_panel
from gpt2lab.viz.plotter import LivePlotter

matplotlib.use("Agg")


def test_logging_configuration(tmp_path: Path) -> None:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    try:
        path = tmp_path / "logs" / "run.log"
        configure_logging(logging.DEBUG, path)
        logger = get_logger("gpt2lab.test")
        logger.info("hello")
        for handler in root.handlers:
            handler.flush()
        assert "hello" in path.read_text(encoding="utf-8")
        assert logging.getLogger("matplotlib").level == logging.WARNING
        configure_logging()
        assert len(root.handlers) == 1
    finally:
        for handler in list(root.handlers):
            root.removeHandler(handler)
            handler.close()
        for handler in original_handlers:
            root.addHandler(handler)


def test_plot_panels_and_axis_style() -> None:
    figure, axes = plt.subplots(1, 2)
    metrics = Metrics([1.0], [0.8], [0.01], [1], [1])
    loss_panel(axes[0], metrics)
    lr_panel(axes[1], metrics)
    assert axes[0].get_title() == "Loss"
    assert len(axes[0].lines) == 2
    assert len(axes[1].lines) == 1
    style_axis(axes[0], "Title", "x", "y")
    assert axes[0].get_xlabel() == "x"
    plt.close(figure)


def test_empty_plot_panels() -> None:
    figure, axes = plt.subplots(1, 2)
    loss_panel(axes[0], Metrics())
    lr_panel(axes[1], Metrics())
    assert not axes[0].lines
    assert not axes[1].lines
    plt.close(figure)


def test_live_plotter_callbacks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(plt, "pause", lambda _seconds: None)
    monkeypatch.setattr(plt, "show", lambda: None)
    plotter = LivePlotter(plot_interval=2)
    metrics = Metrics([1.0], [], [0.01], [0], [])
    plotter.on_step_end(1, metrics)
    plotter.on_step_end(2, metrics)
    plotter.on_train_end(metrics)
    plt.close(plotter.fig)
    with pytest.raises(ValueError, match="positive"):
        LivePlotter(0)

"""Composition root – wires everything together from config."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

import tiktoken

from .config.experiment import ExperimentConfig
from .config.sections import ModelConfig
from .data.corpus import prepare_dataset
from .data.loader import DataLoader
from .models.gpt import GPT
from .training.trainer import Trainer
from .training.callbacks import PrintCallback, Callback
from .viz.plotter import LivePlotter
from .utils.logging import get_logger

logger = get_logger()


def _load_config_dict(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Load configs/default.py CONFIG dict and apply overrides."""
    cfg_path = Path(__file__).resolve().parent.parent.parent / "configs" / "default.py"
    spec = importlib.util.spec_from_file_location("default_cfg", cfg_path)
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    flat: dict[str, Any] = dict(mod.CONFIG)
    if overrides:
        flat.update(overrides)
    return flat


class Experiment:
    """Fully-wired experiment ready to .run()."""

    def __init__(self, cfg: ExperimentConfig, model: GPT,
                 train_loader: DataLoader, val_loader: DataLoader,
                 callbacks: list[Callback]) -> None:
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.callbacks = callbacks

    def run(self, resume_from: str | None = None):
        trainer = Trainer(self.cfg, self.model, self.train_loader,
                          self.val_loader, self.callbacks)
        return trainer.run(resume_from=resume_from)


def build_experiment(overrides: dict[str, Any] | None = None) -> Experiment:
    flat = _load_config_dict(overrides)
    cfg = ExperimentConfig.from_flat_dict(flat)

    logger.info("Preparing dataset '%s' …", cfg.data.dataset)
    enc = tiktoken.get_encoding("gpt2")
    train_ids, val_ids = prepare_dataset(cfg.data.dataset, enc)

    train_loader = DataLoader(train_ids, cfg.data.block_size, cfg.data.batch_size,
                              cfg.training.device)
    val_loader = DataLoader(val_ids, cfg.data.block_size, cfg.data.batch_size,
                            cfg.training.device)

    model = GPT(cfg.model)
    logger.info("Model parameters: %s", f"{model.num_parameters():,}")

    callbacks: list[Callback] = [PrintCallback(cfg.eval.log_interval)]
    if cfg.viz.plot:
        callbacks.append(LivePlotter(cfg.viz.plot_interval))

    return Experiment(cfg, model, train_loader, val_loader, callbacks)


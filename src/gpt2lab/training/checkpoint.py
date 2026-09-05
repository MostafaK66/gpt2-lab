"""Validated checkpoint persistence behind an injectable storage boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, cast

import torch

from gpt2lab.config import ExperimentConfig
from gpt2lab.errors import CheckpointError
from gpt2lab.training.metrics import Metrics

SaveFunction = Callable[[object, Path], None]
LoadFunction = Callable[[Path, str], object]


def _torch_save(payload: object, path: Path) -> None:
    torch.save(payload, path)


def _torch_load(path: Path, device: str) -> object:
    # Checkpoints contain optimizer state as well as tensors. Only load trusted files.
    return torch.load(path, map_location=device, weights_only=False)


class CheckpointManager:
    """Save, restore, and rotate training checkpoints."""

    def __init__(
        self,
        directory: Path,
        keep_last: int,
        *,
        save_function: SaveFunction = _torch_save,
        load_function: LoadFunction = _torch_load,
    ) -> None:
        if keep_last <= 0:
            raise ValueError("keep_last must be positive")
        self.directory = Path(directory)
        self.keep_last = keep_last
        self._save = save_function
        self._load = load_function

    def save(
        self,
        name: str,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        step: int,
        config: ExperimentConfig,
        metrics: Metrics,
    ) -> Path:
        if Path(name).name != name or not name.endswith(".pt"):
            raise CheckpointError("checkpoint name must be a local .pt file name")
        destination = self.directory / name
        temporary = destination.with_suffix(".pt.tmp")
        payload: dict[str, object] = {
            "format_version": 1,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "config": config.to_dict(),
            "metrics": {
                "train_losses": metrics.train_losses,
                "val_losses": metrics.val_losses,
                "learning_rates": metrics.learning_rates,
                "steps": metrics.steps,
                "val_steps": metrics.val_steps,
            },
        }
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
            temporary.unlink(missing_ok=True)
            self._save(payload, temporary)
            temporary.replace(destination)
        except (OSError, RuntimeError) as exc:
            raise CheckpointError(f"cannot save checkpoint {destination}: {exc}") from exc
        finally:
            temporary.unlink(missing_ok=True)
        if name.startswith("step_"):
            self._prune()
        return destination

    def restore(
        self,
        path: Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None,
        device: torch.device,
    ) -> tuple[int, Metrics]:
        try:
            raw = self._load(Path(path), str(device))
            if not isinstance(raw, Mapping):
                raise CheckpointError("checkpoint root must be a mapping")
            if raw.get("format_version") != 1:
                raise CheckpointError("unsupported checkpoint format version")
            step = raw["step"]
            model_state = raw["model"]
            optimizer_state = raw.get("optimizer")
            metrics_state = raw["metrics"]
            if not isinstance(step, int) or step < 0:
                raise CheckpointError("checkpoint step must be a non-negative integer")
            if not isinstance(model_state, Mapping) or not isinstance(
                metrics_state, Mapping
            ):
                raise CheckpointError("checkpoint model or metrics state is invalid")
            model.load_state_dict(model_state)
            if optimizer is not None and optimizer_state is not None:
                if not isinstance(optimizer_state, Mapping):
                    raise CheckpointError("checkpoint optimizer state is invalid")
                optimizer.load_state_dict(cast(dict[str, Any], optimizer_state))
            metrics = Metrics(
                train_losses=list(cast(list[float], metrics_state["train_losses"])),
                val_losses=list(cast(list[float], metrics_state["val_losses"])),
                learning_rates=list(cast(list[float], metrics_state["learning_rates"])),
                steps=list(cast(list[int], metrics_state["steps"])),
                val_steps=list(cast(list[int], metrics_state["val_steps"])),
            )
        except CheckpointError:
            raise
        except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
            raise CheckpointError(f"cannot load checkpoint {path}: {exc}") from exc
        return step, metrics

    def _prune(self) -> None:
        checkpoints = sorted(
            self.directory.glob("step_*.pt"),
            key=lambda path: path.stat().st_mtime_ns,
            reverse=True,
        )
        for stale in checkpoints[self.keep_last :]:
            stale.unlink()

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import pytest
import torch

from gpt2lab.config import ExperimentConfig, OptimizerConfig, RuntimeConfig
from gpt2lab.data import SequentialTokenLoader
from gpt2lab.errors import CheckpointError, DeviceUnavailableError, TrainingError
from gpt2lab.models import GPT
from gpt2lab.training.callbacks import PrintCallback
from gpt2lab.training.checkpoint import CheckpointManager
from gpt2lab.training.metrics import Metrics
from gpt2lab.training.optim import build_optimizer, learning_rate_at
from gpt2lab.training.runtime import (
    autocast_context,
    configure_runtime,
    select_device,
)
from gpt2lab.training.trainer import Trainer


def make_model(config: ExperimentConfig, vocab_size: int = 16) -> GPT:
    return GPT(config.model, vocab_size)


def make_loader() -> SequentialTokenLoader:
    return SequentialTokenLoader(torch.arange(16) % 16, 1, 3)


def test_metrics_log_and_csv(tmp_path: Path) -> None:
    metrics = Metrics()
    metrics.log_train(0, 1.5, 0.01)
    metrics.log_validation(0, 1.25)
    path = tmp_path / "nested" / "metrics.csv"
    metrics.write_csv(path)
    assert path.read_text(encoding="utf-8").splitlines() == [
        "step,train_loss,learning_rate,validation_loss",
        "0,1.5,0.01,1.25",
    ]


def test_print_callback() -> None:
    stream = io.StringIO()
    callback = PrintCallback(2, stream)
    callback.on_step_end(0, Metrics())
    metrics = Metrics()
    metrics.log_train(0, 1.0, 0.1)
    callback.on_step_end(0, metrics)
    callback.on_step_end(1, metrics)
    callback.on_train_end(metrics)
    assert stream.getvalue().splitlines() == [
        "step     2 | loss 1.0000 | lr 1.00e-01",
        "Training complete (1 steps).",
    ]
    with pytest.raises(ValueError, match="positive"):
        PrintCallback(0)


def test_optimizer_groups_parameters(tiny_config: ExperimentConfig) -> None:
    model = make_model(tiny_config)
    frozen = model.transformer["wpe"].weight
    frozen.requires_grad_(False)
    optimizer = build_optimizer(model, tiny_config.optimizer)
    assert len(optimizer.param_groups) == 2
    assert optimizer.param_groups[0]["weight_decay"] == 0.01
    assert optimizer.param_groups[1]["weight_decay"] == 0.0
    all_parameters = {
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    }
    assert id(frozen) not in all_parameters


def test_learning_rate_schedules() -> None:
    constant = OptimizerConfig(schedule="constant", warmup_steps=0)
    assert learning_rate_at(4, 10, constant) == constant.learning_rate
    cosine = OptimizerConfig(
        learning_rate=1.0,
        min_learning_rate=0.1,
        schedule="cosine",
        warmup_steps=2,
    )
    assert learning_rate_at(0, 10, cosine) == 0.5
    assert learning_rate_at(1, 10, cosine) == 1.0
    assert learning_rate_at(10, 10, cosine) == 0.1
    assert 0.1 < learning_rate_at(5, 10, cosine) < 1.0
    with pytest.raises(ValueError, match="negative"):
        learning_rate_at(-1, 10, cosine)
    with pytest.raises(ValueError, match="positive"):
        learning_rate_at(0, 0, cosine)


def test_device_selection_all_paths() -> None:
    def yes() -> bool:
        return True

    def no() -> bool:
        return False

    assert select_device("auto", cuda_available=yes, mps_available=no).type == "cuda"
    assert select_device("auto", cuda_available=no, mps_available=yes).type == "mps"
    assert select_device("auto", cuda_available=no, mps_available=no).type == "cpu"
    assert select_device("cpu", cuda_available=no, mps_available=no).type == "cpu"
    assert select_device("cuda", cuda_available=yes, mps_available=no).type == "cuda"
    assert select_device("mps", cuda_available=no, mps_available=yes).type == "mps"
    with pytest.raises(DeviceUnavailableError, match="CUDA"):
        select_device("cuda", cuda_available=no)
    with pytest.raises(DeviceUnavailableError, match="MPS"):
        select_device("mps", mps_available=no)
    with pytest.raises(DeviceUnavailableError, match="unsupported"):
        select_device("tpu")


def test_configure_runtime_and_autocast(monkeypatch: pytest.MonkeyPatch) -> None:
    seeded: list[int] = []
    monkeypatch.setattr(torch, "manual_seed", seeded.append)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    device = configure_runtime(RuntimeConfig(seed=7, device="cpu", deterministic=True))
    assert device.type == "cpu"
    assert seeded == [7]
    with autocast_context(device, RuntimeConfig(device="cpu")):
        result = torch.ones(1) + 1
    assert result.item() == 2


def test_checkpoint_round_trip_and_rotation(
    tiny_config: ExperimentConfig, tmp_path: Path
) -> None:
    model = make_model(tiny_config)
    optimizer = build_optimizer(model, tiny_config.optimizer)
    metrics = Metrics([1.0], [0.9], [0.01], [0], [0])
    manager = CheckpointManager(tmp_path, 1)
    first = manager.save("step_00000001.pt", model, optimizer, 0, tiny_config, metrics)
    second = manager.save("step_00000002.pt", model, optimizer, 1, tiny_config, metrics)
    assert not first.exists()
    assert second.exists()
    restored_model = make_model(tiny_config)
    restored_optimizer = build_optimizer(restored_model, tiny_config.optimizer)
    step, restored_metrics = manager.restore(
        second, restored_model, restored_optimizer, torch.device("cpu")
    )
    assert step == 1
    assert restored_metrics.train_losses == [1.0]
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            model.state_dict().values(), restored_model.state_dict().values(), strict=True
        )
    )


def test_checkpoint_validation_and_wrapped_save_errors(
    tiny_config: ExperimentConfig, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match="keep_last"):
        CheckpointManager(tmp_path, 0)
    manager = CheckpointManager(tmp_path, 1)
    model = make_model(tiny_config)
    optimizer = build_optimizer(model, tiny_config.optimizer)
    with pytest.raises(CheckpointError, match="name"):
        manager.save("../bad.pt", model, optimizer, 0, tiny_config, Metrics())

    def fail_save(_payload: object, _path: Path) -> None:
        raise OSError("disk full")

    failing = CheckpointManager(tmp_path, 1, save_function=fail_save)
    with pytest.raises(CheckpointError, match="disk full"):
        failing.save("valid.pt", model, optimizer, 0, tiny_config, Metrics())


@pytest.mark.parametrize(
    "payload,message",
    [
        ([], "root"),
        ({"format_version": 2}, "format"),
        (
            {"format_version": 1, "step": -1, "model": {}, "metrics": {}},
            "step",
        ),
        (
            {"format_version": 1, "step": 0, "model": [], "metrics": {}},
            "model or metrics",
        ),
    ],
)
def test_checkpoint_rejects_invalid_payloads(
    tiny_config: ExperimentConfig,
    tmp_path: Path,
    payload: object,
    message: str,
) -> None:
    manager = CheckpointManager(tmp_path, 1, load_function=lambda _path, _device: payload)
    with pytest.raises(CheckpointError, match=message):
        manager.restore(
            tmp_path / "bad.pt", make_model(tiny_config), None, torch.device("cpu")
        )


def test_checkpoint_wraps_loader_and_state_errors(
    tiny_config: ExperimentConfig, tmp_path: Path
) -> None:
    def fail_load(_path: Path, _device: str) -> object:
        raise OSError("unreadable")

    manager = CheckpointManager(tmp_path, 1, load_function=fail_load)
    with pytest.raises(CheckpointError, match="unreadable"):
        manager.restore(
            tmp_path / "bad.pt", make_model(tiny_config), None, torch.device("cpu")
        )


def test_trainer_runs_evaluates_saves_and_writes_metrics(
    tiny_config: ExperimentConfig,
) -> None:
    trainer = Trainer(
        tiny_config,
        make_model(tiny_config),
        make_loader(),
        make_loader(),
        torch.device("cpu"),
        callbacks=[],
    )
    metrics = trainer.run()
    assert metrics.steps == [0, 1]
    assert metrics.val_steps == [0, 1]
    assert tiny_config.training.metrics_csv is not None
    assert tiny_config.training.metrics_csv.exists()
    assert (tiny_config.training.checkpoint_dir / "final.pt").exists()
    assert len(list(tiny_config.training.checkpoint_dir.glob("step_*.pt"))) == 1


def test_trainer_resume_continues_after_completed_step(
    tiny_config: ExperimentConfig,
) -> None:
    original = Trainer(
        tiny_config,
        make_model(tiny_config),
        make_loader(),
        make_loader(),
        torch.device("cpu"),
    )
    original.run()
    resumed = Trainer(
        tiny_config,
        make_model(tiny_config),
        make_loader(),
        make_loader(),
        torch.device("cpu"),
    )
    metrics = resumed.run(tiny_config.training.checkpoint_dir / "final.pt")
    assert metrics.steps == [0, 1]


def test_trainer_rejects_checkpoint_beyond_run(
    tiny_config: ExperimentConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    trainer = Trainer(
        tiny_config,
        make_model(tiny_config),
        make_loader(),
        make_loader(),
        torch.device("cpu"),
    )
    monkeypatch.setattr(
        trainer.checkpoints,
        "restore",
        lambda *_args: (tiny_config.training.num_steps, Metrics()),
    )
    with pytest.raises(TrainingError, match="beyond"):
        trainer.run(Path("unused.pt"))


def test_trainer_stops_on_non_finite_loss(
    tiny_config: ExperimentConfig, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = make_model(tiny_config)

    def non_finite(*_args: Any, **_kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        return torch.zeros(1), torch.tensor(float("nan"), requires_grad=True)

    monkeypatch.setattr(model, "forward", non_finite)
    trainer = Trainer(
        tiny_config,
        model,
        make_loader(),
        make_loader(),
        torch.device("cpu"),
    )
    with pytest.raises(TrainingError, match="non-finite"):
        trainer.run()

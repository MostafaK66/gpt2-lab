from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import pytest

from gpt2lab.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    SamplingConfig,
    TrainingConfig,
)
from gpt2lab.errors import ConfigurationError


def test_default_config_is_immutable_and_serializable() -> None:
    config = ExperimentConfig()
    assert config.total_tokens == 640_000
    assert config.to_dict()["name"] == "gpt2-practice"
    assert '"checkpoint_dir": "checkpoints"' in config.to_json()
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.name = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"file_name": ""}, "file_name"),
        ({"file_name": "../secret"}, "file_name"),
        ({"tokenizer_name": " "}, "tokenizer_name"),
        ({"source_url": "http://example.com/a"}, "HTTPS"),
        ({"batch_size": 0}, "batch_size"),
        ({"sequence_length": -1}, "sequence_length"),
        ({"validation_fraction": 0.0}, "validation_fraction"),
        ({"validation_fraction": 1.0}, "validation_fraction"),
    ],
)
def test_data_config_rejects_invalid_values(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        DataConfig(**kwargs)


def test_data_config_properties_and_path_conversion() -> None:
    config = DataConfig(cache_dir=Path("cache"), file_name="text.txt", batch_size=2)
    assert config.file_path == Path("cache/text.txt")
    assert config.tokens_per_step == 256


@pytest.mark.parametrize(
    "kwargs",
    [
        {"device": "tpu"},
        {"mixed_precision_dtype": "float32"},
    ],
)
def test_runtime_config_rejects_unknown_choices(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        RuntimeConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"block_size": 0},
        {"n_layer": 0},
        {"n_head": 0},
        {"n_embd": 0},
        {"mlp_ratio": 0},
        {"init_std": 0.0},
        {"dropout": -0.1},
        {"dropout": 1.0},
        {"n_embd": 7, "n_head": 2},
    ],
)
def test_model_config_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        ModelConfig(**kwargs)


def test_model_config_derived_sizes() -> None:
    config = ModelConfig(n_embd=12, n_head=3, mlp_ratio=2)
    assert config.head_size == 4
    assert config.mlp_hidden_size == 24


@pytest.mark.parametrize(
    "kwargs",
    [
        {"learning_rate": 0.0},
        {"weight_decay": -1.0},
        {"beta1": 1.0},
        {"beta2": -0.1},
        {"epsilon": 0.0},
        {"warmup_steps": -1},
        {"min_learning_rate": -1.0},
        {"grad_clip_norm": 0.0},
        {"schedule": "linear"},
        {"learning_rate": 0.1, "min_learning_rate": 0.2},
    ],
)
def test_optimizer_config_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        OptimizerConfig(**kwargs)


def test_optimizer_betas() -> None:
    assert OptimizerConfig(beta1=0.8, beta2=0.9).betas == (0.8, 0.9)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_steps": 0},
        {"log_every": 0},
        {"eval_steps": 0},
        {"keep_last_checkpoints": 0},
        {"eval_every": 0},
        {"checkpoint_every": 0},
    ],
)
def test_training_config_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        TrainingConfig(**kwargs)


def test_training_paths_are_normalized() -> None:
    config = TrainingConfig(checkpoint_dir=Path("ckpt"), metrics_csv=Path("metrics.csv"))
    assert config.checkpoint_dir == Path("ckpt")
    assert config.metrics_csv == Path("metrics.csv")
    assert TrainingConfig(metrics_csv=None).metrics_csv is None


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_new_tokens": 0},
        {"temperature": 0.0},
        {"top_k": 0},
    ],
)
def test_sampling_config_validation(kwargs: dict[str, Any]) -> None:
    with pytest.raises(ConfigurationError):
        SamplingConfig(**kwargs)


def test_experiment_cross_validation() -> None:
    with pytest.raises(ConfigurationError, match="name"):
        ExperimentConfig(name=" ")
    with pytest.raises(ConfigurationError, match="sequence_length"):
        ExperimentConfig(
            data=DataConfig(sequence_length=5), model=ModelConfig(block_size=4)
        )
    with pytest.raises(ConfigurationError, match="warmup_steps"):
        ExperimentConfig(
            optimizer=OptimizerConfig(warmup_steps=11),
            training=TrainingConfig(num_steps=10),
        )
    with pytest.raises(ConfigurationError, match="after warmup"):
        ExperimentConfig(
            optimizer=OptimizerConfig(warmup_steps=10),
            training=TrainingConfig(num_steps=10),
        )


def test_vocabulary_validation() -> None:
    config = ExperimentConfig(sampling=SamplingConfig(top_k=5))
    with pytest.raises(ConfigurationError, match="positive"):
        config.validate_vocabulary(0)
    with pytest.raises(ConfigurationError, match="top_k"):
        config.validate_vocabulary(4)
    ExperimentConfig(sampling=SamplingConfig(top_k=None)).validate_vocabulary(1)


def test_from_mapping_and_toml(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        'name = "mapped"\n[data]\ncache_dir = "cache"\n'
        '[optimizer]\nschedule = "constant"\nwarmup_steps = 0\n',
        encoding="utf-8",
    )
    config = ExperimentConfig.from_toml(path)
    assert config.name == "mapped"
    assert config.data.cache_dir == Path("cache")
    assert config.optimizer.schedule == "constant"


@pytest.mark.parametrize(
    ("mapping", "message"),
    [
        ([], "root"),
        ({"wat": 1}, "top-level"),
        ({"data": []}, "TOML table"),
        ({"data": {"wat": 1}}, "unknown key"),
        ({"data": {"batch_size": "bad"}}, "invalid.*data"),
        ({"name": 4}, "name"),
    ],
)
def test_from_mapping_rejects_invalid_input(mapping: object, message: str) -> None:
    with pytest.raises(ConfigurationError, match=message):
        ExperimentConfig.from_mapping(mapping)


def test_from_toml_reports_file_errors(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        ExperimentConfig.from_toml(tmp_path / "missing.toml")
    invalid = tmp_path / "invalid.toml"
    invalid.write_text("[", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="cannot read"):
        ExperimentConfig.from_toml(invalid)

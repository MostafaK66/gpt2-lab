"""Composed experiment configuration and safe TOML loading."""

from __future__ import annotations

import json
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar, cast

from gpt2lab.config.sections import (
    DataConfig,
    ModelConfig,
    OptimizerConfig,
    RuntimeConfig,
    SamplingConfig,
    TrainingConfig,
)
from gpt2lab.errors import ConfigurationError

_Section = TypeVar("_Section")


def _make_section(cls: type[_Section], value: object, name: str) -> _Section:
    if not isinstance(value, dict):
        raise ConfigurationError(f"[{name}] must be a TOML table")
    allowed = {item.name for item in fields(cast(Any, cls))}
    unknown = set(value) - allowed
    if unknown:
        names = ", ".join(sorted(str(item) for item in unknown))
        raise ConfigurationError(f"unknown key(s) in [{name}]: {names}")
    try:
        return cls(**value)
    except (TypeError, ValueError) as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(f"invalid [{name}] configuration: {exc}") from exc


def _json_value(value: object) -> object:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    """Complete validated description of one run."""

    name: str = "gpt2-practice"
    data: DataConfig = field(default_factory=DataConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    sampling: SamplingConfig = field(default_factory=SamplingConfig)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ConfigurationError("name must not be empty")
        if self.data.sequence_length > self.model.block_size:
            raise ConfigurationError(
                "data.sequence_length cannot exceed model.block_size"
            )
        if self.optimizer.warmup_steps > self.training.num_steps:
            raise ConfigurationError(
                "optimizer.warmup_steps cannot exceed training.num_steps"
            )
        if (
            self.optimizer.schedule == "cosine"
            and self.optimizer.warmup_steps == self.training.num_steps
        ):
            raise ConfigurationError("cosine decay needs a step after warmup")

    @property
    def total_tokens(self) -> int:
        return self.data.tokens_per_step * self.training.num_steps

    def validate_vocabulary(self, vocab_size: int) -> None:
        if vocab_size <= 0:
            raise ConfigurationError("tokenizer vocabulary must be positive")
        if self.sampling.top_k is not None and self.sampling.top_k > vocab_size:
            raise ConfigurationError("sampling.top_k exceeds tokenizer vocabulary")

    def to_dict(self) -> dict[str, object]:
        return cast(dict[str, object], _json_value(asdict(self)))

    def to_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_mapping(cls, value: object) -> ExperimentConfig:
        if not isinstance(value, dict):
            raise ConfigurationError("configuration root must be a TOML table")
        allowed = {
            "name",
            "data",
            "runtime",
            "model",
            "optimizer",
            "training",
            "sampling",
        }
        unknown = set(value) - allowed
        if unknown:
            names = ", ".join(sorted(str(item) for item in unknown))
            raise ConfigurationError(f"unknown top-level key(s): {names}")
        try:
            return cls(
                name=value.get("name", "gpt2-practice"),
                data=_make_section(DataConfig, value.get("data", {}), "data"),
                runtime=_make_section(RuntimeConfig, value.get("runtime", {}), "runtime"),
                model=_make_section(ModelConfig, value.get("model", {}), "model"),
                optimizer=_make_section(
                    OptimizerConfig, value.get("optimizer", {}), "optimizer"
                ),
                training=_make_section(
                    TrainingConfig, value.get("training", {}), "training"
                ),
                sampling=_make_section(
                    SamplingConfig, value.get("sampling", {}), "sampling"
                ),
            )
        except TypeError as exc:
            raise ConfigurationError(f"invalid configuration: {exc}") from exc

    @classmethod
    def from_toml(cls, path: Path) -> ExperimentConfig:
        try:
            with Path(path).open("rb") as stream:
                data = tomllib.load(stream)
        except FileNotFoundError as exc:
            raise ConfigurationError(f"configuration file not found: {path}") from exc
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigurationError(f"cannot read configuration {path}: {exc}") from exc
        return cls.from_mapping(data)

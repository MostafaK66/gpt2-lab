"""Immutable, self-validating configuration sections."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from gpt2lab.errors import ConfigurationError

DeviceName = Literal["auto", "cpu", "cuda", "mps"]
PrecisionName = Literal["float16", "bfloat16"]
ScheduleName = Literal["constant", "cosine"]


def _positive(value: int | float, name: str) -> None:
    if value <= 0:
        raise ConfigurationError(f"{name} must be greater than zero; got {value!r}")


def _non_negative(value: int | float, name: str) -> None:
    if value < 0:
        raise ConfigurationError(f"{name} cannot be negative; got {value!r}")


def _probability(value: float, name: str) -> None:
    if not 0.0 <= value < 1.0:
        raise ConfigurationError(f"{name} must be in [0, 1); got {value!r}")


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Corpus location, tokenizer, split, and mini-batch geometry."""

    source_url: str | None = None
    file_name: str = "corpus.txt"
    tokenizer_name: str = "gpt2"
    cache_dir: Path = Path("data")
    batch_size: int = 1
    sequence_length: int = 128
    validation_fraction: float = 0.1
    pin_memory: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "cache_dir", Path(self.cache_dir))
        if Path(self.file_name).name != self.file_name or not self.file_name.strip():
            raise ConfigurationError("data.file_name must be a non-empty file name")
        if not self.tokenizer_name.strip():
            raise ConfigurationError("data.tokenizer_name must not be empty")
        if self.source_url is not None:
            parsed = urlparse(self.source_url)
            if parsed.scheme != "https" or not parsed.netloc:
                raise ConfigurationError("data.source_url must be an HTTPS URL")
        _positive(self.batch_size, "data.batch_size")
        _positive(self.sequence_length, "data.sequence_length")
        if not 0.0 < self.validation_fraction < 1.0:
            raise ConfigurationError("data.validation_fraction must be in (0, 1)")

    @property
    def file_path(self) -> Path:
        return self.cache_dir / self.file_name

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.sequence_length


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    """Reproducibility, device selection, and numerical precision."""

    seed: int = 1337
    device: DeviceName = "auto"
    deterministic: bool = False
    use_mixed_precision: bool = True
    mixed_precision_dtype: PrecisionName = "float16"

    def __post_init__(self) -> None:
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ConfigurationError(f"unsupported runtime.device: {self.device!r}")
        if self.mixed_precision_dtype not in {"float16", "bfloat16"}:
            raise ConfigurationError(
                "runtime.mixed_precision_dtype must be 'float16' or 'bfloat16'"
            )


@dataclass(frozen=True, slots=True)
class ModelConfig:
    """GPT-2 architecture settings independent of tokenizer vocabulary size."""

    block_size: int = 128
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 256
    mlp_ratio: int = 4
    dropout: float = 0.0
    bias: bool = True
    init_std: float = 0.02

    def __post_init__(self) -> None:
        for name in ("block_size", "n_layer", "n_head", "n_embd", "mlp_ratio"):
            _positive(getattr(self, name), f"model.{name}")
        _positive(self.init_std, "model.init_std")
        _probability(self.dropout, "model.dropout")
        if self.n_embd % self.n_head:
            raise ConfigurationError("model.n_embd must be divisible by model.n_head")

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head

    @property
    def mlp_hidden_size(self) -> int:
        return self.mlp_ratio * self.n_embd


@dataclass(frozen=True, slots=True)
class OptimizerConfig:
    """AdamW, gradient clipping, and learning-rate schedule settings."""

    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-8
    grad_clip_norm: float | None = 1.0
    schedule: ScheduleName = "cosine"
    warmup_steps: int = 100
    min_learning_rate: float = 3e-5

    def __post_init__(self) -> None:
        _positive(self.learning_rate, "optimizer.learning_rate")
        _non_negative(self.weight_decay, "optimizer.weight_decay")
        _probability(self.beta1, "optimizer.beta1")
        _probability(self.beta2, "optimizer.beta2")
        _positive(self.epsilon, "optimizer.epsilon")
        _non_negative(self.warmup_steps, "optimizer.warmup_steps")
        _non_negative(self.min_learning_rate, "optimizer.min_learning_rate")
        if self.grad_clip_norm is not None:
            _positive(self.grad_clip_norm, "optimizer.grad_clip_norm")
        if self.schedule not in {"constant", "cosine"}:
            raise ConfigurationError(f"unsupported optimizer.schedule: {self.schedule!r}")
        if self.min_learning_rate > self.learning_rate:
            raise ConfigurationError(
                "optimizer.min_learning_rate cannot exceed learning_rate"
            )

    @property
    def betas(self) -> tuple[float, float]:
        return self.beta1, self.beta2


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    """Loop, evaluation, logging, and checkpoint settings."""

    num_steps: int = 5000
    log_every: int = 10
    eval_every: int | None = 250
    eval_steps: int = 20
    checkpoint_every: int | None = 1000
    checkpoint_dir: Path = Path("checkpoints")
    keep_last_checkpoints: int = 3
    save_final_checkpoint: bool = True
    stop_on_non_finite_loss: bool = True
    metrics_csv: Path | None = Path("runs/metrics.csv")

    def __post_init__(self) -> None:
        object.__setattr__(self, "checkpoint_dir", Path(self.checkpoint_dir))
        if self.metrics_csv is not None:
            object.__setattr__(self, "metrics_csv", Path(self.metrics_csv))
        for name in ("num_steps", "log_every", "eval_steps", "keep_last_checkpoints"):
            _positive(getattr(self, name), f"training.{name}")
        if self.eval_every is not None:
            _positive(self.eval_every, "training.eval_every")
        if self.checkpoint_every is not None:
            _positive(self.checkpoint_every, "training.checkpoint_every")


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Autoregressive text-generation settings."""

    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int | None = 40

    def __post_init__(self) -> None:
        _positive(self.max_new_tokens, "sampling.max_new_tokens")
        _positive(self.temperature, "sampling.temperature")
        if self.top_k is not None:
            _positive(self.top_k, "sampling.top_k")

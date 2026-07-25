"""
Configuration sections.

Each dataclass owns one coherent group of settings and validates itself in
``__post_init__``.  Rules that span two sections (for example
``sequence_length <= block_size``) live in
:class:`gpt2lab.config.experiment.ExperimentConfig` instead.

Every section is frozen, so a configuration object cannot silently change
half-way through a run.  Use :func:`dataclasses.replace` to derive a variant.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

__all__ = [
    "DataConfig",
    "MetricsConfig",
    "ModelConfig",
    "OptimizerConfig",
    "PlotConfig",
    "RuntimeConfig",
    "TrainingConfig",
    "VALID_AMP_DTYPES",
    "VALID_DEVICE_PREFERENCES",
    "VALID_MATMUL_PRECISIONS",
    "VALID_PLOT_MODES",
    "VALID_SCHEDULES",
]


VALID_DEVICE_PREFERENCES = frozenset({"auto", "cuda", "cpu"})
VALID_MATMUL_PRECISIONS = frozenset({"highest", "high", "medium"})
VALID_AMP_DTYPES = frozenset({"float16", "bfloat16"})
VALID_SCHEDULES = frozenset({"constant", "cosine"})
VALID_PLOT_MODES = frozenset({"off", "window", "file"})


# ---------------------------------------------------------------------------
# Small validation helpers
# ---------------------------------------------------------------------------

def _require_positive(value: float, name: str) -> None:
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero (received {value!r}).")


def _require_non_negative(value: float, name: str) -> None:
    if value < 0:
        raise ValueError(f"{name} cannot be negative (received {value!r}).")


def _require_half_open_unit(value: float, name: str) -> None:
    if not 0.0 <= value < 1.0:
        raise ValueError(f"{name} must satisfy 0 <= {name} < 1 (received {value!r}).")


def _require_choice(value: Any, name: str, allowed: Iterable[Any]) -> None:
    allowed = set(allowed)
    if value not in allowed:
        options = ", ".join(sorted(repr(option) for option in allowed))
        raise ValueError(f"{name} must be one of {options} (received {value!r}).")


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string (received {value!r}).")


# ---------------------------------------------------------------------------
# Sections
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DataConfig:
    """Corpus location, tokenizer and mini-batch geometry."""

    source_url: str
    file_name: str
    tokenizer_name: str = "gpt2"
    cache_dir: Path = Path("data")

    # B: number of independent sequences per batch.
    batch_size: int = 1

    # T: number of tokens in every sequence.
    sequence_length: int = 1024

    # Page-locked host memory speeds up host -> CUDA transfers.
    pin_memory: bool = True

    def __post_init__(self) -> None:
        _require_text(self.source_url, "source_url")
        _require_text(self.file_name, "file_name")
        _require_text(self.tokenizer_name, "tokenizer_name")
        _require_positive(self.batch_size, "batch_size")
        _require_positive(self.sequence_length, "sequence_length")

    @property
    def file_path(self) -> Path:
        """Absolute-or-relative path of the cached corpus file."""
        return Path(self.cache_dir) / self.file_name

    @property
    def tokens_per_step(self) -> int:
        return self.batch_size * self.sequence_length


@dataclass(frozen=True)
class RuntimeConfig:
    """Device selection, reproducibility and numerical precision."""

    seed: int = 1337

    # "auto" -> CUDA when available, otherwise CPU
    # "cuda" -> require a CUDA GPU
    # "cpu"  -> always CPU
    device_preference: str = "auto"

    deterministic: bool = False

    # Internal precision of float32 matmuls: "highest" | "high" | "medium".
    float32_matmul_precision: str = "high"

    use_mixed_precision: bool = True

    # "float16" for Turing (T4) and older, "bfloat16" for Ampere and newer.
    mixed_precision_dtype: str = "float16"

    def __post_init__(self) -> None:
        _require_choice(
            self.device_preference, "device_preference", VALID_DEVICE_PREFERENCES
        )
        _require_choice(
            self.float32_matmul_precision,
            "float32_matmul_precision",
            VALID_MATMUL_PRECISIONS,
        )
        _require_choice(
            self.mixed_precision_dtype, "mixed_precision_dtype", VALID_AMP_DTYPES
        )


@dataclass(frozen=True)
class ModelConfig:
    """GPT architecture.

    ``vocab_size`` deliberately does not live here: it is dictated by the
    tokenizer, not by the experimenter, and is passed to :class:`~gpt2lab.models.gpt.GPT`
    separately.
    """

    block_size: int = 1024
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    mlp_ratio: int = 4
    init_std: float = 0.02

    def __post_init__(self) -> None:
        _require_positive(self.block_size, "block_size")
        _require_positive(self.n_layer, "n_layer")
        _require_positive(self.n_head, "n_head")
        _require_positive(self.n_embd, "n_embd")
        _require_positive(self.mlp_ratio, "mlp_ratio")
        _require_positive(self.init_std, "init_std")

        if self.n_embd % self.n_head != 0:
            raise ValueError(
                "n_embd must be divisible by n_head "
                f"(received n_embd={self.n_embd}, n_head={self.n_head})."
            )

    @property
    def head_size(self) -> int:
        return self.n_embd // self.n_head

    @property
    def mlp_hidden_size(self) -> int:
        return self.mlp_ratio * self.n_embd


@dataclass(frozen=True)
class OptimizerConfig:
    """AdamW settings, gradient clipping and the learning-rate schedule."""

    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8

    # None disables clipping; the global gradient norm is still measured.
    grad_clip_norm: float | None = 1.0

    # "constant" reproduces the original notebook behaviour.
    schedule: str = "constant"
    warmup_steps: int = 0
    min_learning_rate: float = 0.0

    # Use the fused AdamW kernel when running on CUDA.
    fused: bool = True

    # Exclude tensors with fewer than 2 dimensions (biases, LayerNorm gains)
    # from weight decay, as in the GPT-2 / nanoGPT recipe.
    decay_matrices_only: bool = True

    def __post_init__(self) -> None:
        _require_positive(self.learning_rate, "learning_rate")
        _require_non_negative(self.weight_decay, "weight_decay")
        _require_half_open_unit(self.beta1, "beta1")
        _require_half_open_unit(self.beta2, "beta2")
        _require_positive(self.epsilon, "epsilon")
        _require_choice(self.schedule, "schedule", VALID_SCHEDULES)
        _require_non_negative(self.warmup_steps, "warmup_steps")
        _require_non_negative(self.min_learning_rate, "min_learning_rate")

        if self.grad_clip_norm is not None:
            _require_positive(self.grad_clip_norm, "grad_clip_norm")

        if self.min_learning_rate > self.learning_rate:
            raise ValueError(
                "min_learning_rate cannot exceed learning_rate "
                f"(received min_learning_rate={self.min_learning_rate}, "
                f"learning_rate={self.learning_rate})."
            )

    @property
    def betas(self) -> tuple[float, float]:
        return self.beta1, self.beta2


@dataclass(frozen=True)
class TrainingConfig:
    """Loop length, logging cadence and checkpointing."""

    num_steps: int = 10
    log_every: int = 1

    # None disables periodic checkpoints; a final checkpoint is still written
    # when save_final_checkpoint is True.
    checkpoint_every: int | None = None
    checkpoint_dir: Path = Path("checkpoints")
    keep_last_checkpoints: int = 3
    save_final_checkpoint: bool = True

    # Abort instead of silently training through a NaN / Inf loss.
    stop_on_non_finite_loss: bool = True

    # None disables the CSV dump of the metric history.
    metrics_csv: Path | None = Path("runs/metrics.csv")

    def __post_init__(self) -> None:
        _require_positive(self.num_steps, "num_steps")
        _require_positive(self.log_every, "log_every")
        _require_positive(self.keep_last_checkpoints, "keep_last_checkpoints")

        if self.checkpoint_every is not None:
            _require_positive(self.checkpoint_every, "checkpoint_every")


@dataclass(frozen=True)
class MetricsConfig:
    """Derived training metrics."""

    top_k_accuracy: int = 5

    # Guards against exp() overflow when converting loss to perplexity.
    perplexity_loss_cap: float = 20.0

    def __post_init__(self) -> None:
        _require_positive(self.top_k_accuracy, "top_k_accuracy")
        _require_positive(self.perplexity_loss_cap, "perplexity_loss_cap")


@dataclass(frozen=True)
class PlotConfig:
    """Live plotting.

    ``mode``:
        ``"off"``    - no plots at all
        ``"window"`` - interactive matplotlib window (the PyCharm default)
        ``"file"``   - re-render a PNG after every update (headless friendly)
    """

    mode: str = "window"

    # None plots every completed step; an integer keeps only the latest N.
    window: int | None = None

    every: int = 1
    output_dir: Path = Path("plots")
    file_name: str = "training_progress.png"
    figure_size: tuple[float, float] = (14.0, 9.0)
    dpi: int = 110

    # Keep the interactive window open when training finishes.
    block_on_finish: bool = True

    def __post_init__(self) -> None:
        _require_choice(self.mode, "mode", VALID_PLOT_MODES)
        _require_positive(self.every, "every")
        _require_positive(self.dpi, "dpi")
        _require_text(self.file_name, "file_name")

        if self.window is not None:
            _require_positive(self.window, "window")

        if len(self.figure_size) != 2:
            raise ValueError("figure_size must contain exactly two values.")

        for extent in self.figure_size:
            _require_positive(extent, "figure_size")

    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    @property
    def file_path(self) -> Path:
        return Path(self.output_dir) / self.file_name


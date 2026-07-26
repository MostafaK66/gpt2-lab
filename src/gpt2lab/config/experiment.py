"""
The composed experiment configuration.

:class:`ExperimentConfig` groups the individual sections and enforces the rules
that only make sense once several sections are known.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from gpt2lab.config.sections import (
    DataConfig,
    MetricsConfig,
    ModelConfig,
    OptimizerConfig,
    PlotConfig,
    RuntimeConfig,
    TrainingConfig,
)

__all__ = ["ExperimentConfig"]


def _jsonify(value: Any) -> Any:
    """Convert a nested ``asdict`` result into JSON-serialisable primitives."""
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    return value


@dataclass(frozen=True)
class ExperimentConfig:
    """A complete, validated description of one training run."""

    data: DataConfig
    name: str = "gpt2-tinyshakespeare"
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    plotting: PlotConfig = field(default_factory=PlotConfig)

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("name must be a non-empty string.")

        if self.data.sequence_length > self.model.block_size:
            raise ValueError(
                "sequence_length cannot exceed block_size (received "
                f"sequence_length={self.data.sequence_length}, "
                f"block_size={self.model.block_size})."
            )

        if self.optimizer.warmup_steps > self.training.num_steps:
            raise ValueError(
                "warmup_steps cannot exceed num_steps (received "
                f"warmup_steps={self.optimizer.warmup_steps}, "
                f"num_steps={self.training.num_steps})."
            )

        if (
            self.optimizer.schedule == "cosine"
            and self.optimizer.warmup_steps == self.training.num_steps
        ):
            raise ValueError(
                "A cosine schedule needs at least one step after warmup; "
                "reduce warmup_steps or increase num_steps."
            )

    # -- derived quantities -------------------------------------------------

    @property
    def tokens_per_step(self) -> int:
        return self.data.tokens_per_step

    @property
    def total_tokens(self) -> int:
        return self.tokens_per_step * self.training.num_steps

    def validate_against_vocabulary(self, vocab_size: int) -> None:
        """Checks that can only run once the tokenizer has been loaded."""
        if vocab_size <= 0:
            raise ValueError(f"vocab_size must be positive (received {vocab_size}).")

        if self.metrics.top_k_accuracy > vocab_size:
            raise ValueError(
                "top_k_accuracy cannot exceed the tokenizer vocabulary size "
                f"(received top_k_accuracy={self.metrics.top_k_accuracy}, "
                f"vocab_size={vocab_size})."
            )

    # -- serialisation ------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return _jsonify(asdict(self))

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def describe(self) -> str:
        """A compact, human-readable summary for the run log."""
        rows: list[tuple[str, Any]] = [
            ("Experiment", self.name),
            ("Seed", self.runtime.seed),
            ("Device preference", self.runtime.device_preference),
            ("Tokenizer", self.data.tokenizer_name),
            ("Corpus", self.data.file_path),
            ("Batch size (B)", self.data.batch_size),
            ("Sequence length (T)", self.data.sequence_length),
            ("Tokens per step", f"{self.tokens_per_step:,}"),
            ("Block size", self.model.block_size),
            ("Layers", self.model.n_layer),
            ("Heads", self.model.n_head),
            ("Embedding size", self.model.n_embd),
            ("MLP hidden size", self.model.mlp_hidden_size),
            ("Learning rate", self.optimizer.learning_rate),
            ("LR schedule", self.optimizer.schedule),
            ("Weight decay", self.optimizer.weight_decay),
            ("Grad clip norm", self.optimizer.grad_clip_norm),
            ("Training steps", self.training.num_steps),
            ("Total tokens", f"{self.total_tokens:,}"),
            ("Mixed precision", self.runtime.use_mixed_precision),
            ("AMP dtype", self.runtime.mixed_precision_dtype),
            ("Plot mode", self.plotting.mode),
            ("Plot window", self.plotting.window),
        ]

        width = max(len(label) for label, _ in rows)
        lines = ["Experiment configuration", "-" * 24]
        lines += [f"{label:<{width}} : {value}" for label, value in rows]
        return "\n".join(lines)


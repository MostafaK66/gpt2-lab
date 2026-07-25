"""Seven self-validating configuration dataclasses."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    block_size: int = 256
    dropout: float = 0.2
    bias: bool = False

    def __post_init__(self) -> None:
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.n_layer > 0
        assert self.vocab_size > 0
        assert 0.0 <= self.dropout < 1.0


@dataclass
class DataConfig:
    dataset: str = "shakespeare"
    batch_size: int = 64
    block_size: int = 256

    def __post_init__(self) -> None:
        assert self.batch_size > 0
        assert self.block_size > 0


@dataclass
class TrainingConfig:
    max_steps: int = 5000
    learning_rate: float = 3e-4
    weight_decay: float = 1e-1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0
    warmup_steps: int = 100
    lr_decay_steps: int = 5000
    min_lr: float = 3e-5
    device: str = "cpu"

    def __post_init__(self) -> None:
        assert self.max_steps > 0
        assert self.learning_rate > 0
        assert 0.0 <= self.beta1 < 1.0
        assert 0.0 <= self.beta2 < 1.0


@dataclass
class EvalConfig:
    eval_interval: int = 250
    eval_steps: int = 20
    log_interval: int = 10

    def __post_init__(self) -> None:
        assert self.eval_interval > 0
        assert self.eval_steps > 0


@dataclass
class CheckpointConfig:
    checkpoint_dir: str = "checkpoints"
    save_interval: int = 1000

    def __post_init__(self) -> None:
        assert self.save_interval > 0


@dataclass
class VizConfig:
    plot: bool = True
    plot_interval: int = 50

    def __post_init__(self) -> None:
        assert self.plot_interval > 0


@dataclass
class SamplingConfig:
    max_new_tokens: int = 200
    temperature: float = 0.8
    top_k: int = 40

    def __post_init__(self) -> None:
        assert self.temperature > 0
        assert self.top_k >= 0


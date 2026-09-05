"""Composition root for training and sampling."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch

from gpt2lab.config import ExperimentConfig
from gpt2lab.data import (
    Downloader,
    SequentialTokenLoader,
    TextCorpus,
    TiktokenTokenizer,
    Tokenizer,
    split_tokens,
)
from gpt2lab.errors import CorpusError
from gpt2lab.models import GPT
from gpt2lab.training import CheckpointManager, Metrics, Trainer
from gpt2lab.training.callbacks import Callback, PrintCallback
from gpt2lab.training.runtime import configure_runtime

TokenizerFactory = Callable[[str], Tokenizer]


@dataclass(slots=True)
class Experiment:
    """A completely wired training run."""

    trainer: Trainer
    corpus: TextCorpus

    def run(self, resume_from: Path | None = None) -> Metrics:
        return self.trainer.run(resume_from)


def build_experiment(
    config: ExperimentConfig,
    *,
    tokenizer_factory: TokenizerFactory = TiktokenTokenizer,
    downloader: Downloader | None = None,
    device: torch.device | None = None,
    callbacks: list[Callback] | None = None,
) -> Experiment:
    """Create application objects; callers may replace every external boundary."""
    tokenizer = tokenizer_factory(config.data.tokenizer_name)
    config.validate_vocabulary(tokenizer.n_vocab)
    corpus = TextCorpus(config.data, tokenizer, downloader)
    train_tokens, validation_tokens = split_tokens(
        corpus.tokens(), config.data.validation_fraction
    )
    try:
        train_loader = SequentialTokenLoader(
            train_tokens,
            config.data.batch_size,
            config.data.sequence_length,
            pin_memory=config.data.pin_memory and torch.cuda.is_available(),
        )
        validation_loader = SequentialTokenLoader(
            validation_tokens,
            config.data.batch_size,
            config.data.sequence_length,
            pin_memory=config.data.pin_memory and torch.cuda.is_available(),
        )
    except ValueError as exc:
        raise CorpusError(
            f"corpus split cannot produce configured batches: {exc}"
        ) from exc
    selected_device = device if device is not None else configure_runtime(config.runtime)
    model = GPT(config.model, tokenizer.n_vocab)
    trainer = Trainer(
        config,
        model,
        train_loader,
        validation_loader,
        selected_device,
        callbacks=(
            callbacks
            if callbacks is not None
            else [PrintCallback(config.training.log_every)]
        ),
    )
    return Experiment(trainer, corpus)


def sample_text(
    config: ExperimentConfig,
    checkpoint: Path,
    prompt: str,
    *,
    tokenizer_factory: TokenizerFactory = TiktokenTokenizer,
    device: torch.device | None = None,
) -> str:
    """Load a trusted checkpoint and generate text from a prompt."""
    tokenizer = tokenizer_factory(config.data.tokenizer_name)
    config.validate_vocabulary(tokenizer.n_vocab)
    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        raise CorpusError("prompt produced no tokens")
    selected_device = device if device is not None else configure_runtime(config.runtime)
    model = GPT(config.model, tokenizer.n_vocab).to(selected_device)
    manager = CheckpointManager(checkpoint.parent, 1)
    manager.restore(checkpoint, model, None, selected_device)
    context = torch.tensor([token_ids], dtype=torch.long, device=selected_device)
    generated = model.generate(
        context,
        config.sampling.max_new_tokens,
        config.sampling.temperature,
        config.sampling.top_k,
    )
    return tokenizer.decode(generated[0].tolist())

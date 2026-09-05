from __future__ import annotations

from pathlib import Path

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


@pytest.fixture
def tiny_config(tmp_path: Path) -> ExperimentConfig:
    return ExperimentConfig(
        name="test-run",
        data=DataConfig(
            file_name="corpus.txt",
            cache_dir=tmp_path,
            batch_size=1,
            sequence_length=3,
            validation_fraction=0.25,
            pin_memory=False,
        ),
        runtime=RuntimeConfig(device="cpu", use_mixed_precision=False),
        model=ModelConfig(
            block_size=4,
            n_layer=1,
            n_head=2,
            n_embd=8,
            mlp_ratio=2,
        ),
        optimizer=OptimizerConfig(
            learning_rate=0.01,
            weight_decay=0.01,
            schedule="cosine",
            warmup_steps=1,
            min_learning_rate=0.001,
        ),
        training=TrainingConfig(
            num_steps=2,
            log_every=1,
            eval_every=1,
            eval_steps=1,
            checkpoint_every=1,
            checkpoint_dir=tmp_path / "checkpoints",
            keep_last_checkpoints=1,
            metrics_csv=tmp_path / "metrics.csv",
        ),
        sampling=SamplingConfig(max_new_tokens=1, temperature=1.0, top_k=4),
    )

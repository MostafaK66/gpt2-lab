from __future__ import annotations

from pathlib import Path

import pytest
import torch

from gpt2lab import cli
from gpt2lab.config import ExperimentConfig
from gpt2lab.errors import CorpusError
from gpt2lab.experiment import build_experiment, sample_text
from gpt2lab.models import GPT
from gpt2lab.training import CheckpointManager, Metrics
from gpt2lab.training.optim import build_optimizer


class FakeTokenizer:
    n_vocab = 16

    def encode(self, text: str) -> list[int]:
        return [ord(character) % self.n_vocab for character in text]

    def decode(self, tokens: list[int]) -> str:
        return " ".join(map(str, tokens))


def tokenizer_factory(_name: str) -> FakeTokenizer:
    return FakeTokenizer()


def test_build_experiment_is_offline_and_runs(
    tiny_config: ExperimentConfig, tmp_path: Path
) -> None:
    tiny_config.data.file_path.write_text(
        "abcdefghijklmnopqrstuvwxyz012345", encoding="utf-8"
    )
    experiment = build_experiment(
        tiny_config,
        tokenizer_factory=tokenizer_factory,
        device=torch.device("cpu"),
        callbacks=[],
    )
    assert experiment.corpus.path == tiny_config.data.file_path
    assert experiment.run().steps == [0, 1]


def test_build_experiment_reports_impossible_batches(
    tiny_config: ExperimentConfig,
) -> None:
    tiny_config.data.file_path.write_text("12345678", encoding="utf-8")
    with pytest.raises(CorpusError, match="cannot produce"):
        build_experiment(
            tiny_config,
            tokenizer_factory=tokenizer_factory,
            device=torch.device("cpu"),
        )


def test_sample_text_loads_checkpoint(
    tiny_config: ExperimentConfig, tmp_path: Path
) -> None:
    model = GPT(tiny_config.model, 16)
    optimizer = build_optimizer(model, tiny_config.optimizer)
    checkpoint = CheckpointManager(tmp_path, 1).save(
        "sample.pt", model, optimizer, 0, tiny_config, Metrics()
    )
    torch.manual_seed(1)
    sampled = sample_text(
        tiny_config,
        checkpoint,
        "abc",
        tokenizer_factory=tokenizer_factory,
        device=torch.device("cpu"),
    )
    assert len(sampled.split()) == 4
    with pytest.raises(CorpusError, match="prompt"):
        sample_text(
            tiny_config,
            checkpoint,
            "",
            tokenizer_factory=tokenizer_factory,
            device=torch.device("cpu"),
        )


def test_cli_config_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config = tmp_path / "config.toml"
    config.write_text(
        '[optimizer]\nschedule = "constant"\nwarmup_steps = 0\n', encoding="utf-8"
    )
    assert cli.main(["config", "--config", str(config)]) == 0
    assert '"name": "gpt2-practice"' in capsys.readouterr().out


def test_cli_train_and_sample_dispatch(
    tiny_config: ExperimentConfig,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[optimizer]\nschedule = "constant"\nwarmup_steps = 0\n', encoding="utf-8"
    )
    calls: list[object] = []

    class FakeExperiment:
        def run(self, resume: Path | None) -> Metrics:
            calls.append(resume)
            return Metrics()

    monkeypatch.setattr(cli, "build_experiment", lambda _config: FakeExperiment())
    monkeypatch.setattr(cli, "sample_text", lambda *_args: "generated")
    assert (
        cli.main(
            [
                "train",
                "--config",
                str(config_path),
                "--resume",
                str(tmp_path / "resume.pt"),
            ]
        )
        == 0
    )
    assert calls == [tmp_path / "resume.pt"]
    assert (
        cli.main(
            [
                "sample",
                "--config",
                str(config_path),
                "--checkpoint",
                str(tmp_path / "model.pt"),
                "--prompt",
                "Hi",
            ]
        )
        == 0
    )
    assert capsys.readouterr().out.strip() == "generated"


def test_cli_reports_domain_errors(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["config", "--config", str(tmp_path / "missing.toml")]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_parser_requires_command() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])

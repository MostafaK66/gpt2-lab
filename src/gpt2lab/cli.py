"""Thin command-line interface for gpt2-lab."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from gpt2lab.config import ExperimentConfig
from gpt2lab.errors import GPT2LabError
from gpt2lab.experiment import build_experiment, sample_text


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gpt2lab", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train a model")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--resume", type=Path)

    sample = commands.add_parser("sample", help="sample from a trusted checkpoint")
    sample.add_argument("--config", type=Path, required=True)
    sample.add_argument("--checkpoint", type=Path, required=True)
    sample.add_argument("--prompt", required=True)

    describe = commands.add_parser("config", help="validate and print configuration")
    describe.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = ExperimentConfig.from_toml(args.config)
        if args.command == "train":
            build_experiment(config).run(args.resume)
        elif args.command == "sample":
            print(sample_text(config, args.checkpoint, args.prompt))
        else:
            print(config.to_json())
    except (GPT2LabError, ValueError) as exc:
        print(f"gpt2lab: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

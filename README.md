# gpt2-lab

`gpt2-lab` is an educational, from-scratch GPT-2 training repository built on
PyTorch. It keeps model code small enough to study while treating configuration,
data retrieval, checkpoints, errors, tests, and packaging as production concerns.

The default example trains a compact decoder-only transformer on Tiny Shakespeare.
No datasets, model weights, or checkpoints are included in this repository.

## Requirements

- Python 3.11 or 3.12
- CPU, CUDA, or Apple Metal (MPS) for training
- Network access only when a configured corpus is not already cached

A CPU is sufficient for tests and small experiments. The example configuration is
intended for practice; 5,000 steps can still take substantial time on a CPU.

## Installation

### Linux and macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

Install development tools and plotting support with:

```bash
python -m pip install -e ".[dev,plot]"
```

### Windows PowerShell

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
```

For CUDA, select the PyTorch build appropriate for your system using the official
PyTorch installation instructions before installing this package.

## Configuration

Copy [`configs/example.toml`](configs/example.toml), keep local overrides in a file
ending in `.local.toml`, and validate the result:

```bash
cp configs/example.toml configs/practice.local.toml
gpt2lab config --config configs/practice.local.toml
```

Configuration is immutable after loading. Unknown fields, unsafe corpus URLs,
invalid dimensions, impossible warmup schedules, and incompatible batch geometry
fail early with actionable errors.

Set `data.source_url` to an HTTPS URL for download-once caching. To use an existing
local UTF-8 corpus without network access, omit `source_url` and set `cache_dir` and
`file_name` to its location. The file name cannot contain path traversal segments.

## Train

```bash
gpt2lab train --config configs/practice.local.toml
```

Resume from a checkpoint created by this repository:

```bash
gpt2lab train \
  --config configs/practice.local.toml \
  --resume checkpoints/step_00001000.pt
```

The trainer moves batches to the selected device, uses mixed precision only on
CUDA, applies gradient clipping, evaluates periodically, writes metrics to CSV,
and rotates numbered checkpoints.

## Sample

```bash
gpt2lab sample \
  --config configs/practice.local.toml \
  --checkpoint checkpoints/final.pt \
  --prompt "ROMEO:"
```

PyTorch checkpoints can execute code while loading. Only load checkpoints you
created or obtained from a source you trust. Model downloads and pretrained GPT-2
weight conversion are intentionally outside this repository's scope.

## Architecture

- `config/` validates frozen data, runtime, model, optimizer, training, and sampling
  settings and safely loads TOML with the Python standard library.
- `data/` separates HTTPS retrieval, tokenization, corpus caching, splitting, and
  deterministic sequential batching. Protocols make network-free tests possible.
- `models/` implements learned position embeddings, pre-normalized causal attention,
  GELU MLPs, residual connections, and tied token/output embeddings.
- `training/` owns device selection, AMP, AdamW grouping, cosine scheduling,
  checkpoint rotation, metrics, and the training loop.
- `experiment.py` is the composition root; `cli.py` only parses arguments and
  reports domain errors.

Hardware-dependent behavior is isolated in `training/runtime.py`. Network access is
isolated behind `Downloader`, and tokenizer construction is injected at the
experiment boundary. Unit tests use tiny tensors and models and require neither a
network connection nor an accelerator.

## Development

```bash
make lint
make typecheck
make test
make build
make quality
```

The coverage gate is branch-aware and fails below 90%. CI runs Ruff, pytest with
coverage, strict mypy, bytecode compilation, and a package build on Python 3.11 and
3.12.

## License and attribution

MIT licensed. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE). This repository keeps
the history of the original `gpt2-lab` practice project and does not bundle OpenAI
model weights.

"""Corpus retrieval and tokenization behind injectable boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import requests
import tiktoken
import torch

from gpt2lab.config import DataConfig
from gpt2lab.errors import CorpusError


class Tokenizer(Protocol):
    """The tokenizer behavior required by the application."""

    @property
    def n_vocab(self) -> int: ...

    def encode(self, text: str) -> list[int]: ...

    def decode(self, tokens: list[int]) -> str: ...


class Downloader(Protocol):
    """A replaceable network-to-filesystem boundary."""

    def download(self, url: str, destination: Path) -> None: ...


class TiktokenTokenizer:
    """Adapter around a named tiktoken encoding."""

    def __init__(self, name: str) -> None:
        try:
            self._encoding = tiktoken.get_encoding(name)
        except (KeyError, ValueError) as exc:
            raise CorpusError(f"unknown tokenizer encoding {name!r}") from exc

    @property
    def n_vocab(self) -> int:
        return self._encoding.n_vocab

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode_ordinary(text)

    def decode(self, tokens: list[int]) -> str:
        try:
            return self._encoding.decode(tokens)
        except (KeyError, ValueError) as exc:
            raise CorpusError("token ids cannot be decoded by this tokenizer") from exc


class HTTPSDownloader:
    """Download an HTTPS resource with a timeout and atomic caller handoff."""

    def __init__(self, timeout_seconds: float = 30.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._timeout_seconds = timeout_seconds

    def download(self, url: str, destination: Path) -> None:
        try:
            with requests.get(
                url, timeout=self._timeout_seconds, stream=True
            ) as response:
                response.raise_for_status()
                with destination.open("wb") as stream:
                    for chunk in response.iter_content(chunk_size=64 * 1024):
                        stream.write(chunk)
        except (OSError, requests.RequestException) as exc:
            raise CorpusError(f"failed to download corpus from {url}: {exc}") from exc


class TextCorpus:
    """A cached UTF-8 corpus encoded lazily with an injected tokenizer."""

    def __init__(
        self,
        config: DataConfig,
        tokenizer: Tokenizer,
        downloader: Downloader | None = None,
    ) -> None:
        self.config = config
        self.tokenizer = tokenizer
        self.downloader = downloader or HTTPSDownloader()
        self._tokens: torch.Tensor | None = None

    @property
    def path(self) -> Path:
        return self.config.file_path

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.n_vocab

    def ensure_available(self) -> Path:
        if self.path.is_file():
            return self.path
        if self.path.exists():
            raise CorpusError(f"corpus path is not a regular file: {self.path}")
        if self.config.source_url is None:
            raise CorpusError(
                f"corpus file is missing and no source_url is configured: {self.path}"
            )

        partial = self.path.with_name(f"{self.path.name}.part")
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            partial.unlink(missing_ok=True)
            self.downloader.download(self.config.source_url, partial)
            if not partial.is_file() or partial.stat().st_size == 0:
                raise CorpusError("downloaded corpus is empty")
            partial.replace(self.path)
        except (OSError, CorpusError) as exc:
            raise CorpusError(f"could not prepare corpus at {self.path}: {exc}") from exc
        finally:
            partial.unlink(missing_ok=True)
        return self.path

    def tokens(self) -> torch.Tensor:
        if self._tokens is not None:
            return self._tokens
        path = self.ensure_available()
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            raise CorpusError(f"cannot read UTF-8 corpus {path}: {exc}") from exc
        try:
            token_ids = self.tokenizer.encode(text)
        except (KeyError, TypeError, ValueError) as exc:
            raise CorpusError(f"cannot tokenize corpus {path}: {exc}") from exc
        if not token_ids:
            raise CorpusError(f"corpus {path} produced no tokens")
        self._tokens = torch.tensor(token_ids, dtype=torch.long)
        return self._tokens

    def decode(self, token_ids: list[int]) -> str:
        try:
            return self.tokenizer.decode(token_ids)
        except (KeyError, TypeError, ValueError) as exc:
            raise CorpusError(f"cannot decode token ids: {exc}") from exc


def split_tokens(
    tokens: torch.Tensor, validation_fraction: float
) -> tuple[torch.Tensor, torch.Tensor]:
    """Split a 1-D token stream into non-overlapping train and validation data."""
    if tokens.ndim != 1:
        raise CorpusError(f"tokens must be 1-D; got shape {tuple(tokens.shape)}")
    if not 0.0 < validation_fraction < 1.0:
        raise CorpusError("validation_fraction must be in (0, 1)")
    split_at = int(tokens.numel() * (1.0 - validation_fraction))
    if split_at < 2 or tokens.numel() - split_at < 2:
        raise CorpusError("corpus is too small to create train and validation splits")
    return tokens[:split_at], tokens[split_at:]

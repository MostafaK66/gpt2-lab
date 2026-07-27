"""Download, cache and tokenize a plain-text corpus."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve

import tiktoken
import torch

from gpt2lab.config import DataConfig
from gpt2lab.utils.logging import get_logger

__all__ = ["TextCorpus"]

LOGGER = get_logger(__name__)


class TextCorpus:
    """A UTF-8 text file plus the tokenizer used to encode it.

    The download happens at most once; the token tensor is built lazily and
    then cached on the instance.
    """

    def __init__(self, config: DataConfig) -> None:
        self._config = config
        self._encoding = tiktoken.get_encoding(config.tokenizer_name)
        self._tokens: torch.Tensor | None = None

    # -- properties ---------------------------------------------------------

    @property
    def path(self) -> Path:
        return self._config.file_path

    @property
    def tokenizer_name(self) -> str:
        return self._config.tokenizer_name

    @property
    def vocab_size(self) -> int:
        return self._encoding.n_vocab

    @property
    def encoding(self) -> tiktoken.Encoding:
        return self._encoding

    # -- behaviour ----------------------------------------------------------

    def download_if_missing(self) -> Path:
        """Fetch the corpus unless it is already cached.

        The download is written to a temporary file first, so an interrupted
        run can never leave a truncated corpus behind.
        """
        path = self.path

        if path.exists():
            LOGGER.info("Corpus already cached at %s", path.resolve())
            return path

        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.parent / (path.name + ".part")

        LOGGER.info("Downloading corpus from %s", self._config.source_url)
        try:
            urlretrieve(self._config.source_url, partial)
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)

        LOGGER.info("Corpus saved to %s", path.resolve())
        return path

    def tokens(self) -> torch.Tensor:
        """Return the whole corpus as a 1-D ``int64`` tensor of token ids."""
        if self._tokens is not None:
            return self._tokens

        path = self.download_if_missing()
        text = path.read_text(encoding="utf-8")

        # encode_ordinary ignores special tokens instead of raising on them,
        # which keeps arbitrary corpora safe to load.
        token_ids = self._encoding.encode_ordinary(text)

        if not token_ids:
            raise ValueError(f"Corpus {path} produced zero tokens.")

        self._tokens = torch.tensor(token_ids, dtype=torch.long)

        LOGGER.info(
            "Tokenized %s characters into %s tokens (tokenizer=%s, vocab=%s)",
            f"{len(text):,}",
            f"{self._tokens.numel():,}",
            self.tokenizer_name,
            f"{self.vocab_size:,}",
        )
        return self._tokens

    def decode(self, token_ids: list[int]) -> str:
        return self._encoding.decode(token_ids)


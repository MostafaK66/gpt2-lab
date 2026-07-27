"""Corpus handling and batching."""

from gpt2lab.data.corpus import TextCorpus
from gpt2lab.data.loader import SequentialTokenLoader

__all__ = ["SequentialTokenLoader", "TextCorpus"]
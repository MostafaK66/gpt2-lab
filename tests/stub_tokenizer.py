"""Stub tokenizer for offline testing."""
from __future__ import annotations


class StubTokenizer:
    """Mimics tiktoken interface without network access."""

    def __init__(self, vocab_size: int = 256) -> None:
        self.vocab_size = vocab_size

    def encode(self, text: str) -> list[int]:
        return [b % self.vocab_size for b in text.encode("utf-8")]

    def decode(self, tokens: list[int]) -> str:
        return bytes(t % 256 for t in tokens).decode("utf-8", errors="replace")


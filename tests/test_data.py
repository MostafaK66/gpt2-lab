from __future__ import annotations

from pathlib import Path
from typing import BinaryIO

import pytest
import torch

from gpt2lab.config import DataConfig
from gpt2lab.data import (
    HTTPSDownloader,
    SequentialTokenLoader,
    TextCorpus,
    TiktokenTokenizer,
    split_tokens,
)
from gpt2lab.errors import CorpusError


class FakeTokenizer:
    n_vocab = 32

    def encode(self, text: str) -> list[int]:
        return [ord(character) % self.n_vocab for character in text]

    def decode(self, tokens: list[int]) -> str:
        return ",".join(str(token) for token in tokens)


class FakeDownloader:
    def __init__(self, content: bytes = b"downloaded") -> None:
        self.content = content
        self.calls: list[tuple[str, Path]] = []

    def download(self, url: str, destination: Path) -> None:
        self.calls.append((url, destination))
        destination.write_bytes(self.content)


class BrokenDownloader:
    def download(self, url: str, destination: Path) -> None:
        destination.write_bytes(b"partial")
        raise CorpusError("network down")


def test_existing_corpus_is_cached_and_decoded(tmp_path: Path) -> None:
    path = tmp_path / "text.txt"
    path.write_text("abc", encoding="utf-8")
    downloader = FakeDownloader()
    corpus = TextCorpus(
        DataConfig(cache_dir=tmp_path, file_name="text.txt"),
        FakeTokenizer(),
        downloader,
    )
    first = corpus.tokens()
    assert first.tolist() == [1, 2, 3]
    assert corpus.tokens() is first
    assert corpus.vocab_size == 32
    assert corpus.decode([1, 2]) == "1,2"
    assert downloader.calls == []


def test_missing_corpus_is_downloaded_atomically(tmp_path: Path) -> None:
    downloader = FakeDownloader(b"hello")
    config = DataConfig(
        source_url="https://example.com/text",
        cache_dir=tmp_path,
        file_name="text.txt",
    )
    corpus = TextCorpus(config, FakeTokenizer(), downloader)
    assert corpus.ensure_available().read_bytes() == b"hello"
    assert downloader.calls[0][0] == "https://example.com/text"
    assert not (tmp_path / "text.txt.part").exists()


def test_corpus_availability_errors_are_actionable(tmp_path: Path) -> None:
    missing = TextCorpus(
        DataConfig(cache_dir=tmp_path, file_name="missing.txt"), FakeTokenizer()
    )
    with pytest.raises(CorpusError, match="no source_url"):
        missing.ensure_available()

    directory = tmp_path / "directory"
    directory.mkdir()
    wrong_type = TextCorpus(
        DataConfig(cache_dir=tmp_path, file_name="directory"), FakeTokenizer()
    )
    with pytest.raises(CorpusError, match="regular file"):
        wrong_type.ensure_available()

    broken = TextCorpus(
        DataConfig(
            source_url="https://example.com/text",
            cache_dir=tmp_path,
            file_name="broken.txt",
        ),
        FakeTokenizer(),
        BrokenDownloader(),
    )
    with pytest.raises(CorpusError, match="network down"):
        broken.ensure_available()
    assert not (tmp_path / "broken.txt.part").exists()


def test_empty_download_and_empty_tokenization_are_rejected(tmp_path: Path) -> None:
    empty_download = TextCorpus(
        DataConfig(
            source_url="https://example.com/empty",
            cache_dir=tmp_path,
            file_name="empty.txt",
        ),
        FakeTokenizer(),
        FakeDownloader(b""),
    )
    with pytest.raises(CorpusError, match="empty"):
        empty_download.ensure_available()

    path = tmp_path / "blank.txt"
    path.write_text("", encoding="utf-8")
    empty_tokens = TextCorpus(
        DataConfig(cache_dir=tmp_path, file_name="blank.txt"), FakeTokenizer()
    )
    with pytest.raises(CorpusError, match="no tokens"):
        empty_tokens.tokens()


def test_invalid_utf8_is_reported(tmp_path: Path) -> None:
    (tmp_path / "bad.txt").write_bytes(b"\xff")
    corpus = TextCorpus(
        DataConfig(cache_dir=tmp_path, file_name="bad.txt"), FakeTokenizer()
    )
    with pytest.raises(CorpusError, match="UTF-8"):
        corpus.tokens()


def test_https_downloader_uses_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def raise_for_status(self) -> None:
            return None

        def iter_content(self, *, chunk_size: int) -> list[bytes]:
            observed["chunk_size"] = chunk_size
            return [b"bo", b"dy"]

    def fake_get(url: str, *, timeout: float, stream: bool) -> Response:
        observed.update(url=url, timeout=timeout, stream=stream)
        return Response()

    monkeypatch.setattr("gpt2lab.data.corpus.requests.get", fake_get)
    output = tmp_path / "output"
    HTTPSDownloader(2.5).download("https://example.com", output)
    assert output.read_bytes() == b"body"
    assert observed == {
        "url": "https://example.com",
        "timeout": 2.5,
        "stream": True,
        "chunk_size": 65_536,
    }
    with pytest.raises(ValueError, match="positive"):
        HTTPSDownloader(0)


def test_https_downloader_wraps_os_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_url: str, *, timeout: float, stream: bool) -> BinaryIO:
        assert stream
        raise OSError(f"timeout {timeout}")

    monkeypatch.setattr("gpt2lab.data.corpus.requests.get", fail)
    with pytest.raises(CorpusError, match="failed to download"):
        HTTPSDownloader().download("https://example.com", tmp_path / "out")


def test_tiktoken_adapter_round_trip() -> None:
    tokenizer = TiktokenTokenizer("gpt2")
    ids = tokenizer.encode("hello")
    assert tokenizer.n_vocab > 0
    assert tokenizer.decode(ids) == "hello"
    with pytest.raises(CorpusError, match="unknown tokenizer"):
        TiktokenTokenizer("not-an-encoding")


def test_split_tokens() -> None:
    train, validation = split_tokens(torch.arange(10), 0.2)
    assert train.tolist() == list(range(8))
    assert validation.tolist() == [8, 9]
    with pytest.raises(CorpusError, match="1-D"):
        split_tokens(torch.ones((2, 2)), 0.2)
    with pytest.raises(CorpusError, match="fraction"):
        split_tokens(torch.arange(10), 1.0)
    with pytest.raises(CorpusError, match="too small"):
        split_tokens(torch.arange(3), 0.5)


@pytest.mark.parametrize(
    "tokens,batch_size,sequence_length,message",
    [
        (torch.arange(10), 0, 2, "batch_size"),
        (torch.arange(10), 1, 0, "sequence_length"),
        (torch.ones((2, 2), dtype=torch.long), 1, 2, "1-D"),
        (torch.arange(10, dtype=torch.int32), 1, 2, "long"),
        (torch.arange(4), 2, 2, "requires"),
    ],
)
def test_loader_validation(
    tokens: torch.Tensor, batch_size: int, sequence_length: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SequentialTokenLoader(tokens, batch_size, sequence_length)


def test_loader_batches_wrap_and_iterate() -> None:
    loader = SequentialTokenLoader(torch.arange(7), 1, 3, pin_memory=True)
    assert loader.batches_per_epoch == 2
    inputs, targets = loader.next_batch()
    assert inputs.tolist() == [[0, 1, 2]]
    assert targets.tolist() == [[1, 2, 3]]
    assert loader.position == 3
    second_inputs, _ = next(iter(loader))
    assert second_inputs.tolist() == [[3, 4, 5]]
    assert loader.position == 0
    loader.reset()
    assert loader.position == 0

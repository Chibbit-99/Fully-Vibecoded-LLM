"""Unit tests for the tokenizer (no checkpoint needed)."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinyllm.tokenizer import ByteBPE, split_chunks


def test_chunking_basic():
    assert split_chunks(b"First Citizen") == [b"First", b" ", b"Citizen"]
    assert split_chunks(b"abc123") == [b"abc", b"123"]
    assert split_chunks(b"  \n\t") == [b"  \n\t"]
    assert split_chunks(b"") == []


def test_chunking_keeps_non_ascii_bytes_separate():
    chunks = split_chunks("héllo".encode())
    assert b"".join(chunks) == "héllo".encode()
    # every non-classifiable byte is its own chunk -> guaranteed losslessness
    for c in chunks:
        assert len(c) == 1 or all(32 <= b < 127 or b in (9, 10, 13) for b in c)


def test_chunking_reconstructs_input():
    rng = random.Random(0)
    data = bytes(rng.randrange(256) for _ in range(2000))
    assert b"".join(split_chunks(data)) == data


@pytest.fixture(scope="module")
def tokenizer():
    text = Path(__file__).resolve().parent.parent / "data" / "tinyshakespeare.txt"
    if not text.exists():
        pytest.skip("dataset not available")
    return ByteBPE.train(text.read_text()[:200_000], vocab_size=600)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "hello",
        "ROMEO:\nShall I hear more?",
        "héllo wörld ✨",
        "1234567890",
        "\n\n\n\t ",
        "ALL CAPS!",
        "mixed123abc",
        "\x00\x01\x02",
        "🙂🙃 emoji",
    ],
)
def test_encode_decode_roundtrip(tokenizer, text):
    ids = tokenizer.encode(text)
    assert all(0 <= i < tokenizer.vocab_size for i in ids)
    assert tokenizer.decode(ids) == text


def test_vocab_size(tokenizer):
    assert tokenizer.vocab_size == 256 + len(tokenizer.merges) == 600


def test_compression(tokenizer):
    text = Path(__file__).resolve().parent.parent / "data" / "tinyshakespeare.txt"
    if not text.exists():
        pytest.skip("dataset not available")
    sample = text.read_text()[:50_000]
    ids = tokenizer.encode(sample)
    assert len(ids) < len(sample) * 0.8, "BPE should compress natural language"


def test_serialisation_roundtrip(tokenizer, tmp_path):
    path = tmp_path / "tok.json"
    tokenizer.save(path)
    loaded = ByteBPE.load(path)
    text = "ROMEO: I would thou wert"
    assert loaded.encode(text) == tokenizer.encode(text)
    assert loaded.merges == tokenizer.merges


def test_frequency_ordering():
    """Frequent pairs must be merged before rare ones."""
    tok = ByteBPE.train("the the the the cat", vocab_size=270)
    assert tok.merges, "expected at least one merge"
    ids = tok.encode("the")
    assert len(ids) == 1, f"'the' should collapse to one token, got {ids}"


def test_invalid_token_id(tokenizer):
    with pytest.raises(ValueError):
        tokenizer.decode([tokenizer.vocab_size + 5])

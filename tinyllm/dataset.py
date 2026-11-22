"""Dataset loading and batching for Tiny Shakespeare.

The corpus is stored as raw UTF-8 text in ``data/tinyshakespeare.txt`` and is
tokenized on the fly with the BPE tokenizer.  Token ids are cached in memory
for the duration of a training run.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np

from .tokenizer import ByteBPE

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "tinyshakespeare.txt"


def load_text(path: str | Path = DEFAULT_DATA_PATH) -> str:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"dataset not found at {path}. Run `python scripts/download_data.py`."
        )
    return path.read_text(encoding="utf-8")


def train_val_split(text: str, val_fraction: float = 0.1) -> tuple[str, str]:
    """Contiguous split: the tail of the corpus is validation (no leakage)."""
    n = len(text)
    cut = int(n * (1.0 - val_fraction))
    # snap to a newline so we don't cut a word in half
    cut = text.rfind("\n", 0, cut) or cut
    return text[:cut], text[cut:]


@dataclass
class TokenData:
    train: np.ndarray
    val: np.ndarray
    tokenizer: ByteBPE

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size


def encode_corpus(tokenizer: ByteBPE, text_path: str | Path = DEFAULT_DATA_PATH) -> TokenData:
    text = load_text(text_path)
    train_text, val_text = train_val_split(text)
    train_ids = np.array(tokenizer.encode(train_text), dtype=np.uint16)
    val_ids = np.array(tokenizer.encode(val_text), dtype=np.uint16)
    return TokenData(train=train_ids, val=val_ids, tokenizer=tokenizer)


def get_batch(
    data: np.ndarray,
    batch_size: int,
    block_size: int,
    rng: np.random.Generator,
    device: str = "cpu",
):
    """Sample (x, y) where y is x shifted one token to the right."""
    import torch

    if len(data) <= block_size + 1:
        raise ValueError("corpus too small for block_size")
    ix = rng.integers(0, len(data) - block_size - 1, size=batch_size)
    x = np.stack([data[i : i + block_size] for i in ix]).astype(np.int64)
    y = np.stack([data[i + 1 : i + block_size + 1] for i in ix]).astype(np.int64)
    return (
        torch.from_numpy(x).to(device),
        torch.from_numpy(y).to(device),
    )


def iter_sequential_batches(data: np.ndarray, batch_size: int, block_size: int) -> List[tuple[int, int]]:
    """Spans covering the whole array (used for validation loss)."""
    spans = []
    stride = block_size * batch_size
    for start in range(0, len(data) - block_size - 1, stride):
        spans.append((start, min(start + stride, len(data) - block_size - 1)))
    return spans

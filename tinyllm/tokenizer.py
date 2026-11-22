"""A byte-level BPE tokenizer, implemented from scratch and ported to JS.

Design goals
------------
* Lossless: any input round-trips exactly (we start from raw UTF-8 bytes).
* Dependency free: no `regex`, no `tokenizers`, no `tiktoken`.
* Portable: the exact same algorithm ships in ``web/tokenizer.js`` so the
  browser can tokenise identically to Python.  Tests assert parity.

How it works
------------
1. The text is UTF-8 encoded into bytes.
2. Bytes are split into *chunks* on character classes: runs of ASCII letters,
   runs of digits, runs of whitespace, and single other bytes.  This mirrors
   the "pre-tokenisation" step of GPT-2 but with hand-written rules so Python
   and JavaScript behave identically (no difference in regex flavours).
3. Inside every chunk, the highest-priority byte-pair merge is applied
   repeatedly until no merge applies.
4. The resulting merge ids are mapped to vocabulary ids:
   ``id < 256`` -> the raw byte, ``id >= 256`` -> BPE merge ``id - 256``.

Training counts pairs over unique chunks weighted by frequency (the classic
word-level BPE trick) which keeps training fast in plain Python.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------

LETTER = 1
DIGIT = 2
SPACE = 3
OTHER = 4


def _byte_class(b: int) -> int:
    if 65 <= b <= 90 or 97 <= b <= 122:  # A-Z a-z
        return LETTER
    if 48 <= b <= 57:  # 0-9
        return DIGIT
    if b in (32, 9, 10, 13):  # space, tab, LF, CR
        return SPACE
    return OTHER


def split_chunks(data: bytes) -> List[bytes]:
    """Split encoded bytes into pre-tokenisation chunks (deterministic)."""
    chunks: List[bytes] = []
    if not data:
        return chunks
    start = 0
    prev = _byte_class(data[0])
    for i in range(1, len(data)):
        cur = _byte_class(data[i])
        # OTHER bytes are always kept as single-byte chunks so that every
        # byte can be emitted on its own (guarantees losslessness).
        if cur != prev or cur == OTHER:
            chunks.append(data[start:i])
            start = i
        prev = cur
    chunks.append(data[start:])
    return chunks


def _apply_merges(chunk: bytes, ranks: Dict[Tuple[int, int], int]) -> List[int]:
    """Greedy BPE on a single chunk: repeatedly merge the lowest-rank pair."""
    parts: List[int] = list(chunk)
    while len(parts) > 1:
        best_rank = None
        best_i = -1
        for i in range(len(parts) - 1):
            r = ranks.get((parts[i], parts[i + 1]))
            if r is not None and (best_rank is None or r < best_rank):
                best_rank = r
                best_i = i
        if best_i < 0:
            break
        merged = 256 + best_rank  # type: ignore[operator]
        parts[best_i : best_i + 2] = [merged]
    return parts


# ---------------------------------------------------------------------------
# tokenizer
# ---------------------------------------------------------------------------


class ByteBPE:
    """Byte-level BPE tokenizer with a JSON-serialisable vocabulary."""

    def __init__(self, merges: Sequence[Tuple[int, int]] = ()) -> None:
        # merges[i] == (a, b) means a + b -> 256 + i
        self.merges: List[Tuple[int, int]] = [tuple(m) for m in merges]  # type: ignore[misc]
        self.ranks: Dict[Tuple[int, int], int] = {m: i for i, m in enumerate(self.merges)}
        self.vocab_size = 256 + len(self.merges)

    # -- training ----------------------------------------------------------
    @classmethod
    def train(cls, text: str, vocab_size: int = 1280, verbose: bool = False) -> "ByteBPE":
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256")
        merges: List[Tuple[int, int]] = []
        ranks: Dict[Tuple[int, int], int] = {}

        chunk_counts = Counter(split_chunks(text.encode("utf-8")))
        # unique chunk -> list of ids (start as raw bytes)
        words: Dict[bytes, List[int]] = {c: list(c) for c in chunk_counts}

        for i in range(vocab_size - 256):
            pair_counts: Counter = Counter()
            for chunk, ids in words.items():
                freq = chunk_counts[chunk]
                for a, b in zip(ids, ids[1:]):
                    pair_counts[(a, b)] += freq
            if not pair_counts:
                break
            # deterministic: highest count, then lexicographically smallest pair
            best_pair, best_count = max(
                pair_counts.items(), key=lambda kv: (kv[1], (-kv[0][0], -kv[0][1]))
            )
            if best_count < 3:
                break
            new_id = 256 + len(merges)
            merges.append(best_pair)
            ranks[best_pair] = len(merges) - 1
            # rewrite the unique chunks
            updated: Dict[bytes, List[int]] = {}
            a, b = best_pair
            for chunk, ids in words.items():
                out: List[int] = []
                j = 0
                n = len(ids)
                while j < n:
                    if j < n - 1 and ids[j] == a and ids[j + 1] == b:
                        out.append(new_id)
                        j += 2
                    else:
                        out.append(ids[j])
                        j += 1
                updated[chunk] = out
            words = updated
            if verbose and (i + 1) % 128 == 0:
                print(f"  merge {i + 1:4d}: {best_pair} count={best_count}")

        return cls(merges)

    # -- encoding ----------------------------------------------------------
    def encode(self, text: str) -> List[int]:
        out: List[int] = []
        for chunk in split_chunks(text.encode("utf-8")):
            out.extend(_apply_merges(chunk, self.ranks))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        data = bytearray()
        for i in ids:
            if i < 0 or i >= self.vocab_size:
                raise ValueError(f"token id {i} outside vocabulary")
            data.extend(self._expand(i))
        return data.decode("utf-8", errors="replace")

    def _expand(self, token: int) -> bytes:
        if token < 256:
            return bytes((token,))
        a, b = self.merges[token - 256]
        return self._expand(a) + self._expand(b)

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> Dict:
        return {"type": "byte-bpe", "vocab_size": self.vocab_size, "merges": [list(m) for m in self.merges]}

    @classmethod
    def from_dict(cls, data: Dict) -> "ByteBPE":
        return cls([tuple(m) for m in data["merges"]])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPE":
        return cls.from_dict(json.loads(Path(path).read_text()))

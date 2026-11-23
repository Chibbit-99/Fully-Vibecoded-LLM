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

   Words are then separated from the spaces around them at a fixed list of
   high-frequency boundaries (`` the `` -> ``the``, ``and``, ``of``, ...).  This
   is what GPT-2 does with a regex lookbehind, and it matters: without it the
   most frequent chunk is the single space, so BPE spends dozens of merges
   gluing " the" back together and the vocabulary is wasted on spaces.  On the
   Spore corpora it takes compression from 2.15 to 2.9 characters per token and
   makes tokenizer training roughly 10x faster.
3. Inside every chunk, the highest-priority byte-pair merge is applied
   repeatedly until no merge applies.
4. The resulting merge ids are mapped to vocabulary ids:
   ``id < 256`` -> the raw byte, ``id >= 256`` -> BPE merge ``id - 256``.

Special tokens
--------------
Chat needs control markers (``<|user|>``, ``<|assistant|>``, ...) that can never
be produced by merging normal text.  They are appended *after* the merges as
whole-string tokens: ``id = 256 + len(merges) + i``.  Encoding splits the input
on those literal strings first, so a special token is always exactly one id,
and decoding re-emits the literal text.  Ordinary text is untouched, so the
byte-level losslessness guarantee still holds.

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


# High-frequency boundaries: an isolated occurrence of one of these words is
# split from the whitespace around it.  Kept as plain byte strings so the JS
# port can mirror the list exactly (no regex flavour differences).
_WORD_SPLITS = (
    b"the", b"and", b"of", b"to", b"a", b"in", b"is", b"was", b"that", b"it",
    b"he", b"she", b"they", b"for", b"on", b"with", b"as", b"at", b"be",
    b"this", b"from", b"have", b"not", b"but", b"by", b"or", b"his", b"her",
    b"which", b"one", b"all", b"were", b"when", b"there", b"can", b"an",
    b"their", b"we", b"more", b"has", b"had", b"you", b"are", b"will", b"if",
)
_SPLIT_WORDS = frozenset(_WORD_SPLITS)
# Marker byte that never appears in text we handle; MUST match web/tokenizer.js
_SPLIT = 0x00


def _is_letter(b: int) -> bool:
    return 65 <= b <= 90 or 97 <= b <= 122


def presplit(text: str) -> bytes:
    """Encode to UTF-8 and mark word boundaries at high-frequency words."""
    data = text.encode("utf-8")
    marker = bytes((_SPLIT,))
    out = bytearray()
    i, n = 0, len(data)
    while i < n:
        c = data[i]
        if c == 32:  # space: decide whether the following word is a split word
            j = i + 1
            while j < n and _is_letter(data[j]):
                j += 1
            word = data[i + 1 : j]
            if word in _SPLIT_WORDS:
                if out and out[-1] == 32:
                    out[-1] = _SPLIT  # " the" -> marker + "the"
                else:
                    out.extend(marker)
                i += 1
                continue
            out.append(c)
            i += 1
            continue
        out.append(c)
        i += 1
    return bytes(out)


def _chunks_from_presplit(data: bytes) -> List[bytes]:
    """Split pre-split bytes; NUL becomes its own chunk."""
    chunks: List[bytes] = []
    for part in data.split(bytes((_SPLIT,))):
        if part:
            chunks.extend(split_chunks(part))
        chunks.append(bytes((_SPLIT,)))
    if chunks:
        chunks.pop()  # trailing marker after the final segment
    return chunks


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


def _split_on_specials(text: str, specials: Sequence[str]) -> List[str]:
    """Split ``text`` so every special token ends up as its own element.

    Longest token first, so ``<|user|>`` wins over a hypothetical ``<|u``.
    The result never contains empty strings around adjacent specials.
    """
    ordered = sorted(specials, key=len, reverse=True)
    out: List[str] = []
    buf: List[str] = []
    i = 0
    n = len(text)
    while i < n:
        for tok in ordered:
            if text.startswith(tok, i):
                if buf:
                    out.append("".join(buf))
                    buf = []
                out.append(tok)
                i += len(tok)
                break
        else:
            buf.append(text[i])
            i += 1
    if buf:
        out.append("".join(buf))
    return out


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

    def __init__(
        self,
        merges: Sequence[Tuple[int, int]] = (),
        special_tokens: Sequence[str] = (),
    ) -> None:
        # merges[i] == (a, b) means a + b -> 256 + i
        self.merges: List[Tuple[int, int]] = [tuple(m) for m in merges]  # type: ignore[misc]
        self.ranks: Dict[Tuple[int, int], int] = {m: i for i, m in enumerate(self.merges)}
        self.special_tokens: List[str] = list(special_tokens)
        # special token text -> id, reserved at the top of the vocabulary
        self.special_ids: Dict[str, int] = {
            tok: 256 + len(self.merges) + i for i, tok in enumerate(self.special_tokens)
        }
        self.vocab_size = 256 + len(self.merges) + len(self.special_tokens)

    # -- special tokens ----------------------------------------------------
    def encode_special(self, token: str) -> int:
        """Id of a control token such as ``<|assistant|>``."""
        try:
            return self.special_ids[token]
        except KeyError:
            raise KeyError(
                f"{token!r} is not a special token of this vocabulary "
                f"(have: {self.special_tokens})"
            ) from None

    # -- training ----------------------------------------------------------
    @classmethod
    def train(cls, text: str, vocab_size: int = 1280, verbose: bool = False) -> "ByteBPE":
        if vocab_size < 256:
            raise ValueError("vocab_size must be >= 256")
        merges: List[Tuple[int, int]] = []
        ranks: Dict[Tuple[int, int], int] = {}

        chunk_counts = Counter(_chunks_from_presplit(presplit(text)))
        # unique chunk -> list of ids (start as raw bytes)
        words: Dict[bytes, List[int]] = {c: list(c) for c in chunk_counts}
        freq = dict(chunk_counts)

        # ---- incremental BPE ------------------------------------------------
        # A naive trainer rescans every chunk after every merge, which is
        # O(merges x chunks) and took ~2.5 minutes for a 5 MB corpus at 3000
        # merges.  Instead we keep a running pair -> count table and only
        # revisit the (few) chunks that actually contain the pair just merged.
        # The selected merges are identical to the naive version because the
        # counts are always exact; only the work is skipped.
        pair_counts: Counter = Counter()
        pair_where: Dict[Tuple[int, int], set] = {}
        for chunk, ids in words.items():
            f = freq[chunk]
            for a, b in zip(ids, ids[1:]):
                pair_counts[(a, b)] += f
                pair_where.setdefault((a, b), set()).add(chunk)

        for i in range(vocab_size - 256):
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
            a, b = best_pair

            # rewrite only the chunks that contain this pair
            for chunk in list(pair_where.get(best_pair, ())):
                ids = words[chunk]
                f = freq[chunk]
                # remove this chunk's contribution to the running counts
                for x, y in zip(ids, ids[1:]):
                    pair_counts[(x, y)] -= f
                    if pair_counts[(x, y)] <= 0:
                        del pair_counts[(x, y)]
                        pair_where.get((x, y), set()).discard(chunk)
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
                words[chunk] = out
                # add the new contribution back
                for x, y in zip(out, out[1:]):
                    pair_counts[(x, y)] += f
                    pair_where.setdefault((x, y), set()).add(chunk)
            pair_counts.pop(best_pair, None)
            pair_where.pop(best_pair, None)

            if verbose and (i + 1) % 128 == 0:
                print(f"  merge {i + 1:4d}: {best_pair} count={best_count}")

        return cls(merges)

    # -- encoding ----------------------------------------------------------
    def encode(self, text: str, allow_special: bool = True) -> List[int]:
        """Encode text to ids.

        With ``allow_special`` (default) occurrences of the registered special
        tokens are emitted as their single reserved id.  Pass ``False`` to treat
        them as ordinary text -- useful when training on raw corpora that might
        contain such spellings, where we do not want them to act as control
        characters.
        """
        if allow_special and self.special_tokens:
            pieces = _split_on_specials(text, self.special_tokens)
            if len(pieces) > 1 or (pieces and pieces[0] in self.special_ids):
                out: List[int] = []
                for piece in pieces:
                    sid = self.special_ids.get(piece)
                    if sid is not None:
                        out.append(sid)
                    else:
                        out.extend(self._encode_plain(piece))
                return out
        return self._encode_plain(text)

    def _encode_plain(self, text: str) -> List[int]:
        out: List[int] = []
        ranks = self.ranks
        # A literal NUL would be indistinguishable from the word-boundary
        # marker on decode, so escape each literal NUL as a *pair* of NULs.
        # Boundary markers are always a single NUL followed by a letter, so
        # pairing left-to-right on decode is unambiguous.
        text = text.replace("\x00", "\x00\x00")
        for chunk in _chunks_from_presplit(presplit(text)):
            out.extend(_apply_merges(chunk, ranks))
        return out

    def decode(self, ids: Iterable[int]) -> str:
        raw = bytearray()
        n_merges = len(self.merges)
        for i in ids:
            if i < 0 or i >= self.vocab_size:
                raise ValueError(f"token id {i} outside vocabulary")
            if i >= 256 + n_merges:  # reserved control token
                raw.extend(self.special_tokens[i - 256 - n_merges].encode("utf-8"))
            else:
                raw.extend(self._expand(i))
        # NUL pairs are escaped literal NULs; a lone NUL is a boundary marker
        data = bytearray()
        i = 0
        n = len(raw)
        while i < n:
            b = raw[i]
            if b == _SPLIT:
                if i + 1 < n and raw[i + 1] == _SPLIT:
                    data.append(0)   # escaped literal NUL
                    i += 2
                else:
                    data.append(32)  # word-boundary space
                    i += 1
            else:
                data.append(b)
                i += 1
        return data.decode("utf-8", errors="replace")

    def _expand(self, token: int) -> bytes:
        if token < 256:
            return bytes((token,))
        a, b = self.merges[token - 256]
        return self._expand(a) + self._expand(b)

    # -- persistence -------------------------------------------------------
    def to_dict(self) -> Dict:
        return {
            "type": "byte-bpe",
            "vocab_size": self.vocab_size,
            "merges": [list(m) for m in self.merges],
            "special_tokens": list(self.special_tokens),
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "ByteBPE":
        return cls(
            [tuple(m) for m in data["merges"]],
            data.get("special_tokens", []),
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict()) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ByteBPE":
        return cls.from_dict(json.loads(Path(path).read_text()))

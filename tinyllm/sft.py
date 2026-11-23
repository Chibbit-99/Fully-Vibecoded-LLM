"""Chat formatting and supervised-fine-tuning (SFT) data for the Spore models.

Spore is trained in two stages:

1. **Pretraining** on mixed prose (stories + encyclopaedia) so the model learns
   English, facts and long-range structure.
2. **Chat tuning** on rendered dialogues, so it learns the *shape* of a
   conversation: when to answer, when to stop, and how to use the retrieved
   context it is given.

Stage 2 is a real SFT loss: only the tokens the assistant actually said
contribute to the loss, everything else is masked to ``-100``.  That single
detail is what makes a 12M-parameter model behave like a chat model instead of
a document completer.

Wire format
-----------
Turns are rendered into one flat token stream using control tokens that the
byte-level BPE can never produce by accident (see :mod:`tinyllm.tokenizer`)::

    <|user|>What is the capital of France?<|assistant|>Paris.<|end|>
    <|user|>And of Japan?<|assistant|>Tokyo.<|end|>

The model is trained to emit ``<|end|>`` when an answer is finished, which is
also how generation knows to stop.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

# Control tokens, in vocabulary order.  Kept short on purpose: every extra
# token is a row of the embedding matrix and a column of the output head.
USER = "<|user|>"
ASSISTANT = "<|assistant|>"
END = "<|end|>"
# Marks the boundary between the retrieved knowledge block and the question,
# so the model can tell "this is reference material" from "this is the task".
CONTEXT = "<|context|>"

SPECIAL_TOKENS: Tuple[str, ...] = (USER, ASSISTANT, END, CONTEXT)

# Ignore index for cross-entropy: positions the model is not asked to predict.
IGNORE_INDEX = -100

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_QA_PATH = REPO_ROOT / "data" / "spore_qa.jsonl"


# ---------------------------------------------------------------------------
# chat templates
# ---------------------------------------------------------------------------
def render_prompt(user_text: str, context: str = "") -> str:
    """Everything *before* the assistant speaks (what inference feeds in)."""
    if context:
        return f"{USER}{context}{CONTEXT}{user_text}{ASSISTANT}"
    return f"{USER}{user_text}{ASSISTANT}"


def render_example(user_text: str, answer: str, context: str = "") -> str:
    """A full training record: prompt + answer + end-of-turn marker."""
    return render_prompt(user_text, context) + f"{answer}{END}"


def render_history(turns: Sequence[Tuple[str, str]], context: str = "") -> str:
    """Render a multi-turn history (list of ``(role, text)``) as a prompt.

    Roles are ``"user"`` and ``"assistant"``.  The returned string ends with
    ``<|assistant|>`` so the model continues the reply.
    """
    parts: List[str] = []
    first = True
    for role, text in turns:
        if role == "assistant":
            parts.append(f"{ASSISTANT}{text}{END}")
        else:
            if first and context:
                parts.append(f"{USER}{context}{CONTEXT}{text}")
            else:
                parts.append(f"{USER}{text}")
            first = False
    if turns and turns[-1][0] == "assistant":
        parts.append(USER)  # nudge it to speak again rather than stop
    return "".join(parts) + ASSISTANT


# ---------------------------------------------------------------------------
# tokenised SFT corpus
# ---------------------------------------------------------------------------
@dataclass
class SFTData:
    """Token ids plus a loss mask: ``labels[i] == -100`` means "do not train"."""

    input_ids: np.ndarray
    labels: np.ndarray
    tokenizer: object = None
    examples: int = 0
    supervised_tokens: int = 0
    # window starts that contain at least one assistant token; computed once
    valid_windows: List[int] = field(default_factory=list)

    @property
    def length(self) -> int:
        return int(len(self.input_ids))

    def window(self, start: int, block_size: int) -> Tuple[np.ndarray, np.ndarray]:
        """One ``(x, y)`` window; ``y`` is already masked for loss."""
        x = self.input_ids[start : start + block_size].astype(np.int64)
        y = self.labels[start + 1 : start + block_size + 1].astype(np.int64)
        if len(x) < block_size or len(y) < block_size:
            pad = block_size - len(x)
            x = np.concatenate([x, np.zeros(pad, dtype=np.int64)])
            y = np.concatenate([y, np.full(pad, IGNORE_INDEX, dtype=np.int64)])
        return x, y


def valid_window_starts(labels: np.ndarray, block_size: int) -> List[int]:
    """Window starts whose target span holds at least one assistant token.

    Without this filter most windows fall entirely inside a long context block
    where every label is ``-100``; cross-entropy over "no targets" is NaN, and a
    NaN in an AdamW step poisons every weight.  Two thirds of the windows in the
    Spore corpus would have been NaN -- the run would have looked alive and
    learned nothing.
    """
    # prefix count of supervised positions -> O(1) per window
    live = (labels != IGNORE_INDEX).astype(np.int64)
    cum = np.concatenate([[0], np.cumsum(live)])
    starts: List[int] = []
    n = len(labels)
    for s in range(0, n - block_size - 1):
        # labels[s+1 : s+block_size+1] is the target span
        if cum[s + block_size + 1] - cum[s + 1] > 0:
            starts.append(s)
    return starts


def _encode_supervised(
    tokenizer,
    full_text: str,
    answer_start: int,
    answer_end: int,
) -> Tuple[List[int], List[int]]:
    """Tokenise one record and mask everything before the answer.

    Character offsets are converted to token offsets by tokenising the prefix
    separately.  Because the tokenizer is byte-level and deterministic, the
    prefix encoding is always a true prefix of the full encoding -- the test
    suite asserts this for every record in the corpus.
    """
    ids = tokenizer.encode(full_text)
    prefix_ids = tokenizer.encode(full_text[:answer_start])
    if ids[: len(prefix_ids)] != prefix_ids:  # pragma: no cover - guard
        raise AssertionError("prefix tokenisation is not a prefix of the full text")
    start = len(prefix_ids)
    end = len(tokenizer.encode(full_text[:answer_end]))
    labels = [IGNORE_INDEX] * len(ids)
    for i in range(start, min(end, len(ids))):
        labels[i] = ids[i]
    return ids, labels


def build_sft_data(
    tokenizer,
    records: Sequence[Dict[str, str]],
    *,
    repeat: int = 1,
    context_for: Optional[Dict[str, Dict[str, str]]] = None,
    shuffle_seed: int = 0,
) -> SFTData:
    """Turn rendered dialogue records into one long masked token stream.

    ``records`` are dicts with ``user`` / ``assistant`` (and optional
    ``context``).  Each is rendered with :func:`render_example`; ``repeat``
    duplicates them with fresh shuffling, which is how a small corpus is
    stretched over many epochs without presenting the same order every time.
    """
    all_ids: List[int] = []
    all_labels: List[int] = []
    rng = random.Random(shuffle_seed)
    pool = list(records) * max(1, repeat)
    rng.shuffle(pool)
    supervised = 0
    for rec in pool:
        user_text = rec["user"]
        answer = rec["assistant"]
        # Alternate long and short grounding: long blocks teach retrieval use,
        # short blocks keep the assistant token density high enough to learn
        # from.  With only long blocks, 7% of the stream is supervised.
        if rng.random() < 0.5 and rec.get("context_long"):
            context = rec["context_long"]
        else:
            context = rec.get("context", "")
        if context_for:
            context = context_for.get(rec.get("topic", ""), {}).get("text", context)
        full = render_example(user_text, answer, context)
        prompt = render_prompt(user_text, context)
        ids, labels = _encode_supervised(tokenizer, full, len(prompt), len(prompt) + len(answer))
        all_ids.extend(ids)
        all_labels.extend(labels)
        supervised += sum(1 for x in labels if x != IGNORE_INDEX)
    labels_arr = np.array(all_labels, dtype=np.int32)
    return SFTData(
        input_ids=np.array(all_ids, dtype=np.uint16),
        labels=labels_arr,
        tokenizer=tokenizer,
        examples=len(pool),
        supervised_tokens=supervised,
        valid_windows=[],  # filled lazily by the trainer (needs block_size)
    )


# ---------------------------------------------------------------------------
# corpus file helpers
# ---------------------------------------------------------------------------
def load_qa(path: str | Path = DEFAULT_QA_PATH) -> List[Dict[str, str]]:
    """Read the generated Q&A corpus (one JSON object per line)."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Q&A corpus not found at {path}. Run `python scripts/build_spore_data.py`."
        )
    out: List[Dict[str, str]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def save_qa(records: Iterable[Dict[str, str]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    return path


@dataclass
class ChatStats:
    """Small report printed by the trainer and the data builder."""

    examples: int = 0
    supervised_tokens: int = 0
    total_tokens: int = 0
    avg_answer_tokens: float = 0.0
    topics: int = 0
    extra: Dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, float]:
        d = {
            "examples": self.examples,
            "supervised_tokens": self.supervised_tokens,
            "total_tokens": self.total_tokens,
            "avg_answer_tokens": round(self.avg_answer_tokens, 1),
            "topics": self.topics,
        }
        d.update(self.extra)
        return d

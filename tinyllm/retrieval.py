"""Retrieval for Spore: find the passage that answers a question.

Why retrieval
-------------
A 12M-parameter model cannot store the world's facts.  What it *can* learn --
and what the chat-tuning stage teaches it -- is how to read a short passage and
answer from it.  So the server (and the browser demo) look up the most relevant
passage from the factbook built alongside the training corpus, prepend it as a
``<|context|>`` block, and let the model answer from evidence.  That is the
difference between a model that invents "Paris is a small town in Wales" and one
that answers correctly because the sentence is right there.

The factbook has one passage per article/story, so retrieval answers "which
*article* is this about" precisely and then hands over the article's opening
sentences.  Fine-grained paragraph retrieval would need a much bigger
factbook; for a 12M-parameter demo this is the right trade (the alternative --
letting the model answer from memory -- is what produces invented facts).

The ranker is deliberately dependency-free (no numpy, no sklearn, no network):
BM25 over a small inverted index.  With ~10k passages it ranks in a couple of
milliseconds and needs no download at inference time.

    from tinyllm.retrieval import Factbook
    book = Factbook.load("data/spore/factbook.json")
    hits = book.search("What is the capital of France?", k=2)
    hits[0]["text"], hits[0]["score"]
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FACTBOOK = REPO_ROOT / "data" / "spore" / "factbook.json"

# Very small stop list: enough to stop function words dominating BM25, small
# enough not to need a download.
STOPWORDS = frozenset(
    """a an the and or but if then than that this these those of in on at to for
    from by with without as is are was were be been being do does did doing have
    has had having i you he she it we they me him her them my your his its our
    their what which who whom whose when where why how not no nor so such only
    own same too very can will just should now about into over after before
    again further once here there all any both each few more most other some""".split()
)

_TOKEN = re.compile(r"[a-z0-9][a-z0-9'\-]*")


def tokenize(text: str) -> List[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in STOPWORDS and len(t) > 1]


def _stem(word: str) -> str:
    """Crude suffix stripping so "rivers"/"river" and "spelled"/"spell" match.

    Deliberately tiny: a real stemmer (Porter) would be 100 lines and this is
    only used to widen recall on a small index.
    """
    for suf in ("ing", "ed", "es", "s"):
        if len(word) > len(suf) + 3 and word.endswith(suf):
            return word[: -len(suf)]
    return word


# Facts should come from the encyclopaedia, stories from the stories.  Without
# this prior a short TinyStory ("Once upon a time there was a little boat")
# outranks a Wikipedia article for "tell me about the Boat Race", because BM25
# rewards short documents.
SOURCE_PRIOR = {"wikipedia": 1.25, "tinystories": 0.85}


class Factbook:
    """BM25 index over retrieval passages."""

    def __init__(self, passages: List[Dict[str, str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.passages = passages
        self.k1 = k1
        self.b = b
        self.postings: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
        self.doc_len: List[int] = []
        self.priors: List[float] = []
        for i, p in enumerate(passages):
            toks = [_stem(t) for t in tokenize(p.get("title", "") + " " + p.get("text", ""))]
            # a title hit is worth more than a body hit: repeat the title tokens
            toks += [_stem(t) for t in tokenize(p.get("title", ""))] * 2
            self.doc_len.append(len(toks))
            self.priors.append(SOURCE_PRIOR.get(str(p.get("source", "")), 1.0))
            counts = Counter(toks)
            for term, tf in counts.items():
                self.postings[term].append((i, tf))
        self.avg_len = (sum(self.doc_len) / len(self.doc_len)) if self.doc_len else 0.0
        self.n = len(passages)

    # -- persistence -------------------------------------------------------
    @classmethod
    def load(cls, path: str | Path = DEFAULT_FACTBOOK) -> "Factbook":
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"factbook not found at {path}; run `python scripts/build_spore_data.py`"
            )
        payload = json.loads(path.read_text(encoding="utf-8"))
        passages = payload["passages"] if isinstance(payload, dict) else payload
        return cls(passages)

    # -- search ------------------------------------------------------------
    def find(self, title: str) -> Optional[Dict[str, str]]:
        """Exact / prefix lookup by article title (fast path for "What is X?")."""
        t = title.lower().strip(" .?!")
        if not t:
            return None
        for p in self.passages:
            if str(p.get("title", "")).lower() == t:
                return p
        for p in self.passages:
            if str(p.get("title", "")).lower().startswith(t):
                return p
        return None

    def search(
        self,
        query: str,
        k: int = 2,
        min_score: float = 0.6,
        source: Optional[str] = None,
    ) -> List[Dict[str, object]]:
        """Return the ``k`` best passages with a relevance ``score``.

        ``min_score`` keeps weak matches out: it is better to answer with no
        context than to hand the model an unrelated paragraph and invite it to
        hallucinate a link.
        """
        terms = [_stem(t) for t in tokenize(query)]
        if not terms or not self.passages:
            return []
        scores: Dict[int, float] = defaultdict(float)
        for term in set(terms):
            posting = self.postings.get(term)
            if not posting:
                continue
            idf = math.log(1 + (self.n - len(posting) + 0.5) / (len(posting) + 0.5))
            for doc, tf in posting:
                dl = self.doc_len[doc] or 1
                denom = tf + self.k1 * (1 - self.b + self.b * dl / max(1e-9, self.avg_len))
                scores[doc] += idf * tf * (self.k1 + 1) / denom
        for doc in scores:
            scores[doc] *= self.priors[doc]
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        out: List[Dict[str, object]] = []
        for doc, score in ranked:
            if score < min_score:
                continue
            if source and self.passages[doc].get("source") != source:
                continue
            if len(out) >= max(1, k):
                break
            p = dict(self.passages[doc])
            p["score"] = round(score, 3)
            out.append(p)
        return out


_TITLE_Q = re.compile(
    r"^(?:what|who|where)\s+(?:is|are|was|were)\s+(?:the\s+|a\s+|an\s+)?(.+?)[?.]?$", re.I
)
_TELL_Q = re.compile(r"^tell me about\s+(.+?)[?.]?$", re.I)


def title_from_question(question: str) -> Optional[str]:
    """Extract the entity a question is about ("What is the Boat Race?")."""
    q = question.strip()
    for pat in (_TITLE_Q, _TELL_Q):
        m = pat.match(q)
        if m:
            return m.group(1).strip()
    return None


# Common concepts that the committed corpus happens not to contain as an
# article.  Small, hand-written, and deliberately boring: they exist so the
# demo answers everyday "what is X" questions correctly instead of pulling an
# unrelated Wikipedia paragraph.  They are *not* used for training.
BUILTIN_FACTS: Dict[str, str] = {
    "water": "Water is a clear liquid made of hydrogen and oxygen. All animals and plants need water to live.",
    "fire": "Fire is the heat and light given off when something burns.",
    "sun": "The Sun is the star at the centre of the solar system. It gives the Earth light and heat.",
    "moon": "The Moon is the natural satellite that orbits the Earth. It is about a quarter of the Earth's width.",
    "rain": "Rain is water that falls from clouds in the sky.",
    "snow": "Snow is frozen water that falls from clouds as small white ice crystals.",
    "wind": "Wind is air moving across the surface of the Earth.",
    "earth": "The Earth is the third planet from the Sun and the only planet known to have life.",
    "computer": "A computer is a machine that follows instructions to work with information.",
    "book": "A book is a set of printed pages that tells a story or gives information.",
    "language": "A language is a system of words that people use to speak and write to each other.",
    "model": "A model is a small program that has learned patterns from examples.",
    "cat": "A cat is a small furry animal that people often keep as a pet.",
    "dog": "A dog is a friendly animal that people often keep as a pet and as a helper.",
    "tree": "A tree is a tall plant with a wooden trunk and branches that grow leaves.",
    "river": "A river is a long stream of water that flows across the land towards the sea.",
    "mountain": "A mountain is a very high part of the land, much higher than the hills around it.",
    "school": "A school is a place where children go to learn.",
    "story": "A story is a description of people and events, real or imagined, told to entertain or inform.",
}


def builtin_fact(title: str) -> Optional[str]:
    """A hand-written definition for a common concept, if we have one."""
    key = title.lower().strip(" .?!")
    key = re.sub(r"^(the|a|an)\s+", "", key).rstrip("s")
    return BUILTIN_FACTS.get(key)


# "What is your name?" must not be answered from the encyclopaedia.
_NON_LOOKUP = re.compile(
    r"\b(your|you|my|me|i|we|us|our|yourself)\b", re.I
)


def context_for(question: str, book: Optional[Factbook], max_chars: int = 700, min_score: float = 6.0) -> str:
    """Grounding block for a question (empty string when nothing matches).

    Strategy: try an exact title match first (the model answers "What is X?"
    perfectly when X's opening sentence is in front of it), then fall back to
    BM25.  Returning nothing is better than returning unrelated text -- the
    model was trained to say "I do not know that" when it has no context.
    """
    if book is None:
        return ""
    # questions about the assistant itself are answered from the ability set,
    # never from retrieval
    if _NON_LOOKUP.search(question):
        return ""
    title = title_from_question(question)
    if title:
        fact = builtin_fact(title)
        if fact:
            return fact[:max_chars].strip()
        hit = book.find(title)
        if hit:
            return str(hit.get("text", ""))[:max_chars].strip()
    hits = book.search(question, k=1)
    # a weak match is worse than none: it invites the model to invent a link
    if not hits or float(hits[0].get("score", 0)) < min_score:
        return ""
    return str(hits[0].get("text", ""))[:max_chars].strip()

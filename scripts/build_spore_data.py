#!/usr/bin/env python3
"""Build the Spore training corpora: prose pretraining text + a Q&A chat corpus.

Why this exists
---------------
The previous model saw only Tiny Shakespeare, so it could finish a sentence of
early-modern English and nothing else.  Spore needs three things to answer
questions:

1. **English and facts** -- encyclopaedia prose (WikiText-2: real Wikipedia
   articles) plus narrative prose (TinyStories).
2. **The shape of a conversation** -- ``<|user|>...<|assistant|>...<|end|>``
   turns, so it learns to answer and then *stop*.
3. **Grounding** -- answers that are supported by a short context block, so at
   inference time the server can retrieve a relevant passage and the model
   quotes from it instead of inventing.  A 12M-parameter model cannot memorise
   world knowledge; retrieval is what makes it useful.

The Q&A corpus is *generated* from the two prose corpora: every article and
story is mined for sentences and turned into question/answer pairs with
deterministic templates.  Because the answers are verbatim sentences from the
source, the model learns to answer from evidence rather than to hallucinate.
No network access is needed at build time -- the raw text is committed under
``data/``.

Outputs (in ``data/spore/``)
----------------------------
    prose_train.txt       80% split of the mixed prose corpus (pretraining)
    prose_val.txt         20% held-out split (validation loss)
    qa.jsonl              chat records: {user, assistant, context, kind, topic}
    qa_stats.json         counts, so the trainer can log corpus composition
    factbook.json         retrieval passages (one per article/story) used by
                          the server and the HTML demo to ground answers

Usage
-----
    python scripts/build_spore_data.py                 # full build (~1 min)
    python scripts/build_spore_data.py --small         # fast, for tests
    python scripts/build_spore_data.py --stats-only    # inspect, write nothing
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

DATA = REPO_ROOT / "data"
OUT = DATA / "spore"

TINYSTORIES = DATA / "raw_tinystories.txt"
WIKITEXT = DATA / "raw_wikitext2.txt"

# ---------------------------------------------------------------------------
# cleaning
# ---------------------------------------------------------------------------
# WikiText was released already tokenised: "side @-@" means "side-by-side",
# "<unk>" is a dropped rare word, and headings are nested runs of "=" tokens
# (``= Article =``, ``= = Section = =``).  On top of that, the tokenisation
# doubles balanced quotes and leaves stray quote-only lines behind.
_SPACED_PUNCT = re.compile(r"\s+([,.;:!?%)\]])")
_OPEN_PUNCT = re.compile(r"([(\[])\s+")
_DOUBLE_SPACE = re.compile(r"[ \t]{2,}")
_PAIRED_DOUBLE = re.compile(r'""\s*([^"]{1,60}?)\s*""')
_PAIRED_SINGLE = re.compile(r"''\s*([^']{1,60}?)\s*''")
# A fragment created by splitting a line at "<unk>" must be at least this many
# words to count as prose. Below 6 the output is mostly sentence stubs like
# "season of", which add noise rather than signal.
UNK_MIN_WORDS = 6

_QUOTE_JUNK = re.compile(r'^[\s"\'`\-–—.,;:!?()\[\]{}|/\\*#=+_~^·•]+$')


def parse_heading(line: str) -> Tuple[int, str] | Tuple[None, None]:
    """Return ``(level, title)`` for a WikiText heading line, else ``(None, None)``.

    WikiText renders heading level as repeated ``=`` *tokens*::

        = Article =              -> (1, "Article")
        = = Background = =       -> (2, "Background")
        = = = August = = =       -> (3, "August")

    Treating the nested forms as new articles (the obvious regex mistake) turns
    every section of every article into a bogus "article" whose title is the
    rest of the heading -- which then produces nonsense questions.
    """
    tokens = line.split()
    if len(tokens) < 3:
        return None, None
    lead = 0
    while lead < len(tokens) and tokens[lead] == "=":
        lead += 1
    trail = 0
    while trail < len(tokens) and tokens[-1 - trail] == "=":
        trail += 1
    if lead == 0 or trail == 0 or lead + trail >= len(tokens):
        return None, None
    title = " ".join(tokens[lead : len(tokens) - trail]).strip()
    if not title:
        return None, None
    return min(lead, trail), title


def clean_wikitext(raw: str) -> str:
    """Turn tokenised WikiText into prose with ``##``/``###`` headings."""
    out_lines: List[str] = []
    for line in raw.split("\n"):
        s = line.strip()
        if not s:
            out_lines.append("")
            continue
        if _QUOTE_JUNK.match(s):
            continue  # stray quote/punctuation lines left by the tokeniser
        level, title = parse_heading(s)
        if level is not None:
            # A heading whose title is a rare word arrives as "### <unk>".
            # Emitting it would inject the literal token into the corpus and
            # leave a meaningless section marker; drop the heading instead.
            if "<unk>" in title:
                continue
            # the hyphen glue below is applied to body lines only, so headings
            # kept raw markers: "In Through the Looking @-@ Glass"
            title = re.sub(r"\s*@-@\s*", "-", title).strip()
            if not title:
                continue
            # level 1 = article, level >= 2 = section inside the article
            marker = "##" if level == 1 else "###"
            out_lines.append(f"\n{marker} {title}\n")
            continue
        # hyphenated compounds: "side @-@ by @-@ side" -> "side-by-side"
        s = re.sub(r"\s*@-@\s*", "-", s)
        # "<unk>" marks a rare word the WikiText tokeniser could not represent.
        # Dropping the whole line (what this used to do) discards 86% of the
        # corpus: 14,334 of 24,974 non-blank lines contain it and they hold
        # 9.32 MB of the 10.77 MB total. Splitting at the hole keeps the
        # surrounding prose; fragments too short to be sentences are dropped
        # because a 3-word stub teaches nothing.
        if "<unk>" in s:
            fragments = [f.strip() for f in s.split("<unk>")]
            keep = [f for f in fragments if len(f.split()) >= UNK_MIN_WORDS]
            if not keep:
                continue
            s = " ".join(keep)
        # "Joseph Michael "" Joe "" Nathan" -> 'Joseph Michael "Joe" Nathan'
        s = _PAIRED_DOUBLE.sub(lambda m: f'"{m.group(1)}"', s)
        s = _PAIRED_SINGLE.sub(lambda m: f"'{m.group(1)}'", s)
        s = s.replace('""', '"').replace("''", "'")
        # the tokeniser splits "year 's" into two tokens; glue clitics back on
        s = re.sub(r"\s+'s\b", "'s", s)
        s = re.sub(r"\s+'t\b", "'t", s)
        # a quote left stranded before punctuation ("word ' .") is noise
        s = re.sub(r"\s+['’\"]\s*([.,;:!?])", r"\1", s)
        s = _OPEN_PUNCT.sub(r"\1", s)
        s = _SPACED_PUNCT.sub(r"\1", s)
        s = _DOUBLE_SPACE.sub(" ", s)
        # the tokeniser separates "2013 – 14" with spaces around the dash
        s = re.sub(r"(\d)\s+([–—])\s+(\d)", r"\1\2\3", s)
        out_lines.append(s)
    text = "\n".join(out_lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() + "\n"


def clean_tinystories(raw: str) -> List[str]:
    """TinyStories is already clean prose, separated by <|endoftext|>."""
    stories = []
    for chunk in raw.split("<|endoftext|>"):
        story = chunk.strip()
        if len(story) < 120:
            continue
        # the fixture has a "Story N:" or similar preamble line in places
        story = re.sub(r"^(Story|Title)\s*\d*\s*:\s*", "", story)
        stories.append(story)
    return stories


def split_articles(text: str) -> List[Tuple[str, str]]:
    """Split cleaned WikiText into ``(title, body)`` per article.

    Sub-headings (``###``) stay inside the body as plain text: only ``##`` marks
    the start of a new article.
    """
    articles: List[Tuple[str, str]] = []
    title = ""
    body: List[str] = []
    for line in text.split("\n"):
        if line.startswith("## ") and not line.startswith("###"):
            if title and body:
                articles.append((title, "\n".join(body).strip()))
            title = line[3:].strip()
            body = []
        elif line.strip():
            body.append(line.strip())
    if title and body:
        articles.append((title, "\n".join(body).strip()))
    return articles


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'(])")


def sentences(text: str, min_len: int = 45, max_len: int = 320) -> List[str]:
    """Split into usable sentences (answers must be a single clean sentence)."""
    out: List[str] = []
    for para in text.split("\n"):
        for s in SENT_SPLIT.split(para.strip()):
            s = s.strip().strip('"').strip()
            if min_len <= len(s) <= max_len and s.count("(") == s.count(")"):
                if not re.search(r"[{}<>|]", s) and "@" not in s:
                    out.append(s)
    return out


# ---------------------------------------------------------------------------
# Wikipedia article -> Q&A
# ---------------------------------------------------------------------------
# Only question forms whose answer is unambiguous and verbatim in the sentence
# are generated.  A tiny model cannot learn from questions it has no way to
# answer, and a mismatched pair teaches hallucination.
_YEAR = re.compile(r"\b(1[0-9]{3}|20[0-2][0-9])\b")
_MONTH = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|"
    r"October|November|December)\b"
)

# "X was Y" / "X is Y" -> "What is X?"  only when X is a short noun phrase.
_COPULA = re.compile(r"^(?P<subj>[A-Z][^,;:]{2,70}?)\s+(?P<verb>was|is|were|are)\s+(?P<rest>.{10,})$")
# "X was born/died/began/..." -> "When did X ...?" / "When was X born?"
_EVENT = re.compile(
    r"^(?P<subj>[A-Z][^,;:]{2,70}?)\s+(?P<verb>was born|were born|was released|was published|"
    r"was founded|was established|was built|died|began|started|took place|became|won|"
    r"was elected|was appointed)\s*(?P<rest>.*)$"
)
_LOCATED = re.compile(r"\b(city|town|village|country|county|state|region|island|river|mountain|district)\b", re.I)


_LEADING_ARTICLE = re.compile(r"^(The|A|An|This|These|Those)\s+", re.I)
_BARE_ARTICLE = re.compile(r"^(The|A|An|This|These|Those)$", re.I)


def _clean_subject(subj: str) -> str:
    """Drop a leading article so questions read naturally.

    A subject that is *only* an article is not a subject at all. WikiText has
    articles literally titled "The", and without this they produced questions
    like "What did The win?".
    """
    subj = subj.strip()
    if _BARE_ARTICLE.match(subj):
        return ""
    return _LEADING_ARTICLE.sub("", subj)


# A subject must look like a name or a noun phrase, not a sentence fragment or a
# temporal adverb: "At the time of its construction it", "Overall there" and
# "oldest school still standing in the town" were all leaking into questions.
_BAD_SUBJ_START = re.compile(
    r"^(at|in|on|by|after|before|during|over|under|from|with|as|while|when|"
    r"overall|however|although|though|there|here|this|these|those|it|its|"
    r"they|he|she|his|her|their|also|then|both|each|many|most|some|all|"
    r"oldest|first|last|second|third|new|former|other|another|such|"
    r"three|four|five|six|seven|eight|nine|ten|several|many|later|"
    r"serial|major|minor|main|chief|senior|junior)\b",
    re.I,
)
_BAD_SUBJ_CHARS = re.compile(r"[()\[\]{}<>\"“”@=]|\d\s*$")
_SUBJ_VERBS = re.compile(
    r"\b(was|were|is|are|had|has|have|did|does|do|won|began|became|extended|"
    r"announced|died|released|founded|named|appointed|elected|known|used|"
    r"played|held|took|made|said|went|came)\b",
    re.I,
)


def _valid_subject(subj: str, max_words: int = 6) -> bool:
    """Conservative filter: better to skip a fact than to emit a broken question.

    A question subject must be a name or a short noun phrase.  Anything holding
    a verb, adverb or clause boundary ("Spain won", "Overall there", "Minnesota
    Territory extended far into what") produced ungrammatical questions and is
    rejected before it can reach the corpus.
    """
    if not subj or not subj[0].isupper():
        return False
    if _BARE_ARTICLE.match(subj.strip()):
        return False
    if not (1 <= len(subj.split()) <= max_words):
        return False
    if _BAD_SUBJ_START.match(subj) or _BAD_SUBJ_CHARS.search(subj):
        return False
    if _SUBJ_VERBS.search(subj):
        return False
    if "," in subj or " – " in subj or "/" in subj:
        return False
    # a verb phrase masquerading as a subject ("the team win", "X become")
    if re.search(r"\s(the team|the club|the company)\b", subj, re.I):
        return False
    return True


# Only these predicate heads earn a "Who is X?" question.
_PERSON_WORDS = re.compile(
    r"\b(footballer|player|pitcher|quarterback|actor|actress|writer|author|poet|"
    r"singer|musician|composer|politician|scientist|physicist|chemist|artist|"
    r"painter|sculptor|king|queen|emperor|prince|princess|director|producer|"
    r"engineer|athlete|swimmer|boxer|runner|general|officer|soldier|admiral|"
    r"judge|lawyer|physician|doctor|novelist|journalist|architect|"
    r"mathematician|philosopher|economist|businessman|entrepreneur|activist)\b",
    re.I,
)
_PERSON_LEAD = re.compile(
    r"^[A-Z][\w'’.-]+(?:\s+(?:[A-Z][\w'’.-]+|[a-z]{1,3}|van|von|de|da|del|der)){1,4}"
    r"\s*(?:\([^)]*\)\s*)?(?:was|is)\b"
)
# "X (born 1978) is a former professional footballer" -> person
_PERSON_BORN = re.compile(r"^[A-Z][^(]{1,60}?\(\s*(?:born|b\.)\s", re.I)


# A sentence that is *about* a work, not a person, regardless of what nouns
# appear later in it. "The Food Album is a compilation album by American
# singer-songwriter Weird Al Yankovic" contains "singer", but the subject is
# an album: matching any person noun anywhere produced "Who is The Food Album?".
_WORK_LEAD = re.compile(
    r"\b(is|was)\s+(?:a|an|the)\s+[\w\s-]{0,40}?\b("
    r"album|song|single|film|movie|book|novel|magazine|newspaper|journal|"
    r"game|video game|series|episode|painting|sculpture|building|bridge|"
    r"company|organisation|organization|band|route|road|highway|station|"
    r"school|university|college|village|town|city|county|province|state|"
    r"species|genus|family|disease|ship|vessel|aircraft)\b",
    re.I,
)


def _is_person_lead(sent: str) -> bool:
    """True only for a genuine biography opening sentence.

    Two independent guards, because one is not enough: the subject must match
    the name shape in _PERSON_LEAD, *and* the sentence must not describe a work
    or place. The person noun has to appear in the predicate near the subject,
    not merely somewhere in the first 220 characters.
    """
    if _PERSON_BORN.match(sent):
        return True
    if _WORK_LEAD.search(sent[:220]):
        return False
    if not _PERSON_LEAD.match(sent):
        return False
    # "X is a footballer" -> person; "X is a compilation album by a singer"
    # already returned above. Look only at the start of the predicate.
    return _PERSON_WORDS.search(sent[:120]) is not None


def _question_subject(subj: str) -> Optional[str]:
    """Return a subject safe to embed in a question, else ``None``."""
    clean = _clean_subject(subj)
    # strip a trailing date band: "X ( 1902 – 1983 )" -> "X"
    clean = re.sub(r"\s*\([^)]*\)\s*$", "", clean).strip()
    return clean if _valid_subject(clean) else None


def wiki_qa(title: str, body: str) -> List[Dict[str, str]]:
    """Build extractive Q&A records for one article.

    Kinds:
      * ``lead``   -- "What is <title>?" -> the opening definition sentence.
      * ``who-is`` -- "Who is <title>?" for biographical leads.
      * ``when``   -- the sentence states a date (year or month).
      * ``where``  -- the sentence locates a place.

    Every answer is a complete sentence copied verbatim from the article, and
    the context block always contains that sentence (it is built from the
    article text, never truncated before the answer), so the record is
    self-consistent: given the context, the answer is recoverable.
    """
    records: List[Dict[str, str]] = []
    sents = sentences(body)
    if not sents:
        return records

    # Context = the article's opening, extended until it holds the whole
    # answer whenever the answer is nearby.  Long articles are cut only at a
    # point where the answer is already fully inside.
    def context_for(answer: str, budget: int) -> str:
        """A grounding block of at most ``budget`` characters holding the answer.

        The block starts at the beginning of the sentence that contains the
        answer and grows forward; if the budget is tight it is allowed to start
        mid-article so the answer always fits.  Never truncated before the
        answer -- that invariant is what makes the corpus learnable.
        """
        text = " ".join(sents)
        pos = text.find(answer)
        if pos < 0:
            return text[:budget]
        # start at the sentence containing the answer (or the article start)
        start = text.rfind(". ", 0, pos)
        start = 0 if start < 0 else start + 2
        if pos - start + len(answer) > budget:
            start = max(0, pos - budget // 3)
        block = text[start : start + budget]
        if block and block[-1] not in ".!?":
            cut = max(block.rfind(". "), block.rfind("! "), block.rfind("? "))
            if cut > len(block) // 2:
                block = block[: cut + 1]
        return block.strip()

    seen: set[str] = set()

    def add(user: str, answer: str, kind: str) -> None:
        if len(user) < 8 or user in seen:
            return
        if len(answer) < 25:
            return
        seen.add(user)
        # Two context variants per record.  The short one keeps chat training
        # dense (a 512-token window can hold several short examples); the long
        # one teaches the model to work with the larger blocks the server and
        # the HTML demo actually pass in.  Only the short form is validated
        # verbatim because it may be trimmed around the answer.
        short_ctx = context_for(answer, 280)
        if answer not in short_ctx:
            # a sentence longer than the budget: the answer itself is the
            # context, which keeps "answer always inside context" true
            short_ctx = answer
        records.append(
            {
                "user": user,
                "assistant": answer,
                "context": short_ctx,                      # dense chat windows
                "context_long": context_for(answer, 1400),  # longer grounding
                "kind": kind,
                "topic": title,
            }
        )

    def definitional(sent: str) -> bool:
        """True when ``sent`` really defines the article subject.

        Articles whose text begins with a section heading ("### History and
        Purpose") have no definitional first sentence; asking "What is X?"
        there returns an unrelated fact and teaches the model to answer
        off-topic.
        """
        if not re.match(rf"^{re.escape(title)}\b", sent):
            return False
        return re.search(r"\b(is|was|are|were|refers to|may refer to)\b", sent[:220], re.I) is not None

    lead = sents[0]
    if len(lead) >= 60 and definitional(lead):
        add(f"What is {title}?", lead, "lead")
        # "Who is X?" is only valid when the sentence actually calls the subject
        # a person.  Matching "is a major freeway ... linking Providence" as a
        # biography produced "Who is Rhode Island Route 4?".
        if _is_person_lead(lead):
            add(f"Who is {title}?", lead, "who-is")

    for s in sents[1:]:
        m = _EVENT.match(s)
        if m and (_YEAR.search(s) or _MONTH.search(s)):
            subj = _question_subject(m.group("subj"))
            verb = m.group("verb")
            if subj:
                if verb in ("was born", "were born"):
                    add(f"When was {subj} born?", s, "when")
                elif verb == "died":
                    add(f"When did {subj} die?", s, "when")
                elif verb in ("began", "started", "took place"):
                    base = {"began": "begin", "started": "start", "took place": "take place"}[verb]
                    add(f"When did {subj} {base}?", s, "when")
                elif verb in (
                    "was released", "was published", "was founded",
                    "was established", "was built",
                ):
                    add(f"When was {subj} {' '.join(verb.split()[1:])}?", s, "when")
                elif verb in ("was elected", "was appointed"):
                    add(f"When was {subj} {' '.join(verb.split()[1:])}?", s, "when")
                elif verb in ("became", "won"):
                    # "X won the 2001 ARIA Awards" -> "What did X win?"
                    past = {"won": "win", "became": "become"}[verb]
                    add(f"What did {subj} {past}?", s, "when")

        m = _COPULA.match(s)
        if m and _LOCATED.search(m.group("rest")):
            subj = _question_subject(m.group("subj"))
            if subj and "'s " not in subj and not re.search(
                r"\b(built environment|population|economy|climate|history|"
                r"culture|government|area|region of|part of)\b", subj, re.I
            ):
                add(f"Where is {subj}?", s, "where")

        if m and len(records) < 9:
            subj = _clean_subject(m.group("subj"))
            if subj.lower() == title.lower() and len(m.group("rest")) >= 25:
                add(f"What is {title}?", s, "lead")

    return records


# ---------------------------------------------------------------------------
# Story -> Q&A
# ---------------------------------------------------------------------------
_STORY_NAME = re.compile(
    r"\b(Lily|Tom|Anna|Ben|Lucy|Sam|Mia|Jack|Emma|Max|Sara|Zoey|Leo|Nora|Tim|"
    r"Mom|Dad|Grandma|Grandpa|the little (?:girl|boy|cat|dog|bird|fox|bear))\b"
)


def _slice_around(answer: str, pool: Sequence[str], budget: int = 280) -> str:
    """Pick sentences from ``pool`` until ``answer`` is fully inside ``budget``.

    A plain character cap can cut a long final sentence in half, which silently
    breaks the "the answer is always inside the context" invariant (418 of the
    "how does the story end" records did exactly that).
    """
    if len(answer) >= budget:
        return answer
    text = ""
    # grow backwards from the sentence holding the answer, then forwards
    try:
        i = next(k for k, s in enumerate(pool) if answer in s or s in answer)
    except StopIteration:
        i = 0
    lo, hi = i, i + 1
    text = pool[i]
    while len(text) < budget:
        grew = False
        if hi < len(pool) and len(text) + len(pool[hi]) + 1 <= budget:
            text = text + " " + pool[hi]
            hi += 1
            grew = True
        if lo > 0 and len(text) + len(pool[lo - 1]) + 1 <= budget:
            text = pool[lo - 1] + " " + text
            lo -= 1
            grew = True
        if not grew:
            break
    return text if answer in text else answer


def story_qa(story: str, idx: int) -> List[Dict[str, str]]:
    """Comprehension-style questions over a TinyStories story.

    The context is the *whole* story (stories average ~700 characters, well
    inside the model's 512-token window), so every answer is guaranteed to be
    inside it.  Truncating the context to 900 characters and then quoting a
    sentence from the full story produced answers the model could not see --
    about 40% of records -- which teaches hallucination.
    """
    records: List[Dict[str, str]] = []
    sents = sentences(story, min_len=30, max_len=240)
    if len(sents) < 3:
        return records
    flat = " ".join(story.split())
    # keep whole stories; only hard-cap absurdly long ones at a sentence end
    if len(flat) > 1600:
        cut = flat.rfind(". ", 0, 1600)
        flat = flat[: cut + 1] if cut > 600 else flat[:1600]
    context = flat
    # a short variant: the two sentences around the answer, for dense chat
    # training windows
    sents = sentences(context, min_len=30, max_len=240)
    if len(sents) < 3:
        return records

    # summary: the opening sentence alone.  Concatenating the first and last
    # sentence produced an answer that exists nowhere in the context, so the
    # model could not learn to copy it -- 35% of the corpus was unlearnable.
    records.append(
        {
            "user": "Summarise this story.",
            "assistant": sents[0],
            "context": _slice_around(sents[0], sents),
            "context_long": flat,
            "kind": "summary",
            "topic": story[:60],
        }
    )
    # what happens at the end
    records.append(
        {
            "user": "How does the story end?",
            "assistant": sents[-1],
            "context": _slice_around(sents[-1], sents),
            "context_long": flat,
            "kind": "ending",
            "topic": story[:60],
        }
    )
    # a who-question from a sentence with a named character
    for s in sents[1:-1]:
        m = _STORY_NAME.search(s)
        if m and len(s) < 220:
            who = m.group(1)
            records.append(
                {
                    "user": f"What does {who} do in the story?",
                    "assistant": s,
                    "context": _slice_around(s, sents),
                    "context_long": flat,
                    "kind": "character",
                    "topic": who,
                }
            )
            break
    # a why/how question if the story contains a reason
    for s in sents[1:-1]:
        if s.lower().startswith(("because", "so ", "then ")) or " because " in s:
            records.append(
                {
                    "user": "Why does that happen?",
                    "assistant": s,
                    "context": _slice_around(s, sents),
                    "context_long": flat,
                    "kind": "reason",
                    "topic": story[:60],
                }
            )
            break
    return records


# ---------------------------------------------------------------------------
# small hand-written ability set (conversational glue + spelling + arithmetic)
# ---------------------------------------------------------------------------
def ability_qa() -> List[Dict[str, str]]:
    """A compact set of conversational and procedural skills.

    These are deliberately tiny and high-frequency: greetings, refusals,
    counting and spelling are exactly the behaviours a chat model needs to
    look well-formed, and they cost almost no tokens to learn.
    """
    recs: List[Dict[str, str]] = []

    def add(user: str, assistant: str, kind: str = "chat") -> None:
        recs.append({"user": user, "assistant": assistant, "context": "",
                     "context_long": "", "kind": kind, "topic": "skill"})

    greetings = [
        ("Hello!", "Hello! How can I help you today?"),
        ("Hi there", "Hi! What would you like to know?"),
        ("Good morning", "Good morning! What can I do for you?"),
        ("Good evening", "Good evening! Ask me anything about the stories and articles I know."),
        ("How are you?", "I am a language model, so I do not have feelings, but I am ready to help."),
        ("What is your name?", "My name is Spore. I am a small language model trained from scratch."),
        ("Who made you?", "I was trained from scratch by a small open-source project."),
        ("Thank you", "You are welcome! Ask me another question any time."),
        ("Thanks!", "Glad to help."),
        ("Goodbye", "Goodbye! Come back if you have more questions."),
        ("Can you help me?", "Yes. Ask me a question and I will answer from what I know."),
        ("What can you do?", "I can answer questions about the stories and articles I was trained on, and I can count and spell."),
        ("Are you a human?", "No, I am a small language model running on a computer."),
        ("Do you know everything?", "No. I am very small, so I only know what I read during training."),
        ("I do not understand", "Tell me the question in other words and I will try again."),
    ]
    for u, a in greetings:
        add(u, a)

    # refusals: the model must be allowed to say "I do not know"
    for u in [
        "What is the weather tomorrow?",
        "Who won the football match last night?",
        "What is the price of a house in London?",
        "Tell me a secret.",
        "What is my name?",
        "Can you browse the internet?",
    ]:
        add(u, "I do not know that. I am a small model and I only know the stories and articles in my training data.")

    # arithmetic: small, exact, and checkable
    for a in range(1, 11):
        for b in range(1, 11):
            add(f"What is {a} plus {b}?", f"{a} plus {b} is {a + b}.")
    for a in range(1, 11):
        add(f"What is {a} times {a}?", f"{a} times {a} is {a * a}.")
    add("Count from one to ten.", "One, two, three, four, five, six, seven, eight, nine, ten.")
    add("Count from ten to one.", "Ten, nine, eight, seven, six, five, four, three, two, one.")
    add("What comes after seven?", "Eight comes after seven.")
    add("What comes before three?", "Two comes before three.")

    # spelling / letters
    for word in ("cat", "dog", "tree", "house", "water", "story", "fire", "friend", "school", "river"):
        add(f"How do you spell {word}?", f"{word} is spelled: {' - '.join(word)}.")
    add("How many letters are in the word story?", "The word story has five letters: s, t, o, r, y.")
    add("What is the first letter of the alphabet?", "The first letter of the alphabet is A.")
    add("What is the last letter of the alphabet?", "The last letter of the alphabet is Z.")

    # simple definitions -- a seed for "what is X" from general knowledge
    for word, definition in [
        ("a cat", "A cat is a small furry animal that people keep as a pet."),
        ("a dog", "A dog is a friendly animal that people keep as a pet and as a helper."),
        ("water", "Water is a clear liquid that all animals and plants need to live."),
        ("the sun", "The sun is the star at the centre of our solar system. It gives Earth light and heat."),
        ("the moon", "The moon is the natural satellite that circles the Earth."),
        ("rain", "Rain is water that falls from clouds in the sky."),
        ("a book", "A book is a set of printed pages that tells a story or gives information."),
        ("a computer", "A computer is a machine that follows instructions to work with information."),
        ("a language", "A language is a system of words that people use to speak and write to each other."),
        ("a model", "A model is a small program that has learned patterns from examples."),
    ]:
        add(f"What is {word}?", definition, kind="definition")

    return recs


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------
def write_prose(articles: Sequence[Tuple[str, str]], stories: Sequence[str], out_dir: Path) -> Dict[str, int]:
    """Mix articles and stories, then make a contiguous train/val split."""
    blocks: List[str] = []
    for title, body in articles:
        blocks.append(f"## {title}\n\n{body}\n")
    for s in stories:
        blocks.append(f"{s}\n")
    # interleave deterministically so a training window sees both domains
    rng = random.Random(1234)
    idx = list(range(len(blocks)))
    rng.shuffle(idx)
    mixed = [blocks[i] for i in idx]
    text = "\n".join(mixed)
    cut = text.rfind("\n", 0, int(len(text) * 0.85)) or int(len(text) * 0.85)
    train, val = text[:cut], text[cut:]
    (out_dir / "prose_train.txt").write_text(train, encoding="utf-8")
    (out_dir / "prose_val.txt").write_text(val, encoding="utf-8")
    return {"prose_train_chars": len(train), "prose_val_chars": len(val), "blocks": len(mixed)}


def build_factbook(articles: Sequence[Tuple[str, str]], stories: Sequence[str], out_dir: Path) -> int:
    """One retrieval passage per article/story, for grounding answers at runtime."""
    passages: List[Dict[str, str]] = []
    for title, body in articles:
        sents = sentences(body)
        if len(sents) < 2:
            continue
        # lead with the sentence that actually defines the subject, so a
        # "What is X?" question gets a definition rather than trivia
        order = list(range(len(sents)))
        for i, sent in enumerate(sents[:6]):
            if re.match(rf"^{re.escape(title)}\b", sent) and re.search(
                r"\b(is|was|are|were)\b", sent[:200], re.I
            ):
                order.insert(0, order.pop(i))
                break
        ordered = [sents[i] for i in order]
        passages.append(
            {
                "id": f"wiki-{len(passages)}",
                "title": title,
                "text": " ".join(ordered[:4])[:700],
                "source": "wikipedia",
            }
        )
    for i, story in enumerate(stories):
        passages.append(
            {
                "id": f"story-{i}",
                "title": story.split(". ")[0][:70],
                "text": " ".join(story.split())[:700],
                "source": "tinystories",
            }
        )
    payload = {
        "note": "Retrieval passages used to ground Spore's answers. Built from committed prose, no network.",
        "passages": passages,
    }
    (out_dir / "factbook.json").write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return len(passages)


def build(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    if not TINYSTORIES.exists():
        print(f"missing {TINYSTORIES}", file=sys.stderr)
        return 2
    if not WIKITEXT.exists():
        print(f"missing {WIKITEXT}", file=sys.stderr)
        return 2

    print(f"[read] {WIKITEXT.name} ...", flush=True)
    wiki_clean = clean_wikitext(WIKITEXT.read_text(encoding="utf-8", errors="replace"))
    articles = split_articles(wiki_clean)
    print(f"[wiki] {len(articles):,} articles, {len(wiki_clean):,} chars")

    print(f"[read] {TINYSTORIES.name} ...", flush=True)
    stories = clean_tinystories(TINYSTORIES.read_text(encoding="utf-8", errors="replace"))
    print(f"[stories] {len(stories):,} stories, {sum(len(s) for s in stories):,} chars")

    if args.small:
        articles = articles[: args.limit_articles]
        stories = stories[: args.limit_stories]
        print(f"[small] using {len(articles)} articles / {len(stories)} stories")

    # ---- Q&A ------------------------------------------------------------
    records: List[Dict[str, str]] = []
    for title, body in articles:
        records.extend(wiki_qa(title, body))
    for i, story in enumerate(stories):
        records.extend(story_qa(story, i))
    records.extend(ability_qa())
    rng.shuffle(records)
    print(f"[qa] {len(records):,} records")

    kinds = Counter(r["kind"] for r in records)
    print("[qa] kinds:", dict(kinds))

    if args.stats_only:
        for r in records[:3]:
            print(json.dumps(r)[:300])
        return 0

    save_qa(records, out_dir / "qa.jsonl")
    prose_stats = write_prose(articles, stories, out_dir)
    n_fact = build_factbook(articles, stories, out_dir)

    stats = {
        "records": len(records),
        "kinds": dict(kinds),
        "topics": len({r["topic"] for r in records}),
        "articles": len(articles),
        "stories": len(stories),
        "factbook_passages": n_fact,
        "seed": args.seed,
        **prose_stats,
    }
    (out_dir / "qa_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    # content hash: train_spore.yml folds this into the cache fingerprint so a data
    # change invalidates a cached checkpoint instead of resuming onto it
    h = hashlib.sha256()
    for name in ("prose_train.txt", "qa.jsonl"):
        h.update((out_dir / name).read_bytes())
    stats["content_hash"] = h.hexdigest()[:16]
    (out_dir / "qa_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")

    print("[out]")
    for name in ("prose_train.txt", "prose_val.txt", "qa.jsonl", "factbook.json", "qa_stats.json"):
        p = out_dir / name
        print(f"   {name:18s} {p.stat().st_size/1e6:8.2f} MB")
    print(f"[hash] {stats['content_hash']}")
    return 0


def save_qa(records: Iterable[Dict[str, str]], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--small", action="store_true", help="tiny build for tests")
    p.add_argument("--limit-articles", type=int, default=200)
    p.add_argument("--limit-stories", type=int, default=400)
    p.add_argument("--stats-only", action="store_true", help="print composition, write nothing")
    return p.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(build(parse_args()))

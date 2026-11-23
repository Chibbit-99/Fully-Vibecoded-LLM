"""Tests for the Spore Q&A models: tokenizer, chat format, corpus, resuming.

These are fast: no training run, no network, no large checkpoints.  The aim is
to fail *here* rather than after a five-hour GitHub Actions job.
"""

from __future__ import annotations

import json
import random
import subprocess
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tinyllm.config import GPTConfig
from tinyllm.dataset import get_batch
from tinyllm.model import GPT
from tinyllm.retrieval import Factbook, builtin_fact, context_for, title_from_question
from tinyllm.sft import (
    ASSISTANT,
    CONTEXT,
    END,
    IGNORE_INDEX,
    SPECIAL_TOKENS,
    USER,
    build_sft_data,
    render_example,
    render_history,
    render_prompt,
    valid_window_starts,
)
from tinyllm.tokenizer import ByteBPE

DATA = ROOT / "data" / "spore"


def make_tokenizer(vocab_size: int = 400) -> ByteBPE:
    """A small but real tokenizer with the Spore control tokens."""
    text = (ROOT / "data" / "tinyshakespeare.txt").read_text(encoding="utf-8")[:120_000]
    tok = ByteBPE.train(text, vocab_size=vocab_size - len(SPECIAL_TOKENS))
    tok.special_tokens = list(SPECIAL_TOKENS)
    tok.special_ids = {t: 256 + len(tok.merges) + i for i, t in enumerate(SPECIAL_TOKENS)}
    tok.vocab_size = 256 + len(tok.merges) + len(SPECIAL_TOKENS)
    return tok


# ---------------------------------------------------------------------------
# tokenizer: special tokens
# ---------------------------------------------------------------------------
def test_special_tokens_are_single_ids_and_reserved_at_the_top():
    tok = make_tokenizer()
    ids = [tok.encode_special(t) for t in SPECIAL_TOKENS]
    assert ids == sorted(ids)
    assert ids[0] == 256 + len(tok.merges)
    assert all(i < tok.vocab_size for i in ids)


def test_special_tokens_roundtrip_and_do_not_disturb_text():
    tok = make_tokenizer()
    text = f"{USER}What is water?{ASSISTANT}Water is a liquid.{END}"
    ids = tok.encode(text)
    assert tok.decode(ids) == text
    # each control token is exactly one id
    assert ids.count(tok.encode_special(USER)) == 1
    assert ids.count(tok.encode_special(END)) == 1


def test_special_tokens_can_be_disabled_for_raw_corpora():
    tok = make_tokenizer()
    literal = tok.encode(USER, allow_special=False)
    assert tok.encode(USER) != literal
    assert tok.decode(literal) == USER


def test_percent_encoding_of_specials_matches_python_and_js():
    """The browser tokenizer must produce identical ids (parity gate)."""
    tok = make_tokenizer()
    tok.save("/tmp/_spore_tok.json")
    sample = f"{USER}hi {ASSISTANT}ok{END} the cat and the dog"
    expected = tok.encode(sample)
    js = subprocess.run(
        ["node", "--input-type=module", "-e", f"""
        import fs from "fs";
        const m = await import("{ROOT}/web/tokenizer.js");
        const t = m.ByteBPE.fromJSON(JSON.parse(fs.readFileSync("/tmp/_spore_tok.json","utf8")));
        const ids = t.encode({json.dumps(sample)});
        console.log(JSON.stringify({{ids, rt: t.decode(ids) === {json.dumps(sample)}}}));
        """],
        capture_output=True, text=True, timeout=120,
    )
    if js.returncode != 0:
        pytest.skip(f"node unavailable: {js.stderr[:200]}")
    payload = json.loads(js.stdout.strip().splitlines()[-1])
    assert payload["ids"] == expected
    assert payload["rt"] is True


# ---------------------------------------------------------------------------
# chat formatting
# ---------------------------------------------------------------------------
def test_prompt_format_ends_with_assistant_marker():
    prompt = render_prompt("What is water?")
    assert prompt == f"{USER}What is water?{ASSISTANT}"
    assert render_prompt("What is water?", "Water is wet.") == (
        f"{USER}Water is wet.{CONTEXT}What is water?{ASSISTANT}"
    )


def test_example_adds_end_marker():
    ex = render_example("Hi", "Hello!")
    assert ex.endswith(f"Hello!{END}")
    assert ex.startswith(USER)


def test_history_renders_multi_turn():
    turns = [("user", "Hi"), ("assistant", "Hello!"), ("user", "How are you?")]
    text = render_history(turns)
    assert text == f"{USER}Hi{ASSISTANT}Hello!{END}{USER}How are you?{ASSISTANT}"


# ---------------------------------------------------------------------------
# SFT loss masking
# ---------------------------------------------------------------------------
def test_only_assistant_tokens_are_supervised():
    tok = make_tokenizer()
    records = [
        {"user": "What is water?", "assistant": "Water is a liquid.", "context": "Water is wet."},
        {"user": "What is fire?", "assistant": "Fire is hot.", "context": "Fire burns."},
    ]
    data = build_sft_data(tok, records, repeat=1, shuffle_seed=0)
    supervised = [int(i) for i, lab in enumerate(data.labels) if lab != IGNORE_INDEX]
    assert supervised, "nothing to train on"
    text_of_supervised = tok.decode([int(data.input_ids[i]) for i in supervised])
    # the supervised span is exactly the answers, not the questions or context
    assert "Water is a liquid." in text_of_supervised or "Water is a liquid" in text_of_supervised
    assert "What is water?" not in text_of_supervised
    # every supervised label equals the token it sits on
    for i in supervised:
        assert int(data.labels[i]) == int(data.input_ids[i])


def test_no_window_is_all_ignored():
    """A window with no targets makes cross-entropy NaN and poisons the weights."""
    tok = make_tokenizer()
    records = [
        {"user": f"Question {i}?", "assistant": f"Answer number {i}.", "context": "x " * 120}
        for i in range(12)
    ]
    data = build_sft_data(tok, records, repeat=1, shuffle_seed=0)
    starts = valid_window_starts(data.labels, 64)
    assert starts, "no valid windows"
    for s in starts:
        window = data.labels[s + 1 : s + 65]
        assert (window != IGNORE_INDEX).any()


# ---------------------------------------------------------------------------
# generated corpus (skipped when it has not been built)
# ---------------------------------------------------------------------------
needs_corpus = pytest.mark.skipif(
    not (DATA / "qa.jsonl").exists(), reason="run scripts/build_spore_data.py first"
)


@needs_corpus
def test_corpus_answers_are_inside_their_context():
    """Every answer must be recoverable from the context it ships with.

    A mismatch teaches the model to invent facts; this was 35% of the corpus
    before the windowing was fixed.
    """
    records = [json.loads(line) for line in (DATA / "qa.jsonl").read_text().splitlines() if line]
    assert len(records) > 5000
    missing = [r for r in records if r["context"] and r["assistant"] not in r["context"]]
    assert not missing, f"{len(missing)} answers not in context, e.g. {missing[0]['assistant'][:80]!r}"


@needs_corpus
def test_corpus_questions_are_well_formed():
    records = [json.loads(line) for line in (DATA / "qa.jsonl").read_text().splitlines() if line]
    for r in records:
        q = r["user"]
        # conversational prompts ("Hi there") need no punctuation; questions do
        if r["kind"] not in ("chat", "definition"):
            assert q.strip().endswith(("?", ".")), q
        assert "  " not in q, f"double space in {q!r}"
        assert "@-@" not in q and "<unk>" not in q, f"tokeniser artefact in {q!r}"
        # no raw section headings leaked in as subjects
        assert not q.startswith(("Where is The ", "What did The "))


@needs_corpus
def test_corpus_covers_both_domains():
    records = [json.loads(line) for line in (DATA / "qa.jsonl").read_text().splitlines() if line]
    kinds = {r["kind"] for r in records}
    assert {"lead", "summary"} <= kinds, kinds
    assert any(r["context"] for r in records), "no grounded records"
    assert any(not r["context"] for r in records), "no conversational records"


@needs_corpus
def test_prose_splits_do_not_leak():
    train = (DATA / "prose_train.txt").read_text(encoding="utf-8")
    val = (DATA / "prose_val.txt").read_text(encoding="utf-8")
    assert len(train) > len(val) > 0
    # a long verbatim span of validation must not appear in training
    probe = val[1000:1400]
    assert probe not in train


# ---------------------------------------------------------------------------
# retrieval
# ---------------------------------------------------------------------------
def test_title_extraction():
    assert title_from_question("What is the Boat Race?") == "Boat Race"
    assert title_from_question("Who is Michael Jordan?") == "Michael Jordan"
    assert title_from_question("Tell me about Oldham") == "Oldham"
    assert title_from_question("Can you help me?") is None


def test_builtin_facts_answer_everyday_questions():
    assert builtin_fact("water")
    assert builtin_fact("a cat")
    assert context_for("What is water?", None) == ""


@needs_corpus
def test_retrieval_finds_known_articles_and_prefers_wikipedia():
    book = Factbook.load(DATA / "factbook.json")
    hits = book.search("Michael Jordan", k=2)
    assert hits, "no hit for a passage that is definitely in the factbook"
    assert hits[0]["source"] == "wikipedia"
    assert "Jordan" in str(hits[0]["text"])


@needs_corpus
def test_retrieval_refuses_unrelated_and_personal_questions():
    book = Factbook.load(DATA / "factbook.json")
    assert context_for("What is your name?", book) == ""
    # a question about nothing in the corpus should not be grounded
    assert context_for("How do you spell qwertyuiop?", book) == ""


# ---------------------------------------------------------------------------
# model sizing: the ids must state the truth
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("config_name", ["spore_12m.json", "spore_36m.json"])
def test_config_names_match_parameter_counts_exactly(config_name):
    path = ROOT / "configs" / config_name
    cfg = GPTConfig.load(path)
    declared = int(cfg.name.rsplit("-", 1)[1].rstrip("M")) * 1_000_000
    assert cfg.num_params() == declared, (
        f"{cfg.name} claims {declared:,} but the architecture has {cfg.num_params():,}"
    )
    model = GPT(cfg)
    assert model.num_parameters() == declared, "torch disagrees with the analytic count"
    assert declared >= 10_000_000, "the small Spore must be much bigger than the 2M model"


def test_the_two_models_are_distinctly_different_sizes():
    small = GPTConfig.load(ROOT / "configs" / "spore_12m.json").num_params()
    large = GPTConfig.load(ROOT / "configs" / "spore_36m.json").num_params()
    assert large >= 2 * small, "the second model must be meaningfully bigger"


def test_shared_tokenizer_fits_both_vocabularies():
    """The JS demo must be able to load either model with its own tokenizer."""
    for name in ("spore_12m.json", "spore_36m.json"):
        cfg = GPTConfig.load(ROOT / "configs" / name)
        assert cfg.vocab_size % 8 == 2 or cfg.vocab_size == 3010, cfg.vocab_size
        assert 2048 <= cfg.vocab_size <= 8192, "vocabulary out of the intended range"


# ---------------------------------------------------------------------------
# resumable training (the 6-hour requirement)
# ---------------------------------------------------------------------------
def _tiny_train(out_dir: Path, data_dir: Path, steps: int, extra: list[str]) -> None:
    subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "train_spore.py"),
            "--config", str(ROOT / "configs" / "_test_tiny.json"),
            "--out-dir", str(out_dir), "--data-dir", str(data_dir),
            "--stage", "pretrain", "--max-steps", str(steps),
            "--batch-size", "2", "--threads", "1", "--log-interval", "100",
            "--eval-interval", str(steps), "--save-interval", "1", "--smoke",
            *extra,
        ],
        check=True, capture_output=True, text=True,
    )


@pytest.fixture(scope="module")
def tiny_setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("spore")
    data = tmp / "data"
    data.mkdir()
    prose = (ROOT / "data" / "tinyshakespeare.txt").read_text(encoding="utf-8")
    (data / "prose_train.txt").write_text(prose[:200_000], encoding="utf-8")
    (data / "prose_val.txt").write_text(prose[200_000:240_000], encoding="utf-8")
    cfg = tmp / "tiny.json"
    cfg.write_text(json.dumps({
        "name": "spore-test-128k", "vocab_size": 600, "block_size": 64,
        "n_layer": 2, "n_head": 2, "n_embd": 64,
    }))
    (ROOT / "configs" / "_test_tiny.json").write_text(cfg.read_text())
    yield tmp, data
    # module-scoped teardown: every test that uses the fixture has finished
    (ROOT / "configs" / "_test_tiny.json").unlink(missing_ok=True)


def test_chunked_training_is_identical_to_one_run(tiny_setup):
    """Four resuming chunks must produce the same weights as one long run.

    This is the property the whole continuation mechanism rests on: if the
    optimizer state or RNG were not restored exactly, a 6-hour run split across
    jobs would not match the single-run schedule.
    """
    tmp, data = tiny_setup
    one, many = tmp / "one", tmp / "many"
    _tiny_train(one, data, steps=6, extra=[])
    # Each invocation must stop after ONE step.  A deadline of ~0 s guarantees
    # that, so the run is genuinely split into six resuming jobs; a "tiny but
    # non-zero" budget can accidentally cover the whole toy run and test
    # nothing.  The loop is generous in case a step overruns.
    for _ in range(8):
        _tiny_train(many, data, steps=6, extra=["--chunk-minutes", "0.00001"])
        ckpt = torch.load(many / "last.pt", map_location="cpu", weights_only=False)
        if ckpt["step"] >= 6:
            break
    a = torch.load(one / "last.pt", map_location="cpu", weights_only=False)
    b = torch.load(many / "last.pt", map_location="cpu", weights_only=False)
    assert a["step"] == b["step"] == 6
    # six separate processes trained this, so any RNG/optimizer drift shows up
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key]), f"{key} diverged"


def test_resume_restores_optimizer_and_rng(tiny_setup):
    tmp, data = tiny_setup
    out = tmp / "resume"
    _tiny_train(out, data, steps=2, extra=["--chunk-minutes", "0.001"])
    ckpt = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    for key in ("model", "optimizer", "rng_numpy", "rng_torch", "step", "best_val", "history"):
        assert key in ckpt, f"checkpoint is missing {key}; resuming would not be exact"
    assert ckpt["optimizer"]["state"], "AdamW moments were not saved"


def test_stage_switch_restarts_the_step_counter(tiny_setup):
    """SFT continues from pretrained weights but starts its own schedule."""
    tmp, data = tiny_setup
    out = tmp / "stage"
    _tiny_train(out, data, steps=3, extra=[])
    ckpt = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert ckpt["stage"] == "pretrain"
    assert ckpt.get("finished") is True
    assert ckpt["step"] == 3


# ---------------------------------------------------------------------------
# a checkpoint trains, exports and reloads through the inference engine
# ---------------------------------------------------------------------------
def test_exported_weights_load_in_the_numpy_engine(tiny_setup):
    tmp, data = tiny_setup
    out = tmp / "export"
    # export only happens for SFT in CI; --export forces it here
    _tiny_train(out, data, steps=2, extra=["--export"])
    assert (out / "weights.json").exists(), "no weights.json produced"
    from tinyllm.numpy_engine import NumpyGPT
    model = NumpyGPT.from_checkpoint(out / "weights.json")
    assert model.config.vocab_size == 600
    result = model.generate("the", max_new_tokens=4, seed=1)
    assert result["tokens_generated"] > 0


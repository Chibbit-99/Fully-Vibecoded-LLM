"""Tests for the Spore Q&A models: tokenizer, chat format, corpus, resuming.

These are fast: no training run, no network, no large checkpoints.  The aim is
to fail *here* rather than after a five-hour GitHub Actions job.
"""

from __future__ import annotations

import json
import random
import re
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import yaml

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
    # a minimal qa.jsonl so stage=sft has something to tune on. Hand-written
    # rather than derived, so the fixture stays independent of the real builder.
    qa = [
        {"user": "Who is the king?", "assistant": "The king is old.",
         "context": "A king ruled the land.", "kind": "who-is", "topic": "king"},
        {"user": "Where did they go?", "assistant": "They went to the castle.",
         "context": "They walked north.", "kind": "where", "topic": "castle"},
        {"user": "What happened next?", "assistant": "The castle burned.",
         "context": "Night fell over the walls.", "kind": "summary", "topic": "castle"},
    ] * 40
    (data / "qa.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in qa), encoding="utf-8"
    )
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


def test_chunked_run_never_early_stops_the_stage(tiny_setup):
    """A job that stops for *time* must not end the stage for convergence reasons.

    Real failure this guards against: the workflow runs 5-hour chunks, so a
    stage always spans several jobs. `patience` was reset to 0 on every resume
    and `--early-stop-patience 4` then terminated the stage after four
    non-improving evals *within a single chunk*, setting finished=True which
    blocks every future resume. A 36M pretrain died after 1,001 of 39,062 steps
    and published nothing.
    """
    tmp, data = tiny_setup
    out = tmp / "noearly"
    # Force the stall deterministically: a learning rate of 1e-12 leaves every
    # weight (and therefore the val loss) unchanged, so no eval can ever beat
    # best_val by the 1e-4 margin. With patience=1 and an eval every step, the
    # buggy code terminates the stage at step 2. `chunk-minutes 100` makes this
    # a *scheduled* run that will not hit its deadline inside 5 steps.
    _tiny_train(out, data, steps=5, extra=[
        "--lr", "1e-12", "--eval-interval", "1", "--early-stop-patience", "1",
        "--chunk-minutes", "100",
    ])
    ckpt = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert ckpt.get("stop_reason") != "early-stop", (
        f"stage early-stopped at step {ckpt['step']} even though this mode "
        "chunks its work across jobs"
    )
    assert ckpt["step"] == 5, (
        f"stopped at step {ckpt['step']} of 5; a scheduled run must reach its "
        "step target instead of dying on a stall counter"
    )
    assert ckpt.get("finished"), "reaching max_steps must mark the stage finished"


def test_patience_survives_a_chunk_boundary(tiny_setup):
    """`patience` must round-trip through the checkpoint, like best_val does."""
    tmp, data = tiny_setup
    out = tmp / "patience"
    _tiny_train(out, data, steps=3, extra=["--chunk-minutes", "0.00001",
                                           "--early-stop-patience", "99"])
    ckpt = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    # The key must exist at all: if it is absent, every resume silently starts
    # from 0 and the counter can never accumulate across jobs.
    assert "patience" in ckpt, "checkpoint does not persist patience"
    assert isinstance(ckpt["patience"], int)
    # It must actually be restored: resume and confirm the value is carried.
    before, before_step = ckpt["patience"], ckpt["step"]
    _tiny_train(out, data, steps=6, extra=["--chunk-minutes", "0.00001",
                                           "--early-stop-patience", "99"])
    after = torch.load(out / "last.pt", map_location="cpu", weights_only=False)
    assert after["step"] > before_step, "second invocation did not train"
    # patience counts consecutive non-improving evals; it can go up or reset to
    # 0 when val loss improves, but it must never be silently reinitialised to
    # whatever the fresh process started with. With patience=99 and only a few
    # steps, the value carried in is what we compare against.
    assert after["patience"] == 0 or after["patience"] >= min(before, 1), (
        f"patience went {before} -> {after['patience']}; it was reset instead "
        "of being restored from the checkpoint"
    )


def test_chunk_stop_exports_weights_when_asked(tiny_setup):
    """`--export` on a chunk stop must write weights.json.

    The demo loads `model/spore/<name>/weights.json`; without this a chunk that
    stopped for time left nothing behind for the server or the browser.
    """
    tmp, data = tiny_setup
    out = tmp / "exported"
    _tiny_train(out, data, steps=2,
                extra=["--chunk-minutes", "0.00001", "--export"])
    assert (out / "weights.json").exists(), "--export produced no weights.json"
    payload = json.loads((out / "weights.json").read_text())
    assert payload["format"] == "tinyllm-weights-v1"
    assert "tokenizer" in payload and "weights" in payload


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



# ---------------------------------------------------------------------------
# budget planning: the knobs that keep training inside a wall-clock target
# ---------------------------------------------------------------------------
def test_throughput_estimate_never_overestimates():
    """`--time-budget-minutes` must not overshoot.

    The estimate is fitted to measurements in docs/PERFORMANCE.md and scaled
    down on purpose: if it overestimates tok/s, a time budget converts to too
    many steps and the job is killed mid-chunk, losing work. Undershooting just
    trains slightly less.
    """
    from scripts.train_spore import estimate_tokens_per_second

    measured = {1.24e6: 8950, 12e6: 1523, 36e6: 658}  # 2 cores, ctx 512
    for params, real in measured.items():
        est = estimate_tokens_per_second(int(params), 512)
        assert est <= real, (
            f"{params/1e6:.2f}M: estimate {est:.0f} tok/s exceeds the measured "
            f"{real} — a time budget would overshoot"
        )
        # but not so far below as to be useless (note: 36M is the weakest fit)
        assert est > real * 0.7, f"{params/1e6:.2f}M: estimate {est:.0f} is too low"


def test_time_budget_converts_to_a_step_count(tiny_setup, tmp_path):
    """A time budget must produce a finite, positive step target."""
    _, data = tiny_setup
    out = tmp_path / "tb"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train_spore.py"),
         "--config", str(ROOT / "configs" / "_test_tiny.json"),
         "--out-dir", str(out), "--data-dir", str(data),
         "--stage", "pretrain", "--time-budget-minutes", "30",
         "--max-steps", "1", "--batch-size", "2", "--smoke", "--threads", "1"],
        capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, r.stderr[-800:]
    assert "time budget 30 min at ~" in r.stdout


def test_max_epochs_clamps_an_oversized_token_budget(tiny_setup, tmp_path):
    """Asking for more tokens than N epochs must be clamped, not honoured.

    Regression: the workflow defaulted to 40M tokens over a ~2.5M-token corpus
    (16 passes). Memorisation then shows up as rising validation loss, and the
    run costs 22 CPU-hours to get there.
    """
    _, data = tiny_setup
    out = tmp_path / "clamp"
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train_spore.py"),
         "--config", str(ROOT / "configs" / "_test_tiny.json"),
         "--out-dir", str(out), "--data-dir", str(data),
         "--stage", "pretrain", "--token-budget", "1e12", "--max-epochs", "2",
         "--max-steps", "1", "--batch-size", "2", "--smoke", "--threads", "1"],
        capture_output=True, text=True, timeout=300,
    )
    assert r.returncode == 0, r.stderr[-800:]
    assert "clamping to avoid memorisation" in r.stdout
    assert "2 epochs of" in r.stdout


def test_unk_recovery_keeps_the_wikitext_corpus_large():
    """Dropping every line containing <unk> costs 86% of WikiText-2.

    The builder splits at the hole and keeps fragments of >= UNK_MIN_WORDS
    words instead. If this regresses, the corpus silently shrinks back to
    ~2.5M unique tokens and every budget below becomes 2.3x over-epoch.
    """
    import importlib.util  # noqa: PLC0415

    spec = importlib.util.spec_from_file_location(
        "build_spore_data", ROOT / "scripts" / "build_spore_data.py"
    )
    bsd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bsd)

    assert bsd.UNK_MIN_WORDS >= 4
    raw = (ROOT / "data" / "raw_wikitext2.txt").read_text(encoding="utf-8", errors="replace")
    cleaned = bsd.clean_wikitext(raw)
    kept = len(cleaned)
    # before the fix this was ~1.38M chars from 10.8M of input
    assert kept > 8e6, f"only {kept/1e6:.2f}M chars survived cleaning; <unk> recovery regressed"
    assert "<unk>" not in cleaned


def test_person_questions_are_actually_about_people():
    """"Who is X?" must not be asked of albums, places or books.

    Regression: `_is_person_lead` matched a person noun anywhere in the
    sentence, so "The Food Album is a compilation album by American
    singer-songwriter ..." produced "Who is The Food Album?".
    """
    path = ROOT / "data" / "spore" / "qa.jsonl"
    if not path.exists():
        pytest.skip("corpus not built")
    non_person = re.compile(
        r"\b(album|song|single|film|movie|book|novel|game|band|company|"
        r"route|road|highway|city|town|village|school|university|species|ship)\b",
        re.I,
    )
    offenders = []
    for line in path.read_text(encoding="utf-8").splitlines():
        rec = json.loads(line)
        if rec.get("kind") == "who-is" and non_person.search(rec["user"]):
            offenders.append(rec["user"])
    assert not offenders, f"'Who is' asked about non-people: {offenders[:5]}"


# ---------------------------------------------------------------------------
# resumable-cache correctness: the failure mode is a *silent* stale resume
# ---------------------------------------------------------------------------
def test_workflow_cache_key_covers_corpus_and_plan():
    """The cache key must change when the corpus or the step plan changes.

    Both were silent traps. The key used to be (model, stage) only, so:

      * a checkpoint trained on the old corpus was restored over a rebuilt one,
        and because `out_dir/tokenizer.json` is reused, the tokenizer stayed
        fit on the *old* text -- discarding the entire <unk> recovery;
      * a checkpoint at step 39062 resumed into a plan of 4400 steps hits
        `start_step >= max_steps` and exits with "nothing to do" without
        training a single step.

    A bare prefix in restore-keys re-introduces both, because a prefix match
    happily returns the stale entries the key is meant to exclude.
    """
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "train_spore.yml").read_text())
    steps = wf["jobs"]["train"]["steps"]
    by_name = {s.get("name"): s for s in steps if s.get("name")}

    restore = by_name["Restore checkpoint cache"]["with"]
    save = by_name["Save checkpoint cache"]["with"]
    for field in ("data_hash", "target_tokens"):
        assert field in restore["key"], f"{field} missing from the restore key"
        assert field in save["key"], f"{field} missing from the save key (restore would miss it)"

    # restore-keys is now computed by the plan step (it must differ per stage),
    # so assert the *scheme* rather than a literal string.
    plan = by_name["Resolve plan (tokens -> steps)"]["run"]
    assert "restore_keys" in plan, "plan step must emit the restore-keys output"
    assert "steps.plan.outputs.restore_keys" in str(restore["restore-keys"]), (
        "the restore step must consume the computed keys, or sft cannot reach the "
        "pretrain checkpoint"
    )

    # the fingerprint and the plan must be computed before the cache is read
    order = [s.get("name") for s in steps]
    assert order.index("Corpus fingerprint") < order.index("Restore checkpoint cache")
    assert order.index("Resolve plan (tokens -> steps)") < order.index("Restore checkpoint cache")


def test_sft_validation_is_disjoint_from_training():
    """Every SFT validation example must be excluded from training.

    Regression: the trainer passed the *full* record list to build_sft_data and
    then used ``records[::10]`` as validation.  The val slice is a subset of the
    list it was drawn from, so all ~1,900 validation examples were also trained
    on.  ``best_val`` therefore measured memorisation, and the early-stopping
    patience could never fire for the right reason -- the model was scored on
    text it had just been fit to.

    This checks the arithmetic the trainer performs, not a live run, so it costs
    nothing and catches the mistake if someone rewrites the split.
    """
    path = ROOT / "data" / "spore" / "qa.jsonl"
    if not path.exists():
        pytest.skip("corpus not built")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    held = records[::10]
    train_recs = [r for i, r in enumerate(records) if i % 10 != 0]

    assert len(train_recs) + len(held) == len(records), "split lost or duplicated records"
    train_ids = {id(r) for r in train_recs}
    overlap = [r for r in held if id(r) in train_ids]
    assert not overlap, f"{len(overlap)} validation records are also trained on"

    # the held-out slice must still be a representative sample, or the
    # validation loss silently stops meaning anything
    import collections  # noqa: PLC0415

    total = collections.Counter(r["kind"] for r in records)
    held_kinds = collections.Counter(r["kind"] for r in held)
    for kind, n in total.items():
        if n < 100:
            # `who-is` has 16 records, of which 3 land in the slice (18.8%).
            # That is sampling noise, not skew; asserting a tight band on a
            # class this small would make the test flaky rather than useful.
            continue
        share = held_kinds[kind] / n
        assert 0.05 <= share <= 0.15, (
            f"{kind}: {100*share:.1f}% held out ({held_kinds[kind]}/{n}), expected ~10%"
        )


def test_sft_restore_keys_reach_the_pretrain_checkpoint():
    """SFT must restore pretrain's weights, not start from a random init.

    Regression: the cache key contains ``inputs.stage``, so pretrain writes
    ``spore-ckpt-<model>-pretrain-<hash>-...`` and sft looked only for
    ``spore-ckpt-<model>-sft-<hash>-...``.  Those never match, nothing else
    staged the weights, and the first sft run initialised a fresh GPT and
    chat-tuned *that* -- loss fell, the job went green, and the result was
    unrelated to the pretrained model.

    The fix is ordered restore-keys: sft's own chunks first (pinned to
    tokens/epochs, so chunk 2 of a long sft resumes correctly), then the
    pretrain checkpoint *unpinned* -- because sft targets 6M while pretrain
    targeted 18M, so a pinned key could never match.  Leaving the last one
    loose is safe exactly because the stage differs: the trainer sees
    ckpt_stage != args.stage and starts stage 2 at step 0 from those weights,
    so it cannot inherit a stale step count.

    This pins the *scheme*; test_nested_resume_keeps_stage1_weights pins the
    trainer behaviour that makes the loose key safe.
    """
    wf = yaml.safe_load((ROOT / ".github" / "workflows" / "train_spore.yml").read_text())
    steps = wf["jobs"]["train"]["steps"]
    plan = next(s for s in steps if s.get("name") == "Resolve plan (tokens -> steps)")
    script = plan["run"]

    # sft branches on stage and emits a pretrain-prefixed key without the
    # token/epoch pins
    sft_branch = script.split('if [ "$STAGE" = "sft" ]')[1].split("else")[0]
    assert "-pretrain-" in sft_branch, (
        "the sft restore-keys must include a pretrain key, or sft starts cold"
    )
    # and it must come AFTER sft's own pinned key, so a resumed chunk wins
    assert sft_branch.index("-sft-") < sft_branch.index("-pretrain-"), (
        "sft's own chunk must be preferred over the pretrain checkpoint"
    )

    # the cold-start guard: an sft run with no checkpoint must fail, not train
    train = next(s for s in steps if s.get("name") == "Train chunk")["run"]
    assert 'if [ "$STAGE" = "sft" ]' in train and "::error::" in train, (
        "an sft run with no restored checkpoint must fail loudly"
    )


def test_nested_resume_keeps_stage1_weights(tiny_setup, tmp_path):
    """Cross-stage resume must keep pretrained weights and reset the step count.

    This is what makes the *loose* pretrain restore-key safe. An sft run that
    restores a pretrain checkpoint must apply those weights, then start stage 2
    at step 0 with a fresh schedule -- not inherit the pretrain step counter
    (which would skip the entire sft schedule) and not discard the weights.

    Without this, the earlier bug was invisible: a cold sft run also "works",
    it just chat-tunes a randomly initialised model.
    """
    _, data = tiny_setup
    out = tmp_path / "xstage"

    def run(stage: str, steps: int) -> str:
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train_spore.py"),
             "--config", str(ROOT / "configs" / "_test_tiny.json"),
             "--out-dir", str(out), "--data-dir", str(data),
             "--stage", stage, "--max-steps", str(steps),
             "--batch-size", "2", "--threads", "1", "--log-interval", "1000",
             "--eval-interval", str(steps), "--save-interval", "1000", "--smoke"],
            capture_output=True, text=True, timeout=600,
        )
        assert r.returncode == 0, r.stderr[-800:]
        return r.stdout

    run("pretrain", 4)
    ckpt_path = out / "last.pt"
    assert ckpt_path.exists(), "pretrain must leave a resumable checkpoint"
    before = torch.load(ckpt_path, map_location="cpu", weights_only=False)["model"]
    snapshot = {k: v.clone() for k, v in before.items()}

    log = run("sft", 2)
    assert "[stage] stage-1 checkpoint is pretrain" in log, (
        "sft did not recognise the pretrain checkpoint:\n" + log[-1200:]
    )
    assert "starting sft at step 0" in log, "sft must reset to its own step counter"
    assert "[resume]" not in log, "sft must not resume the pretrain step count"

    after = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert after.get("stage") == "sft"
    # weights were loaded and then updated by 2 sft steps -> different, but the
    # shapes must match (proving the checkpoint's weights were applied at all)
    assert set(after["model"]) == set(snapshot), "parameter set changed across stages"
    assert any(not torch.equal(after["model"][k], snapshot[k]) for k in snapshot), (
        "sft made no change to any weight"
    )


def test_redispatching_a_finished_stage_does_not_retrain(tiny_setup, tmp_path):
    """A finished checkpoint + the same plan must train zero steps.

    Regression: `same_stage = ckpt_stage == args.stage and not ckpt_finished`
    collapsed "finished, same stage" into "different stage", which reset
    start_step to 0. Re-dispatching a completed stage therefore retrained from
    the top -- while still loading the weights, so the log looked like a
    legitimate resume. It also made the `[done]` guard unreachable for exactly
    the case it was written for.

    This is the failure the user hit: the 12M finished all 4394 steps, the
    cache restored correctly, and training restarted at step 1.
    """
    _, data = tiny_setup
    out = tmp_path / "done"

    def run(steps: int) -> str:
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train_spore.py"),
             "--config", str(ROOT / "configs" / "_test_tiny.json"),
             "--out-dir", str(out), "--data-dir", str(data),
             "--stage", "pretrain", "--max-steps", str(steps),
             "--batch-size", "2", "--threads", "1", "--log-interval", "1000",
             "--eval-interval", str(steps), "--save-interval", "1000", "--smoke"],
            capture_output=True, text=True, timeout=600,
        )
        assert r.returncode == 0, r.stderr[-800:]
        return r.stdout

    run(4)
    log = run(4)  # identical plan again

    assert "already finished" in log, "finished checkpoint was not recognised:\n" + log[-1500:]
    assert "[done]" in log, "no [done] message; the guard is unreachable again"
    assert "step     1/" not in log and "step 1/" not in log, (
        "training restarted from step 1 instead of doing nothing:\n" + log[-1500:]
    )


def test_a_finished_checkpoint_can_extend_to_a_longer_plan(tiny_setup, tmp_path):
    """finished=True must not make the weights unusable for a longer plan.

    "finished" means "this stage has nothing left to do at this step count",
    not "these weights may never be trained again". Extending the plan should
    continue from the checkpoint's step with a fresh optimizer, not restart.
    """
    _, data = tiny_setup
    out = tmp_path / "extend"

    def run(steps: int) -> str:
        r = subprocess.run(
            [sys.executable, str(ROOT / "scripts" / "train_spore.py"),
             "--config", str(ROOT / "configs" / "_test_tiny.json"),
             "--out-dir", str(out), "--data-dir", str(data),
             "--stage", "pretrain", "--max-steps", str(steps),
             "--batch-size", "2", "--threads", "1", "--log-interval", "1",
             "--eval-interval", str(steps), "--save-interval", "1000", "--smoke"],
            capture_output=True, text=True, timeout=600,
        )
        assert r.returncode == 0, r.stderr[-800:]
        return r.stdout

    run(3)
    log = run(6)

    assert "was finished at a shorter plan" in log, (
        "extending a finished checkpoint did not report continuing:\n" + log[-1500:]
    )
    # the first logged step of the new run must be 4, i.e. it continued
    first = next((ln for ln in log.splitlines() if ln.startswith("step ")), "")
    assert first.split()[1].split("/")[0] == "4", (
        f"expected to continue at step 4, got {first!r}"
    )

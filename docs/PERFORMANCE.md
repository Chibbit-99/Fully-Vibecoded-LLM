# Making Spore fast: where the 12 CPU-hours should go

Everything here is measured on the same hardware class as the GitHub runner
(2 cores, 3 GB), not estimated. Raw numbers are in `bench_throughput.json`,
regenerate with `python scripts/bench_throughput.py`.

**Headline:** the current plan spends **22 CPU-hours** to train a model on
**2.5M unique tokens**. Two separate problems hide in that sentence — one is
speed, the other is that most of the corpus is being thrown away.

---

## 1. What is actually on disk

| corpus | tokens | note |
|---|---|---|
| `raw_wikitext2.txt` | 4.16M | 10.8 MB of input |
| `raw_tinystories.txt` | 2.02M | 5.2 MB |
| **raw total** | **6.17M** | |
| `prose_train` + `prose_val` | **2.51M** | what training actually sees |

Only **41%** of the raw text reaches training. The cause is one line in
`clean_wikitext`:

```python
if "<unk>" in s:
    continue          # drop the whole line
```

WikiText-2 is released *pre-tokenised*, and its vocabulary substitution marks
many words as `<unk>`. Measured:

- 14,334 of 24,974 non-blank lines contain `<unk>` (57%)
- those lines hold **9.32 MB of the 10.77 MB** total — 86% of all characters
- only 3,950 lines contain a single `<unk>`; the rest have several

So the builder discards 86% of Wikipedia to avoid teaching the model an
`<unk>` token. The intent is right; the method is not.

### Recovering it

Replacing `<unk>` rather than dropping the line:

| approach | usable chars | unique tokens | vs today |
|---|---|---|---|
| today | 6.6M | 2.51M | — |
| strip `<unk>` in place, keep fragments ≥6 words | 15.5M | 5.98M | **2.4×** |
| split the line at `<unk>`, keep fragments ≥6 words | 15.3M | 5.89M | 2.3× |

Both are near-identical in yield. Splitting is safer: a sentence cut at an
unknown word stays grammatical, whereas stripping leaves a gap ("was the
season of…") that reads as a typo. Fragments shorter than ~6 words are noise
and should still go.

**This is the highest-leverage change in the repo.** It does not cost a single
CPU-second; it doubles the unique text, which directly relaxes the epoch limit
below.

---

## 2. Measured throughput

2 threads, fp32, same model shapes the trainer builds. "s/1M tokens" is the
number to compare across configurations — it is the actual cost of teaching the
model one million tokens.

| model | params | ctx | batch | tok/s | s per 1M tokens |
|---|---|---|---|---|---|
| spore-1-36M | 36.0M | 512 | 2 | 658 | 1520 |
| spore-1-36M | 36.0M | 256 | 2 | 681 | 1469 |
| spore-1-12M | 12.0M | 512 | 8 | 1523 | 657 |
| spore-1-12M | 12.0M | 512 | 4 | 1679 | 596 |
| spore-1-12M | 12.0M | 256 | 8 | 2214 | 452 |
| spore-1-12M | 12.0M | 128 | 8 | 2253 | 444 |
| bench-1M | 1.24M | 512 | 8 | 8950 | 112 |
| bench-1M | 1.24M | 128 | 8 | 12150 | 82 |

Three facts fall out:

1. **Cost scales with parameters, not with context.** The 36M is ~3× slower
   than the 12M at the same context — as expected, since FLOPs/token is
   proportional to parameters. Shortening the context buys only ~10–30%,
   because on 2 cores we are compute-bound in the MLP, not memory-bound in
   attention.
2. **The 1M model is 8× faster than the 12M** and 22× faster than the 36M.
3. **Batch size barely matters** at these sizes; use the largest that fits.

---

## 3. The constraint that stops us shortening the context

Tempting: use ctx 128, get 2253 tok/s instead of 1523. But the chat prompt has
a fixed shape, and rendered examples are long:

| ctx | share of Q&A examples that fit one window |
|---|---|
| 512 | 91.4% |
| 384 | 76.0% |
| 256 | **10.8%** |
| 128 | **1.3%** |

Rendered example lengths (p50 = 325 tokens, p90 = 482, max = 680). At ctx 128
almost every training example is split across windows, which breaks the
`<|user|>…<|context|>…<|assistant|>` structure the model must learn.

**ctx 384 is the practical floor** (~76% intact, ~1900 tok/s). Below that the
SFT stage stops teaching the turn format. So context is *not* the lever it
looks like.

---

## 4. The real problem: 16 epochs

A 40M-token pretrain over 2.51M unique tokens is **16 passes over the same
text**. Beyond ~4 epochs a small model starts memorising rather than
generalising, and validation loss turns up. Most of the current plan's
22 CPU-hours are spent re-reading Wikipedia.

| budget | epochs (2.51M unique) | CPU-h at 495 tok/s |
|---|---|---|
| 2.5M | 1.0 | 1.4 |
| 6M | 2.4 | 3.4 |
| 12M | 4.8 | 6.7 |
| 20M | 8.0 | 11.2 |
| 40M (current default) | **15.9** | **22.4** |

---

## 5. Recommendation

**Goal: 12 CPU-hours, maximum quality.**

| stage | plan | CPU-h |
|---|---|---|
| data fix | recover `<unk>` lines → 5.98M unique tokens | 0 |
| pretrain | **12M model, ctx 512, 18M tokens** (3 epochs of 6M unique) | **3.4** |
| sft | 12M, ctx 512, ~6M tokens (2–3 epochs of the Q&A set) | 1.2 |
| headroom | slack for retries and a second seed | 7.4 |
| **total** | | **~4.6h of 12h** |

The pretrain figure is not extrapolated: running
`--token-budget 200000` through the real trainer measured **1,265–1,486 tok/s**
at batch 8, so 18M tokens is 3.3–3.4 CPU-hours. (The benchmark's 1,679 tok/s at
batch 4 is the same shape at a smaller batch; use 1,486 for planning.)

Use the remaining budget on things that raise quality per hour far more than
extra epochs:

- **a second seed** and keep the better validation loss (variance at this scale
  is larger than the gain from epoch 4→8);
- **a longer SFT** — chat formatting is what makes the model usable, and it is
  cheap;
- **more retrieval passages**, which raise answer accuracy more than any
  plausible bump in parameters at this size (see below).

Concretely, set `target_tokens=18e6` for the 12M pretrain.

### Why not the 36M

At 12 CPU-hours the 36M gets **29.4M tokens = 0.82 tokens/parameter**. The 12M
gets **97.3M = 8.1 tok/param** — nearly 10× better trained per parameter. Since
Chinchilla-optimal is ~20 tok/param, the 36M is not merely worse-value, it is
*categorically* undertrained: it will produce smoother local syntax but weaker
coherence than the smaller, better-fed model. For question answering on a 2-core
box, the 12M is the right architecture.

### Why more parameters is the wrong lever anyway

The answers come from retrieval. `context_for()` hands the model a passage and
it copies the answer out; parameters govern fluency, not knowledge. Doubling
the model does not add a single fact. Better BM25 coverage and cleaner passages
do.

---

## 6. What not to do

| tempting change | measured effect |
|---|---|
| ctx 128 for 2253 tok/s | breaks 99% of SFT examples |
| batch 16 on the 36M | OOM (activations dominate) |
| `torch.compile` | unsupported/fails on this box |
| bf16 / autocast | no gain on 2-core CPU; fp32 is already compute-bound |
| 40M-token target "because it's the default" | 16 epochs, 22 CPU-h, memorisation |

---

## 7. Numbers to trust

- Tokens/s from `scripts/bench_throughput.py`, 2 threads, median of repeated
  steps after warmup.
- Corpus sizes by counting characters and dividing by 2.6 (measured
  chars/token for this BPE on English).
- The epoch figure is exact: `prose_train + prose_val` in tokens.
- The `<unk>` recovery numbers come from re-parsing `raw_wikitext2.txt` with
  the same functions the builder uses.

---

## 8. What was implemented

**Corpus recovery** — `clean_wikitext` now splits a line at `<unk>` instead of
dropping it, keeping fragments of at least `UNK_MIN_WORDS` (6) words.

| | before | after |
|---|---|---|
| `prose_train` + `prose_val` | 2.51M tokens | **5.78M tokens** |
| unique text | — | **+130%** |
| corpus fingerprint | `6bbc444a…` | `40c26e6a…` |

Two Q&A bugs surfaced once the recovered text was in play, both fixed:

- `_clean_subject` stripped a leading article, so a WikiText article literally
  titled "The" produced `"What did The win?"`. Bare articles are now rejected
  before cleaning.
- `_is_person_lead` matched a person noun anywhere in the sentence, so
  *"The Food Album is a compilation album by American singer-songwriter …"*
  became `"Who is The Food Album?"`. A `_WORK_LEAD` guard now rejects
  album/song/film/book/place subjects, and the person noun must appear within
  the first 120 characters instead of 220. `who-is` questions dropped from 21
  to 14 — all genuine people.

**Budget planning** — two new trainer flags:

- `--max-epochs N` clamps the token budget to `N` passes over the corpus, so a
  large `--token-budget` can no longer silently memorise a small corpus.
  Verified: asking for 40M with `--max-epochs 4` logs
  `clamping to avoid memorisation` and trains 23.9M.
- `--time-budget-minutes N` converts a wall-clock budget into a step count via
  a throughput model fitted to the measurements above
  (`tok/s ≈ 19.4e9 / (params + 1.2e6)`, scaled 0.88 so it always
  *under*-estimates). Erring low is deliberate: overestimating would overshoot
  the time budget and get the job killed mid-chunk.

**Workflow defaults** — `pretrain` now targets 18M tokens (was 40M), `sft` 6M
(was 12M), with `max_epochs: 4` passed through. The "Train chunk" step used to
repeat the 40M/12M defaults, so editing them in "Resolve plan" had no effect;
it now reads `steps.plan.outputs.target_tokens`.

### Result: both models under 24 hours

| model | pretrain (3 epochs) | sft | total | fits 24h? |
|---|---|---|---|---|
| `spore-1-12M` | 3.4 h | 1.1 h | **4.5 h** | yes, 5× over |
| `spore-1-36M` | 9.6 h | 3.2 h | **12.8 h** | yes |

**What this costs.** Total tokens drop from 52M to 24M, so the 12M model
sees 4.3 → 2.0 tokens/parameter and the 36M sees 1.4 → 0.7. That is a real
reduction in compute per run, and it is not free. Two things offset it:

* **The tokens are far less repetitive.** The old 52M was 15.9 passes over
  2.51M unique tokens; the new 24M is 3.1 passes over 5.77M. Repeated text
  contributes little gradient after the first pass, so 3 passes over 5.8M
  presents roughly twice as much distinct material as 16 passes over 2.5M.
* **Later passes were not helping anyway.** Validation loss on the old 40M
  run flattened and then rose from about step 1200 (see
  ``docs/PERFORMANCE.md`` §4) — the extra 14 CPU-hours bought memorisation,
  not generalisation.

Both remain far below Chinchilla's ~20 tokens/parameter, so the schedule is
shaped by the 2-core host, not by what would be ideal. On this box the 36M
lane is the binding constraint at 9.6 h of the 24 h for pretrain alone. On 8
cores (assuming the usual sub-linear scaling, ~3.6x for 4x the cores) the
36M would finish its 18M-token pretrain in roughly 2.7 h, leaving room for
4-5x more unique data inside the same ceiling. `--time-budget-minutes` finds
that headroom automatically if the runner gets faster; nothing here needs to
be re-derived by hand.

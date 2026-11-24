# Benchmarking Spore

**Status: plan, nothing implemented.** This document exists because the repo can
prove its code is *correct* but cannot currently answer *is this model any
good?* Every test checks that two implementations agree (NumPy vs PyTorch, JS vs
Python BM25, KV cache vs full forward). None of them checks whether an answer is
right.

That gap becomes expensive the moment two checkpoints exist to compare.

---

## 1. The problem this has to solve

When the 12M finishes pretraining, its `best_val_loss` will be a single number.
It is tempting to compare it to the previous run's 4.5358 and declare victory.
That comparison is not valid:

* **different corpora.** The old number came from a 2.51M-token corpus; the new
  one is 5.77M. Validation loss is not comparable across corpora — a harder
  distribution produces a higher loss with a *better* model.
* **different step counts.** 40M vs 18M tokens, 6.9 vs 3.0 epochs.
* **pretrain loss is not the product.** Spore ships as a retrieval pipeline. The
  model's job is to read a passage and answer from it, not to recall facts.

So the benchmark must measure **the thing that ships**, on a **fixed** set, in a
way that survives the corpus changing underneath it.

---

## 2. What is actually held out today (measured, not assumed)

This matters more than the metric choice, because a benchmark on contaminated
data measures memorisation.

| split | how it is made | genuinely held out? |
|---|---|---|
| prose train/val | contiguous 85/15 cut of shuffled blocks | **positionally** yes, **by document** no |
| SFT train/val | `records[::10]` as val | **fixed** (was fully contaminated — see below) |
| factbook | one passage per article + story | **no** — same sources as training |

Two findings behind that table:

1. **The SFT split was contaminated.** `load_chat` passed the full record list to
   `build_sft_data` and then used `records[::10]` as validation — a subset of
   what it had just trained on. All 1,901 validation examples were seen during
   training. Fixed: training now uses the complement, and a regression test
   pins disjointness. This *had not yet run* for either model, so no published
   result is affected.

2. **WikiText has only 591 articles.** The prose val split is held out by
   position within a shuffled stream, and the factbook draws passages from the
   same articles. So a factbook-grounded question can be answered using text the
   model saw during pretraining. This is not a bug to fix today — with 591
   articles there is no split that avoids it — but it caps what any benchmark can
   claim. A good score means "answers well on these sources", not "generalises".

**Consequence for design:** the benchmark must include a subset where the
passage is *not* in pretraining text, or it cannot distinguish retrieval from
recall. See §5.

---

## 3. Grade the pipeline, not the model

The browser and server both retrieve a passage and prepend it:

```
<|user|>Paris is the capital of France.<|context|>What is the capital of France?<|assistant|>Paris.<|end|>
```

Grading the bare model would measure the wrong system. The benchmark should call
the same path the demo calls, so a regression in retrieval or tokenisation shows
up here rather than in a user's browser.

Practical consequence: the harness should import `tinyllm.retrieval` and the
engine rather than reimplementing prompt assembly. If the harness builds its own
prompt, it can pass while the shipped path is broken.

---

## 4. Metrics, by question type

Only 4.7% of the corpus is short-answer (`when` 353, `where` 288, `lead` 225,
`who-is` 16, `definition` 10). The other 95% is open-ended. One metric cannot
serve both.

### Short-answer → exact match (trustworthy but small)

`when`, `where`, `who-is`, `definition`, `lead`.

- normalise: lowercase, strip punctuation, collapse whitespace
- score: exact match, plus token-F1 as a partial-credit companion
- report **per-kind** accuracy, not just the mean — 16 `who-is` records can be
  swung by one lucky answer, and a single mean would hide that

This is the only part of the benchmark that produces a number you can defend.

### Open-ended → token-F1 against the reference (noisy but broad)

`summary`, `ending`, `character`, `reason`, `chat`.

Reference answers are *extracted* by the builder, so a "wrong" answer may be the
builder's fault rather than the model's. Report it, label it a trend indicator,
and never quote it as a headline.

### Always report: loss and bits-per-byte

Loss alone is a poor guide (§1) but it is the only metric that needs no
references and so cannot be gamed by a bad extraction. Report **bits per byte**
rather than raw loss: it is comparable across tokenizers and vocabularies, which
matters because the 12M and 36M have different vocab sizes (3010 / 5210).

### Ablations that isolate the model from retrieval

These are what turn a score into a diagnosis:

| condition | what it tells you |
|---|---|
| passage supplied (normal) | the shipped configuration |
| no passage | how much of the score is recall, not reading |
| passage shuffled (wrong passage) | does it read at all, or pattern-match the question? |

If "no passage" scores nearly as well as "passage supplied", the model is
reciting pretraining text and the retrieval is decorative. If "shuffled passage"
scores the same as correct passage, the model is ignoring its context entirely —
which is the failure mode a 12M model is most likely to have.

---

## 5. The contamination problem, and what to do about it

Any question written about a WikiText article is answerable from pretraining
text. There is no clean fix with 591 articles. Three options, in increasing cost:

**A. Accept it and state it.** Report scores with the caveat that passages are
drawn from pretraining sources. Cheap, honest, and adequate for comparing two
checkpoints trained on the same data.

**B. Add a small out-of-corpus set.** ~50 questions about subjects *not* in
WikiText-2 or TinyStories (e.g. facts about the project itself, or recent
events). Nothing in pretraining covers them, so a correct answer proves the
model read the supplied passage. This is the only subset that measures the
actual skill.

**C. Hold out articles entirely.** Rebuild with ~20% of articles excluded from
pretraining, then generate Q&A from them. The rigorous answer, but it changes the
corpus and invalidates checkpoint comparisons.

**Recommendation: A now, B next.** B is cheap (a JSON file of 50 items) and
answers the one question that matters — *can this model read?* — which no
contaminated subset can.

---

## 6. Sizing

| subset | items | why this size |
|---|---|---|
| short-answer, in-corpus | ~300 | enough that ±1 item is not a headline |
| open-ended, in-corpus | ~200 | trend only; more adds noise not signal |
| out-of-corpus (option B) | ~50 | hand-written, each one read by a human |
| ablations | ×3 conditions | same items, different passage |

Sampling must be **stratified by kind** and **seeded**, so two checkpoints are
scored on identical items. A benchmark that re-samples per run cannot compare
runs.

---

## 7. Shape of the implementation (not built)

```
scripts/bench_spore.py                 CLI: python scripts/bench_spore.py --model model/spore/...
  ├─ load checkpoint (NumPy engine, no torch — so it runs anywhere)
  ├─ retrieve passage via tinyllm.retrieval           <- the shipped path
  ├─ generate with the same sampling the demo uses
  ├─ score per kind: exact match / token-F1
  └─ write bench_results.json  (gitignored, like bench_throughput.json)

docs/bench/                            the frozen item sets
  ├─ short_answer.jsonl                {kind, question, answer, passage_id}
  └─ out_of_corpus.jsonl               hand-written, no pretraining overlap
```

Two hard requirements:

1. **Deterministic.** Seeded sampling, temperature 0 or a fixed seed. A
   benchmark whose score moves on rerun is worse than no benchmark.
2. **Does not touch the training path.** It reads a checkpoint and scores it.
   No new flags on `train_spore.py`, no new workflow inputs. If the harness needs
   data the checkpoint lacks, that is a signal to change the exporter
   deliberately, not to bolt an option onto training.

---

## 8. What this will and will not tell you

**Will:** whether checkpoint B beats checkpoint A on a fixed set; whether the
model reads its passage or recites pretraining; per-kind weaknesses (e.g. good
at `when`, bad at `reason`); whether a corpus change helped.

**Will not:** whether the model is good in an absolute sense (no baseline
exists); whether it generalises beyond WikiText (it cannot, see §2); whether the
demo "feels" right (that needs a human reading transcripts).

The last one is worth stating plainly: a model can score 0% exact match and
still be a pleasant demo, and can score well and be tiresome. Numbers answer
whether a change helped, not whether the result is good.

---

## 9. Order of work

1. Fix the SFT split — **done**, with a regression test.
2. Freeze the item sets (§7). Hand-verify the 50 out-of-corpus items.
3. Build `scripts/bench_spore.py` against the shipped retrieval path.
4. Score the first checkpoint to finish; record it as the baseline.
5. Add the three ablation conditions.
6. Only then compare 12M vs 36M — comparing before a baseline exists invites
   reading noise as signal.

Step 4 is the reason to build this now rather than after both models finish: the
first checkpoint scores against *nothing*, and that number is only useful if the
harness that produced it is the same one used later.

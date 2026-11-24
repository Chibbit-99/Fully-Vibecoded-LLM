# Spore — question-answering models you can train yourself

Spore is the second family of models in this repository. The first one learned
Tiny Shakespeare and could do nothing but continue a line of Elizabethan
dialogue. Spore is trained to **answer questions and hold a short conversation**,
and it comes in two sizes:

| model | parameters | layers | width | heads | context | vocab | weights |
|---|---|---|---|---|---|---|---|
| `spore-1-12M` | **12,000,000** | 6 | 384 | 8 | 512 | 3010 | ~24 MB |
| `spore-1-36M` | **36,000,000** | 12 | 480 | 8 | 512 | 5210 | ~72 MB |

The number in the name is the exact parameter count, asserted by
`tests/test_spore.py::test_config_names_match_parameter_counts_exactly`. If a
config drifts, the test fails before a six-hour job can run.

---

## 1. Which workflow to run

Everything happens in **Actions → Train Spore (12M / 36M) → Run workflow**.
The workflow has four inputs:

| input | meaning | first run |
|---|---|---|
| `model` | `spore-1-12M` or `spore-1-36M` | `spore-1-12M` |
| `stage` | `pretrain` (prose) or `sft` (chat) | `pretrain` |
| `chunk_minutes` | max training minutes for **this job** | `300` |
| `target_tokens` | total tokens for the stage; blank = default | *(blank)* |
| `reset` | ignore the cached checkpoint and start the stage over | `false` |

**Run them in this order:**

1. `model=spore-1-12M`, `stage=pretrain` → repeat until the summary says
   **Stage FINISHED** (usually one run).
2. `model=spore-1-12M`, `stage=sft` → repeat until **Stage FINISHED**.
3. `model=spore-1-36M`, `stage=pretrain` → repeat until **Stage FINISHED**
   (expect 2–4 runs).
4. `model=spore-1-36M`, `stage=sft` → repeat until **Stage FINISHED**.

You do **not** have to do them all before the demo works: the moment a model's
`sft` stage finishes, its weights are committed to `model/spore/<name>/` and the
browser demo can load it.

### Why "repeat until FINISHED"

A GitHub-hosted runner is killed after 6 hours. A 36M model needs more CPU time
than that, so each invocation trains for at most `chunk_minutes`, saves a
checkpoint, exits **successfully**, and caches the checkpoint. Re-running the
same workflow restores the cache and continues from the exact step.

`chunk_minutes` must stay well under 355 (the job's `timeout-minutes`) to leave
room for install, corpus building and plotting. The default 300 is safe.

---

### Quick reference: the exact runs

Do these in order, in **Actions → Train Spore (12M / 36M) → Run workflow**.
Re-run any row until its job summary says **Stage FINISHED**.

| # | `model` | `stage` | `chunk_minutes` | `target_tokens` | typical runs |
|---|---|---|---|---|---|
| 1 | `spore-1-12M` | `pretrain` | `300` | *(blank → 40M)* | 1 |
| 2 | `spore-1-12M` | `sft` | `300` | *(blank → 12M)* | 1 |
| 3 | `spore-1-36M` | `pretrain` | `300` | *(blank → 40M)* | 2–4 |
| 4 | `spore-1-36M` | `sft` | `300` | *(blank → 12M)* | 1–2 |

After row 2 finishes, `spore-1-12M` is committed to `model/spore/spore-1-12M/`
and the browser demo works. You do not have to finish the 36M model first.

If you want a **fast first pass** to see it working end-to-end, use
`target_tokens=8000000` for row 1: one ~2-hour job, a weak but functional model.

## 2. What "continuation" actually guarantees

The checkpoint saved at the end of a chunk contains everything needed to resume
*bit-identically*: weights, AdamW moments, NumPy and Torch RNG state, the step
counter, the loss history and the best validation loss so far. The learning-rate
schedule is a function of the **absolute** step against the stage's total, never
of the current chunk, so a run split into four jobs follows exactly the same
curve as one uninterrupted run.

This is not a claim, it is a test:

```python
# tests/test_spore.py
def test_chunked_training_is_identical_to_one_run(tiny_setup):
    ...
    for key in a["model"]:
        assert torch.equal(a["model"][key], b["model"][key])
```

Four separate processes train the toy config step by step; the result is
compared tensor-for-tensor with a single six-step run.

If you stop a run by any other means (a killed runner, a cancelled job), the
last checkpoint is at most `--save-interval` steps old (200 by default, about
5 minutes of work). Nothing restarts from zero.

### A chunked run never early-stops the stage

Early stopping exists to abandon a *converged* stage, but for a schedule split
across several jobs it is actively harmful: each job ends long before the
cosine tail, resumes at a different data position, and would see "no
improvement in 4 evals" as a verdict on the whole stage. Because a stage that
stops this way is marked `finished`, no later job could resume it — the stage
would be dead after roughly 40 minutes of a planned 5 hours.

So `--early-stop-patience` is only honoured when the job owns the complete
schedule. When `--chunk-minutes` is set, a stall is logged as
`[patience] ... continuing` and training proceeds to the step cap. The
`patience` counter is also persisted in the checkpoint, so it accumulates
across jobs rather than resetting to zero on every resume.

Both properties are covered by tests that fail against the previous
implementation:

```
tests/test_spore.py::test_chunked_run_never_early_stops_the_stage
tests/test_spore.py::test_patience_survives_a_chunk_boundary
```

### Chunks publish a preview, not nothing

The workflow passes `--export`, so **every** chunk writes `weights.json` — the
file the server and the browser actually load. A chunk that stops for time is
published as a *preview*: `meta.json` carries `"preview": true`, the README
table says `preview (training)`, and the commit subject includes
`(preview, step N/M)`. Once the stage reaches its step target the same paths
are overwritten with the final artefacts and `preview` becomes `false`.

This means the demo is usable for the whole run instead of only at the end.
It also means a preview build is *weaker* than the final one — it is the
checkpoint as of that chunk, not a converged model.

---

## 3. The data

`scripts/build_spore_data.py` builds everything from two committed raw text files
and needs no network:

| file | what it is | size |
|---|---|---|
| `data/raw_wikitext2.txt` | WikiText-2 train split — real Wikipedia articles | 10.8 MB |
| `data/raw_tinystories.txt` | TinyStories — simple children's stories | 5.2 MB |

It produces, in `data/spore/`:

| output | contents |
|---|---|
| `prose_train.txt` / `prose_val.txt` | 85/15 split of the mixed prose, for pretraining |
| `qa.jsonl` | 18k chat records: `{user, assistant, context, context_long, kind, topic}` |
| `factbook.json` | 7k retrieval passages (one per article/story) |
| `qa_stats.json` | counts + a content hash used as the cache key |

Rebuild it at any time (about 2 seconds):

```bash
python scripts/build_spore_data.py
```

### The three lessons baked into the builder

1. **Headings are nested `=` tokens.** WikiText writes a level-2 section as
   `= = Background = =`. A naive regex treats every section as a new article and
   produces nonsense questions like *"Where is There?"*. `parse_heading()`
   counts the markers instead.
2. **The answer must be inside the context.** Earlier drafts quoted a sentence
   from the full article but gave the model a truncated 900-character block; 35%
   of records had answers the model could not see, which teaches hallucination.
   Now every record ships a short (280-char) and a long (1400-char) context, both
   guaranteed to contain the answer verbatim, and the test suite asserts it.
3. **Questions must be grammatical.** Subject filters reject clauses and adverbs
   (`"What did Three seasons later the team win?"`), and verb forms are mapped
   explicitly (`"When did X began?"` → `"begin"`). Wording is template-based and
   extractive, so no fact is invented at build time.

### Retrieval is what makes a 12M model answer correctly

A 12M model cannot store the world's facts. What it *can* learn is how to read a
passage and answer from it. So the server and the browser both look up a passage
in the factbook (BM25, implemented twice: `tinyllm/retrieval.py` and
`web/retrieval.js`, verified to rank identically) and prepend it:

```
<|user|>Paris is the capital of France.<|context|>What is the capital of France?<|assistant|>Paris.<|end|>
```

The model was trained on exactly that shape, using both context lengths, so it
works whether it is handed a 280-character passage or a 1400-character one.

---

## 4. The two stages

### Stage 1 — pretraining

Next-token prediction on the mixed prose. Learning rate 6e-4 with cosine decay,
batch size chosen from the model size (8 for 12M, 2 for 36M), context 512.

### Stage 2 — chat tuning (SFT)

The same model continues from its pretrained weights but with a **masked loss**:
the label is `-100` everywhere except the assistant's reply, so the model is
never trained to predict the question or the context. Two details matter:

* **Only windows with at least one assistant token are sampled.** With long
  contexts, two thirds of all windows contain no target at all; cross-entropy
  over an empty target set is `NaN`, and one `NaN` in an AdamW step destroys
  every weight. `valid_window_starts()` computes the usable windows once.
  Before this, the SFT loss was `NaN` on most steps and the run would have
  looked healthy while learning nothing.
* **Chat tuning uses a lower peak LR** (3e-4) than pretraining, and its own step
  counter. The trainer records the stage in every checkpoint and starts stage 2
  at step 0 while keeping stage 1's weights — otherwise SFT would inherit the
  finished step count and skip its entire schedule.

---

## 5. Time and memory, measured

Measured on this 2-core sandbox (GitHub's `ubuntu-latest` runner is comparable):

| model | stage | tok/s | 40M tokens | per job (300 min) |
|---|---|---|---|---|
| 12M | pretrain | ~1,235 | ~9 h | ~22M tokens |
| 36M | pretrain | ~580 | ~19 h | ~10M tokens |

Both exceed six hours at the default budget, which is why continuation exists.
Peak memory is about 16 bytes per parameter (weights + AdamW moments): 200 MB for
12M, 600 MB for 36M — comfortable inside 8 GB.

A realistic target is 20–40M tokens for 12M and 15–25M for 36M. More tokens is
better up to roughly 20× parameters; beyond that the model is under-trained on
compute, not data.

---

## 6. Using a trained model

### Python server

```bash
pip install -r requirements.txt
python -m tinyllm.server --checkpoint spore-1-12M --port 8000
```

`--checkpoint` accepts a model name (resolved under `model/spore/`) or a path.
Then:

```bash
curl -s localhost:8000/api/generate -H 'content-type: application/json' \
  -d '{"prompt": "What is water?", "max_tokens": 60}'
```

The response includes `completion`, the `grounding` passage that was retrieved
and `retrieved` (title + score), so you can see *why* it said what it said.

```bash
curl -s localhost:8000/api/models      # which checkpoints exist
curl -s localhost:8000/api/health      # params, chat mode, retrieval status
```

### Browser demo

```bash
python -m tinyllm.server --checkpoint spore-1-12M --port 8000
# open http://localhost:8000/         -> Spore chat

```

Or open the site through any static server after the model is published
(GitHub Pages does this automatically on push to `main`). The whole model runs in
a Web Worker: the 12M checkpoint loads in about 200 ms and generates at roughly
35 tokens/second on a laptop.

### CLI

```bash
python -m tinyllm chat --checkpoint model/spore/spore-1-12M/weights.json
```

---

## 7. Honest expectations

A 12M model trained for half a day on 2 CPU cores is **not** a chatbot you would
compare to anything modern. What it can do:

* answer "What is X?" for X in its factbook, by reading the retrieved passage;
* reply to greetings, refuse questions outside its data ("I do not know that"),
  count small numbers, and spell simple words — those behaviours are in the
  hand-written ability set;
* stop talking when it finishes a turn, because it was trained to emit `<|end|>`.

What it cannot do: reason, do multi-step arithmetic beyond the small set, know
anything that was not in 16 MB of committed text, or stay on topic without
retrieval. The 36M model is better at grammar and recall, not at reasoning.

The training loss will end somewhere around 3.0–4.0 (cross-entropy per token on a
3–5k vocabulary); the chat loss is not comparable to the prose loss because the
masked tokens are the only ones counted.

---

## 8. Files

| path | role |
|---|---|
| `configs/spore_12m.json`, `configs/spore_36m.json` | the two architectures |
| `scripts/build_spore_data.py` | builds prose, Q&A and factbook from the raw text |
| `scripts/train_spore.py` | two-stage, resumable trainer |
| `tinyllm/sft.py` | chat templates, SFT masking, window filtering |
| `tinyllm/retrieval.py` | BM25 factbook + `context_for()` |
| `.github/workflows/train_spore.yml` | the training workflow described above |
| `web/index.html`, `web/spore.js`, `web/retrieval.js` | the playground |
| `tests/test_spore.py` | 25 tests: tokenizer, corpus quality, sizing, resuming |

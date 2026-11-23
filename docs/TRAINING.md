# Training guide

Everything about how the model is trained, tuned, and published.

## 1. The pipeline at a glance

```
data/tinyshakespeare.txt
        │
        ├─ ByteBPE.train()  ──────────────▶ model/tokenizer.json   (deterministic, committed once)
        │
        ├─ encode corpus ─── 520k train tokens / 59k val tokens (90/10 contiguous split)
        │
        ├─ GPT(4 layers, 4 heads, d=256, ctx=192) ──▶ runs/main/model.pt
        │
        ├─ estimate val loss ─▶ train_log.json, loss_curve.svg, samples.txt
        │
        └─ export_checkpoint() ─▶ model/weights.json  (fp16, base64, ~9.4 MB)
                                        │
                                        ├─▶ tinyllm/server.py  (NumPy)
                                        └─▶ web/engine.js      (browser)
```

## 2. Running the GitHub Actions workflow

```bash
gh workflow run train.yml --ref main -f max_steps=2400 -f seed=1337
```

or **Actions → Train TinyLLM → Run workflow**.

| Input | Default | Meaning |
|---|---|---|
| `max_steps` | 3000 | Optimiser steps for this run |
| `batch_size` | 16 | Micro-batch |
| `grad_accum` | 2 | Effective batch = 32 |
| `seed` | 1337 | Full reproducibility of data order + init |
| `time_budget_minutes` | 120 | Hard stop, checkpoint is saved anyway |
| `clear_checkpoints` | false | Start from scratch instead of resuming |

The job caches `runs/main/last.pt` under a key derived from the branch, step
count and seed. Re-running the workflow **resumes** rather than restarts, so a
long training run can be split across several jobs — useful because the
GitHub-hosted runner is killed after 6 hours.

What it produces on the branch:

```
model/weights.json     ← the thing both engines load
model/config.json      ← hyper-parameters
model/tokenizer.json   ← vocabulary
model/meta.json        ← metrics + loss history
model/loss_curve.svg   ← chart for the README
model/samples.txt      ← generations from the final weights
```

Every commit from CI is tagged `[skip ci]` so it does not retrigger itself.

## 3. Running locally

```bash
python scripts/train.py --out-dir runs/main --config configs/small.json --max-steps 2400
```

Useful flags:

* `--time-budget 60` — stop after an hour and still export.
* `--resume runs/main/last.pt` — continue with optimizer + RNG state.
* `--tokenizer model/tokenizer.json` — reuse a vocabulary instead of retraining.

At ~2 800 tokens/s on two CPU cores, 2 400 steps × 32 × 192 tokens ≈ 14.7M
tokens ≈ 90 minutes. On a GitHub runner expect roughly the same or a little
faster; on a beefier box it is CPU-bound on the matrix multiplies, so more
cores (up to `OMP_NUM_THREADS`) helps.

## 4. Choosing the model

| Config | Layers | d_model | Heads | Ctx | Params | Notes |
|---|---|---|---|---|---|---|
| `configs/small.json` (shipped) | 4 | 256 | 4 | 192 | 3.5M | trains in ~1.5 h on CPU, 9.4 MB in the browser |
| `configs/tiny_shakespeare.json` | 6 | 288 | 6 | 256 | 6.4M | ~2.5× slower to train, 17 MB browser payload |

Scaling rules of thumb learned while building this:

1. **Context length is the biggest lever on Shakespeare.** At ctx 64 the model
   cannot keep a speaker's name more than a line; at 192 it can hold a whole
   speech. It costs linear memory and ~linear compute per token.
2. **Width beats depth at this size** on CPU: a 4×256 model trains faster and
   reaches lower loss than 8×176 for the same parameter count.
3. **Batch size 32–48 with lr 9e-4** was stable; below batch 16 the loss curve
   gets noisy.
4. **Gradient accumulation** is supported but on CPU it only saves memory, not
   time — prefer raising `--batch-size` first.

## 5. Reading the metrics

* **`val_loss` (nats/token)** — the training objective. At the shipped size,
  3.5 → 1.6 over 2 400 steps.
* **`bits_per_byte`** — `val_loss / ln(2) / compression_ratio`. This is the only
  number comparable across tokenizers and the one the GPT-2/3 papers quote.
  8.0 = uniform random bytes; 1.5–1.8 = decent small Shakespeare model.
* **`tokens_seen`** — total training tokens; watch this against parameter count.
  A useful heuristic is ≥20× parameters, which above easily clears.

## 6. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `config vocab_size != tokenizer vocab` | mismatched tokenizer/checkpoint | delete `runs/*/tokenizer.json` and let it retrain, or pass the right one |
| loss spikes above 6 after warmup | LR too high for batch size | lower `--lr` or raise `--batch-size` |
| `sequence longer than block_size` | older bug: batching used the CLI default | fixed — `args.block_size` follows the config |
| Actions job "not found" in `gh workflow list` | Actions disabled on the account | see §7 |
| 403 `Resource not accessible by integration` | token lacks `workflow` scope | see §7 |

## 7. If GitHub Actions is disabled for the repository

Symptom: `gh api repos/<owner>/<repo>/actions/workflows` returns
`"total_count": 0` even though `.github/workflows/*.yml` is committed, and
`gh workflow run` returns `404`.

That is an account-level setting, not a problem with the YAML (run
`python scripts/validate_workflows.py` to prove the files are fine). Fix it in
the browser:

1. **Settings → Actions → General** on the repository, choose *Allow all
   actions and reusable workflows*.
2. If the Actions tab shows a banner about the account, open
   **github.com/settings/billing** and confirm Actions are permitted for the
   account (a verified e-mail may also be required).
3. Organisation-owned repos: an owner must enable Actions and allow the
   workflow. If workflows were disabled for the repo long enough, GitHub may
   require a push after re-enabling — a no-op commit does it.
4. If you use the GitHub CLI in a terminal, `gh auth refresh -s workflow` adds
   the missing scope so `gh workflow run` stops returning 403.

Once Actions is on, the first push registers all three workflows
(`Train TinyLLM`, `Tests`, `Deploy demo to GitHub Pages`) and you can run them
from the UI or with `gh workflow run train.yml`.

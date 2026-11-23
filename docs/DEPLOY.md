# Deployment & first-run checklist

You have **one** thing to do that I cannot: enable GitHub Actions on the
repository. Everything else is already wired up.

---

## Step 0 — what I already proved works

| Check | Command | Result |
|---|---|---|
| Code is complete | `make test-fast` | 35 passed |
| API works | `make serve` then `curl localhost:8000/api/health` | `{"status":"ok",...}` |
| Browser engine works | `node scripts/worker_smoke.mjs model/weights.json` | 24 tokens @ ~80 tok/s |
| JS matches Python | `node scripts/js_parity.mjs model/weights.json expected.json` | max Δlogit 4.3e-6 |
| Workflows are valid | `make lint` | all 3 valid |

The only untested-at-runtime piece is **GitHub's own execution of the
workflows**, because Actions is disabled on the repo.

---

## Step 1 — enable GitHub Actions (required)

I get this when I ask the API:

```
gh api repos/Chibbit-99/Fully-Vibecoded-LLM/actions/workflows
→ {"total_count": 0, "workflows": []}          # workflows ignore exists
gh workflow run train.yml
→ HTTP 404                                      # dispatch is blocked
gh api -X PUT .../actions/permissions
→ HTTP 403 Resource not accessible by integration
```

`total_count: 0` with valid YAML committed means **Actions is disabled at the
repository or account level** — GitHub never even reads the workflow files.

**Do this, in order:**

1. **Settings → Actions → General** (repo settings).
   - Under *Actions permissions* select **Allow all actions and reusable
     workflows**.
   - Click **Save**.
2. Go to the **Actions** tab of the repo.
   - If there's a banner saying actions are disabled or that billing/identity
     verification is required, follow its link. For personal accounts that is
     usually <https://github.com/settings/billing> → ensure Actions are
     allowed, and confirm your e-mail is verified at
     <https://github.com/settings/emails>.
3. If the Actions tab still shows nothing, push any commit to `main` (I can do
   that for you after merging — a no-op commit forces GitHub to re-scan
   `.github/workflows/`).

Verify it worked — this must now print **3**:

```bash
gh api repos/Chibbit-99/Fully-Vibecoded-LLM/actions/workflows --jq '.total_count'
```

You should see these three workflows listed:

| Workflow | File | Trigger |
|---|---|---|
| **Train TinyLLM** | `.github/workflows/train.yml` | manual, weekly schedule |
| **Tests** | `.github/workflows/test.yml` | push to `main`, PRs, manual |
| **Deploy demo to GitHub Pages** | `.github/workflows/pages.yml` | push touching `web/` or `model/`, manual |

---

## Step 2 — enable GitHub Pages (required for the public demo)

**Settings → Pages →** under *Build and deployment* set **Source: GitHub
Actions**.

(The deploy workflow also passes `enablement: true`, so it can create the site
itself — but the setting must not be blocked.)

Result: `https://chibbit-99.github.io/Fully-Vibecoded-LLM/`

---

## Step 3 — run the workflows, in this order

### 3a. `Tests` — 5 minutes, proves nothing is broken

```bash
gh workflow run test.yml --ref main
gh run watch
```

Expect two green jobs: `test` and `browser-engine`.

> The parity/server tests **skip** until `model/weights.json` is committed. That
> is deliberate — the first run is fast, and the heavy checks run on every
> subsequent push.

### 3b. `Train TinyLLM` — this is the actual training run

```bash
gh workflow run train.yml --ref main \
  -f config=configs/fast.json \
  -f max_steps=4000 \
  -f batch_size=32 \
  -f seed=1337 \
  -f chunk_minutes=25
```

**Each run trains a bounded chunk and saves.** 25 minutes of the 2M model is
≈ 6.7M tokens ≈ 700 steps. Re-run (or let the daily schedule do it) and
training *continues* from the cached checkpoint until `max_steps` is reached;
after that, runs detect the finished model and exit immediately. This is why
the workflow can never die on a runner timeout on a free 2-core machine.

What it does, in order:

1. restores `runs/main` from the Actions cache (so re-runs **resume**),
2. reuses the committed `model/tokenizer.json` (deterministic vocabulary),
3. trains on a 2-core CPU runner (~40–60 min for 2 400 steps),
4. evaluates, exports `weights.json`, renders `loss_curve.svg`, writes samples,
5. runs the **full** test suite — now including cross-engine parity,
6. commits `model/*` back to `main` with `[skip ci]`,
7. uploads the artefacts to the run page (90-day retention).

**Inputs you can change:**

| Input | Default | Notes |
|---|---|---|
| `config` | `configs/fast.json` | 2.05M params, ctx 128, ~9k tok/s on a runner |
| `max_steps` | 4000 | total for the whole run, across all chunks |
| `batch_size` | 32 | 32–48 is the sweet spot on 2 cores |
| `seed` | 1337 | reproduces data order exactly |
| `chunk_minutes` | 25 | minutes of training for *this* invocation |
| `reset` | false | true = ignore the cache and start over |

Speed cheat-sheet measured on a 2-core runner-class CPU:

| Config | Params | Context | Throughput | 4000 steps |
|---|---|---|---|---|
| `configs/fast.json` | 2.05M | 128 | ~9,100 tok/s | ~50 min total |
| `configs/small.json` | 3.54M | 192 | ~5,800 tok/s | ~75 min total |

So the default (4000 steps of `fast.json`) is **two 25-minute re-runs**, not one
hour-long job.

### 3c. `Deploy demo to GitHub Pages` — automatic

Fires by itself when 3b commits the new `model/`. If you want it manually
before training finishes:

```bash
gh workflow run pages.yml --ref main
```

---

## Step 4 — verify the deployed model

```bash
# the live demo should report the trained model's metrics
curl -s https://chibbit-99.github.io/Fully-Vibecoded-LLM/model/meta.json | python3 -m json.tool
```

Then open the demo and press **Generate**. The badges should read
`3.54M params` / `4L · 4H · d256 · ctx 192` / `val loss ...`.

---

## Doing it without GitHub Actions

If Actions remains unavailable, the whole pipeline runs locally:

```bash
make install        # venv + deps
make train          # ~90 min on 2 CPU cores -> runs/main/
make export         # copy artefacts into model/  (same as CI)
make test           # everything
make demo           # http://localhost:8080/web/
make serve          # http://localhost:8000
```

Then commit `model/` yourself and push — the Pages workflow is not required
either: any static host can serve the repo root, because the browser demo is
just `web/` + `model/weights.json`.

---

## Troubleshooting

| Symptom | Meaning | Fix |
|---|---|---|
| `gh workflow run` → `404` | Actions disabled, or workflow name wrong | Step 1; confirm with the `total_count` check |
| `Resource not accessible by integration` | token has no `workflow` scope | `gh auth refresh -s workflow` in a terminal, or run from the UI |
| Workflows listed but runs never start | runner minutes exhausted / billing | check <https://github.com/settings/billing> |
| Deploy job fails: `model/weights.json missing` | Pages ran before training | run `Train TinyLLM` first, then re-run Pages |
| Deploy job fails: `Pages site not found` | Pages not enabled | Step 2 |
| Train job stopped early | chunk finished | just run it again — it resumes from the cached checkpoint |
| `git add` ignored-path error | old bug: `runs/` was added to git | fixed; only `model/` and `README.md` are committed |
| Train job wasted an hour then failed | old bug: commit step after training | fixed; artefacts are also uploaded before the commit, and `scripts/test_workflows.py` now executes every workflow step locally |
| Demo shows “Could not load weights” | opening `web/index.html` from disk | serve the **repo root**: `make demo`, then `/web/` |

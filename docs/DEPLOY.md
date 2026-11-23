# Deployment & first-run checklist

> **Update after the merge to `main`:** GitHub Actions turned out to be
> **already enabled**. My earlier `total_count: 0` reading was misleading:
> `push`-triggered workflows only register once their YAML exists on the
> **default branch**, and until then the REST API reports no workflows at all.
> Merging PR #1 fixed it — all three workflows are now `active` and have run:
>
> | Workflow | Result on first push to `main` |
> |---|---|
> | **Tests** | ✅ success (44 unit + parity + API tests) |
> | **Train TinyLLM** | ✅ started automatically, training on a runner |
> | **Deploy demo to GitHub Pages** | ❌ `Resource not accessible by integration` — **needs one manual step, below** |

So there is now exactly **one** manual action left: enabling GitHub Pages.

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

## Step 1 — GitHub Actions: already working ✅

Confirmed by the API after merging:

```
gh api repos/Chibbit-99/Fully-Vibecoded-LLM/actions/workflows --jq '.total_count'
→ 3
```

All three workflows are `active` and ran on the merge commit. Nothing to do
here, unless a future run reports *"Actions has been disabled"* — in that case
**Settings → Actions → General → Allow all actions** → **Save**.

Verify at any time:

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

## Step 2 — enable GitHub Pages ← **the one thing only you can do**

The deploy run failed with:

```
Create Pages site failed.
HttpError: Resource not accessible by integration
  https://docs.github.com/rest/pages/pages#create-a-apiname-pages-site
```

That is GitHub refusing to let the workflow's `GITHUB_TOKEN` create the Pages
site — a deliberate guard. I tried both the API and `configure-pages`'s
`enablement: true`; both are blocked for tokens minted by the GitHub App
integration. **A repository owner must flip the switch once:**

1. Open **Settings → Pages** on <https://github.com/Chibbit-99/Fully-Vibecoded-LLM/settings/pages>.
2. Under **Build and deployment → Source**, choose **GitHub Actions**.
3. Save. No workflow file change is needed — just re-run the deploy:

```bash
gh run rerun 45089447109          # or: gh workflow run pages.yml --ref main
```

Result: <https://chibbit-99.github.io/Fully-Vibecoded-LLM/>

If you would rather not use Pages at all, the demo is pure static files —
hosting `web/` + `model/weights.json` on Netlify, Cloudflare Pages or S3 works
identically (both path layouts are supported).

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
# from the Actions UI: Train TinyLLM → Run workflow, or:
gh workflow run train.yml --ref main \
  -f max_steps=2400 \
  -f batch_size=32 \
  -f grad_accum=1 \
  -f seed=1337 \
  -f time_budget_minutes=120
```

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
| `max_steps` | 3000 | raise to 6000–10000 for better text; the cache resumes |
| `batch_size` | 16 | 32 is faster per token; watch runner memory |
| `grad_accum` | 2 | effective batch = batch × accum |
| `seed` | 1337 | reproduces data order exactly |
| `time_budget_minutes` | 120 | hard stop; still saves + exports |
| `clear_checkpoints` | false | true = ignore the cache and start over |

To train longer without losing work, just run it again with a bigger
`max_steps` — it resumes from the cached checkpoint.

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
| Train job killed at 6 h | runner hard limit | lower `time_budget_minutes`, re-run — it resumes from cache |
| Demo shows “Could not load weights” | opening `web/index.html` from disk | serve the **repo root**: `make demo`, then `/web/` |

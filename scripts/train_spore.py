#!/usr/bin/env python3
"""Train a Spore chat model in two stages, on CPU, resumably.

Stage 1 -- pretraining
    Next-token prediction on the mixed prose corpus (WikiText-2 articles +
    TinyStories).  This is where English, facts and long-range structure come
    from.

Stage 2 -- chat tuning (SFT)
    The same model, trained on rendered dialogues with a masked loss: only the
    assistant's tokens count.  This is what turns a document completer into
    something that answers a question and then *stops*.

Both stages support **continuation**.  A GitHub-hosted runner gives ~6 hours,
so a full run is expected to span several jobs:

    job 1:  python scripts/train_spore.py --config configs/spore_12m.json \
                --stage pretrain --chunk-minutes 300
    job 2:  (same command)  -> resumes at the saved step, same loss curve
    job 3:  python scripts/train_spore.py --stage sft ...
    job 4:  python scripts/train_spore.py --stage sft ...   -> finishes

`--chunk-minutes` stops *cleanly*: it saves `last.pt` (weights + optimizer +
RNG state + step counter) and exits 0, so the workflow commits what it has and
the next job picks up exactly where this one stopped.  Nothing is lost and the
learning-rate schedule is identical to one uninterrupted run.

Artefacts (``--out-dir``)::

    tokenizer.json     shared by both Spore models (vocab 3010 / 5210)
    config.json        GPTConfig
    last.pt            resumable checkpoint (weights+optimizer+RNG+step)
    best.pt            lowest-validation-loss checkpoint
    model.pt           final weights of the stage
    weights.json       float16 export for the NumPy server / browser engine
    train_log.json     history + meta (also drives the README / summary)
    loss_curve.svg     dependency-free chart
    samples.txt        chat transcripts at the end of training

Examples
--------
    # 2 minutes of pretraining, then 1 of SFT -- a smoke test, not a model
    python scripts/train_spore.py --config configs/spore_12m.json \
        --stage pretrain --max-steps 20 --chunk-minutes 2 --smoke --out-dir runs/smoke
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from tinyllm.config import GPTConfig  # noqa: E402
from tinyllm.dataset import get_batch  # noqa: E402
from tinyllm.model import GPT  # noqa: E402
from tinyllm.sft import (  # noqa: E402
    ASSISTANT,
    END,
    IGNORE_INDEX,
    SPECIAL_TOKENS,
    build_sft_data,
    load_qa,
    render_prompt,
    valid_window_starts,
)
from tinyllm.tokenizer import ByteBPE  # noqa: E402

DATA = REPO_ROOT / "data" / "spore"
DEFAULT_CHAT_PROMPTS = [
    "Hello!",
    "What is your name?",
    "What is water?",
    "How do you spell cat?",
    "What is 3 plus 4?",
    "Summarise this story.",
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train Spore (pretrain -> SFT) with resumable chunks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    g = p.add_argument_group("model & data")
    g.add_argument("--config", type=Path, required=True, help="configs/spore_12m.json etc.")
    g.add_argument("--out-dir", type=Path, required=True)
    g.add_argument("--data-dir", type=Path, default=DATA)
    g.add_argument("--tokenizer", type=Path, default=None, help="reuse a tokenizer.json")
    g.add_argument("--stage", choices=["pretrain", "sft"], default="pretrain")

    t = p.add_argument_group("training")
    t.add_argument("--max-steps", type=int, default=0, help="0 = derive from --token-budget")
    t.add_argument("--token-budget", type=float, default=0.0,
                   help="total tokens for this stage (e.g. 20e6); sets max-steps")
    t.add_argument("--chunk-minutes", type=float, default=0.0,
                   help="stop cleanly after N minutes and save a resumable checkpoint")
    t.add_argument("--resume", type=Path, default=None, help="checkpoint (default: out-dir/last.pt)")
    t.add_argument("--batch-size", type=int, default=0, help="0 = pick from the model size")
    t.add_argument("--grad-accum", type=int, default=1)
    t.add_argument("--block-size", type=int, default=0, help="0 = from config")
    t.add_argument("--lr", type=float, default=0.0, help="0 = per-stage default")
    t.add_argument("--min-lr", type=float, default=0.0)
    t.add_argument("--warmup", type=int, default=200)
    t.add_argument("--weight-decay", type=float, default=0.1)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--eval-interval", type=int, default=0, help="0 = auto")
    t.add_argument("--eval-batches", type=int, default=8)
    t.add_argument("--log-interval", type=int, default=25)
    t.add_argument("--save-interval", type=int, default=200)
    t.add_argument("--early-stop-patience", type=int, default=4,
                   help="stop when val loss stops improving (0 = never); keeps best.pt")
    t.add_argument("--threads", type=int, default=2)
    t.add_argument("--seed", type=int, default=1337)
    t.add_argument("--sft-repeat", type=int, default=2, help="epochs over the Q&A corpus")

    o = p.add_argument_group("output")
    o.add_argument("--publish", action="store_true", help="copy artefacts to model/spore/<name>/")
    o.add_argument("--export", action="store_true", help="write weights.json even if unfinished")
    o.add_argument("--smoke", action="store_true", help="tiny run for tests")
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def lr_at(step: int, warmup: int, total: int, base: float, floor: float) -> float:
    """Linear warmup + cosine decay, a function of the *absolute* step.

    Deriving the schedule from the run's total (never from the current
    invocation's chunk bound) is what makes continuation exact: a run split
    into five jobs follows the same curve as one uninterrupted run.
    """
    if step < warmup:
        return base * (step + 1) / max(1, warmup)
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return floor + 0.5 * (1.0 + math.cos(math.pi * progress)) * (base - floor)


def save_checkpoint(path: Path, model, optimizer, config, step, best_val, history, rng,
                    extra=None, stage: str = "pretrain") -> None:
    """Atomic, fully resumable: weights, optimizer moments, RNG, step, history."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.to_dict(),
        "step": step,
        "best_val": best_val,
        "history": history,
        "rng_numpy": rng.bit_generator.state,
        "rng_torch": torch.get_rng_state(),
        "stage": stage,
    }
    if extra:
        payload.update(extra)
    torch.save(payload, tmp)
    tmp.replace(path)


def configure_optimizer(model: GPT, weight_decay: float, lr: float) -> torch.optim.Optimizer:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "ln_" in name or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay},
         {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95), eps=1e-8,
    )


def auto_batch_size(params: int, block_size: int) -> int:
    """Micro-batch that fits a 2-core / 8 GB runner without OOM."""
    if params >= 30_000_000:
        return 2
    if params >= 20_000_000:
        return 4
    if params >= 12_000_000:
        return 8
    return 16


def auto_eval_interval(max_steps: int) -> int:
    return max(25, min(250, max_steps // 12 or 25))


# ---------------------------------------------------------------------------
# stage 1: pretraining
# ---------------------------------------------------------------------------
def load_prose(data_dir: Path, tokenizer: ByteBPE) -> Tuple[np.ndarray, np.ndarray]:
    train_path = data_dir / "prose_train.txt"
    val_path = data_dir / "prose_val.txt"
    if not train_path.exists():
        raise SystemExit(
            f"missing {train_path}; run `python scripts/build_spore_data.py` first"
        )
    train_ids = np.array(tokenizer.encode(train_path.read_text(encoding="utf-8")), dtype=np.uint16)
    val_ids = np.array(tokenizer.encode(val_path.read_text(encoding="utf-8")), dtype=np.uint16)
    return train_ids, val_ids


@torch.no_grad()
def eval_prose(model, data: np.ndarray, args, rng: np.random.Generator, n: Optional[int] = None) -> float:
    model.eval()
    total, count = 0.0, 0
    for _ in range(n or args.eval_batches):
        x, y = get_batch(data, args.batch_size, args.block_size, rng)
        _, loss, _ = model(x, y)
        total += float(loss.item())
        count += 1
    model.train()
    return total / max(1, count)


def pretrain_step(model, optimizer, train_ids, args, rng, lr) -> float:
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for _ in range(args.grad_accum):
        x, y = get_batch(train_ids, args.batch_size, args.block_size, rng)
        _, loss, _ = model(x, y)
        (loss / args.grad_accum).backward()
        total += float(loss.item()) / args.grad_accum
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    return total


# ---------------------------------------------------------------------------
# stage 2: chat tuning
# ---------------------------------------------------------------------------
def load_chat(data_dir: Path, tokenizer: ByteBPE, args) -> Tuple[Any, np.ndarray]:
    qa_path = data_dir / "qa.jsonl"
    if not qa_path.exists():
        raise SystemExit(
            f"missing {qa_path}; run `python scripts/build_spore_data.py` first"
        )
    records = load_qa(qa_path)
    train = build_sft_data(tokenizer, records, repeat=args.sft_repeat, shuffle_seed=args.seed)
    # held-out slice for validation: every 10th record, no repeat
    held = records[::10]
    val = build_sft_data(tokenizer, held, repeat=1, shuffle_seed=args.seed + 1)
    return train, val


def ensure_windows(data, block_size: int) -> None:
    """Precompute usable window starts once (see valid_window_starts)."""
    if not data.valid_windows:
        data.valid_windows = valid_window_starts(data.labels, block_size)
        if not data.valid_windows:
            raise SystemExit(
                "no SFT window contains an assistant token -- check the corpus "
                "and block_size"
            )
        print(f"[data] {len(data.valid_windows):,} usable SFT windows of {block_size} tokens")


def sft_batch(data, batch_size: int, block_size: int, rng: random.Random):
    """Return (x, y) with y already masked to the assistant's tokens."""
    starts = data.valid_windows
    ix = [starts[rng.randrange(len(starts))] for _ in range(batch_size)]
    xs = [data.window(s, block_size)[0] for s in ix]
    ys = [data.window(s, block_size)[1] for s in ix]
    return torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))


def sft_step(model, optimizer, train, args, rng, lr) -> float:
    for group in optimizer.param_groups:
        group["lr"] = lr
    optimizer.zero_grad(set_to_none=True)
    total, seen = 0.0, 0
    for _ in range(args.grad_accum):
        x, y = sft_batch(train, args.batch_size, args.block_size, rng)
        _, loss, _ = model(x, y)
        (loss / args.grad_accum).backward()
        total += float(loss.item()) / args.grad_accum
        seen += int((y != IGNORE_INDEX).sum())
    if args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
    optimizer.step()
    return total


@torch.no_grad()
def eval_sft(model, val, args, rng: random.Random, n: Optional[int] = None) -> float:
    model.eval()
    total, count = 0.0, 0
    for _ in range(n or args.eval_batches):
        x, y = sft_batch(val, args.batch_size, args.block_size, rng)
        _, loss, _ = model(x, y)
        if loss is not None and not torch.isnan(loss):
            total += float(loss.item())
            count += 1
    model.train()
    return total / max(1, count)


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------
@torch.no_grad()
def chat_samples(model, tokenizer: ByteBPE, args, out_path: Path) -> List[Dict[str, str]]:
    """Ask the model a few questions and record the transcripts."""
    stop = {tokenizer.encode_special(END), tokenizer.encode_special(ASSISTANT)}
    model.eval()
    out: List[Dict[str, str]] = []
    for i, q in enumerate(DEFAULT_CHAT_PROMPTS):
        ids = tokenizer.encode(render_prompt(q))
        prompt_ids = ids[-model.config.block_size :]
        x = torch.tensor([prompt_ids], dtype=torch.long)
        gen = model.generate(
            x, max_new_tokens=60, temperature=0.7, top_k=40, top_p=0.9,
            seed=args.seed + i, stop_tokens=stop,
        )
        # decode only the *new* tokens (the generate() output includes the prompt)
        text = tokenizer.decode(gen[0].tolist()[len(prompt_ids) :])
        # cut at the turn marker for readability
        for marker in (END, ASSISTANT):
            if marker in text:
                text = text.split(marker)[0]
        out.append({"prompt": q, "completion": text.strip()})
    with open(out_path, "w", encoding="utf-8") as f:
        for s in out:
            f.write("=" * 70 + f"\nPROMPT: {s['prompt']!r}\n" + "-" * 70 + "\n")
            f.write(s["completion"] + "\n\n")
    model.train()
    return out


def svg_loss_curve(history: List[Dict[str, Any]], path: Path, width: int = 720, height: int = 340) -> None:
    train_pts = [(e["step"], e["loss"]) for e in history if "loss" in e]
    val_pts = [(e["step"], e["val_loss"]) for e in history if "val_loss" in e]
    if not train_pts and not val_pts:
        return
    all_pts = train_pts + val_pts
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    xmin, xmax = min(xs), max(xs) or 1
    ymin, ymax = min(ys) * 0.95, max(ys) * 1.02
    pad = 45

    def px(x):
        return pad + (x - xmin) / max(1e-9, xmax - xmin) * (width - 2 * pad)

    def py(y):
        return height - pad - (y - ymin) / max(1e-9, ymax - ymin) * (height - 2 * pad)

    def poly(pts):
        return " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#30363d"/>',
    ]
    if train_pts:
        parts.append(f'<polyline fill="none" stroke="#39d353" stroke-width="2" points="{poly(train_pts)}"/>')
    if val_pts:
        parts.append(f'<polyline fill="none" stroke="#58a6ff" stroke-width="2" points="{poly(val_pts)}"/>')
    parts.append(f'<text x="{pad}" y="24" fill="#8b949e" font-family="monospace" font-size="12">'
                 f'train (green) / val (blue) - {ymax:.2f} .. {ymin:.2f}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        args.max_steps = args.max_steps or 20
        args.sft_repeat = 1
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir: Path = args.data_dir

    torch.set_num_threads(args.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already initialised
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng_np = np.random.default_rng(args.seed)
    rng_py = random.Random(args.seed)

    # ---- config + tokenizer ---------------------------------------------
    config = GPTConfig.load(args.config)
    tok_path = args.tokenizer or (out_dir / "tokenizer.json")
    if tok_path.exists():
        tokenizer = ByteBPE.load(tok_path)
        print(f"[tokenizer] loaded {tok_path} (vocab={tokenizer.vocab_size})")
    else:
        text = (data_dir / "prose_train.txt").read_text(encoding="utf-8")
        print("[tokenizer] training byte-level BPE ...", flush=True)
        t0 = time.time()
        tokenizer = ByteBPE.train(text, vocab_size=config.vocab_size - len(SPECIAL_TOKENS))
        tokenizer.special_tokens = list(SPECIAL_TOKENS)
        tokenizer.special_ids = {
            t: 256 + len(tokenizer.merges) + i for i, t in enumerate(SPECIAL_TOKENS)
        }
        tokenizer.vocab_size = 256 + len(tokenizer.merges) + len(SPECIAL_TOKENS)
        tokenizer.save(out_dir / "tokenizer.json")
        print(f"[tokenizer] done in {time.time()-t0:.1f}s vocab={tokenizer.vocab_size}")

    if tokenizer.vocab_size != config.vocab_size:
        raise SystemExit(
            f"config vocab_size={config.vocab_size} != tokenizer vocab={tokenizer.vocab_size}. "
            f"Re-train the tokenizer or fix {args.config}."
        )
    config.save(out_dir / "config.json")
    args.block_size = args.block_size or config.block_size

    # ---- data ------------------------------------------------------------
    if args.stage == "pretrain":
        train_ids, val_ids = load_prose(data_dir, tokenizer)
        print(f"[data] prose train={len(train_ids):,} val={len(val_ids):,} tokens")
        corpus_tokens = len(train_ids)
        train_data = val_data = None
    else:
        train_data, val_data = load_chat(data_dir, tokenizer, args)
        ensure_windows(train_data, args.block_size)
        ensure_windows(val_data, args.block_size)
        print(f"[data] sft {train_data.examples:,} examples "
              f"({train_data.supervised_tokens:,} supervised tokens), "
              f"val {val_data.examples:,} examples")
        corpus_tokens = train_data.supervised_tokens
        train_ids = val_ids = None

    # ---- model -----------------------------------------------------------
    model = GPT(config)
    params = model.num_parameters()
    args.batch_size = args.batch_size or auto_batch_size(params, args.block_size)
    tokens_per_step = args.batch_size * args.grad_accum * args.block_size
    if args.max_steps <= 0:
        if args.token_budget <= 0:
            raise SystemExit("either --max-steps or --token-budget is required")
        args.max_steps = max(1, int(args.token_budget // tokens_per_step))
    if args.eval_interval <= 0:
        args.eval_interval = auto_eval_interval(args.max_steps)

    if args.stage == "pretrain":
        base_lr = args.lr or 6e-4
        min_lr = args.min_lr or base_lr * 0.1
    else:
        base_lr = args.lr or 3e-4   # chat tuning wants a gentler peak
        min_lr = args.min_lr or base_lr * 0.1

    print(f"[model] {config.name}: {params:,} params | vocab {config.vocab_size} | "
          f"{config.n_layer}L/{config.n_head}H/{config.n_embd}D | ctx {args.block_size}")
    print(f"[plan]  stage={args.stage} steps={args.max_steps} "
          f"batch={args.batch_size}x{args.grad_accum} -> {tokens_per_step:,} tok/step "
          f"({args.max_steps*tokens_per_step/1e6:.1f}M tokens) lr={base_lr}")
    if corpus_tokens:
        print(f"[plan]  ~{args.max_steps*tokens_per_step/max(1,corpus_tokens):.1f} passes over the corpus")

    optimizer = configure_optimizer(model, args.weight_decay, base_lr)
    start_step, best_val, history, patience = 0, float("inf"), [], 0
    # Stage 2 starts its own schedule (step 0, fresh optimizer, fresh history)
    # but continues from stage 1's *weights*.  Without this, SFT would resume
    # the pretraining step counter and skip its entire schedule.
    ckpt_finished = False
    resume_path = args.resume or (out_dir / "last.pt")
    if resume_path and Path(resume_path).exists():
        ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        ckpt_finished = bool(ckpt.get("finished", False))
        ckpt_stage = ckpt.get("stage") or (
            "pretrain" if ckpt.get("stage1", True) else "sft"
        )
        same_stage = ckpt_stage == args.stage and not ckpt_finished
        if same_stage and "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except (ValueError, KeyError) as exc:
                print(f"[resume] optimizer not restored ({exc})")
        if same_stage:
            start_step = int(ckpt.get("step", 0))
            best_val = float(ckpt.get("best_val", best_val))
            history = list(ckpt.get("history", []))
        else:
            print(f"[stage] stage-1 checkpoint is {ckpt_stage} "
                  f"(finished={ckpt_finished}); starting {args.stage} at step 0 "
                  f"from its weights")
        for key, target in (("rng_numpy", "rng_np"), ("rng_torch", "rng_torch")):
            if key in ckpt:
                try:
                    if key == "rng_numpy":
                        rng_np.bit_generator.state = ckpt[key]
                    else:
                        torch.set_rng_state(ckpt[key])
                except Exception:
                    pass
        if start_step:
            print(f"[resume] {resume_path} -> step {start_step}/{args.max_steps} "
                  f"best_val={best_val:.4f}")

    if start_step >= args.max_steps:
        print(f"[done] already at step {start_step} of {args.max_steps} — nothing to do")
    model.train()
    t0 = time.time()
    running, running_n = 0.0, 0
    deadline = t0 + args.chunk_minutes * 60 if args.chunk_minutes > 0 else None
    stop_reason = "target"
    step = start_step
    accumulated_tokens = 0

    while step < args.max_steps:
        lr = lr_at(step, args.warmup, args.max_steps, base_lr, min_lr)
        if args.stage == "pretrain":
            loss = pretrain_step(model, optimizer, train_ids, args, rng_np, lr)
        else:
            loss = sft_step(model, optimizer, train_data, args, rng_py, lr)
        step += 1
        accumulated_tokens += tokens_per_step
        running += loss
        running_n += 1

        if step % args.log_interval == 0 or step == start_step + 1:
            elapsed = time.time() - t0
            tps = accumulated_tokens / max(1e-9, elapsed)
            print(f"step {step:5d}/{args.max_steps} | loss {running/running_n:.4f} | "
                  f"lr {lr:.2e} | {tps:,.0f} tok/s | {elapsed/60:.1f} min", flush=True)
            history.append({"step": step, "loss": round(running / running_n, 5), "lr": round(lr, 8)})
            running, running_n = 0.0, 0

        if step % args.eval_interval == 0 or step == args.max_steps:
            if args.stage == "pretrain":
                val = eval_prose(model, val_ids, args, rng_np)
            else:
                val = eval_sft(model, val_data, args, rng_py)
            print(f"  [eval] step {step} val_loss {val:.4f} (best {best_val:.4f})", flush=True)
            history.append({"step": step, "val_loss": round(val, 5)})
            if val < best_val - 1e-4:
                best_val, patience = val, 0
            else:
                patience += 1
            save_checkpoint(out_dir / "best.pt", model, optimizer, config, step, best_val, history, rng_np,
                            extra={"best_step": step}, stage=args.stage)
            if args.early_stop_patience and patience >= args.early_stop_patience:
                print(f"[early-stop] no improvement for {patience} evals (best {best_val:.4f})", flush=True)
                stop_reason = "early-stop"
                break

        if args.save_interval and step % args.save_interval == 0:
            save_checkpoint(out_dir / "last.pt", model, optimizer, config, step, best_val,
                            history, rng_np, stage=args.stage)

        if deadline is not None and time.time() >= deadline:
            stop_reason = "chunk"
            print(f"[chunk] reached {args.chunk_minutes} min at step {step}/{args.max_steps} — "
                  f"saved; re-run the same command to continue", flush=True)
            break

    total_time = time.time() - t0
    finished = step >= args.max_steps
    save_checkpoint(out_dir / "last.pt", model, optimizer, config, step, best_val, history,
                    rng_np, extra={"finished": finished, "stop_reason": stop_reason},
                    stage=args.stage)
    save_checkpoint(out_dir / "model.pt", model, optimizer, config, step, best_val, history,
                    rng_np, extra={"finished": finished, "stop_reason": stop_reason},
                    stage=args.stage)

    # ---- report ----------------------------------------------------------
    hist = [h for h in history if "val_loss" in h]
    best_step = int(hist[-1]["step"]) if hist else step
    if best_val < float("inf"):
        print(f"[final] step {step}/{args.max_steps} stop={stop_reason} "
              f"best_val={best_val:.4f}")
    meta = {
        "model": config.name,
        "stage": args.stage,
        "params": params,
        "config": config.to_dict(),
        "vocab_size": tokenizer.vocab_size,
        "steps": step,
        "max_steps": args.max_steps,
        "finished": finished,
        "stop_reason": stop_reason,
        "best_val_loss": round(best_val, 5) if best_val < float("inf") else None,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "block_size": args.block_size,
        "tokens_this_invocation": accumulated_tokens,
        "train_seconds": round(total_time, 1),
        "tokens_per_second": round(accumulated_tokens / max(1e-9, total_time), 0),
        "seed": args.seed,
        "stage1": args.stage == "pretrain",
        "sft_repeat": args.sft_repeat if args.stage == "sft" else None,
        "torch": torch.__version__,
    }

    if args.stage == "sft" and (finished or args.export):
        best_path = out_dir / "best.pt"
        use_best = False
        if best_path.exists():
            best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
            if float(best_ckpt.get("best_val", 1e9)) < best_val - 1e-6:
                model.load_state_dict(best_ckpt["model"])
                best_val = float(best_ckpt["best_val"])
                best_step = int(best_ckpt.get("step", step))
                use_best = True
        meta["exported"] = "best" if use_best else "last"
        meta["exported_val_loss"] = round(best_val, 5) if best_val < float("inf") else None
        samples = chat_samples(model, tokenizer, args, out_dir / "samples.txt")
        print("\n--- chat samples ---")
        for s in samples:
            print(f"\nPROMPT: {s['prompt']!r}\n{s['completion'][:300]}")

    (out_dir / "train_log.json").write_text(
        json.dumps({"meta": meta, "history": history}, indent=2) + "\n", encoding="utf-8"
    )
    svg_loss_curve(history, out_dir / "loss_curve.svg")

    if args.stage == "sft" and (finished or args.export):
        from tinyllm.numpy_engine import export_checkpoint
        path = export_checkpoint(model, config, tokenizer, out_dir / "weights.json")
        print(f"[export] {path} ({path.stat().st_size/1e6:.2f} MB)")
    elif args.export:
        # pretraining exports too, so an unfinished stage can still be tried
        from tinyllm.numpy_engine import export_checkpoint
        path = export_checkpoint(model, config, tokenizer, out_dir / "weights.json")
        print(f"[export] {path} ({path.stat().st_size/1e6:.2f} MB)")

    if args.publish:
        import shutil
        dest = REPO_ROOT / "model" / "spore" / config.name
        dest.mkdir(parents=True, exist_ok=True)
        for name in ("weights.json", "config.json", "tokenizer.json", "loss_curve.svg",
                     "samples.txt", "train_log.json"):
            src = out_dir / name
            if src.exists():
                shutil.copy(src, dest / name)
        (dest / "meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
        print(f"[publish] {dest}")

    if not finished:
        print(f"\n[continue] not finished ({stop_reason}). Re-run the same command; "
              f"it resumes from step {step}.")
        return 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

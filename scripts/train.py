#!/usr/bin/env python3
"""Train TinyGPT from scratch on CPU.

Designed to run inside a GitHub Actions runner (2 vCPUs, 8 GB RAM) but works
anywhere.  Produces, under ``--out-dir``:

  tokenizer.json      : the trained byte-level BPE vocabulary
  config.json         : model hyper-parameters
  model.pt            : PyTorch checkpoint (last step)
  best.pt             : checkpoint with the lowest validation loss
  weights.json        : float16 export consumed by the numpy / JS engines
  train_log.json      : loss history + run metadata (plotted in the README)
  loss_curve.svg      : chart rendered without matplotlib
  samples.txt         : sample generations from the final model

Resumable: if ``--resume`` points at a checkpoint, training continues from it
(optimizer state included) so a run can be split across jobs.

Examples
--------
    # full run (default: ~6.4M params, reproducible)
    python scripts/train.py --out-dir runs/main

    # quick smoke test (~1 minute)
    python scripts/train.py --out-dir runs/smoke --max-steps 60 --n-layer 2 \
        --n-embd 64 --n-head 2 --block-size 64
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tinyllm.config import GPTConfig  # noqa: E402
from tinyllm.dataset import (  # noqa: E402
    DEFAULT_DATA_PATH,
    get_batch,
    load_text,
    train_val_split,
)
from tinyllm.model import GPT  # noqa: E402
from tinyllm.tokenizer import ByteBPE  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TinyGPT on CPU", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    g = p.add_argument_group("data & output")
    g.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    g.add_argument("--out-dir", type=Path, default=Path("runs/main"))
    g.add_argument("--tokenizer", type=Path, default=None, help="reuse an existing tokenizer.json")
    g.add_argument("--config", type=Path, default=None, help="load GPTConfig from json")
    g.add_argument("--resume", type=Path, default=None, help="checkpoint to resume from")

    m = p.add_argument_group("model")
    m.add_argument("--n-layer", type=int, default=6)
    m.add_argument("--n-head", type=int, default=6)
    m.add_argument("--n-embd", type=int, default=288)
    m.add_argument("--block-size", type=int, default=256)
    m.add_argument("--dropout", type=float, default=0.0)

    t = p.add_argument_group("training")
    t.add_argument("--vocab-size", type=int, default=1280)
    t.add_argument("--max-steps", type=int, default=6000)
    t.add_argument("--batch-size", type=int, default=16)
    t.add_argument("--grad-accum", type=int, default=2)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--min-lr", type=float, default=1e-4)
    t.add_argument("--warmup", type=int, default=200)
    t.add_argument("--weight-decay", type=float, default=0.1)
    t.add_argument("--grad-clip", type=float, default=1.0)
    t.add_argument("--eval-interval", type=int, default=250)
    t.add_argument("--eval-batches", type=int, default=20)
    t.add_argument("--log-interval", type=int, default=25)
    t.add_argument("--save-interval", type=int, default=1000)
    t.add_argument("--threads", type=int, default=0, help="0 = auto (use nproc on CI)")
    t.add_argument(
        "--compile",
        action="store_true",
        help="torch.compile the model (big CPU speedup on Linux, slower first step)",
    )
    t.add_argument(
        "--fast-math",
        action="store_true",
        help="allow MKLDNN/mkldnn backends (default off for exact parity)",
    )
    t.add_argument(
        "--budget-steps",
        type=int,
        default=0,
        help="stop after this many steps in THIS invocation (0 = run to --max-steps). "
             "Lets CI split a long run across jobs that each fit the time budget.",
    )
    t.add_argument("--seed", type=int, default=1337)
    t.add_argument("--time-budget", type=float, default=0.0, help="stop after N minutes (0 = no limit)")
    t.add_argument("--export", action="store_true", help="write weights.json even if not finished")
    t.add_argument(
        "--publish",
        action="store_true",
        help="also copy artefacts into model/ (what the GitHub workflow does)",
    )
    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def save_checkpoint(path: Path, model, optimizer, config, step: int, best_val: float, history, rng) -> None:
    """Everything needed to resume *exactly*: weights, optimizer, RNG state."""
    import torch

    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.to_dict(),
            "step": step,
            "best_val": best_val,
            "history": history,
            "rng_numpy": rng.bit_generator.state,
            "rng_torch": torch.get_rng_state(),
        },
        tmp,
    )
    tmp.replace(path)  # atomic: a killed job never leaves a half-written file


def lr_at(step: int, args: argparse.Namespace) -> float:
    """Linear warmup then cosine decay to min_lr."""
    if step < args.warmup:
        return args.lr * (step + 1) / max(1, args.warmup)
    progress = (step - args.warmup) / max(1, args.max_steps - args.warmup)
    progress = min(1.0, progress)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return args.min_lr + coeff * (args.lr - args.min_lr)


def configure_optimizer(model: GPT, weight_decay: float, lr: float) -> torch.optim.Optimizer:
    """AdamW with no decay on biases / LayerNorm, and no decay on embeddings."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() < 2 or "ln_" in name or name.endswith(".bias"):
            no_decay.append(param)
        else:
            decay.append(param)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), eps=1e-8)


@torch.no_grad()
def estimate_loss(model: GPT, data: np.ndarray, args: argparse.Namespace, rng: np.random.Generator, device: str) -> float:
    model.eval()
    total, count = 0.0, 0
    for _ in range(args.eval_batches):
        x, y = get_batch(data, args.batch_size, args.block_size, rng, device)
        _, loss, _ = model(x, y)
        total += float(loss.item())
        count += 1
    model.train()
    return total / max(1, count)


@torch.no_grad()
def sample_text(model: GPT, tokenizer: ByteBPE, prompts: List[str], args: argparse.Namespace, device: str, max_new: int = 220) -> List[Dict[str, str]]:
    model.eval()
    outputs = []
    for i, prompt in enumerate(prompts):
        ids = tokenizer.encode(prompt)
        if not ids:
            ids = [tokenizer.encode("\n")[0]]
        x = torch.tensor([ids], dtype=torch.long, device=device)
        out = model.generate(
            x,
            max_new_tokens=max_new,
            temperature=0.8,
            top_k=50,
            top_p=0.95,
            seed=args.seed + i,
        )
        outputs.append({"prompt": prompt, "completion": tokenizer.decode(out[0].tolist())})
    model.train()
    return outputs


def svg_loss_curve(log: List[Dict[str, Any]], path: Path, width: int = 720, height: int = 360) -> None:
    """Tiny dependency-free SVG line chart of the loss history."""
    train_pts = [(e["step"], e["train_loss"]) for e in log if "train_loss" in e]
    val_pts = [(e["step"], e["val_loss"]) for e in log if "val_loss" in e]
    if not train_pts:
        return
    all_pts = train_pts + val_pts
    xs = [p[0] for p in all_pts]
    ys = [p[1] for p in all_pts]
    xmin, xmax = min(xs), max(xs) or 1
    ymin, ymax = min(ys) * 0.95, max(ys) * 1.02
    pad = 45

    def px(x: float) -> float:
        return pad + (x - xmin) / max(1e-9, xmax - xmin) * (width - 2 * pad)

    def py(y: float) -> float:
        return height - pad - (y - ymin) / max(1e-9, ymax - ymin) * (height - 2 * pad)

    def polyline(pts):
        return " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#0d1117"/>',
        f'<line x1="{pad}" y1="{height-pad}" x2="{width-pad}" y2="{height-pad}" stroke="#30363d"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height-pad}" stroke="#30363d"/>',
    ]
    for frac in (0, 0.25, 0.5, 0.75, 1.0):
        y = ymin + frac * (ymax - ymin)
        parts.append(f'<text x="{pad-8}" y="{py(y)+4:.1f}" fill="#8b949e" font-size="11" text-anchor="end" font-family="monospace">{y:.2f}</text>')
    parts.append(f'<polyline fill="none" stroke="#58a6ff" stroke-width="1.5" points="{polyline(train_pts)}"/>')
    parts.append(f'<polyline fill="none" stroke="#3fb950" stroke-width="2.5" points="{polyline(val_pts)}"/>')
    parts.append(f'<text x="{width-pad}" y="{height-pad+28}" fill="#8b949e" font-size="11" text-anchor="end" font-family="monospace">step {xmax}</text>')
    parts.append(f'<text x="{pad+8}" y="{pad-12}" fill="#58a6ff" font-size="12" font-family="monospace">train</text>')
    parts.append(f'<text x="{pad+60}" y="{pad-12}" fill="#3fb950" font-size="12" font-family="monospace">val</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def export_weights(model: GPT, config: GPTConfig, tokenizer: ByteBPE, out_dir: Path) -> Path:
    """Export float16 weights + config + tokenizer as one JS-loadable JSON."""
    from tinyllm.numpy_engine import export_checkpoint

    path = out_dir / "weights.json"
    export_checkpoint(model, config, tokenizer, path)
    return path


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.threads > 0:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)  # avoid inter-op overhead on small models
    if args.fast_math:
        # enables channels_last + better CPU kernels for the linear layers
        torch.backends.mkldnn.enabled = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = "cpu"

    # ---- tokenizer -------------------------------------------------------
    tok_path = args.tokenizer or (out_dir / "tokenizer.json")
    if tok_path.exists():
        tokenizer = ByteBPE.load(tok_path)
        print(f"[tokenizer] loaded {tok_path} (vocab={tokenizer.vocab_size})")
    else:
        print("[tokenizer] training byte-level BPE ...")
        t0 = time.time()
        tokenizer = ByteBPE.train(load_text(args.data), vocab_size=args.vocab_size)
        tokenizer.save(tok_path)
        print(f"[tokenizer] done in {time.time()-t0:.1f}s vocab={tokenizer.vocab_size}")

    # ---- data ------------------------------------------------------------
    text = load_text(args.data)
    train_text, val_text = train_val_split(text)
    train_ids = np.array(tokenizer.encode(train_text), dtype=np.uint16)
    val_ids = np.array(tokenizer.encode(val_text), dtype=np.uint16)
    print(f"[data] train tokens={len(train_ids):,} val tokens={len(val_ids):,} "
          f"(compression {len(text)/ (len(train_ids)+len(val_ids)):.2f}x)")

    # ---- config ----------------------------------------------------------
    if args.config:
        config = GPTConfig.load(args.config)
    else:
        config = GPTConfig(
            vocab_size=tokenizer.vocab_size,
            block_size=args.block_size,
            n_layer=args.n_layer,
            n_head=args.n_head,
            n_embd=args.n_embd,
            dropout=args.dropout,
        )
    if config.vocab_size != tokenizer.vocab_size:
        raise SystemExit(f"config vocab_size={config.vocab_size} != tokenizer vocab={tokenizer.vocab_size}")
    # batching must follow the model's context length, whichever came from the config
    args.block_size = config.block_size
    config.save(out_dir / "config.json")

    model = GPT(config).to(device)
    if args.compile and hasattr(torch, "compile"):
        try:
            model = torch.compile(model)  # type: ignore[assignment]
            print("[model] torch.compile enabled")
        except Exception as exc:  # pragma: no cover - backend dependent
            print(f"[model] torch.compile unavailable ({exc}); using eager mode")
    print(f"[model] {model.num_parameters():,} parameters "
          f"(non-embedding {model.num_parameters(non_embedding=True):,})")

    optimizer = configure_optimizer(model, args.weight_decay, args.lr)
    start_step = 0
    history: List[Dict[str, Any]] = []
    best_val = float("inf")

    if args.resume and args.resume.exists():
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
            except (ValueError, KeyError) as exc:  # param groups changed
                print(f"[resume] could not restore optimizer ({exc}); starting fresh")
        start_step = int(ckpt.get("step", 0))
        history = list(ckpt.get("history", []))
        best_val = float(ckpt.get("best_val", best_val))
        if "rng_numpy" in ckpt:
            try:
                rng.bit_generator.state = ckpt["rng_numpy"]
                torch.set_rng_state(ckpt["rng_torch"])
            except Exception:
                pass
        print(f"[resume] step {start_step} best_val={best_val:.4f}")

    model.train()
    t_start = time.time()
    tokens_per_step = args.batch_size * args.grad_accum * args.block_size
    running = 0.0
    running_n = 0
    stopped_early = False

    stop_step = args.max_steps
    if args.budget_steps > 0:
        stop_step = min(stop_step, start_step + args.budget_steps)
        if stop_step < args.max_steps:
            print(f"[budget] this invocation trains {start_step} -> {stop_step} of {args.max_steps} "
                  f"(resume later with the same total)")

    for step in range(start_step, stop_step):
        lr = lr_at(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(args.grad_accum):
            x, y = get_batch(train_ids, args.batch_size, args.block_size, rng, device)
            _, loss, _ = model(x, y)
            (loss / args.grad_accum).backward()
            step_loss += float(loss.item()) / args.grad_accum
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        running += step_loss
        running_n += 1

        if (step + 1) % args.log_interval == 0 or step == start_step:
            avg = running / max(1, running_n)
            running, running_n = 0.0, 0
            elapsed = time.time() - t_start
            tps = tokens_per_step * (step + 1 - start_step) / max(1e-9, elapsed)
            print(f"step {step+1:5d}/{args.max_steps} | loss {avg:.4f} | lr {lr:.2e} | "
                  f"{tps:,.0f} tok/s | {elapsed/60:.1f} min", flush=True)
            history.append({"step": step + 1, "train_loss": round(avg, 5), "lr": round(lr, 7)})

        if (step + 1) % args.eval_interval == 0 or step + 1 == stop_step:
            val_loss = estimate_loss(model, val_ids, args, rng, device)
            print(f"  [eval] step {step+1} val_loss {val_loss:.4f}", flush=True)
            history.append({"step": step + 1, "val_loss": round(val_loss, 5)})
            if val_loss < best_val:
                best_val = val_loss
            best_val = min(best_val, val_loss)
            save_checkpoint(out_dir / "best.pt", model, optimizer, config, step + 1, best_val, history, rng)

        # Always keep a resumable checkpoint near the end of an invocation: a
        # job may be stopped by the runner at *any* step, so relying only on a
        # save_interval boundary can lose an entire chunk of training.
        near_end = stop_step - (step + 1) <= 2
        if args.save_interval and ((step + 1) % args.save_interval == 0 or near_end):
            save_checkpoint(out_dir / "last.pt", model, optimizer, config, step + 1, best_val, history, rng)

        if args.time_budget > 0 and (time.time() - t_start) / 60 > args.time_budget:
            print(f"[time] budget of {args.time_budget} min reached at step {step+1}", flush=True)
            stopped_early = True
            break

    final_step = int(history[-1]["step"]) if history else start_step
    total_time = time.time() - t_start
    save_checkpoint(out_dir / "model.pt", model, optimizer, config, final_step, best_val, history, rng)
    if final_step > 0:
        # always leave a resumable checkpoint, even if no interval boundary hit
        save_checkpoint(out_dir / "last.pt", model, optimizer, config, final_step, best_val, history, rng)

    reached_target = final_step >= args.max_steps
    if not reached_target:
        print(f"[budget] stopped at step {final_step}/{args.max_steps} — "
              f"re-run with the same flags to continue")

    # ---- final evaluation + samples --------------------------------------
    val_loss = estimate_loss(model, val_ids, args, rng, device)
    print(f"[final] val_loss {val_loss:.4f} (best {best_val:.4f})")

    # bits per character: the honest, tokenizer-independent metric
    bpc = val_loss / math.log(2) / (len(text) / (len(train_ids) + len(val_ids)))
    print(f"[final] ~{bpc:.3f} bits/byte")

    samples = sample_text(
        model, tokenizer,
        ["ROMEO:", "The king", "To be, or not to be", "\n"],
        args, device,
    )
    with open(out_dir / "samples.txt", "w") as f:
        for s in samples:
            f.write("=" * 70 + f"\nPROMPT: {s['prompt']!r}\n" + "-" * 70 + "\n")
            f.write(s["completion"] + "\n\n")

    meta = {
        "config": config.to_dict(),
        "params": model.num_parameters(),
        "params_non_embedding": model.num_parameters(non_embedding=True),
        "steps": final_step,
        "val_loss": round(val_loss, 5),
        "best_val_loss": round(best_val, 5),
        "bits_per_byte": round(bpc, 4),
        "train_tokens": int(len(train_ids)),
        "val_tokens": int(len(val_ids)),
        "vocab_size": tokenizer.vocab_size,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "block_size": config.block_size,
        "tokens_seen": int(final_step * tokens_per_step),
        "train_seconds": round(total_time, 1),
        "seed": args.seed,
        "torch": torch.__version__,
        "device": device,
        "compiled": bool(args.compile),
        "total_steps_target": args.max_steps,
        "finished": reached_target,
        "resumed_from": str(args.resume) if args.resume else None,
    }
    (out_dir / "train_log.json").write_text(json.dumps({"meta": meta, "history": history}, indent=2) + "\n")
    svg_loss_curve(history, out_dir / "loss_curve.svg")

    # ---- export for numpy / JS inference --------------------------------
    path = export_weights(model, config, tokenizer, out_dir)
    print(f"[export] {path} ({path.stat().st_size/1e6:.2f} MB)")

    # ---- publish (mirror of the CI publish step) ------------------------
    if args.publish:
        import shutil

        model_dir = REPO_ROOT / "model"
        model_dir.mkdir(exist_ok=True)
        for name in ("weights.json", "config.json", "tokenizer.json", "loss_curve.svg", "samples.txt"):
            src = out_dir / name
            if src.exists():
                shutil.copy(src, model_dir / name)
        publish_meta = dict(meta)
        publish_meta["history"] = history
        (model_dir / "meta.json").write_text(json.dumps(publish_meta, indent=2) + "\n")
        print(f"[publish] copied artefacts to {model_dir}")

    print("\n--- samples ---")
    for s in samples:
        print(f"\nPROMPT: {s['prompt']!r}\n{s['completion'][:400]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Measure what actually drives wall-clock in train_spore.py.

Every knob here was chosen because it plausibly changes tokens/second or
quality-per-token. The point is to replace guesses with measurements on the
same 2-core CPU class the GitHub runner uses.

    python scripts/bench_throughput.py [--quick]

Prints a table of (config, block_size, batch, precision) -> tok/s and, more
usefully, "seconds per effective token" so the cost of a given token budget is
directly comparable across model shapes.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tinyllm.config import GPTConfig  # noqa: E402
from tinyllm.model import GPT  # noqa: E402


def build(cfg: dict) -> GPT:
    return GPT(GPTConfig.from_dict(cfg))


def time_steps(model, block_size, batch, iters, warmup=2) -> float:
    """Seconds per optimiser step (forward + backward), fp32."""
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    g = torch.Generator().manual_seed(0)
    x = torch.randint(0, 64, (batch, block_size), generator=g)
    y = torch.randint(0, 64, (batch, block_size), generator=g)
    for _ in range(warmup):
        _, loss, _ = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    for _ in range(iters):
        _, loss, _ = model(x, y)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
    return (time.perf_counter() - t0) / iters


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fewer iterations")
    args = ap.parse_args()
    iters = 3 if args.quick else 6

    cfgs = {}
    for name in ("spore_12m", "spore_36m"):
        cfgs[name] = json.loads((ROOT / "configs" / f"{name}.json").read_text())

    # A deliberately small model to show how cost scales with parameters.
    cfgs["tiny_1m"] = {
        "name": "bench-1m", "vocab_size": 3010, "block_size": 512,
        "n_layer": 4, "n_head": 4, "n_embd": 128,
        "dropout": 0.0, "bias": True, "tie_weights": True,
        "gelu_tanh": True, "ln_eps": 1e-5,
    }

    torch.set_num_threads(2)
    rows = []
    # Micro-batches that fit a 2-core / 3 GB box. Activation memory is what
    # kills us, and it scales with batch x block, so cap tokens-per-step.
    BATCH_FOR = {
        "spore_36m": {512: [2, 1], 256: [2, 1], 128: [2, 1]},
        "spore_12m": {512: [8, 4], 256: [8, 4], 128: [8, 4]},
        "tiny_1m":   {512: [16, 8], 256: [16, 8], 128: [16, 8]},
    }
    for cfg_name, cfg in cfgs.items():
        params = sum(p.numel() for p in build(cfg).parameters())
        for block in (512, 256, 128):
            for batch in BATCH_FOR[cfg_name][block]:
                model = build({**cfg, "block_size": block})
                model.train()
                sec = time_steps(model, block, batch, iters)
                tok = batch * block / sec
                rows.append({
                    "model": cfg_name, "params": params, "block": block,
                    "batch": batch, "sec_per_step": sec, "tok_per_s": tok,
                    # seconds of compute per 1M training tokens
                    "s_per_Mtok": 1e6 / tok,
                })
                print(
                    f"{cfg_name:10} {params/1e6:5.2f}M  ctx {block:4}  b{batch:<3} "
                    f"{sec*1000:7.0f} ms/step  {tok:8.0f} tok/s  "
                    f"{1e6/tok:8.1f} s per 1M tokens",
                    flush=True,
                )

    out = ROOT / "bench_throughput.json"
    out.write_text(json.dumps(rows, indent=2) + "\n")
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

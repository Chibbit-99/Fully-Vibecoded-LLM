"""Command line interface.

    python -m tinyllm generate "What is water?" -n 120
    python -m tinyllm chat                      # interactive REPL
    python -m tinyllm info                      # model metadata
    python -m tinyllm tokenize "To be, or not to be"
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .config import GPTConfig
from .numpy_engine import NumpyGPT
from .tokenizer import ByteBPE

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent.parent / "model" / "weights.json"


def load(checkpoint: Path) -> NumpyGPT:
    if not checkpoint.exists():
        raise SystemExit(
            f"checkpoint not found: {checkpoint}\n"
            "Train one with `scripts/train_spore.py` or run the Train Spore workflow."
        )
    return NumpyGPT.from_checkpoint(checkpoint)


def cmd_generate(args: argparse.Namespace) -> int:
    model = load(args.checkpoint)
    out = model.generate(
        args.prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k or None,
        top_p=args.top_p,
        seed=args.seed,
        stop_strings=[args.stop] if args.stop else None,
    )
    print(out["prompt"] + out["text"])
    if args.stats:
        n, tps = out["tokens_generated"], 0.0
        print(f"\n[tinyllm] {n} tokens | temp={args.temperature} top_k={args.top_k} top_p={args.top_p}", file=sys.stderr)
    return 0


def cmd_chat(args: argparse.Namespace) -> int:
    model = load(args.checkpoint)
    print("TinyLLM chat — type a prompt, Ctrl-C or 'quit' to exit.\n")
    while True:
        try:
            prompt = input("you> ")
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if prompt.strip().lower() in {"quit", "exit"}:
            return 0
        out = model.generate(
            prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_k=args.top_k or None,
            top_p=args.top_p,
            seed=args.seed,
        )
        print(f"model> {out['text']}\n")


def cmd_info(args: argparse.Namespace) -> int:
    from .numpy_engine import load_checkpoint

    weights, config, tok = load_checkpoint(args.checkpoint)
    meta_path = Path(args.checkpoint).parent / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    params = sum(v.size for v in weights.values())
    info = {
        "checkpoint": str(args.checkpoint),
        "size_mb": round(Path(args.checkpoint).stat().st_size / 1e6, 2),
        "parameters": params,
        "config": config.to_dict(),
        "tokenizer": {"type": "byte-bpe", "vocab_size": tok.vocab_size, "merges": len(tok.merges)},
        "training": {k: meta.get(k) for k in ("steps", "val_loss", "bits_per_byte", "tokens_seen", "train_seconds")},
    }
    print(json.dumps(info, indent=2))
    # sanity check: analytic count matches stored tensors
    analytical = GPTConfig(**config.to_dict()).num_params()
    if abs(analytical - params) > 2000:
        print(f"warning: analytic param count {analytical} != stored {params}", file=sys.stderr)
    return 0


def cmd_tokenize(args: argparse.Namespace) -> int:
    tok_path = Path(args.checkpoint).parent / "tokenizer.json"
    tok = ByteBPE.load(tok_path) if tok_path.exists() else None
    if tok is None:
        _, _, tok = __import__("tinyllm.numpy_engine", fromlist=["load_checkpoint"]).load_checkpoint(args.checkpoint)
    ids = tok.encode(args.text)
    print(json.dumps({
        "text": args.text,
        "ids": ids,
        "count": len(ids),
        "pieces": [tok.decode([i]) for i in ids],
        "roundtrip_ok": tok.decode(ids) == args.text,
    }, indent=2, ensure_ascii=False))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tinyllm", description="TinyLLM command line tools")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="complete a prompt")
    g.add_argument("prompt", nargs="?", default="What is water?")
    g.add_argument("-n", "--max-tokens", type=int, default=120)
    g.add_argument("-t", "--temperature", type=float, default=0.8)
    g.add_argument("--top-k", type=int, default=40)
    g.add_argument("--top-p", type=float, default=0.95)
    g.add_argument("--seed", type=int, default=None)
    g.add_argument("--stop", default=None)
    g.add_argument("--stats", action="store_true")
    g.set_defaults(func=cmd_generate)

    c = sub.add_parser("chat", help="interactive prompt loop")
    c.add_argument("-n", "--max-tokens", type=int, default=120)
    c.add_argument("-t", "--temperature", type=float, default=0.8)
    c.add_argument("--top-k", type=int, default=40)
    c.add_argument("--top-p", type=float, default=0.95)
    c.add_argument("--seed", type=int, default=None)
    c.set_defaults(func=cmd_chat)

    i = sub.add_parser("info", help="show model metadata")
    i.set_defaults(func=cmd_info)

    t = sub.add_parser("tokenize", help="show BPE tokenization")
    t.add_argument("text")
    t.set_defaults(func=cmd_tokenize)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())

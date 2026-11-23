#!/usr/bin/env python3
"""Fetch Tiny Shakespeare (public domain) into data/.

The file is vendored in the repository so training runs are reproducible and
GitHub Actions never depends on a network download.  This script exists so the
data can be refreshed or fetched into a fresh clone.

    python scripts/download_data.py            # no-op if already present
    python scripts/download_data.py --force    # re-download
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

SOURCES = [
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "https://codeload.github.com/karpathy/char-rnn/zip/refs/heads/master",  # fallback (zip)
]
DEFAULT_OUT = Path(__file__).resolve().parent.parent / "data" / "tinyshakespeare.txt"


def fetch(out: Path) -> int:
    for url in SOURCES:
        try:
            print(f"[data] downloading {url}")
            with urllib.request.urlopen(url, timeout=60) as resp:
                payload = resp.read()
            if url.endswith(".txt"):
                text = payload.decode("utf-8")
            else:
                import io
                import zipfile

                with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                    name = next(n for n in zf.namelist() if n.endswith("tinyshakespeare/input.txt"))
                    text = zf.read(name).decode("utf-8")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(text, encoding="utf-8")
            print(f"[data] wrote {out} ({len(text):,} chars)")
            return 0
        except Exception as exc:  # try the next mirror
            print(f"[data] failed: {exc}")
    print("[data] could not download the dataset", file=sys.stderr)
    return 1


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    if args.out.exists() and not args.force:
        print(f"[data] already present: {args.out} ({args.out.stat().st_size:,} bytes)")
        return 0
    return fetch(args.out)


if __name__ == "__main__":
    raise SystemExit(main())

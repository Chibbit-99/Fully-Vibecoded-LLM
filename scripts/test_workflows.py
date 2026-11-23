#!/usr/bin/env python3
"""Execute the shell bodies of the workflow `run:` steps locally.

Bash inside YAML is where CI jobs silently die — a heredoc that isn't indented
correctly, or a `set -e` tripping over an ignored `git add`. This script pulls
every `run:` block out of the workflows, substitutes the GitHub expressions with
sane defaults, and executes them in a throwaway copy of the repo, so mistakes
surface here instead of after a 25-minute training run.

    python scripts/test_workflows.py            # all safe blocks
    python scripts/test_workflows.py --list     # show what would run
    python scripts/test_workflows.py --step Train

Blocks that train, install packages or push are *stubbed*: the goal is to catch
shell/YAML errors, not to duplicate CI.
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
WORKFLOWS = ROOT / ".github" / "workflows"

# steps that are too slow / destructive to run for real, with a safe stub
STUBS = {
    "Train": "echo '[stub] training skipped'",
    "Install dependencies (CPU PyTorch)": "echo '[stub] pip skipped'",
    "Install": "echo '[stub] pip skipped'",
    "Commit trained model": "echo '[stub] commit skipped'",
    "Set up Python": "echo '[stub] setup-python skipped'",
    "Checkout": "echo '[stub] checkout skipped'",
    # test runners need the CI environment (installed pytest / node deps)
    "Fast unit tests (tokenizer + model shapes)": "echo '[stub] pytest skipped'",
    "Cross-engine parity (needs trained weights)": "echo '[stub] parity skipped'",
    "Server smoke test": "echo '[stub] server tests skipped'",
    "Run fast test suite": "echo '[stub] pytest skipped'",
    "Runner info": "echo '[stub] runner info skipped'",
}

EXPR = re.compile(r"\$\{\{[^}]*\}\}")


def substitute(script: str) -> str:
    """Replace ${{ ... }} expressions with deterministic defaults."""
    defaults = {
        "inputs.config": "configs/fast.json",
        "inputs.max_steps": "4000",
        "inputs.batch_size": "32",
        "inputs.seed": "1337",
        "inputs.chunk_minutes": "25",
        "github.ref_name": "main",
        "github.run_number": "1",
        "github.run_id": "1",
        "inputs.clear_checkpoints": "false",
        "inputs.reset": "false",
    }

    def repl(match: re.Match) -> str:
        expr = match.group(0)[3:-2].strip()
        # `a || 'b'` -> take the literal fallback
        if "||" in expr:
            fallback = expr.split("||", 1)[1].strip().strip("'\"")
            return fallback
        if expr in defaults:
            return defaults[expr]
        if expr.startswith("steps.") or expr.startswith("needs."):
            return "stub"
        return "stub"

    return EXPR.sub(repl, script)


def collect(only: str | None) -> list[tuple[str, str, str]]:
    blocks = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        data = yaml.safe_load(path.read_text())
        for job_name, job in (data.get("jobs") or {}).items():
            for step in job.get("steps") or []:
                name = step.get("name") or step.get("uses") or "unnamed"
                script = step.get("run")
                if not script:
                    continue
                if only and only.lower() not in str(name).lower():
                    continue
                blocks.append((path.name, str(name), script))
    return blocks


def prepare_sandbox() -> Path:
    """Copy the repo (without .git / runs / venv) into a temp dir."""
    tmp = Path(tempfile.mkdtemp(prefix="wf-test-"))
    ignore = shutil.ignore_patterns(".git", "runs", ".venv", "__pycache__", "*.pt", "node_modules")
    shutil.copytree(ROOT, tmp / "repo", ignore=ignore, dirs_exist_ok=True)
    return tmp / "repo"


def run_block(sandbox: Path, workflow: str, name: str, script: str) -> tuple[bool, str]:
    body = substitute(script)
    # inside a temp repo the tokenizer/config already exist; make publish steps
    # resilient because there is no training output
    body = body.replace("runs/main/", "model/")
    full = "set -euo pipefail\n" + body
    proc = subprocess.run(
        ["bash", "-c", full],
        cwd=sandbox,
        capture_output=True,
        text=True,
        env={**os.environ, "PYTHON": sys.executable, "TINYLLM_WF_TEST": "1"},
        timeout=300,
    )
    out = (proc.stdout + proc.stderr).strip()
    return proc.returncode == 0, out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--step", default=None, help="only steps whose name contains this")
    ap.add_argument("--keep", action="store_true", help="keep the sandbox for inspection")
    args = ap.parse_args(argv)

    blocks = collect(args.step)
    if not blocks:
        print("no run blocks found", file=sys.stderr)
        return 1

    if args.list:
        for wf, name, _ in blocks:
            print(f"{wf:16s} {name}")
        return 0

    sandbox = prepare_sandbox()
    failures = 0
    try:
        for wf, name, script in blocks:
            if name in STUBS:
                print(f"SKIP  {wf:16s} {name}  (stubbed)")
                continue
            ok, out = run_block(sandbox, wf, name, script)
            print(f"{'PASS' if ok else 'FAIL'}  {wf:16s} {name}")
            if not ok:
                failures += 1
                for line in out.splitlines()[-25:]:
                    print(f"        {line}")
    finally:
        if not args.keep:
            shutil.rmtree(sandbox.parent, ignore_errors=True)
        else:
            print(f"\nsandbox kept at {sandbox}")

    print()
    if failures:
        print(f"{failures} step(s) failed")
        return 1
    print("all executed workflow steps passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

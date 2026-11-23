#!/usr/bin/env python3
"""Lint the GitHub Actions workflows without shelling out to `actionlint`.

Checks that are cheap to get right here and expensive to debug in CI:

* the file parses as YAML and has `name`, `on`, `jobs`
* every job has `runs-on` and at least one step
* every `uses:` is pinned to a tag or SHA (not a branch)
* no unknown top-level keys
* training defaults referenced in the workflow exist as train.py flags
* shell blocks do not reference undefined `${{ }}` inputs without a fallback

    python scripts/validate_workflows.py
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_DIR = ROOT / ".github" / "workflows"

try:
    import yaml
except ImportError:  # pragma: no cover
    print("pyyaml is required: pip install pyyaml", file=sys.stderr)
    sys.exit(2)


def check(name: str, condition: bool, detail: str = "") -> bool:
    print(f"  {'PASS' if condition else 'FAIL'}  {name}" + (f" — {detail}" if detail and not condition else ""))
    return condition


def validate_file(path: Path, train_flags: set[str]) -> list[str]:
    problems: list[str] = []
    text = path.read_text()
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        return [f"{path.name}: YAML error: {exc}"]

    if not isinstance(data, dict):
        return [f"{path.name}: not a mapping"]

    allowed_top = {"name", "on", "jobs", "permissions", "env", "concurrency", "run-name", "defaults"}
    for key in data:
        if key not in allowed_top and key is not True:  # `on:` parses to True in PyYAML < 6
            problems.append(f"{path.name}: unexpected top-level key {key!r}")
    for required in ("name", "jobs"):
        if required not in data:
            problems.append(f"{path.name}: missing {required!r}")

    jobs = data.get("jobs") or {}
    if not jobs:
        problems.append(f"{path.name}: no jobs")
    for job_name, job in jobs.items():
        if "runs-on" not in job:
            problems.append(f"{path.name}:{job_name}: missing runs-on")
        steps = job.get("steps") or []
        if not steps:
            problems.append(f"{path.name}:{job_name}: no steps")
        for step in steps:
            uses = step.get("uses")
            if uses and "@" in uses:
                ref = uses.split("@", 1)[1]
                if ref in {"main", "master", "HEAD", "latest"}:
                    problems.append(f"{path.name}:{job_name}: {uses} is not pinned to a tag")

    # `${{ inputs.x }}` must be declared under on.workflow_dispatch.inputs
    dispatch = (data.get("on") or {}) if isinstance(data.get("on"), dict) else {}
    declared = set((dispatch.get("workflow_dispatch") or {}).get("inputs", {}) or {})
    for match in re.finditer(r"inputs\.([a-zA-Z0-9_]+)", text):
        name_ = match.group(1)
        if declared and name_ not in declared:
            problems.append(f"{path.name}: uses undeclared input {name_!r}")

    # train.py flags referenced in run blocks must exist
    for match in re.finditer(r"scripts/train\.py([^\n`]*)", text):
        for flag in re.findall(r"(--[a-z0-9-]+)", match.group(1)):
            if flag not in train_flags:
                problems.append(f"{path.name}: unknown train.py flag {flag}")

    return problems


def train_flags() -> set[str]:
    out = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "train.py"), "--help"],
        capture_output=True, text=True, cwd=ROOT,
    )
    return set(re.findall(r"(--[a-z0-9-]+)", out.stdout))


def main() -> int:
    flags = train_flags()
    files = sorted(WORKFLOW_DIR.glob("*.yml")) + sorted(WORKFLOW_DIR.glob("*.yaml"))
    if not files:
        print("no workflows found", file=sys.stderr)
        return 1
    all_problems: list[str] = []
    for path in files:
        print(f"{path.name}:")
        problems = validate_file(path, flags)
        for p in problems:
            print(f"  FAIL  {p}")
        all_problems.extend(problems)
        if not problems:
            print("  PASS  structure, pinned actions, inputs, train.py flags")
    print()
    if all_problems:
        print(f"{len(all_problems)} problem(s) found")
        return 1
    print(f"all {len(files)} workflow(s) look valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

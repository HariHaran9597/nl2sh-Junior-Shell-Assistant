"""Audit eval_results.json for score inflation.

Self-comparison scoring passes a task when generated and reference produce
IDENTICAL stdout + filesystem. On a Windows git-bash sandbox (no Docker),
Linux-only commands can fail IDENTICALLY on both sides (e.g. both
"command not found") -> counted as PASS without the model being right.
This script separates genuine passes from agreeing-failure passes.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
P = ROOT / "data" / "eval_results.json"


def main():
    payload = json.loads(P.read_text(encoding="utf-8"))
    results = payload["results"]
    total = len(results)
    passed = [r for r in results if r.get("passed")]
    failed = [r for r in results if r.get("passed") is False]
    errored = [r for r in results if r.get("passed") is None]

    def is_empty_out(r):
        return not (r.get("stdout") or "").strip()

    agree_fail = [r for r in passed if is_empty_out(r)]
    genuine = [r for r in passed if not is_empty_out(r)]

    print(f"total:   {total}")
    print(f"passed:  {len(passed)} ({len(passed)/total:.4f})  <- raw rate")
    print(f"failed:  {len(failed)}")
    print(f"errored: {len(errored)}")
    print()
    print(f"genuine passes (non-empty stdout): {len(genuine)} ({len(genuine)/total:.4f})")
    print(f"agreeing-failure passes (both empty): {len(agree_fail)} ({len(agree_fail)/total:.4f})")
    print()
    print("--- sample agreeing-failure passes (model NOT proven right) ---")
    for r in agree_fail[:8]:
        print(f"  {r['task_id']}: q={r['query'][:60]!r} gen={r['generated'][:60]!r}")
    print()
    print("--- sample genuine passes ---")
    for r in genuine[:8]:
        print(f"  {r['task_id']}: q={r['query'][:60]!r} gen={r['generated'][:60]!r}")
    print()
    print("--- sample failures ---")
    for r in failed[:8]:
        print(f"  {r['task_id']}: q={r['query'][:60]!r} gen={r['generated'][:60]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

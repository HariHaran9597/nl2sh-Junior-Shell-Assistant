"""Strict audit: decompose passes into proven-right vs agreeing-failure.

Categories for passed tasks:
  A: non-empty stdout matched            -> certainly right (output proof)
  B: empty stdout+stderr, mutating verb  -> silent mutation agreement (cp/touch/mkdir/rm...). Right if fs truly changed; counted separately
  C: non-empty stderr                    -> both errored identically (agreeing failure). NOT right.
  D: empty stdout+stderr, read-only verb -> no observable effect. Inconclusive.

Fails are fails. Prints the defensible numbers.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MUTATING = {"cp", "mv", "rm", "touch", "mkdir", "rmdir", "ln", "tar", "tee",
            "dd", "chmod", "chown", "truncate", "shred", "unzip", "gzip",
            "sed", "awk", "find", "xargs", "rsync", "git", "ln"}


def first_verb(cmd: str) -> str:
    m = re.match(r"\s*(?:sudo\s+)?([a-zA-Z0-9_.-]+)", cmd or "")
    return m.group(1) if m else ""


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="data/eval_results.json")
    a = ap.parse_args()
    payload = json.loads((ROOT / a.file).read_text(encoding="utf-8"))
    results = payload["results"]
    total = len(results)
    passed = [r for r in results if r.get("passed")]
    failed = [r for r in results if r.get("passed") is False]
    A = [r for r in passed if (r.get("stdout") or "").strip()]
    rest = [r for r in passed if not (r.get("stdout") or "").strip()]
    C = [r for r in rest if (r.get("stderr") or "").strip()]
    rest2 = [r for r in rest if not (r.get("stderr") or "").strip()]
    B = [r for r in rest2 if first_verb(r.get("generated", "")) in MUTATING]
    D = [r for r in rest2 if first_verb(r.get("generated", "")) not in MUTATING]
    print(f"total: {total} | raw pass: {len(passed)} ({len(passed)/total:.4f}) | fail: {len(failed)}")
    print(f"A output-proof passes:      {len(A)} ({len(A)/total:.4f})")
    print(f"B silent-mutation passes:   {len(B)} ({len(B)/total:.4f})")
    print(f"C agreeing-failure passes:  {len(C)} ({len(C)/total:.4f})  <- subtract these")
    print(f"D inconclusive (no effect): {len(D)} ({len(D)/total:.4f})")
    strict = len(A) + len(B)
    print(f"STRICT (A+B): {strict} ({strict/total:.4f})")
    honest = len(A) + len(B)
    print(f"RAW minus agreeing-failures: {(len(passed)-len(C))} ({(len(passed)-len(C))/total:.4f})")
    print("\n--- sample C (agreeing failures, model wrong but scored pass) ---")
    for r in C[:6]:
        print(f"  {r['task_id']}: gen={r['generated'][:70]!r} err={(r.get('stderr') or '')[:70]!r}")
    print("\n--- sample B (silent mutations) ---")
    for r in B[:6]:
        print(f"  {r['task_id']}: gen={r['generated'][:70]!r}")
    print("\n--- sample D (inconclusive) ---")
    for r in D[:6]:
        print(f"  {r['task_id']}: q={r['query'][:55]!r} gen={r['generated'][:60]!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

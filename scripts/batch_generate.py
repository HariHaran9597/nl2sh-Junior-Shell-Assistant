"""Batch-generate commands for benchmark cases (resumable, stateless).

Usage: python scripts/batch_generate.py [--limit N] [--base-url URL] [--model NAME]
Writes data/generated.jsonl incrementally (flush per row) so interrupts lose nothing.
Re-running skips task_ids already present. Uses plain queries (NO context injection)
to match single-turn benchmark conditions.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from cli.nl2sh import LlamaCppBackend, _extract  # noqa: E402

CASES = ROOT / "data" / "benchmark_cases.jsonl"
OUT = ROOT / "data" / "generated.jsonl"


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="nl2sh")
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--out", default=str(OUT),
                    help="output JSONL (resumable by task_id)")
    a = ap.parse_args()
    OUT = Path(a.out)

    cases = [json.loads(l) for l in CASES.open(encoding="utf-8") if l.strip()]
    if a.limit > 0:
        cases = cases[: a.limit]
    done = set()
    if OUT.exists():
        for l in OUT.open(encoding="utf-8"):
            if l.strip():
                try:
                    done.add(json.loads(l)["task_id"])
                except Exception:
                    pass
    todo = [c for c in cases if c["task_id"] not in done]
    print(f"cases: {len(cases)}, done: {len(done)}, todo: {len(todo)}")
    if not todo:
        print("nothing to do")
        return 0

    backend = LlamaCppBackend(a.base_url, a.model, a.timeout)
    t0 = time.time()
    with OUT.open("a", encoding="utf-8") as f:
        for i, c in enumerate(todo, 1):
            try:
                raw = backend.generate(c["query"])
                cmd = _extract(raw)
            except Exception as e:
                cmd = ""
                print(f"[{i}/{len(todo)}] ERROR {c['task_id']}: {str(e)[:120]}")
            f.write(json.dumps({"task_id": c["task_id"], "query": c["query"],
                                "reference": c["reference"], "generated": cmd},
                               ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0 or i == len(todo):
                el = time.time() - t0
                print(f"[{i}/{len(todo)}] {c['task_id']} -> {cmd[:70]} "
                      f"({el / 60:.1f} min elapsed)")
    print(f"finished {len(todo)} in {(time.time() - t0) / 60:.1f} min -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

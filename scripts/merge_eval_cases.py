"""Join generated commands with benchmark metadata (fixture/difficulty/reference2).

Usage: python scripts/merge_eval_cases.py
Reads data/generated.jsonl + data/benchmark_cases.jsonl (by task_id),
writes data/eval_cases.jsonl for the evaluator.
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--generated", default="data/generated.jsonl")
    ap.add_argument("--out", default="data/eval_cases.jsonl")
    a = ap.parse_args()
    bench = {}
    for l in (ROOT / "data" / "benchmark_cases.jsonl").open(encoding="utf-8"):
        if l.strip():
            c = json.loads(l)
            bench[c["task_id"]] = c
    out = []
    for l in (ROOT / a.generated).open(encoding="utf-8"):
        if not l.strip():
            continue
        g = json.loads(l)
        b = bench.get(g["task_id"], {})
        g["reference2"] = b.get("reference2", "")
        g["difficulty"] = b.get("difficulty", 0)
        g["fixture"] = b.get("fixture", "fs4")
        out.append(g)
    with (ROOT / a.out).open("w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"wrote {len(out)} to {a.out}")
    print("fixtures:", dict(Counter(o["fixture"] for o in out)))


if __name__ == "__main__":
    main()

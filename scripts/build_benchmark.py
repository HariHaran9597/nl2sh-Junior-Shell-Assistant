"""Build benchmark_cases.jsonl from the official InterCode-ALFA test split.

Includes per-task fixture family (fs1..fs5, mapped via the official
nl2bash_fs_*.json assets), difficulty, and the alternative reference
(command is correct if it matches bash OR bash2).

Fixture families (see InterCode-ALFA main.py::index_to_img):
  fs1 (153): /testbed rich tree      fs2 (49): /system
  fs3 (57):  /workspace + /backup    fs4 (23): no fixtures (dummy env)
  fs5 (18):  /testbed small tree

5 HF rows differ slightly from the fs assets; overrides below assign them
by path reference (verified by hand).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "benchmark_cases.jsonl"
ASSETS = ROOT / "data" / "raw" / "icalfa_pkg" / "src" / "icalfa" / "assets" / "datasets"

# (query, gold) differs from fs assets for these 5; fixture assigned by path
OVERRIDES = {
    "alfa-38": "fs4",   # echo -n 'hello' | base64 (no paths; empty env is safe)
    "alfa-100": "fs1",  # reads setup_nl2b_fs_1.sh (fs1 image root)
    "alfa-150": "fs1",  # /testbed/.../FooBar (only in fs1 tree)
    "alfa-190": "fs2",  # /system/...
    "alfa-284": "fs1",  # /testbed/dir1/textfile1.txt (same content in fs1/fs5; fs1 chosen)
}


def main():
    from datasets import load_dataset

    pair2fx = {}
    for i in range(1, 6):
        rows = json.loads((ASSETS / f"nl2bash_fs_{i}.json").read_text(encoding="utf-8"))
        for r in rows:
            pair2fx[(r["query"], r["gold"])] = f"fs{i}"

    ds = load_dataset("westenfelder/NL2SH-ALFA", "test", split="train")
    cases = []
    for i, ex in enumerate(ds):
        tid = f"alfa-{i}"
        fx = pair2fx.get((ex["nl"], ex["bash"]), OVERRIDES.get(tid))
        if fx is None:
            raise SystemExit(f"no fixture for {tid}: {ex['nl'][:60]!r}")
        cases.append({
            "task_id": tid,
            "query": ex["nl"],
            "reference": ex["bash"],
            "reference2": ex.get("bash2") or "",
            "difficulty": int(ex.get("difficulty", 0)),
            "fixture": fx,
        })
    with OUT.open("w", encoding="utf-8") as f:
        for c in cases:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    from collections import Counter
    print(f"wrote {len(cases)} to {OUT}")
    print("fixtures:", dict(Counter(c["fixture"] for c in cases)))
    print("difficulty:", dict(Counter(c["difficulty"] for c in cases)))


if __name__ == "__main__":
    main()

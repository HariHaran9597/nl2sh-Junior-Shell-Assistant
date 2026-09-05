"""Builds the Kaggle CPU scoring bundle:
  - kaggle_eval_bundle.zip  -> upload as a Kaggle DATASET (as-is, zipped)
  - notebooks/kaggle_score_base.ipynb -> File -> Import Notebook on Kaggle

On Kaggle (CPU only, NO gpu, NO pip installs, NO tokens): unzips, finishes
the remaining base-model tasks (resume from 140 included), writes
/kaggle/working/eval_results_base.json for download.
"""
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BUNDLE_FILES = [
    "src/evaluate.py",
    "src/__init__.py",
    "data/eval_cases_base.jsonl",
    "data/eval_partial_base.jsonl",
    "data/fixtures/setup_nl2b_fs_1.sh",
    "data/fixtures/setup_nl2b_fs_2.sh",
    "data/fixtures/setup_nl2b_fs_3.sh",
    "data/fixtures/setup_nl2b_fs_5.sh",
]
ZIP_OUT = ROOT / "kaggle_eval_bundle.zip"
NB_OUT = ROOT / "notebooks" / "kaggle_score_base.ipynb"

C0 = """# nl2sh+ base-model scoring on Kaggle CPU (~30-40 min, unattended)

Finishes the untuned-base 300-task benchmark. **CPU ONLY** - do NOT turn on
GPU (this step runs shell commands, not the model; GPU would just burn quota).
No pip installs, no tokens needed.

Setup (once):
1. Kaggle -> Datasets -> New Dataset -> upload `kaggle_eval_bundle.zip` (as-is) -> Create (any title, e.g. `nl2sh-eval`)
2. This notebook -> Add-ons -> Add Input -> your `nl2sh-eval` dataset -> attach
3. Run All. When done: Save Version -> download `eval_results_base.json` from Output.
"""

C1 = """import glob, shutil, subprocess
from pathlib import Path

hits = glob.glob("/kaggle/input/*/src/evaluate.py")
zips = glob.glob("/kaggle/input/*/*.zip")
W = Path("/kaggle/working/nl2sh")
if hits:
    src = Path(hits[0]).parent.parent
    print("dataset files found at:", src)
    shutil.rmtree(W, ignore_errors=True)
    shutil.copytree(src, W)
else:
    assert zips, "Attach the nl2sh-eval dataset (Add-ons -> Add Input) and re-run this cell."
    print("unzipping:", zips[0])
    shutil.rmtree(W, ignore_errors=True)
    W.mkdir(parents=True)
    subprocess.run(["unzip", "-o", zips[0], "-d", str(W)], check=True)
    print("unzipped to:", W)

need = ["src/evaluate.py", "data/eval_cases_base.jsonl", "data/eval_partial_base.jsonl",
        "data/fixtures/setup_nl2b_fs_1.sh", "data/fixtures/setup_nl2b_fs_5.sh"]
for f in need:
    assert (W / f).exists(), f"MISSING {f} - re-upload the bundle zip"
    print("ok:", f)
"""

C2 = """import sys
sys.path.insert(0, "/kaggle/working/nl2sh")
import json
from pathlib import Path
from src.evaluate import _find_bash, _msys_root, FIXTURES

W = Path("/kaggle/working/nl2sh")
cases = [json.loads(l) for l in open(W / "data/eval_cases_base.jsonl") if l.strip()]
prior = [json.loads(l) for l in open(W / "data/eval_partial_base.jsonl") if l.strip()]
print("cases:", len(cases), "| already scored:", len(prior), "| to go:", len(cases) - len(prior))
print("bash:", _find_bash(), "| root:", _msys_root())
import subprocess
subprocess.run(["bash", "-c", "mkdir -p /testbed && touch /testbed/wtest && rm /testbed/wtest && echo FIXTURES_WRITABLE"], check=True)
print("ALL CHECKS PASSED - run the next cell (the long one).")
"""

C3 = """import subprocess, sys
# Scores the remaining tasks (~30-40 min). Resume is built in: re-running
# this cell after any interruption continues where it stopped.
r = subprocess.run(
    [sys.executable, "-m", "src.evaluate",
     "--cases", "data/eval_cases_base.jsonl",
     "--mode", "self", "--no-docker",
     "--out", "/kaggle/working/eval_results_base.json",
     "--partial", "data/eval_partial_base.jsonl"],
    cwd="/kaggle/working/nl2sh")
print("exit:", r.returncode)
"""

C4 = """import json
from collections import Counter, defaultdict
p = json.load(open("/kaggle/working/eval_results_base.json"))
res = p["results"]
scored = [r for r in res if r.get("passed") is not None]
ok = [r for r in scored if r.get("passed")]
print(f\"TOTAL: {len(res)} scored={len(scored)} passed={len(ok)} RAW={len(ok)/len(scored):.4f}\")
by = defaultdict(list)
for r in scored:
    by[r.get("difficulty", 0)].append(1 if r.get("passed") else 0)
for d in sorted(by):
    print(f\"  level {d}: {sum(by[d])}/{len(by[d])} = {sum(by[d])/len(by[d]):.4f}\")
agree = [r for r in ok if not (r.get('stdout') or '').strip() and (r.get('stderr') or '').strip()]
print(f"agreeing-failure passes (both errored): {len(agree)}")
print("gradeable-only rate:", f"{(len(ok)-len(agree))/(len(scored)-len(agree)):.4f}")
"""

C5 = """### Done - download this file

Kaggle top bar -> **Save Version** -> Save -> Output panel -> download
**`eval_results_base.json`** -> send it back (this is the untuned-base score;
the fine-tune delta = fine-tune score − this score).
"""


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "source": src,
            "execution_count": None, "outputs": []}


cells = [md(C0), code(C1), code(C2), code(C3), code(C4), md(C5)]
nb = {
    "nbformat": 4, "nbformat_minor": 4,
    "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    "cells": cells,
}
NB_OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")

import ast
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        ast.parse("\n".join(l for l in c["source"].splitlines() if not l.strip().startswith("!")))

with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as z:
    for f in BUNDLE_FILES:
        p = ROOT / f
        assert p.exists(), f"missing {f}"
        z.write(p, f)
names = zipfile.ZipFile(ZIP_OUT).namelist()

full = "\n".join(c["source"] for c in cells)
checks = {
    "no gpu needed note": "CPU ONLY" in full,
    "dataset attach guide": "Add Input" in full,
    "resume-aware run": "--partial" in full,
    "honest audit math": "agreeing-failure" in full,
    "difficulty split": "level {" in full,
    "bundle has evaluate.py": "src/evaluate.py" in names,
    "bundle has cases": "data/eval_cases_base.jsonl" in names,
    "bundle has partial(resume)": "data/eval_partial_base.jsonl" in names,
    "bundle has 4 fixtures": sum(1 for n in names if "setup_nl2b_fs_" in n) == 4,
}
bad = [k for k, v in checks.items() if not v]
for k, v in checks.items():
    print(("PASS" if v else "FAIL"), "-", k)
assert not bad, bad
print(f"wrote {ZIP_OUT} ({ZIP_OUT.stat().st_size//1024} KB) + {NB_OUT}")
print("ALL BUNDLE CHECKS PASSED")

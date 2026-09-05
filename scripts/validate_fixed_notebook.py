"""Validate nl2sh_FIXED.ipynb: valid JSON, all code cells compile, all fixes present."""
import ast
import json
from pathlib import Path

NB = Path(__file__).resolve().parent.parent / "notebooks" / "nl2sh_FIXED.ipynb"

nb = json.loads(NB.read_text(encoding="utf-8"))
assert nb["nbformat"] == 4, "bad nbformat"
cells = nb["cells"]
print("cells:", len(cells))

for i, c in enumerate(cells):
    assert c["cell_type"] in ("markdown", "code"), i
    if c["cell_type"] == "code":
        lines = [ln for ln in c["source"].splitlines() if not ln.strip().startswith("!")]
        ast.parse("\n".join(lines))
        print(f"cell {i} compiles OK ({len(c['source'])} chars)")

full = "\n".join(c["source"] for c in cells)
checks = {
    "seq 512 (model + trainer)": "max_seq_length=512" in full and "max_length=512" in full,
    "1 epoch": "num_train_epochs=1" in full,
    "push_to_hub": "push_to_hub=True" in full,
    "hub checkpoint strategy": "hub_strategy" in full and "checkpoint" in full,
    "hub resume via snapshot_download": "snapshot_download" in full,
    "backup pruning": "pruned" in full,
    "assert adapter exists": "assert adapters" in full,
    "save_steps 500": "save_steps=500" in full,
    "save_total_limit 2": "save_total_limit=2" in full,
    "smoke test ask()": "def ask(q)" in full,
    "gguf guarded try/except": "GGUF export skipped" in full,
    "warmup 100": "warmup_steps=100" in full,
    "batch 2x4": "per_device_train_batch_size=2" in full and "gradient_accumulation_steps=4" in full,
    "r16 alpha16": "r=16" in full and "lora_alpha=16" in full,
    "dynamic hub username (whoami)": "whoami" in full,
    "local-only fallback on hub failure": "LOCAL-ONLY" in full,
    "recovery cell restores tar": "extractall" in full and "READY - re-run the TRAIN cell" in full,
    "loud local-only warning": "LOCAL-ONLY MODE" in full,
}
failed = [k for k, v in checks.items() if not v]
for k, v in checks.items():
    print(("PASS" if v else "FAIL"), "-", k)
assert not failed, f"missing fixes: {failed}"
print("ALL CHECKS PASSED")

import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.risk import classify, render_risk
def test_low(): assert classify("ls -la").level=="LOW"
def test_med(): assert classify("find . -name '*.log' -delete").level=="MED"
def test_high(): assert classify("rm -rf /").level=="HIGH"
def test_explain(): assert "Explain" in render_risk("tar -czf a.tar.gz .")
def test_dry_run(): assert classify("find . -name '*.log' -delete").dry_run_cmd is not None
def test_golden_200_none_low():
    # all 200 golden-dangerous commands must trip HIGH or MED; none may pass as LOW.
    # (39 are find-delete variants, correctly MED by dataset design.)
    p = Path(__file__).resolve().parent.parent / "data" / "processed" / "golden_dangerous.jsonl"
    rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    assert len(rows) == 200
    levels = [classify(r["output"]).level for r in rows]
    assert "LOW" not in levels, [r for r in rows if classify(r["output"]).level == "LOW"][:3]
    assert set(levels) <= {"HIGH", "MED"}

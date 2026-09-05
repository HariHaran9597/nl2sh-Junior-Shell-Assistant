"""Tests for benchmark fixtures (official InterCode-ALFA testbeds)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import FIXTURES, _abs_root, _prep_fixtures, _roots_for, run_eval


def _cases():
    rows = [json.loads(l) for l in
            open(Path(__file__).resolve().parent.parent / "data" / "benchmark_cases.jsonl",
                 encoding="utf-8") if l.strip()]
    return rows


def test_all_cases_mapped():
    rows = _cases()
    assert len(rows) == 300
    for r in rows:
        assert r["fixture"] in FIXTURES, r["task_id"]
        assert r["difficulty"] in (0, 1, 2)
        assert "reference2" in r


def test_roots_known():
    assert _roots_for("fs1") == ["/testbed"]
    assert _roots_for("fs2") == ["/system"]
    assert _roots_for("fs3") == ["/workspace", "/backup"]
    assert _roots_for("fs4") == []
    assert _roots_for("unknown") == []


def test_fs5_setup_builds_tree(tmp_path):
    # smallest tree; runs official setup script, then cleans the global root
    _prep_fixtures("fs5", tmp_path)
    marker = _abs_root("/testbed")
    assert marker is not None
    assert (marker / "dir1" / "textfile1.txt").exists()
    assert (marker / "dir1" / "textfile1.txt").read_text().strip() == "Hello, World!"
    # setup script must also be visible (cwd + root mirrors of Docker COPY)
    assert (tmp_path / "setup_nl2b_fs_5.sh").exists()
    assert (marker.parent / "setup_nl2b_fs_5.sh").exists()
    # cleanup global state
    import shutil
    shutil.rmtree(marker, ignore_errors=True)
    (marker.parent / "setup_nl2b_fs_5.sh").unlink(missing_ok=True)


def test_touch_delta_passes():
    # `touch` changes only mtime (differs across runs) with identical content:
    # must PASS via content-compared deltas.
    cases = [{"task_id": "t-touch", "query": "t", "generated": "touch newfile123",
              "reference": "touch newfile123", "difficulty": 0, "fixture": "fs4"}]
    res = run_eval(cases, mode="self", use_docker=False)
    assert res[0].passed is True, res[0].to_dict()

def test_content_diff_fails():
    cases = [{"task_id": "t-diff", "query": "t", "generated": "echo a > f123",
              "reference": "echo b > f123", "difficulty": 0, "fixture": "fs4"}]
    res = run_eval(cases, mode="self", use_docker=False)
    assert res[0].passed is False

def test_pristine_noise_ignored():
    # same command twice in a row sees different pristine states (timestamps);
    # per-run deltas must still agree -> PASS
    import time
    cases = [{"task_id": "t-n1", "query": "t", "generated": "echo x > n1_f",
              "reference": "echo x > n1_f", "difficulty": 0, "fixture": "fs1"}]
    time.sleep(1.1)  # force different mtimes across the two runs
    res = run_eval(cases, mode="self", use_docker=False)
    assert res[0].passed is True, res[0].to_dict()

def test_templates_preserve_content_and_mtime():
    import shutil
    import src.evaluate as ev
    ev._build_templates()
    assert "fs5" in ev._TEMPLATES and ev._TEMPLATES["fs5"]["roots"]
    # reset twice from template; marker content identical, mtimes preserved
    import tempfile
    for _ in range(2):
        with tempfile.TemporaryDirectory() as tmp:
            ev._prep_fixtures("fs5", Path(tmp))
            f = ev._abs_root("/testbed") / "dir1" / "textfile1.txt"
            assert f.read_text().strip() == "Hello, World!"

def test_ref2_accepted():
    # gen matches the ALTERNATIVE reference only -> PASS via reference2
    cases = [{"task_id": "t-r2", "query": "hi", "generated": "echo hi",
              "reference": "printf 'bye'", "reference2": "echo hi",
              "difficulty": 0, "fixture": "fs4"}]
    res = run_eval(cases, mode="self", use_docker=False)
    assert res[0].passed is True
    assert res[0].matched == "reference2"


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))

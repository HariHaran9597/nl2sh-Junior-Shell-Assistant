import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.evaluate import _norm, score_expected
import time
import src.evaluate as ev
def test_norm(): assert _norm("  ls  -la  \n\n")=="ls -la"
def test_score(): assert score_expected("3 data.txt","3 data.txt")
def test_score_mismatch(): assert not score_expected("4","3")
def test_eval_records_stdout():
    # regression: run_eval once silently dropped stdout (tuple typo),
    # turning scoring into fs-only comparison. This must stay green.
    cases = [{"task_id": "t-hi", "query": "print hi", "generated": "echo hi",
              "expected": "hi"}]
    res = ev.run_eval(cases, mode="expected", use_docker=False)
    assert len(res) == 1
    assert res[0].stdout.strip() == "hi", repr(res[0].stdout)
    assert res[0].passed is True

def test_hang_returns_timeout_fast():
    # regression: orphaned grandchildren (ping/netstat style) must not wedge the runner.
    # sleep 20 with a 3s timeout must return 124 in well under 20s.
    old = ev.TIMEOUT
    ev.TIMEOUT = 3
    try:
        t0 = time.time()
        code, so, se, snap = ev.run_local("sleep 20")
        dt = time.time() - t0
    finally:
        ev.TIMEOUT = old
    assert code == 124, (code, so, se)
    assert se == "TIMEOUT"
    assert dt < 15, f"took {dt:.1f}s - grandchildren wedged communicate()"

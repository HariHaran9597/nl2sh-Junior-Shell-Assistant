import sys, tempfile, os
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src import context

def test_build_prompt_no_history(monkeypatch, tmp_path):
    monkeypatch.setattr(context, "HIST_PATH", tmp_path/"history.json")
    assert context.build_prompt("show large files")=="show large files"
    context.record("/tmp","tar -xzf app.tar.gz -C /tmp")
    context.record("/tmp","ls /tmp")
    p=context.build_prompt("now show large files")
    assert "- tar -xzf app.tar.gz -C /tmp\n- ls /tmp" in p  # live history as bullets, not echoable list
    assert "do NOT repeat" in p  # anti-echo instruction
    assert "Working directory:" in p and "/tmp" in p
    assert "Request: now show large files" in p
    assert "Example:" in p  # 1-shot wrapper so the fine-tune follows the format

def test_history_capped(monkeypatch, tmp_path):
    monkeypatch.setattr(context, "HIST_PATH", tmp_path/"h.json")
    for i in range(10): context.record("/tmp", f"cmd {i}")
    _, hist = context.get_context()
    assert len(hist)==5
    assert hist[-1]=="cmd 9"

def test_windows_pwd_is_bash_compatible():
    assert context.shell_path(r"D:\\Model_finetuing") == "/d/Model_finetuing"
    assert context.shell_path("/tmp/work") == "/tmp/work"

def test_windows_pwd_is_normalized_in_prompt():
    p = context.build_prompt("show files", pwd=r"D:\\work\\project", history=["pwd"])
    assert "Working directory: /d/work/project" in p
    assert r"D:\\work\\project" not in p

def test_explicit_pwd_is_not_overwritten(monkeypatch, tmp_path):
    monkeypatch.setattr(context, "HIST_PATH", tmp_path/"history.json")
    context.record("/stored", "ls")
    p = context.build_prompt("next", pwd="/explicit", history=None)
    assert "Working directory: /explicit" in p

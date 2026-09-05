"""
Context-aware prompt injection – the first differentiator.

Saves pwd + last 5 commands to ~/.nl2sh/history.json and injects them
into the prompt *without retraining*. This beats the original single-turn,
stateless design.

Previous commands (context only, do NOT repeat them):
- tar -xzf app.tar.gz -C /tmp
- ls /tmp
Working directory: /tmp
Request: now show large files
Command: find /tmp -type f -exec du -h {} + | sort -rh | head
"""
from __future__ import annotations
import json
import os
from pathlib import Path
from datetime import datetime
import re

HIST_PATH = Path.home() / ".nl2sh" / "history.json"
MAX_HISTORY = 5

def _load() -> dict:
    if HIST_PATH.exists():
        try: return json.loads(HIST_PATH.read_text(encoding="utf-8"))
        except Exception: pass
    return {"history": [], "pwd": "", "updated": ""}

def _save(data: dict):
    HIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    data["updated"] = datetime.now().isoformat()
    HIST_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    try: HIST_PATH.chmod(0o600)
    except Exception: pass

def record(pwd: str, command: str):
    d = _load()
    d["pwd"] = pwd
    hist = d.get("history", [])
    hist.append(command)
    d["history"] = hist[-MAX_HISTORY:]
    _save(d)

def get_context() -> tuple[str, list[str]]:
    d = _load()
    return d.get("pwd",""), d.get("history",[])

def shell_path(pwd: str) -> str:
    """Return a Bash-friendly working directory for model context.

    The CLI can run on Windows while the generated command targets Bash
    (Git Bash, WSL, or a Linux server). Never place a raw ``D:\\...`` path in
    a Bash prompt; use Git Bash's ``/d/...`` form instead.
    """
    value = (pwd or "").strip()
    if re.match(r"^[A-Za-z]:[\\/]", value):
        value = value.replace("\\", "/")
        return "/" + value[0].lower() + re.sub(r"/+", "/", value[2:])
    if value.startswith("\\\\"):
        return value.replace("\\", "/")
    return re.sub(r"/+", "/", value.replace("\\", "/"))

# One fixed example teaches the format. The base fine-tune never saw ANY
# context template (single-turn ChatML only), and live testing showed the
# model echoes a `History: [...]` inline list back as its answer. Bullet
# format + explicit do-not-repeat + a matching 1-shot example fixes most of
# it; cli/nl2sh.py::_is_echo catches the rest (regen once without context).
FEWSHOT_EXAMPLE = (
    "Example:\n"
    "Previous commands (context only, do NOT repeat them):\n"
    "- tar -xzf app.tar.gz -C /tmp\n"
    "Working directory: /tmp\n"
    "Request: now show large files\n"
    "Command: find /tmp -type f -exec du -h {} + | sort -rh | head\n"
    "Now answer:\n"
)

def build_prompt(user_query: str, pwd: str | None=None, history: list[str] | None=None) -> str:
    """Return the context-injected prompt string to feed the model."""
    stored_pwd, stored_history = get_context()
    if pwd is None:
        pwd = stored_pwd or os.getcwd()
    if history is None:
        history = stored_history
    if not history:
        return user_query
    pwd = shell_path(pwd)
    lines = "\n".join(f"- {h}" for h in history[-MAX_HISTORY:])
    return (FEWSHOT_EXAMPLE +
            "Previous commands (context only, do NOT repeat them):\n"
            f"{lines}\nWorking directory: {pwd}\nRequest: {user_query}\nCommand:")

def clear():
    if HIST_PATH.exists(): HIST_PATH.unlink()

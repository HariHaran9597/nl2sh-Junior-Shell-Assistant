"""
MCP server – third differentiator. 80 lines.
Claude Desktop / AgentDK can call get_shell_command(natural_language, pwd, history).

Add to mcp.json:
{
  "mcpServers": {
    "nl2sh": {"command":"python","args":["D:/Model_finetuing/mcp_server/server.py"],"env":{}}
  }
}
"""
from __future__ import annotations
import os, sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    from mcp.server.fastmcp import FastMCP  # mcp<2
except ImportError:
    from mcp.server.mcpserver import MCPServer as FastMCP  # mcp>=2
from src.context import build_prompt
from src.risk import classify
from src.safety import check, Level

mcp = FastMCP("nl2sh+")

SYSTEM = ("You translate natural language into exactly one bash one-liner. Output only the command. "
"If previous commands are shown for context, NEVER repeat, continue, or combine them - they are background only. "
"Output ONLY the new command for the current request. No explanations, no lists, no Windows paths.")

def _generate(prompt: str, base_url: str|None=None, model: str|None=None) -> str:
    # Prefer local llama.cpp / Ollama; fallback mock
    base_url = (base_url or os.environ.get("NL2SH_BASE_URL","http://127.0.0.1:8080/v1")).rstrip("/")
    if base_url.endswith("/v1"): base_url = base_url[:-3]  # avoid /v1/v1/... 404
    model = model or os.environ.get("NL2SH_MODEL","nl2sh")
    try:
        import json, urllib.request
        payload={"model":model,"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],"temperature":0.0,"max_tokens":64}
        req=urllib.request.Request(f"{base_url}/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            data=json.loads(r.read().decode()); return data["choices"][0]["message"]["content"].strip()
    except Exception:
        # mock fallback – useful for testing without model server
        tbl={"large files":"find . -type f -exec du -h {} + | sort -rh | head","compress":"tar -czf archive.tar.gz .","extract":"tar -xzf archive.tar.gz -C /tmp"}
        pl=prompt.lower()
        for k,v in tbl.items():
            if k in pl: return v
        return "ls -la"

@mcp.tool()
def get_shell_command(natural_language: str, pwd: str = "", history: list[str] | None = None) -> dict:
    """Translate natural_language to a bash command with risk, explain, dry-run.
    Provide pwd and last commands for context-aware generation."""
    history = history or []
    prompt = build_prompt(natural_language, pwd=pwd or os.getcwd(), history=history)
    raw = _generate(prompt)
    # strip fences
    cmd = raw.strip().strip("`")
    if cmd.startswith("bash\n"): cmd=cmd[5:]
    cmd=cmd.splitlines()[0].strip().lstrip("$ ").strip()
    lvl, findings = check(cmd)
    risk = classify(cmd)
    # safety overrides risk HIGH if blocklist fires
    if lvl==Level.DANGER: risk.level="HIGH"
    return {
        "command": cmd,
        "risk": risk.level,
        "risk_reason": risk.reason,
        "explain": risk.explain,
        "dry_run": risk.dry_run_cmd,
        "safety": [{"level":f.level.value,"rule":f.rule,"detail":f.detail} for f in findings],
        "context_used": {"pwd": pwd or os.getcwd(), "history": history[-5:]},
    }

@mcp.tool()
def explain_command(command: str) -> dict:
    """Explain flags and risk for an existing command."""
    r = classify(command)
    lvl, finds = check(command)
    if lvl == Level.DANGER:
        r.level = "HIGH"
        r.reason = "safety policy matched a dangerous pattern"
    return {"command":command,"risk":r.level,"explain":r.explain,"dry_run":r.dry_run_cmd,"safety":[{"level":f.level.value,"rule":f.rule} for f in finds]}

if __name__ == "__main__":
    mcp.run()

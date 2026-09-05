"""
Gradio Demo – HF Space deploy (Day 10)
Run: python app.py   -> http://127.0.0.1:7860
Deploy: huggingface-cli upload HariHaran9597/nl2sh-demo app.py
"""
import os, json, urllib.request
import gradio as gr
from src.risk import classify
from src.safety import check, Level
from src.context import build_prompt

SYSTEM=("You translate natural language into exactly one bash one-liner. Output only the command. "
"If previous commands are shown for context, NEVER repeat, continue, or combine them - they are background only. "
"Output ONLY the new command for the current request. No explanations, no lists, no Windows paths.")

def _gen(prompt):
    base=os.environ.get("NL2SH_BASE_URL","http://127.0.0.1:8080/v1").rstrip("/")
    if base.endswith("/v1"): base=base[:-3]  # avoid /v1/v1/... 404
    model=os.environ.get("NL2SH_MODEL","nl2sh")
    try:
        payload={"model":model,"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],"temperature":0.0,"max_tokens":64}
        req=urllib.request.Request(f"{base}/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type":"application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())["choices"][0]["message"]["content"].strip()
    except Exception as e:
        # demo fallback
        tbl={"large files":"find /tmp -type f -exec du -h {} + | sort -rh | head","compress":"tar -czf archive.tar.gz .","extract":"tar -xzf archive.tar.gz -C /tmp","disk usage":"du -sh *","delete":"find . -name '*.log' -print"}
        pl=prompt.lower()
        for k,v in tbl.items():
            if k in pl: return v
        return "ls -la  # model server unavailable; preview only"

def infer(nl, pwd, hist_str):
    hist=[h.strip() for h in hist_str.split(",") if h.strip()] if hist_str else []
    pwd=pwd or "/tmp"
    prompt=build_prompt(nl, pwd=pwd, history=hist)
    raw=_gen(prompt); cmd=raw.strip().strip("`")
    if cmd.startswith("bash\n"): cmd=cmd[5:]
    cmd=cmd.splitlines()[0].strip().lstrip("$ ")
    r=classify(cmd); lvl,_=check(cmd)
    # safety badge
    badge="HIGH" if lvl==Level.DANGER else r.level
    color={"LOW":"green","MED":"orange","HIGH":"red"}[badge]
    risk_html=f"<span style='background:{color};color:white;padding:4px 10px;border-radius:999px'>{badge}</span>"
    explain=r.explain
    dry=r.dry_run_cmd or "(no dry-run for LOW)"
    return cmd, risk_html, explain, dry, prompt

with gr.Blocks(title="nl2sh+ Junior Shell Assistant") as demo:
    gr.Markdown("# nl2sh+ — Qwen2.5-Coder-1.5B NL→Bash | Context + advisory risk + MCP\nAsk in plain English, get a bash one-liner with risk badge + explain + dry-run.")
    with gr.Row():
        nl=gr.Textbox(label="Natural language", placeholder="extract tar.gz to /tmp, now show large files", lines=2)
        pwd=gr.Textbox(label="Pwd (context)", value="/tmp")
    hist=gr.Textbox(label="History (comma-separated last 5)", placeholder="tar -xzf app.tar.gz -C /tmp, ls /tmp")
    btn=gr.Button("Translate →")
    cmd=gr.Code(label="Command", language="shell")
    risk=gr.HTML(label="Risk")
    explain=gr.Textbox(label="Explain (flag meanings)")
    dry=gr.Textbox(label="Dry-run")
    prompt_dbg=gr.Textbox(label="Prompt sent to model (debug)", lines=3)
    btn.click(infer, inputs=[nl,pwd,hist], outputs=[cmd,risk,explain,dry,prompt_dbg])
    gr.Examples([["extract tar.gz to /tmp","/tmp",""], ["now show large files","/tmp","tar -xzf app.tar.gz -C /tmp, ls /tmp"], ["delete all logs","/var/log",""]], inputs=[nl,pwd,hist])

if __name__=="__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT","7860")))

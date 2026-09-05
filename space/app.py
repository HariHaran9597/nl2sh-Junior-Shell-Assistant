"""nl2sh+ HF Space - Gradio demo served by the real fine-tuned GGUF on CPU.

Runs on free cpu-basic (2 vCPU, 16 GB): Qwen2.5-Coder-1.5B Q4_K_M (~1 GB).
No chat-template dependence: prompts are formatted as raw ChatML (the exact
training format) with stop=["<|im_end|>"].
"""
import os
from functools import lru_cache

import gradio as gr
from huggingface_hub import snapshot_download
from llama_cpp import Llama

try:
    import spaces
except ImportError:
    # Keep local CPU runs working while allowing ZeroGPU to detect the
    # decorated Gradio handler when the `spaces` package is present.
    class _LocalSpaces:
        @staticmethod
        def GPU(*args, **kwargs):
            if args and callable(args[0]) and len(args) == 1 and not kwargs:
                return args[0]
            return lambda function: function
    spaces = _LocalSpaces()

import risk
import safety
from context import build_prompt

# The model is intentionally CPU-only.  A free Hugging Face Space may still
# need one decorated function to pass ZeroGPU's startup check, but wrapping the
# real handler would route every request through a GPU worker and abort this
# CPU workload.  Keep this probe unused; it consumes no GPU quota.
@spaces.GPU(duration=1)
def _zerogpu_hosting_probe():
    return None

REPO = os.environ.get("NL2SH_GGUF_REPO", "justhariharan/nl2sh-1.5b-Q4_K_M-GGUF")
FILE = os.environ.get("NL2SH_GGUF_FILE", "nl2sh-1.5b.Q4_K_M.gguf")
REVISION = os.environ.get("NL2SH_GGUF_REVISION", "main")
CACHE_DIR = os.environ.get(
    "NL2SH_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "nl2sh"),
)
CHATML = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"

@lru_cache(maxsize=1)
def _load_model():
    """Load the pinned public artifact once, on the first user request."""
    print(f"downloading GGUF {REPO}@{REVISION} (cache={CACHE_DIR})...")
    model_dir = snapshot_download(
        repo_id=REPO,
        revision=REVISION,
        allow_patterns=[FILE],
        cache_dir=CACHE_DIR,
    )
    print("loading model on CPU...")
    model = Llama(model_path=os.path.join(model_dir, FILE), n_ctx=512,
                  n_threads=min(4, os.cpu_count() or 1), verbose=False)
    print("ready.")
    return model


def generate(prompt: str) -> str:
    formatted = CHATML.format(instruction=prompt, output="")[:-len("<|im_end|>")]
    out = _load_model().create_completion(formatted, max_tokens=64, temperature=0.0,
                                stop=["<|im_end|>"])
    text = out["choices"][0]["text"].strip()
    if text.startswith("```"):
        text = text.strip("`")
    return text.splitlines()[0].strip().lstrip("$ ").strip()


def infer(nl, pwd, hist_str):
    hist = [h.strip() for h in hist_str.split(",") if h.strip()] if hist_str else []
    pwd = pwd or "/tmp"
    prompt = build_prompt(nl, pwd=pwd, history=hist)
    # Send the full context block. The model was trained on single-turn data,
    # so the context is framed as user content and remains advisory only.
    cmd = generate(prompt or nl)
    r = risk.classify(cmd)
    lvl, _ = safety.check(cmd)
    badge = "HIGH" if lvl == safety.Level.DANGER else r.level
    color = {"LOW": "green", "MED": "orange", "HIGH": "red"}[badge]
    risk_html = f"<span style='background:{color};color:white;padding:4px 10px;border-radius:999px'>{badge}</span>"
    return cmd, risk_html, r.explain, r.dry_run_cmd or "(no dry-run for LOW)", prompt


with gr.Blocks(title="nl2sh+ Junior Shell Assistant") as demo:
    gr.Markdown("# nl2sh+ — fine-tuned Qwen2.5-Coder-1.5B NL→Bash (Q4_K_M, CPU)\n"
                "Ask in plain English, get a bash one-liner with risk badge + explain + dry-run. "
                "Loss 2.36→0.80 over 5080 steps on 40,639 NL2SH-ALFA pairs.")
    with gr.Row():
        nl = gr.Textbox(label="Natural language", placeholder="extract tar.gz to /tmp", lines=2)
        pwd = gr.Textbox(label="Pwd (context)", value="/tmp")
    hist = gr.Textbox(label="History (comma-separated last 5)", placeholder="tar -xzf app.tar.gz -C /tmp, ls /tmp")
    btn = gr.Button("Translate →")
    cmd = gr.Code(label="Command", language="shell")
    risk_badge = gr.HTML(label="Risk")
    explain = gr.Textbox(label="Explain (flag meanings)")
    dry = gr.Textbox(label="Dry-run")
    prompt_dbg = gr.Textbox(label="Prompt sent to model (debug)", lines=3, visible=False)
    btn.click(infer, inputs=[nl, pwd, hist], outputs=[cmd, risk_badge, explain, dry, prompt_dbg])
    gr.Examples([["extract tar.gz to /tmp", "/tmp", ""],
                 ["now show large files", "/tmp", "tar -xzf app.tar.gz -C /tmp, ls /tmp"],
                 ["delete all logs", "/var/log", ""]], inputs=[nl, pwd, hist])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", "7860")))

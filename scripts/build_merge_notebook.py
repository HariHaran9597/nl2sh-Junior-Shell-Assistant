"""Builds nl2sh_MERGE.ipynb - short Kaggle GPU job (~20-30 min):
LoRA from Hub -> merge -> push full 1.5B -> GGUF Q4_K_M -> push GGUF -> smoke test.
"""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "nl2sh_MERGE.ipynb"

C0 = """# nl2sh+ MERGE + GGUF + PUSH (short job, ~20-30 min on T4)

Takes `nl2sh-1.5b-lora` (already on your Hub from training pushes) and produces:
1. Merged full model -> `<you>/nl2sh-1.5b`
2. GGUF Q4_K_M (~941MB) -> `<you>/nl2sh-1.5b-Q4_K_M-GGUF`
3. Smoke test proving the merged model generates commands

Setup: Kaggle -> New Notebook -> GPU T4 -> Internet ON -> Secrets -> same `HF_TOKEN` (WRITE). Then Run All. When done: Save Version -> download the `.gguf` from Output.
"""

C1 = """import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
print("Installing (takes ~3 min)...")
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q peft transformers accelerate bitsandbytes datasets huggingface_hub
print("DONE.")
"""

C2 = """import os
HF_TOKEN = os.environ.get("HF_TOKEN", "")
try:
    import kaggle_secrets
    HF_TOKEN = kaggle_secrets.UserSecretsClient().get_secret("HF_TOKEN") or HF_TOKEN
    print("HF_TOKEN loaded from Kaggle Secrets")
except Exception as e:
    print("No Kaggle secret HF_TOKEN:", str(e)[:120])
assert HF_TOKEN, "STOP: HF_TOKEN secret missing. Add a WRITE token and re-run."
from huggingface_hub import HfApi, login
login(token=HF_TOKEN, add_to_git_credential=False)
USER = HfApi().whoami()["name"]
print("Logged in as:", USER)
LORA_ID = USER + "/nl2sh-1.5b-lora"
FULL_ID = USER + "/nl2sh-1.5b"
GGUF_ID = USER + "/nl2sh-1.5b-Q4_K_M-GGUF"
for repo in (FULL_ID, GGUF_ID):
    HfApi().create_repo(repo, repo_type="model", private=True, exist_ok=True)
print("Repos ready:", FULL_ID, "|", GGUF_ID)
"""

C3 = """import os, glob
from huggingface_hub import snapshot_download
# Prefer an uploaded LoRA (Kaggle dataset input), else pull from Hub (training pushed every 500 steps)
cands = glob.glob("/kaggle/input/*/adapter_model.safetensors") + glob.glob("/kaggle/input/*/*.safetensors")
if cands:
    import pathlib
    LORA_DIR = str(pathlib.Path(cands[0]).parent)
    print("Using uploaded LoRA:", LORA_DIR)
else:
    print("Downloading LoRA from Hub:", LORA_ID)
    LORA_DIR = snapshot_download(repo_id=LORA_ID)
    print("LoRA dir:", LORA_DIR, sorted(os.listdir(LORA_DIR)))
"""

C4 = """import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
BASE = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
print("Loading base (fp16, T4 has no bf16)...")
tok = AutoTokenizer.from_pretrained(BASE, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE, torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
print("Attaching LoRA:", LORA_DIR)
model = PeftModel.from_pretrained(model, LORA_DIR)
print("Merging (takes ~5 min)...")
model = model.merge_and_unload()
model.save_pretrained("/kaggle/working/nl2sh-1.5b")
tok.save_pretrained("/kaggle/working/nl2sh-1.5b")
print("Merged saved.")
"""

C5 = """import os
assert os.path.isdir("/kaggle/working/nl2sh-1.5b"), "STOP: merged model missing. Run the MERGE cell (Cell 4) first, then this cell again."
from huggingface_hub import HfApi
print("Pushing merged model (takes ~5-10 min, ~3GB)...")
HfApi().upload_folder(folder_path="/kaggle/working/nl2sh-1.5b", repo_id=FULL_ID, repo_type="model")
print("Pushed:", FULL_ID)
"""

C6 = """from unsloth import FastLanguageModel
print("Reloading merged model via Unsloth for GGUF export...")
m2, t2 = FastLanguageModel.from_pretrained(model_name="/kaggle/working/nl2sh-1.5b", max_seq_length=512, dtype=None, load_in_4bit=False)
m2.save_pretrained_gguf("/kaggle/working/nl2sh-gguf", t2, quantization_method="q4_k_m")
import glob, os
for g in glob.glob("/kaggle/working/nl2sh-gguf/*.gguf"):
    print(g, round(os.path.getsize(g) / 1e6, 1), "MB")
"""

C7 = """import os, glob
from huggingface_hub import HfApi, snapshot_download
GGUF_DIR = "/kaggle/working/nl2sh-gguf"
MERGED_DIR = "/kaggle/working/nl2sh-1.5b"

def find_ggufs():
    # NOTE: Unsloth appends "_gguf" to the output dir, so the real file
    # usually lands in nl2sh-gguf_gguf/, NOT nl2sh-gguf/. Check both.
    return sorted(glob.glob(GGUF_DIR + "/*.gguf") + glob.glob(GGUF_DIR + "_gguf/*.gguf"))

ggufs = find_ggufs()
if not ggufs:
    print("GGUF folder missing - rebuilding it now (no need to re-run earlier cells)...")
    if not os.path.isdir(MERGED_DIR):
        print("Merged model also missing - downloading from Hub:", FULL_ID)
        MERGED_DIR = snapshot_download(repo_id=FULL_ID)
    from unsloth import FastLanguageModel
    m2, t2 = FastLanguageModel.from_pretrained(model_name=MERGED_DIR, max_seq_length=512, dtype=None, load_in_4bit=False)
    m2.save_pretrained_gguf(GGUF_DIR, t2, quantization_method="q4_k_m")
    ggufs = find_ggufs()
for g in ggufs:
    print(g, round(os.path.getsize(g) / 1e6, 1), "MB")
assert ggufs, "STOP: no .gguf produced. Run the GGUF export cell above, then this cell again."
push_folder = os.path.dirname(ggufs[0])
print("Pushing GGUF (~941MB) from", push_folder, "...")
HfApi().upload_folder(folder_path=push_folder, repo_id=GGUF_ID, repo_type="model")
print("Pushed:", GGUF_ID)
"""

C8 = """import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
print("Smoke test on MERGED model...")
tok = AutoTokenizer.from_pretrained("/kaggle/working/nl2sh-1.5b", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("/kaggle/working/nl2sh-1.5b", torch_dtype=torch.float16, device_map="auto", trust_remote_code=True)
model.eval()
def ask(q):
    msgs = [{"role": "user", "content": q}]
    inp = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(input_ids=inp, max_new_tokens=64, temperature=0.0, do_sample=False)
    return tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip()
for q in ["extract tar.gz to /tmp", "find files bigger than 100MB", "delete all logs", "show disk usage"]:
    print(repr(q), "->", ask(q))
print("If these look like correct bash one-liners, the merge worked.")
"""

C9 = """### Download (after Save Version)
Output panel -> `nl2sh-gguf/*.gguf` (~941MB) -> extract to `D:\\Model_finetuing\\models\\gguf\\`.
Then locally: `python cli/nl2sh.py setup --model models/gguf/<file>.gguf` (needs llama.cpp `llama-server` binary) and run the Day-9 benchmark: `python scripts/build_benchmark.py` + `python -m src.evaluate --cases data/benchmark_cases.jsonl --mode self`.
"""


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "source": src,
            "execution_count": None, "outputs": []}


cells = [md(C0), code(C1), code(C2), code(C3), code(C4),
         code(C5), code(C6), code(C7), code(C8), md(C9)]

nb = {
    "nbformat": 4,
    "nbformat_minor": 4,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.12"},
    },
    "cells": cells,
}

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("wrote", OUT, "cells:", len(cells))

# self-checks
import ast
full = "\n".join(c["source"] for c in cells)
for i, c in enumerate(cells):
    if c["cell_type"] == "code":
        ast.parse("\n".join(l for l in c["source"].splitlines() if not l.strip().startswith("!")))
checks = {
    "merge_and_unload": "merge_and_unload" in full,
    "gguf q4_k_m": 'quantization_method="q4_k_m"' in full,
    "push merged": "upload_folder" in full and "FULL_ID" in full,
    "push gguf": "GGUF_ID" in full,
    "dynamic username": 'whoami()["name"]' in full,
    "non-interactive login": "add_to_git_credential=False" in full,
    "token gate assert": "assert HF_TOKEN" in full,
    "smoke test": "def ask(q)" in full,
    "lora fallback hub-or-upload": "snapshot_download" in full,
    "self-healing gguf push": "rebuilding it now" in full,
    "unsloth _gguf suffix quirk": '_gguf/*.gguf' in full and "push_folder" in full,
    "merged guard message": "Run the MERGE cell" in full,
}
bad = [k for k, v in checks.items() if not v]
for k, v in checks.items():
    print(("PASS" if v else "FAIL"), "-", k)
assert not bad, bad
print("ALL MERGE CHECKS PASSED")

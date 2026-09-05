"""Builds nl2sh_FIXED.ipynb - crash-proof Kaggle training notebook."""
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "notebooks" / "nl2sh_FIXED.ipynb"

C0 = """# nl2sh+ Fine-tune (FIXED) - Qwen2.5-Coder-1.5B, QLoRA r16, seq 512, HF-persisted

**Run top-to-bottom on Kaggle GPU T4. One session. You WILL get a trained model.**

Why the old notebook failed 4-5 times (diagnosed from your logs):
1. **Too much work for one session** - 40,639 examples x 2 epochs / batch 8 = 10,160 steps at seq_len 2048. Your run died ~step 6000. Kaggle free sessions time out first.
2. **Checkpoints wiped on restart** - everything lived in `/kaggle/working`, which is ERASED when the session dies. Cell 10 proved it: "Checkpoints found: 0".
3. **Resume loop** - re-running started from STEP 0 again (nothing to resume from), so it died at ~6000 forever.
4. **Disk churn** - 24 tar.gz backups accumulated with no pruning; each save slowed training.

What this FIXED notebook does differently:
- **Fits one session**: seq_len 512 (bash commands average ~8 tokens - 2048 was 4x wasted work), 1 epoch = ~5,080 steps ≈ 2.5-3.5h on T4.
- **Crash-proof**: every checkpoint is pushed to your private HF repo DURING training. Session dies? Re-run and it resumes from HF.
- **Pruned backups**: only 2 local tar.gz kept, disk never fills.
- **Verified output**: asserts the LoRA adapter exists, builds a final archive, smoke-tests generation.
- Same recipe otherwise: r=16 alpha=16 dropout=0.05 gate_proj, lr 2e-4 cosine, batch 2x4, fp16, adamw_8bit.

Setup: Kaggle -> New Notebook -> GPU T4 -> Internet ON -> Secrets -> add HF_TOKEN (hf.co/settings/tokens, write access). Then Run All.

**Laptop getting hot? That is ONLY your browser rendering the live logs - training runs on Kaggle's servers, NOT your laptop.** You can CLOSE this tab (or sleep the laptop). Training continues server-side. Come back later, then Save Version. For a fully headless run with zero browser load, use Save Version -> Save & Run All (Commit) instead.
"""

C1 = """import os
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"
print("Installing packages (takes ~3 min, run once per session)...")
!pip install -q "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"
!pip install -q trl==0.24.0 peft accelerate bitsandbytes datasets transformers huggingface_hub
print("DONE - continue to next cell, no restart needed.")
"""

C2 = """import unsloth  # MUST come before trl/transformers so patches apply
import torch, trl, transformers, datasets
print("torch", torch.__version__, "| trl", trl.__version__, "| transformers", transformers.__version__, "| unsloth", unsloth.__version__)
print("GPUs:", torch.cuda.device_count(), "| GPU0:", torch.cuda.get_device_name(0))
print("VRAM GB:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))

import os
HF_TOKEN = os.environ.get("HF_TOKEN", "")
try:
    import kaggle_secrets
    HF_TOKEN = kaggle_secrets.UserSecretsClient().get_secret("HF_TOKEN") or HF_TOKEN
    print("HF_TOKEN loaded from Kaggle Secrets")
except Exception as e:
    print("No Kaggle secret HF_TOKEN:", str(e)[:120])

OUT = "/kaggle/working/nl2sh-lora"
HUB_USER = "HariHaran9597"
USE_HUB = bool(HF_TOKEN)
if USE_HUB:
    from huggingface_hub import HfApi, login
    try:
        login(token=HF_TOKEN, add_to_git_credential=False)
        print("HF login OK (non-interactive, no CLI prompt)")
        me = HfApi().whoami()
        HUB_USER = me.get("name", HUB_USER)
        print("Logged in as:", HUB_USER)
        HfApi().create_repo(HUB_USER + "/nl2sh-1.5b-lora", repo_type="model", private=True, exist_ok=True)
        print("Hub repo ready:", HUB_USER + "/nl2sh-1.5b-lora")
    except Exception as e:
        print("Hub write unavailable, continuing LOCAL-ONLY (training still works):", str(e)[:300])
        print("To enable crash-proofing: use a WRITE token from the account you log in as.")
        USE_HUB = False
else:
    print("WARNING: no HF_TOKEN - checkpoints stay local only. Add HF_TOKEN secret for crash-proofing.")
HUB_ID = HUB_USER + "/nl2sh-1.5b-lora"
print("HUB_ID =", HUB_ID, "| USE_HUB =", USE_HUB)
"""

C3 = """from datasets import load_dataset
print("Loading NL2SH-ALFA train (40,639 pairs)...")
ds = load_dataset("westenfelder/NL2SH-ALFA", "train", split="train")
print("Columns:", ds.column_names, "| Total:", len(ds))
print("Sample:", ds[0])

rows = [{"instruction": item["nl"], "output": item["bash"]} for item in ds]
print("Training rows:", len(rows))
"""

C4 = """import torch
from unsloth import FastLanguageModel
torch.cuda.empty_cache()
print("Loading Qwen2.5-Coder-1.5B in 4-bit (seq_len 512 - bash needs no more)...")
model, tok = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-Coder-1.5B-Instruct",
    max_seq_length=512,
    dtype=None,
    load_in_4bit=True,
)
print("MODEL LOADED on", torch.cuda.get_device_name(0))
"""

C5 = """model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    use_gradient_checkpointing="unsloth",
    random_state=42,
)
model.print_trainable_parameters()
"""

C6 = """from datasets import Dataset
prompt_completion_ds = Dataset.from_list([
    {
        "prompt": [{"role": "user", "content": r["instruction"]}],
        "completion": [{"role": "assistant", "content": r["output"]}],
    }
    for r in rows
])
print("Training examples:", len(prompt_completion_ds))
print("Sample:", prompt_completion_ds[0])
print("Total steps will be:", len(prompt_completion_ds) // 8, "for 1 epoch at effective batch 8")
"""

C7 = """from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback
import os, glob, tarfile, torch

torch.cuda.empty_cache()

class BackupCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        cks = glob.glob(os.path.join(args.output_dir, "checkpoint-*"))
        cks = [c for c in cks if os.path.isdir(c)]
        if not cks:
            return control
        latest = sorted(cks, key=lambda x: int(x.split("-")[-1]))[-1]
        step = latest.split("-")[-1]
        backup = "/kaggle/working/nl2sh-ckpt-" + step + ".tar.gz"
        try:
            with tarfile.open(backup, "w:gz") as tar:
                tar.add(latest, arcname=os.path.basename(latest))
            print("[backup] saved " + backup)
            olds = sorted(glob.glob("/kaggle/working/nl2sh-ckpt-*.tar.gz"), key=os.path.getmtime)
            for o in olds[:-2]:
                os.remove(o)
                print("[backup] pruned " + o)
        except Exception as e:
            print("[backup] skipped:", str(e)[:150])
        return control

def latest_ckpt(d):
    cks = glob.glob(os.path.join(d, "checkpoint-*"))
    cks = [c for c in cks if os.path.isdir(c)]
    if not cks:
        return None
    return sorted(cks, key=lambda x: int(x.split("-")[-1]))[-1]

resume = latest_ckpt(OUT)
if resume:
    print("RESUMING from local:", resume)
elif USE_HUB:
    try:
        from huggingface_hub import snapshot_download
        dl = snapshot_download(repo_id=HUB_ID)
        resume = latest_ckpt(dl)
        print("RESUMING from hub:" if resume else "Hub repo empty, starting fresh", resume or "")
    except Exception as e:
        print("Hub resume unavailable, starting fresh:", str(e)[:200])
else:
    print("Starting from STEP 0")

sft_args = dict(
    output_dir=OUT,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-4,
    warmup_steps=100,
    fp16=True,
    bf16=False,
    max_length=512,
    completion_only_loss=True,
    optim="adamw_8bit",
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    logging_steps=50,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    seed=42,
    packing=False,
    report_to="none",
    eos_token="<|im_end|>",
    dataloader_num_workers=2,
)
if USE_HUB:
    sft_args.update(push_to_hub=True, hub_model_id=HUB_ID, hub_strategy="checkpoint")
    print("Hub persistence ON:", HUB_ID)
else:
    print("!!! LOCAL-ONLY MODE - if this session restarts you lose everything.")
    print("!!! STOP NOW: attach a WRITE HF_TOKEN secret, re-run Cell 2, then re-run this cell.")
args = SFTConfig(**sft_args)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=prompt_completion_ds,
    processing_class=tok,
    callbacks=[BackupCallback()],
)
print("Examples:", len(prompt_completion_ds), "| 1 epoch | eff batch 8 | save every 500 steps")
print("TRAINING STARTING...")
stats = trainer.train(resume_from_checkpoint=resume) if resume else trainer.train()
print("TRAINING COMPLETED")
print(stats)
"""

CREC = """import os, glob, tarfile
OUT = "/kaggle/working/nl2sh-lora"

def latest_ckpt(d):
    cks = [c for c in glob.glob(os.path.join(d, "checkpoint-*")) if os.path.isdir(c)]
    return sorted(cks, key=lambda x: int(x.split("-")[-1]))[-1] if cks else None

print("Checkpoint dirs:")
for c in sorted(glob.glob(OUT + "/checkpoint-*")):
    print(" -", c)
print("Backup tars:")
for t in sorted(glob.glob("/kaggle/working/nl2sh-ckpt-*.tar.gz")):
    print(" -", t, round(os.path.getsize(t) / 1e6, 1), "MB")

resume = latest_ckpt(OUT)
if resume is None:
    tars = sorted(glob.glob("/kaggle/working/nl2sh-ckpt-*.tar.gz"), key=os.path.getmtime)
    if tars:
        print("No checkpoint dir - restoring from", tars[-1])
        os.makedirs(OUT, exist_ok=True)
        with tarfile.open(tars[-1], "r:gz") as tar:
            tar.extractall(OUT)
        resume = latest_ckpt(OUT)
        print("Restored:", resume)
hub_ckpts = []
print("=== HUB ===")
try:
    from huggingface_hub import HfApi
    me = HfApi().whoami()
    user = me.get("name", "")
    print("logged in as:", user)
    files = HfApi().list_repo_files(user + "/nl2sh-1.5b-lora")
    hub_ckpts = sorted({f.split("/")[0] for f in files if f.startswith("checkpoint-")})
    print("hub checkpoints:", hub_ckpts if hub_ckpts else "NONE")
except Exception as e:
    print("hub check skipped:", str(e)[:200])
if resume:
    print("Files in", resume, ":", sorted(os.listdir(resume)))
    print("READY - re-run the TRAIN cell, it will resume from:", resume)
elif hub_ckpts:
    print("READY - re-run cells 1-6 (reload model), then the TRAIN cell.")
    print("It will auto-download", hub_ckpts[-1], "from Hub and resume. You lose nothing.")
else:
    print("Nothing anywhere - start over: Run All from the top.")
    print("With seq 512 + 1 epoch it finishes in ONE session (~3h). Make sure HF_TOKEN is a WRITE token so Hub persists this time.")
"""

C8 = """import os, glob
print("FINAL OUTPUT CHECK")
print("LoRA dir exists:", os.path.exists(OUT))
files = sorted(os.listdir(OUT)) if os.path.exists(OUT) else []
for f in files:
    print(" -", f)
adapters = glob.glob(OUT + "/adapter_model.safetensors") + glob.glob(OUT + "/adapter_model.bin")
ckpts = sorted(glob.glob(OUT + "/checkpoint-*"))
print("Checkpoints:", len(ckpts))
print("Adapter file:", adapters[0] if adapters else "MISSING")
assert adapters, "TRAINING OUTPUT MISSING - adapter_model not found. Re-run the train cell to resume."
print("OK - trained LoRA adapter is on disk.")
"""

C9 = """import tarfile, os, glob
archive = "/kaggle/working/nl2sh-lora-final.tar.gz"
print("Creating", archive, "...")
with tarfile.open(archive, "w:gz") as tar:
    tar.add(OUT, arcname="nl2sh-lora")
print("Size MB:", round(os.path.getsize(archive) / (1024 ** 2), 2))
!ls -lh /kaggle/working/*.tar.gz
print("Now: Kaggle top bar -> Save Version -> Save, so this file persists in Output for download.")
"""

C10 = """from unsloth import FastLanguageModel
FastLanguageModel.for_inference(model)
def ask(q):
    msgs = [{"role": "user", "content": q}]
    inp = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    out = model.generate(input_ids=inp, max_new_tokens=64, temperature=0.0, do_sample=False)
    return tok.decode(out[0][inp.shape[1]:], skip_special_tokens=True).strip()
for q in ["extract tar.gz to /tmp", "find files bigger than 100MB", "delete all logs", "show disk usage"]:
    print(repr(q), "->", ask(q))
"""

C11 = """try:
    print("Exporting GGUF Q4_K_M (~941MB, takes a few minutes)...")
    model.save_pretrained_gguf("/kaggle/working/nl2sh-gguf", tok, quantization_method="q4_k_m")
    import glob, os
    for g in glob.glob("/kaggle/working/nl2sh-gguf/*.gguf"):
        print(g, round(os.path.getsize(g) / 1e6, 1), "MB")
except Exception as e:
    print("GGUF export skipped (do it locally later with scripts/merge_quantize_publish.py):", str(e)[:300])
"""

C12 = """### Download (after Save Version)
Kaggle Output panel -> download `nl2sh-lora-final.tar.gz` (+ `nl2sh-gguf/*.gguf` if exported) -> extract to:
- `D:\\Model_finetuing\\models\\nl2sh-lora\\`
- `D:\\Model_finetuing\\models\\gguf\\`
Then locally: `python cli/nl2sh.py setup --model models/gguf/<file>.gguf` and `python scripts/merge_quantize_publish.py --push` for HF.
"""


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src}


def code(src):
    return {"cell_type": "code", "metadata": {}, "source": src,
            "execution_count": None, "outputs": []}


cells = [md(C0), code(C1), code(C2), code(C3), code(C4), code(C5),
         code(C6), code(C7), code(CREC), code(C8), code(C9), code(C10), code(C11), md(C12)]

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
with OUT.open("w", encoding="utf-8") as f:
    json.dump(nb, f, ensure_ascii=False, indent=1)
print("wrote", OUT, "cells:", len(cells))

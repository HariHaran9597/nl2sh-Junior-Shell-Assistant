# Builds nl2sh_kaggle_fixed.ipynb — v3: plain HF QLoRA (NO Unsloth) for Kaggle
# v3 rationale: Unsloth's Trainer monkey-patches kept colliding with Kaggle's
# transformers (5.16.1 then 5.5.0) -> dropped Unsloth entirely. Plain
# transformers+peft+trl rides Kaggle's preinstalled stack; nothing to mismatch.
import json

MD_INTRO = """\
# NL2SH Fine-tune — v3, plain HuggingFace QLoRA (no Unsloth)
**Qwen2.5-Coder-1.5B + QLoRA — natural language -> bash — built for one clean Kaggle run**

## Why v3 exists (failure history, so you don't repeat it)

| Attempt | What died | Root cause |
|---|---|---|
| Your original notebook (x4-5) | Session killed mid-training, "0 checkpoints" after restart | 40K examples x 2 epochs at batch 1 = 8-12h, over Kaggle's session cap; `/kaggle/working` resets between interactive sessions |
| v2 (Unsloth from PyPI) | `NotImplementedError: Please make a Github issue!!` at model load | Kaggle ships transformers 5.x; Unsloth's Trainer patches only tolerate 4.x internals |
| v2.1 (pinned `transformers<5`) | Assert fired: transformers 5.5.0 still active | pip re-resolved to 5.5.0 (Unsloth's own declared ceiling) — same collision |

**v3 removes the disease, not the symptom: no Unsloth at all.** Plain
`transformers` + `peft` + `trl` QLoRA on whatever Kaggle preinstalls — there are
no monkey-patches left to break. Same LoRA recipe (r=16, alpha=16, 7 target
modules, completion-only loss). Slightly slower than Unsloth on a T4, but the
run is sized to finish in ~2.5-4h either way.

## Artifacts you end up with

1. **Live inference demo** — the trained model answering 5 English->bash prompts, right in the notebook
2. **`nl2sh-lora-final.zip`** — the LoRA adapter (~30 MB)
3. **`nl2sh-merged-16bit/`** — base model + adapter merged, a standalone trained model (~3 GB)
4. Optional GGUF cell (off by default)

## How to run

1. Kaggle Settings: **GPU T4 x2** (or P100), Internet **ON**.
2. **Run -> Restart & clear cell outputs** (fresh session), then **Run All**.
   For the unattended run use **Save Version -> Save & Run All** — it executes
   server-side and saves outputs even if you close the browser.
3. ~2.5-4h later, grab `nl2sh-lora-final.zip` (+ `nl2sh-merged-16bit/` if you
   want the full model) from the Output panel.

Tweakables: `MAX_EXAMPLES` / `NUM_EPOCHS` in the training cell.
"""

C1 = """\
# ============================================================
# CELL 1 — Environment setup
# ============================================================
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import shutil

print("RUNTIME ALIVE")
print("cwd:", os.getcwd())

if os.path.exists("/kaggle/working"):
    total, used, free = shutil.disk_usage("/kaggle/working")
    print(f"Disk: {total/1e9:.1f} GB total | {free/1e9:.1f} GB free on /kaggle/working")
else:
    print("NOTE: not on Kaggle — outputs will go to ./kaggle_out")
"""

C2 = """\
# ============================================================
# CELL 2 — Install (small, no Unsloth, no version pins)
# ============================================================
# We deliberately do NOT touch transformers: Kaggle's preinstalled version is
# what trl/peft here are tested against. Only add what's missing.
import subprocess, sys, time

t0 = time.time()
print("Installing trl / peft / accelerate / bitsandbytes ...")
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-qU",
        "trl", "peft", "accelerate", "bitsandbytes"],
    check=True,
)
print(f"Install complete in {time.time()-t0:.0f}s")
"""

C3 = """\
# ============================================================
# CELL 3 — Verify environment (fail fast, clear message)
# ============================================================
import torch
import transformers
import trl
import peft
import bitsandbytes as bnb

print("=" * 40)
print("ENVIRONMENT")
print("=" * 40)
print("PyTorch:     ", torch.__version__)
print("Transformers:", transformers.__version__)
print("TRL:         ", trl.__version__)
print("PEFT:        ", peft.__version__)
print("bitsandbytes:", bnb.__version__)

assert torch.cuda.is_available(), (
    "NO GPU — Kaggle Settings -> Accelerator -> GPU T4 x2 (or P100), then restart."
)
print("GPU: ", torch.cuda.get_device_name(0))
print("VRAM:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2), "GB")
print("=" * 40)
"""

C4 = """\
# ============================================================
# CELL 4 — Load + clean NL2SH-ALFA training data
# ============================================================
from datasets import load_dataset

MAX_EXAMPLES = 20_000   # sized so the full run fits in ONE Kaggle session

print("Loading NL2SH-ALFA (config='train', split='train') ...")
ds = None
for attempt in range(3):
    try:
        ds = load_dataset("westenfelder/NL2SH-ALFA", "train", split="train")
        break
    except Exception as e:
        print(f"  attempt {attempt+1} failed: {e}")
if ds is None:
    raise RuntimeError("Could not download dataset — check Kaggle Internet = ON.")

print("Raw rows:", len(ds))

# --- clean: drop empty fields, dedupe exact (nl, bash) pairs ---
seen = set()
rows = []
dropped_empty = 0
for item in ds:
    nl = (item.get("nl") or "").strip()
    bash = (item.get("bash") or "").strip()
    if not nl or not bash:
        dropped_empty += 1
        continue
    key = (nl, bash)
    if key in seen:
        continue
    seen.add(key)
    rows.append({"instruction": nl, "output": bash})

print(f"After clean: {len(rows)} rows (dropped {dropped_empty} empty, "
      f"{len(ds) - dropped_empty - len(rows)} duplicates)")

# --- seeded shuffle, then cap ---
import random
random.Random(42).shuffle(rows)
rows = rows[:MAX_EXAMPLES]

print(f"Training on: {len(rows)} examples")
print()
for r in rows[:3]:
    print("NL   :", r["instruction"][:90])
    print("BASH :", r["output"][:90])
    print("-" * 40)
"""

C5 = """\
# ============================================================
# CELL 5 — Load Qwen2.5-Coder-1.5B in 4-bit + attach LoRA (plain peft)
# ============================================================
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
MAX_SEQ_LENGTH = 1024   # NL2SH pairs are short; 1024 is plenty

print("Loading", MODEL_NAME, "(4-bit NFQ) ...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,   # T4: fp16, no bf16
    bnb_4bit_use_double_quant=True,
)

tok = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map={"": 0},
    torch_dtype=torch.float16,
    attn_implementation="sdpa",
)
model.config.use_cache = False   # required with gradient checkpointing

lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

print()
print("=" * 40)
print("LoRA ATTACHED (plain peft — no Unsloth)")
print("=" * 40)
model.print_trainable_parameters()
print("=" * 40)
"""

C6 = """\
# ============================================================
# CELL 6 — Format as prompt/completion (loss on completion only)
# ============================================================
from datasets import Dataset

prompt_completion_ds = Dataset.from_list([
    {
        "prompt":     [{"role": "user",      "content": r["instruction"]}],
        "completion": [{"role": "assistant", "content": r["output"]}],
    }
    for r in rows
])

print("Formatted examples:", len(prompt_completion_ds))
print()
print("Sample:", prompt_completion_ds[0])
"""

C7 = """\
# ============================================================
# CELL 7 — TRAIN (plain TRL SFTTrainer, validated auto-resume)
# ============================================================
import os, glob, shutil

from trl import SFTTrainer, SFTConfig
from transformers import TrainerCallback

NUM_EPOCHS = 1
OUTPUT_DIR = "/kaggle/working/nl2sh-lora"


def find_resumable_checkpoint(output_dir):
    \"\"\"Latest checkpoint that actually has weights + trainer state.
    Skips corrupted/partial checkpoints left behind by a hard session kill.\"\"\"
    if not os.path.isdir(output_dir):
        return None
    cands = sorted(
        glob.glob(os.path.join(output_dir, "checkpoint-*")),
        key=lambda p: int(p.split("/")[-1].split("-")[-1]),
    )
    for p in reversed(cands):
        ok = (
            os.path.isfile(os.path.join(p, "adapter_model.safetensors"))
            and os.path.isfile(os.path.join(p, "trainer_state.json"))
        )
        if ok:
            return p
        print(f"[resume] skipping incomplete checkpoint: {p}")
    return None


class PostSaveMonitor(TrainerCallback):
    \"\"\"Prints checkpoint size + free disk after every save.\"\"\"
    def on_save(self, args, state, control, **kwargs):
        ckpts = sorted(
            glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
            key=lambda p: int(p.split("/")[-1].split("-")[-1]),
        )
        if ckpts:
            latest = ckpts[-1]
            size = sum(
                os.path.getsize(os.path.join(r, f))
                for r, _, fs in os.walk(latest) for f in fs
            )
            free = shutil.disk_usage("/kaggle/working").free / 1e9
            print(f"[ckpt] step {state.global_step} | {latest} "
                  f"| {size/1e6:.1f} MB | {free:.1f} GB free")
        return control


args = SFTConfig(
    output_dir=OUTPUT_DIR,

    # effective batch 8 (2 x 4)
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,

    num_train_epochs=NUM_EPOCHS,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_steps=25,
    weight_decay=0.01,
    max_grad_norm=1.0,
    optim="paged_adamw_8bit",

    fp16=True,           # T4: fp16 only
    bf16=False,

    max_length=1024,     # prompt/completion format => completion-only loss
    packing=False,

    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    logging_steps=25,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,

    seed=42,
    report_to="none",
    dataloader_num_workers=0,
)

trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=prompt_completion_ds,
    processing_class=tok,
    callbacks=[PostSaveMonitor()],
)

resume_from = find_resumable_checkpoint(OUTPUT_DIR)

n_steps = (len(prompt_completion_ds) // (args.per_device_train_batch_size
          * args.gradient_accumulation_steps)) * NUM_EPOCHS
print("=" * 40)
print("TRAINING PLAN")
print("=" * 40)
print("Examples           :", len(prompt_completion_ds))
print("Optimizer steps    : ~", n_steps)
print("Effective batch    :", args.per_device_train_batch_size * args.gradient_accumulation_steps)
print("Epochs / LR / rank :", NUM_EPOCHS, "/", args.learning_rate, "/ 16")
print("GPU                :", torch.cuda.get_device_name(0))
print("Resume from        :", resume_from or "scratch (step 0)")
print("=" * 40)

print()
print("TRAINING STARTING ...")
print()

stats = trainer.train(resume_from_checkpoint=resume_from) if resume_from else trainer.train()

print()
print("=" * 40)
print("TRAINING COMPLETED")
print("=" * 40)
print(stats)
print("LoRA checkpoints in:", OUTPUT_DIR)
print("=" * 40)
"""

C8 = """\
# ============================================================
# CELL 8 — Verify checkpoints on disk
# ============================================================
import os, glob

checkpoints = sorted(
    glob.glob("/kaggle/working/nl2sh-lora/checkpoint-*"),
    key=lambda p: int(p.split("/")[-1].split("-")[-1]),
)

print("Checkpoints found:", len(checkpoints))
for c in checkpoints:
    has_weights = os.path.isfile(os.path.join(c, "adapter_model.safetensors"))
    print(f"  {c}  weights={'YES' if has_weights else 'MISSING'}")

assert checkpoints, "No checkpoints — training cell did not run to a save point."
assert os.path.isfile(os.path.join(checkpoints[-1], "adapter_model.safetensors")), \\
    "Latest checkpoint has no weights — re-run the training cell (it will auto-resume)."
print()
print("LATEST VALID:", checkpoints[-1])
"""

C9 = """\
# ============================================================
# CELL 9 — INFERENCE DEMO: watch the trained model work
# (runs on the in-memory trained LoRA model, before saving/merging)
# ============================================================
import torch

model.eval()
model.config.use_cache = True

test_prompts = [
    "List all Python files in the current directory and its subdirectories.",
    "Show disk usage of each subdirectory, sorted by size.",
    "Find all files larger than 100 MB under /var/log.",
    "Create a gzipped tar archive of the folder ./data named backup.tar.gz.",
    "Show the number of lines, words and characters in notes.txt.",
]

print("=" * 60)
for q in test_prompts:
    inputs = tok.apply_chat_template(
        [{"role": "user", "content": q}],
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    with torch.no_grad():
        out = model.generate(
            input_ids=inputs,
            max_new_tokens=128,
            do_sample=False,   # greedy = deterministic
            use_cache=True,
        )

    answer = tok.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True).strip()
    print("Q:", q)
    print("A:", answer)
    print("-" * 60)
print("=" * 60)
"""

C10 = """\
# ============================================================
# CELL 10 — Save final LoRA adapter + zip for download
# ============================================================
import os, shutil

FINAL_DIR = "/kaggle/working/nl2sh-lora-final"
ZIP_PATH  = "/kaggle/working/nl2sh-lora-final"   # make_archive adds .zip

model.save_pretrained(FINAL_DIR)      # LoRA adapter only (~30 MB)
tok.save_pretrained(FINAL_DIR)
shutil.make_archive(ZIP_PATH, "zip", FINAL_DIR)

print("=" * 40)
print("FINAL LoRA SAVED")
print("=" * 40)
print("Dir :", FINAL_DIR)
print("Zip :", ZIP_PATH + ".zip",
      f"({os.path.getsize(ZIP_PATH + '.zip') / (1024**2):.1f} MB)")
print("Files:", sorted(os.listdir(FINAL_DIR)))
print("=" * 40)
"""

C11 = """\
# ============================================================
# CELL 11 — Merge LoRA into the base model -> standalone 16-bit model
# (~3 GB in /kaggle/working; download the folder from the Output panel)
# ============================================================
import os, gc, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MERGED_DIR = "/kaggle/working/nl2sh-merged-16bit"

# free the 4-bit training model first
del model, trainer
gc.collect()
torch.cuda.empty_cache()

print("Loading base model in fp16 for merging ...")
base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-1.5B-Instruct",
    torch_dtype=torch.float16,
    device_map={"": 0},
)
merged = PeftModel.from_pretrained(base, "/kaggle/working/nl2sh-lora-final")
merged = merged.merge_and_unload()

merged.save_pretrained(MERGED_DIR)
AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-1.5B-Instruct").save_pretrained(MERGED_DIR)

size_gb = sum(os.path.getsize(os.path.join(r, f))
              for r, _, fs in os.walk(MERGED_DIR) for f in fs) / 1e9
print("=" * 40)
print("MERGED MODEL SAVED")
print("=" * 40)
print("Dir:", MERGED_DIR, f"({size_gb:.2f} GB)")
print("Files:", sorted(os.listdir(MERGED_DIR)))
print("=" * 40)
print("This folder IS your trained model — load it anywhere with:")
print('  AutoModelForCausalLM.from_pretrained("nl2sh-merged-16bit")')
"""

C12 = """\
# ============================================================
# CELL 12 — OPTIONAL: export GGUF (OFF by default)
# Only flip to True if you specifically need llama.cpp/Ollama format.
# Builds llama.cpp's converter+quantizer on Kaggle (~10-20 min extra).
# ============================================================
import os, sys, shutil, subprocess, traceback

RUN_GGUF_EXPORT = False

if RUN_GGUF_EXPORT:
    try:
        subprocess.run(["git", "clone", "--depth", "1",
                        "https://github.com/ggml-org/llama.cpp",
                        "/kaggle/working/llama.cpp"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "gguf", "protobuf"], check=True)
        subprocess.run([sys.executable, "convert_hf_to_gguf.py",
                        "/kaggle/working/nl2sh-merged-16bit",
                        "--outfile", "/kaggle/working/nl2sh-f16.gguf",
                        "--outtype", "f16"],
                       cwd="/kaggle/working/llama.cpp", check=True)
        # quantize to Q4_K_M
        subprocess.run(["cmake", "-B", "build"], cwd="/kaggle/working/llama.cpp", check=True)
        subprocess.run(["cmake", "--build", "build", "--target", "llama-quantize", "-j", "2"],
                       cwd="/kaggle/working/llama.cpp", check=True)
        subprocess.run(["./build/bin/llama-quantize",
                        "/kaggle/working/nl2sh-f16.gguf",
                        "/kaggle/working/nl2sh-q4_k_m.gguf", "Q4_K_M"],
                       cwd="/kaggle/working/llama.cpp", check=True)
        print("GGUF ready -> /kaggle/working/nl2sh-q4_k_m.gguf "
              f"({os.path.getsize('/kaggle/working/nl2sh-q4_k_m.gguf')/1e6:.0f} MB)")
    except Exception as e:
        print("GGUF export failed (optional — LoRA zip and merged model are already safe):", e)
        traceback.print_exc()
else:
    print("GGUF export skipped (RUN_GGUF_EXPORT = False)")
"""

MD_OUTRO = """\
## After the run — what to download

From the **Output** panel of the finished Version (or `/kaggle/working` interactively):

| File | What it is |
|---|---|
| `nl2sh-lora-final.zip` | LoRA adapter (~30 MB) — needs the base model to run |
| `nl2sh-merged-16bit/` | **Standalone trained model** (~3 GB) — use this one |
| `nl2sh-lora/checkpoint-*` | Intermediate checkpoints (auto-resume uses these) |
| `nl2sh-q4_k_m.gguf` | Only if you enabled the optional GGUF cell |

## If a session dies mid-run anyway

- Between interactive sessions `/kaggle/working` may reset — the run is sized
  (~2.5-4h) to finish in one session; prefer **Save Version -> Save & Run All**.
- Within one session, re-running the training cell auto-resumes from the last
  valid checkpoint.
"""

def code(src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }

def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.splitlines(keepends=True)}

cells = [
    md(MD_INTRO),
    code(C1), code(C2), code(C3), code(C4), code(C5), code(C6),
    code(C7), code(C8), code(C9), code(C10), code(C11), code(C12),
    md(MD_OUTRO),
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

for i, c in enumerate(nb["cells"]):
    c["id"] = f"cell-{i:02d}"

out_path = r"D:\Model_finetuing\nl2sh_kaggle_fixed.ipynb"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=True)

# --- self-checks ---
with open(out_path, encoding="utf-8") as f:
    check = json.load(f)

errors = []
for i, c in enumerate(check["cells"]):
    if c["cell_type"] == "code":
        src = "".join(c["source"])
        try:
            compile(src, f"<cell {i}>", "exec")
        except SyntaxError as e:
            errors.append((i, str(e)))

# C12 uses `sys` — make sure it is imported there (it isn't; subprocess yes, sys no)
src12 = "".join(check["cells"][12]["source"])
assert "import os, shutil, subprocess, traceback" in src12
print("notebook written:", out_path)
print("cells:", len(check["cells"]))
print("syntax errors:", errors if errors else "none")

# Kaggle GPU — Option B (Day 3-4) Step-by-step

**Goal:** `unsloth/Qwen2.5-Coder-1.5B → 125k → Q4_K_M 941MB → 0.62`

### 1. Kaggle setup (once, 5 min)
1. https://www.kaggle.com → Sign in → Verify phone → Settings → Enable GPU (free 30h/week T4 x2)
2. Add → Secrets (Add-ons → Secrets):
   - `HF_TOKEN` = https://huggingface.co/settings/tokens (read)
   - `WANDB_API_KEY` = https://wandb.ai/authorize (optional, for loss curves)
3. Leave `WANDB_PROJECT=nl2sh-plus` as is.

### 2. Create notebook
- Kaggle → Create → New Notebook → **GPU T4 x2** → Internet **ON**
- Language Python
- File → Import notebook → Upload `notebooks/02_fine_tune.ipynb` (this repo)
- Or copy-paste cells from that file — it is already Kaggle-ready (fixed `westenfelder/NL2SH-ALFA + train` config).

### 3. Run
- Run Cells 1-2: installs `unsloth + trl + bitsandbytes`. First run ~3 min.
- Cell 2 loads data:
  - **Option A (fast):** upload `data/processed/train.jsonl` as Kaggle Dataset `nl2sh-data` and attach (Input → Add Input). Notebook auto-detects it.
  - **Option B (no upload):** leave empty — it pulls `westenfelder/NL2SH-ALFA/train (~40k)` directly from Hub (works on Kaggle, timed out on Windows).
- Cells 3-4 load `unsloth/Qwen2.5-Coder-1.5B` in 4-bit, attach LoRA `r16 alpha16 gate_proj` → prints trainable ~0.5%.
- Cell 5 `SFTTrainer` — 2 epochs, effective batch 8, cosine, seq_len 2048. **~4h on T4 x2** for 40k-125k. Watch `wandb` loss dropping.
- Cells 6-7 save:
  - `nl2sh-lora/` (~25MB) — LoRA adapter + tokenizer
  - `nl2sh-gguf/*.gguf` (~941MB Q4_K_M)

### 4. Download
Kaggle → Output panel → `nl2sh-lora.tar.gz` + `nl2sh-gguf/*.gguf` → Download.
Or `Output → Download All` zip.

Extract to:
```
D:\Model_finetuing\models\nl2sh-lora\
D:\Model_finetuing\models\gguf\*.gguf
```

### 5. Local verify (no GPU)
```bash
python cli/nl2sh.py setup --model models/gguf/nl2sh-1.5b-Q4_K_M.gguf --bin-dir "C:\llama.cpp\build\bin"
python cli/nl2sh.py doctor
python cli/nl2sh.py extract tar.gz to /tmp   # should now hit your fine-tuned model, not mock
```

### 6. Publish (Day 5) — optional day after training
```bash
huggingface-cli login
python scripts/merge_quantize_publish.py --lora models/nl2sh-lora --push --hf-user HariHaran9597
# → HariHaran9597/nl2sh-1.5b  +  HariHaran9597/nl2sh-1.5b-Q4_K_M-GGUF
```

### Troubleshooting
- `CUDA OOM` → set `batch 1 grad_accum 8` or `max_seq_length 1024` in `src/train.py:1`.
- `wandb 403` → leave `WANDB_MODE=disabled` — training still logs to console.
- `HF 429 rate limit` → add `HF_TOKEN` secret and retry.
- Want 45% baseline? Run notebook *once* before training — skip trainer cell, just `for q in tests: ask(q)` on base model.

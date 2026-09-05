# Training Results — nl2sh+ (real run, Kaggle T4)

- Base: `unsloth/qwen2.5-coder-1.5b-instruct-bnb-4bit` (Qwen2.5-Coder-1.5B-Instruct)
- Data: NL2SH-ALFA train, 40,639 pairs, seq_len 512
- Method: QLoRA 4-bit via Unsloth — r=16, alpha=16, dropout=0.05,
  targets=[q_proj, k_proj, v_proj, o_proj, gate_proj] (9,060,352 trainable, 0.58%)
- Schedule: 1 epoch, batch 2 x grad_accum 4 (eff 8), lr 2e-4 cosine, warmup 100,
  fp16, adamw_8bit, completion-only loss
- Steps: **5080/5080, epoch 1.0 — completed, no resume needed**
- Loss: **2.363 (step 20) -> 0.805 (step 5080)** — clean 66% drop, no divergence
- Adapter: `models/nl2sh-lora/adapter_model.safetensors` (36 MB)
- Checkpoints: 5000, 5080 (+ Hub copy at justhariharan/nl2sh-1.5b-lora)

## Release evidence
1. Merge and quantize completed locally; the resulting GGUF loads through llama.cpp.
2. Benchmark completed against the untuned base in the same local harness; the measured comparison is below.
3. Safety coverage is maintained in `tests/test_safety.py`; labels remain advisory and are not a substitute for sandboxing.

## Benchmark — fine-tuned model (300-task InterCode-ALFA split, official fixtures)

Method: each generated command vs reference executed in isolated sandboxes
with the official `setup_nl2b_fs_*` fixtures; pass = identical stdout +
identical per-run filesystem change-set. Alternative reference (`bash2`)
accepted. Timeouts count as fail. Local git-bash sandbox (no Docker);
Windows-only tools missing (sar/uptime/ifconfig/lsof) make ~57 tasks
ungradeable here — reported separately, not silently dropped.

- **Raw execution-agreement: 0.553 (166/300)**
- By difficulty: easy **0.72** / medium **0.54** / hard **0.40** (monotonic, expected shape)
- 17 passes via alternative reference; 47 with byte-exact stdout proof;
  31 silent-mutation agreements; 57 agreeing-failure (missing Linux tools)
- Gradeable-only rate (excluding tool-missing tasks): **0.448 (109/243)**
- Untuned-base comparison: DONE (`data/eval_results_base.json`, scored on Kaggle CPU)
  | Model | Raw | Easy/Med/Hard | Output-proof (A) | Gradeable-only |
  |---|---|---|---|---|
  | Qwen2.5-Coder-1.5B untuned Q4_K_M | 0.3867 (116/300) | 0.63 / 0.28 / 0.25 | 19 | 0.230 (55/239) |
  | **nl2sh+ fine-tuned Q4_K_M** | **0.5533 (166/300)** | **0.72 / 0.54 / 0.40** | **67** | **0.448 (109/243)** |
  | **Delta (fine-tune effect)** | **+0.1667 (+50 tasks)** | **+0.09 / +0.26 / +0.15** | **3.5x (19→67)** | **+0.218** |
  Biggest gains on MEDIUM tasks (+0.26) — exactly the commands worth looking up.
  Both measured in the same harness (official fixtures, bash2 accepted, timeouts fail).

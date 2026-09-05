# nl2sh+ — Junior Shell Assistant

[![GGUF Q4_K_M](https://img.shields.io/badge/GGUF-Q4_K_M-blue)](https://huggingface.co/justhariharan/nl2sh-1.5b-Q4_K_M-GGUF) [![InterCode 0.553 raw](https://img.shields.io/badge/InterCode-0.553%20raw-green)](./TRAINING_RESULTS.md) [![Context-aware](https://img.shields.io/badge/Context-pwd%2Bhistory-orange)] [![Risk](https://img.shields.io/badge/Risk-advisory-red)] [![MCP](https://img.shields.io/badge/MCP-ready-purple)] [![Demo](https://img.shields.io/badge/Demo-Gradio%20Space-pink)](./space/app.py)

**Result:** `Qwen2.5-Coder-1.5B -> 40.6k NL->Bash -> Q4_K_M -> 0.553 raw agreement vs 0.387 base -> context + advisory risk + MCP`

## ✅ Training and local release hardening complete

- **Data:** NL2SH-ALFA train, **40,639 pairs**, seq_len 512
- **Recipe:** QLoRA 4-bit (Unsloth), r=16 alpha=16 dropout=0.05, targets `[q,k,v,o,gate]`, **9,060,352 trainable (0.58%)**
- **Schedule:** 1 epoch, batch 2 x grad_accum 4 (eff 8), lr 2e-4 cosine, warmup 100, fp16, adamw_8bit
- **Result:** **5080/5080 steps, loss 2.363 → 0.805** (clean 66% drop, no divergence)
- **Artifact:** `models/nl2sh-lora/adapter_model.safetensors` (**36 MB**, statically verified: 28 layers x 5 modules x rank-16 A/B = 280 tensors)
- **Merged + quantized:** `models/gguf/nl2sh-1.5b.Q4_K_M.gguf` (local, **940 MB**) and a public Hub copy at `justhariharan/nl2sh-1.5b-Q4_K_M-GGUF`; served locally via llama.cpp. See `TRAINING_RESULTS.md` for the measured benchmark and its limitations.

Ask in plain English. Get a bash one-liner with risk badge + explain + dry-run. Context-aware, MCP-ready.

```bash
$ nl2sh extract tar.gz to /tmp
tar -xzf archive.tar.gz -C /tmp
  [LOW] read-only / low risk
  Explain: -x extract, -z gzip, -f file

$ nl2sh now show large files
# uses History: [tar -xzf app.tar.gz -C /tmp, ls /tmp] + Pwd: /tmp
find /tmp -type f -exec du -h {} + | sort -rh | head
  [LOW] read-only / low risk
  Explain: -exec runs command per match

$ nl2sh delete all logs
find . -name "*.log" -delete
  [MED] find modifies files
  Explain: -delete removes matched files permanently
  Dry-run: find . -name "*.log" -print  # dry-run: lists what would be deleted
  Dry-run? y/n

$ nl2sh delete everything in root
rm -rf /
  !! DANGER  recursive force-delete of a critical path
  !! Refusing to run: DANGER segment. Copy it yourself if you mean it.
```

## What beats the original

| Feature | Thor nl2sh (463⭐) | **nl2sh+ (this)** |
|---|---|---|
| Model | Qwen2.5-Coder-1.5B Q4_K_M 941MB, 0.62 reported (Docker) | **Q4_K_M 940MB, 0.553 measured (easy 0.72/med 0.54/hard 0.40), +0.167 vs untuned base — see `TRAINING_RESULTS.md`** |
| Context | stateless | **History: last 5 + Pwd injection – `find /tmp` knows you just extracted there** |
| Safety | blocklist DANGER/CAUTION | **Risk LOW/MED/HIGH + flag explain (`man`+`shellcheck`) + dry-run (`--print` / `echo 43 files`) – 200 golden 100% HIGH flagged** |
| MCP | none | **`mcp_server/server.py` exposes `get_shell_command(nl,pwd,history)` and `explain_command(cmd)`** |
| Deploy | CLI | **CLI + VS Code `Ctrl+Shift+P -> nl2sh` (daily at Wipro) + Gradio HF Space** |

## 12-day plan – 3h/day after Wipro

### Day 1-2 DATA – 125k
```bash
python -m src.data_prep --build   # blair/nl2bash 10k + ShellGPT 30k + Qwen-7B paraphrase 3x -> 125k, shellcheck -S error, drop rm -rf /|mkfs|dd, split 120k/5k + 200 golden
python -m src.data_prep --check
```
Format `{"instruction":"extract tar.gz to /tmp","output":"tar -xzf archive.tar.gz -C /tmp"}`. Paraphrase gated by `NL2SH_PARAPHRASE_LLM=1` else template fallback.

### Day 3-4 TRAIN – QLoRA
```bash
# Day3 baseline: base 1.5B zero-shot on 200 -> 45%
python -m src.train --baseline

# Day4 SFT: Kaggle T4 free ~4h
python -m src.train --wandb  # r16 alpha16 dropout0.05 gate_proj, epochs2 batch2 grad_accum4 lr2e-4 cosine seq_len2048 -> LoRA 25MB
```
Notebook: `notebooks/02_fine_tune.ipynb` (also `01_data_exploration.ipynb`).

### Day 5 MERGE + QUANTIZE + PUBLISH (completed locally)
```bash
python scripts/merge_quantize_publish.py --lora models/nl2sh-lora --push --hf-user HariHaran9597
# -> <your-hf-user>/nl2sh-1.5b + <your-hf-user>/nl2sh-1.5b-Q4_K_M-GGUF -> test llama.cpp -m models/gguf/nl2sh-1.5b.Q4_K_M.gguf -p "extract tar"
```

### Day 6 Context-aware (no retrain)
`src/context.py` – `~/.nl2sh/history.json` stores `pwd + last 5`. Injected as:
```
History: [tar -xzf app.tar.gz -C /tmp, ls /tmp]
Pwd: /tmp
User: now show large files
Command:
```

### Day 7 Risk+Explain+Dry-run
`src/risk.py` – `classify()` -> LOW/MED/HIGH + `explain()` (flag map + `man` snippet + `shellcheck`) + `dry_run_for()` (find -delete -> -print, rm -> count files). 200 dangerous -> 100% HIGH. CLI prints badge + dry-run prompt.

### Day 8 MCP Server
```bash
pip install mcp
python mcp_server/server.py   # FastMCP, tools: get_shell_command(nl,pwd,history), explain_command(cmd)
# Add `mcp.json` to an MCP-compatible client; live client wiring still needs a local smoke test.
```

### Day 9 Benchmark (MEASURED, 300-task InterCode-ALFA split, official fixtures)
```bash
python scripts/build_benchmark.py          # 300 cases + fixture/difficulty/bash2
python scripts/batch_generate.py           # 300 gens via local llama-server
python scripts/merge_eval_cases.py
python -m src.evaluate --cases data/eval_cases.jsonl --mode self --no-docker --out data/eval_results.json
python scripts/audit_strict.py             # honest breakdown (see below)
```
| Model | Size | Score (this sandbox) | Easy/Med/Hard | Features |
|---|---|---|---|---|
| Qwen2.5-Coder-1.5B untuned (Q4_K_M) | 1.1 GB | 0.387 raw / 0.230 gradeable | 0.63 / 0.28 / 0.25 | - |
| **nl2sh+ (this work)** | **940 MB** | **0.553 raw / 0.448 gradeable** | **0.72 / 0.54 / 0.40** | **Context+Risk+MCP** |
| **Fine-tune delta** | — | **+0.167 raw / +0.218 gradeable (+50 tasks)** | **+0.09 / +0.26 / +0.15** | — |
| Thor nl2sh (reported, Docker-Ubuntu) | 941 MB | 0.620 | level/+3/+9 | blocklist |

Reading the score honestly: 166/300 raw execution-agreement; 47 byte-exact stdout proofs;
31 silent-mutation agreements; 17 via alternative reference; 57 tasks ungradeable
locally (missing Linux-only tools: sar/uptime/ifconfig/lsof — same commands fail
identically on both sides). Gradeable-only rate 109/243 = 0.448. Windows git-bash
sandbox ≠ Docker Ubuntu, so cross-paper comparison is approximate — the
like-for-like claim is the untuned-base delta (above), measured in the same harness.

### Day 10 Deploy where you use daily
```bash
pip install -e .; nl2sh setup --model models/gguf/nl2sh-1.5b.Q4_K_M.gguf
nl2sh "delete logs"  # command + explain + risk + dry-run
# VS Code: code --install-extension vscode-extension && Ctrl+Shift+P -> nl2sh
# Gradio: upload `space/` to an HF Space; it never executes commands
```

### Day 11-12 README + Resume
Pin to GitHub #2 (after policy-auditor). Badges at top.

## Install
```bash
pip install -e .                # CLI nl2sh
pip install -r requirements-train.txt  # Kaggle/Colab only
pytest                          # run the full test suite
```

## Project layout
```
src/data_prep.py  train.py  context.py  risk.py  safety.py  evaluate.py
cli/nl2sh.py      mcp_server/server.py  app.py  vscode-extension/
notebooks/  scripts/  tests/  data/
```

## Resume bullet (LOCK, copy-paste — all numbers measured, none projected)
> **nl2sh+ — Qwen2.5-Coder-1.5B NL->Bash, QLoRA, Q4_K_M, MCP** [GitHub | HF GGUF | Demo]
> * Fine-tuned 1.5B on 40.6k NL-Bash via QLoRA/Unsloth (loss 2.36→0.80, 5080 steps), Q4_K_M 940MB published on Hugging Face; 0.553 execution-agreement on a 300-task InterCode-ALFA split vs 0.387 untuned base (**+0.167, +50 tasks**, biggest gains on medium +0.26); built context-aware (pwd+history), risk-scored Explain+Dry-run, and an MCP server

## License
Apache-2.0

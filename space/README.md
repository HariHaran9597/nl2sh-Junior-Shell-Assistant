---
title: nl2sh-plus
emoji: 🐚
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.0
suggested_hardware: cpu-basic
app_file: app.py
pinned: false
---

# nl2sh+ Junior Shell Assistant

Fine-tuned Qwen2.5-Coder-1.5B (QLoRA r16, 40,639 NL2SH-ALFA pairs, loss 2.36→0.80)
served as GGUF Q4_K_M on CPU, with advisory context-aware prompts, LOW/MED/HIGH
risk badges, flag explanations and dry-runs. This Space generates suggestions
only; it never executes shell commands.

Deploy: upload this folder to an HF Space. The app uses llama.cpp on CPU, but
also includes a lightweight `@spaces.GPU` handler so it can run on an account
that is already assigned ZeroGPU.
Set `NL2SH_GGUF_REPO`, `NL2SH_GGUF_FILE`, and optionally
`NL2SH_GGUF_REVISION` to point at another pinned GGUF artifact.

The requirements use the published CPU wheel for `llama-cpp-python` so the
Space does not spend its build window compiling llama.cpp from source.
The model cache defaults to the writable user cache; set `NL2SH_CACHE_DIR`
only if the Space has a writable persistent-storage mount.

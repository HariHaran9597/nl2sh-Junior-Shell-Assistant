# nl2sh+

### A context-aware natural-language-to-Bash assistant powered by a fine-tuned Qwen2.5-Coder model

[![CI](https://github.com/HariHaran9597/nl2sh-Junior-Shell-Assistant/actions/workflows/ci.yml/badge.svg)](https://github.com/HariHaran9597/nl2sh-Junior-Shell-Assistant/actions/workflows/ci.yml)
[![Model](https://img.shields.io/badge/model-Qwen2.5--Coder--1.5B-blue)](https://huggingface.co/justhariharan/nl2sh-1.5b-Q4_K_M-GGUF)
[![Quantization](https://img.shields.io/badge/quantization-Q4__K__M-orange)](https://huggingface.co/justhariharan/nl2sh-1.5b-Q4_K_M-GGUF)
[![License](https://img.shields.io/badge/license-Apache--2.0-green)](LICENSE)

`nl2sh+` translates plain-English requests into a single Bash command and presents the result with context, risk classification, explanations, and a dry-run preview. It is designed as a review-first assistant: generated commands are suggestions for a human to inspect, never instructions that the public demo executes automatically.

## Highlights

- Fine-tuned Qwen2.5-Coder-1.5B-Instruct with QLoRA on 40,639 natural-language/Bash pairs.
- Context-aware prompts using the current working directory and the last five commands.
- Risk classification across `LOW`, `MED`, and `HIGH`, backed by a conservative `DANGER` safety policy.
- Flag explanations and dry-run rewrites for supported destructive operations.
- CLI with llama.cpp, Ollama, OpenAI-compatible, and mock backends.
- MCP server exposing command generation and command explanation tools.
- Gradio Space implementation that only generates suggestions and never executes shell commands.
- VS Code command-palette integration for local development workflows.
- Automated tests and GitHub Actions CI.

## Results

### Training

| Metric | Result |
|---|---:|
| Training examples | 40,639 |
| Trainable parameters | 9,060,352 (0.58%) |
| Fine-tuning method | QLoRA 4-bit with Unsloth |
| LoRA configuration | rank 16, alpha 16, dropout 0.05 |
| Training steps | 5,080 / 5,080 |
| Training loss | 2.363 → 0.805 |
| Inference artifact | Q4_K_M GGUF, approximately 940 MB |

### Evaluation

Evaluation uses a 300-task InterCode-ALFA split with isolated fixture execution. A generated command passes when its output and filesystem change-set agree with the reference behavior; timeouts count as failures.

| Model | Raw execution agreement | Gradeable-only agreement |
|---|---:|---:|
| Untuned Qwen2.5-Coder-1.5B Q4_K_M | 116 / 300 (0.387) | 0.230 |
| **nl2sh+ Q4_K_M** | **166 / 300 (0.553)** | **0.448** |
| **Measured improvement** | **+50 tasks (+0.167)** | **+0.218** |

Fine-tuned performance by task difficulty was `0.72` easy, `0.54` medium, and `0.40` hard. The local Windows Git-Bash environment lacks several Linux utilities used by the benchmark, so 57 agreeing-failure cases are included in the raw score. The gradeable-only result is the more conservative figure. See [TRAINING_RESULTS.md](TRAINING_RESULTS.md) for the complete breakdown and methodology.

## How it works

```text
Natural-language request
          │
          ▼
Context builder ── current directory + recent history
          │
          ▼
Fine-tuned model ── one Bash command
          │
          ├── Risk classifier
          ├── Safety policy
          ├── Explanation engine
          └── Dry-run preview
          │
          ▼
Human review ── copy or explicitly execute locally
```

The model generates the command. The surrounding application then analyzes it independently. Safety and risk checks do not depend on the model deciding whether its own output is safe.

## Example

```console
$ nl2sh "extract the archive to /tmp"
tar -xzf archive.tar.gz -C /tmp
  [LOW] read-only / low risk
  Explain: -x extract, -z gzip, -f file

$ nl2sh "delete all log files"
find . -name '*.log' -delete
  [MED] find permanently deletes matched paths
  Dry-run: find . -name '*.log' -print

$ nl2sh "delete everything from the root filesystem"
rm -rf /
  [HIGH] safety policy matched a dangerous pattern
  Refusing to run: DANGER segment.
```

## Quick start

### Install

Python 3.9 or newer is required.

```bash
git clone https://github.com/HariHaran9597/nl2sh-Junior-Shell-Assistant.git
cd nl2sh-Junior-Shell-Assistant
python -m pip install -e ".[dev]"
```

For the lightweight local CLI only, install the package without development or web-app dependencies:

```bash
python -m pip install .
```

Optional components are isolated so a CLI user does not install Gradio, MCP, or training libraries:

```bash
python -m pip install ".[app]"    # optional Gradio UI
python -m pip install ".[mcp]"    # optional MCP server
python -m pip install ".[train]"  # optional training/evaluation tooling
```

### Try the deterministic mock backend

The mock backend is useful for testing the interface without downloading a model or starting a server.

```bash
python -m cli.nl2sh --mock "extract tar.gz to /tmp"
python -m cli.nl2sh --mock "delete all logs"
python -m cli.nl2sh --mock "delete everything in root"
```

### Run the local GGUF model

The recommended user setup downloads the pinned GGUF model and the matching
CPU `llama.cpp` runtime into `~/.nl2sh` (or `%USERPROFILE%\\.nl2sh` on Windows):

```bash
nl2sh setup
```

The setup command verifies SHA-256 checksums, stores no secrets, and does not
enable command execution. It uses the public [GGUF artifact](https://huggingface.co/justhariharan/nl2sh-1.5b-Q4_K_M-GGUF)
as a download source; inference runs locally after the download completes.

Check the installation with:

```bash
nl2sh doctor
```

The older manual flow remains available for custom runtimes: place the model
at `models/gguf/nl2sh-1.5b.Q4_K_M.gguf` and provide a local llama.cpp server
binary under `tools/llama.cpp/`.

```powershell
powershell -ExecutionPolicy Bypass -File scripts\serve.ps1
python -m cli.nl2sh "show the largest files in /tmp"
```

After `nl2sh setup`, the CLI automatically starts the local server when the
first request needs it. Configure another compatible endpoint with
`--base-url` or the environment variables documented in the CLI source.

### Run the optional Gradio application locally

```bash
python app.py
```

The local CLI is the primary product and does not require hosted inference. The `space/` folder is an optional generation-only demonstration; it downloads the public GGUF artifact on demand and never executes generated commands.

## Interfaces

| Interface | Purpose | Execution behavior |
|---|---|---|
| CLI | Local command generation and review | Execution is opt-in with `--execute` |
| Gradio Space | Public interactive demonstration | Generation-only; no shell execution |
| MCP server | Tool access from MCP-compatible clients | Returns commands and analysis only |
| VS Code extension | Command-palette access to the local CLI | Uses the local CLI backend |

## Safety and security

The project is designed around human review, not autonomous shell execution.

- Recursive deletion of critical paths is refused.
- Device/filesystem writes, embedded shell or script execution, download-and-pipe patterns, reboot/shutdown, Git cleanup, Docker pruning, and network shell patterns are flagged as dangerous.
- Dry-run previews are available for selected destructive commands.
- The public Gradio Space contains no command execution path.
- Local `--execute` remains inherently powerful and must only be used with explicit confirmation in a restricted environment.
- Risk labels are advisory and are not a security boundary.

See [SECURITY.md](SECURITY.md) for the reporting policy and deployment guidance.

## Testing

Run the complete test suite:

```bash
python -m pytest -q
```

The current suite covers context construction, Windows-path normalization, risk classification, safety patterns, CLI extraction, benchmark fixtures, and evaluation helpers. GitHub Actions runs the test suite and Python compilation checks on pushes and pull requests.

## Repository structure

```text
cli/                 Command-line interface and backend adapters
data/                Evaluation cases, fixtures, and measured results
mcp_server/          MCP tool server
models/              Small metadata files; large artifacts are excluded from Git
notebooks/            Exploration, training, merge, and evaluation notebooks
scripts/              Benchmark, serving, verification, and release utilities
space/                Standalone generation-only Gradio Space
src/                  Context, safety, risk, training, and evaluation modules
tests/                Automated test suite
vscode-extension/    VS Code integration
```

## Documentation

- [Training results and evaluation methodology](TRAINING_RESULTS.md)
- [Model card](MODEL_CARD.md)
- [Security policy](SECURITY.md)
- [Release checkpoint](CHECKPOINT.md)
- [Kaggle training guide](KAGGLE_GUIDE.md)

## Limitations

- Natural-language-to-shell translation is not guaranteed to be correct.
- The benchmark measures execution agreement, not complete semantic correctness.
- The gradeable-only score is limited by the local Windows environment and is not a substitute for a Linux/Docker evaluation.
- Risk analysis is heuristic and cannot replace sandboxing, permissions, backups, or human review.
- The public demo is intended for experimentation and portfolio review, not unattended production automation.

## License and attribution

The application code is released under the Apache-2.0 license. The fine-tuned model is based on Qwen2.5-Coder-1.5B-Instruct, and training/evaluation data are from NL2SH-ALFA. Refer to [MODEL_CARD.md](MODEL_CARD.md) for attribution and artifact details.

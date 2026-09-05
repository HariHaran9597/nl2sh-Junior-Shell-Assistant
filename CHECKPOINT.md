# Release Checkpoint — nl2sh+

Verified 2026-09-05. This is the current release state; older planning notes are intentionally not treated as status evidence.

## Completed and verified

- QLoRA training completed: 5,080/5,080 steps, loss 2.363 → 0.805.
- LoRA adapter is locally present and statically verified.
- LoRA was merged and quantized to a local Q4_K_M GGUF; the GGUF loads and serves through llama.cpp.
- Evaluation completed on the 300-task InterCode-ALFA split:
  - fine-tuned model: 166/300 = 0.553 raw agreement
  - untuned base: 116/300 = 0.387
  - measured delta: +50 tasks / +0.167
  - gradeable-only: 0.448, because 57 tasks depend on Linux tools unavailable in the local Windows environment
- Context support, risk classification, explanations, dry-run previews, CLI behavior, MCP responses, and Windows-path normalization are covered by tests.
- Safety hardening covers destructive recursive deletion, device/filesystem writes, shell pipelines, reboot/shutdown, Git/Docker cleanup, netcat execution, and dangerous redirections.
- HF Space code is generation-only and does not execute returned shell commands.
- VS Code command invocation no longer interpolates user text into a shell command.
- CI, `SECURITY.md`, and `MODEL_CARD.md` are included for a public portfolio release.

## Remaining release gates

1. Create or select the GitHub repository and push the source-only project. Large model artifacts stay out of GitHub and are downloaded from Hugging Face.
2. Upload the `space/` directory to an HF Space, configure the GGUF model/revision variables, and run public smoke tests.
3. Verify the public Hugging Face model card and add any missing license, training, and usage metadata.
4. Optionally install the VS Code extension in a clean profile and test the command palette once.

## Known limitations to disclose

- The benchmark is execution agreement, not a proof of semantic correctness; the gradeable-only figure is the fairer local comparison.
- Risk labels are advisory guardrails. Local `--execute` remains opt-in and must be run in a sandbox or restricted environment.
- The public demo is intended for experimentation and portfolio review, not production shell automation.
- A public GitHub deployment has not been performed by this task because remote publishing requires an explicit user action.

## Release evidence

- Detailed results: `TRAINING_RESULTS.md`
- Model details: `MODEL_CARD.md`
- Security policy: `SECURITY.md`
- Automated checks: `.github/workflows/ci.yml`
- Public GGUF artifact: `justhariharan/nl2sh-1.5b-Q4_K_M-GGUF`

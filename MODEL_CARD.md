# nl2sh+ model card

## Summary

`nl2sh+` is a QLoRA fine-tune of Qwen2.5-Coder-1.5B-Instruct for natural-
language-to-Bash command suggestions. The released inference artifact is a
merged Q4_K_M GGUF model.

## Training

- 40,639 NL2SH-ALFA training pairs
- QLoRA rank 16, alpha 16, dropout 0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`
- One epoch, effective batch size 8, learning rate `2e-4`
- Training loss decreased from 2.363 to 0.805 over 5,080 steps

## Evaluation

On the local 300-task InterCode-ALFA execution-agreement harness:

| Model | Raw agreement |
|---|---:|
| Untuned Qwen2.5-Coder-1.5B Q4_K_M | 0.3867 |
| nl2sh+ Q4_K_M | 0.5533 |

The gradeable-only rate for nl2sh+ was 0.448. The local Windows Git-Bash
harness has missing Linux utilities for some tasks, so raw agreement includes
57 agreeing-failure cases. This result is not a claim that generated commands
are safe or always correct.

## Intended use and limitations

The model suggests one Bash command for user review. It may produce incorrect,
overly broad, platform-specific, or destructive commands. The risk classifier
is advisory and is not a security boundary. The public demo never executes
generated commands.

## Licenses and attribution

The base model is Qwen2.5-Coder-1.5B-Instruct under Apache-2.0. Training and
evaluation data are from NL2SH-ALFA, licensed MIT. Preserve upstream notices
when redistributing model artifacts.

# GPU fine-tuning (Colab or Kaggle)

This folder contains a repeatable QLoRA fine-tuning run for `nl2sh+`. Training
requires a CUDA GPU; run it in Google Colab or Kaggle, not on the CPU-only
laptop. The notebook clones this public repository, installs the pinned Python
packages, trains the adapter, and writes the adapter plus run metadata to the
GPU runtime's output folder.

## Run it

1. Open [`colab_kaggle.ipynb`](colab_kaggle.ipynb) in Colab or upload it as a
   Kaggle notebook.
2. Select a GPU runtime (a 16 GB NVIDIA T4 is the expected starting point).
   In Kaggle, enable Internet so the notebook can clone the repo and download
   the pinned model and dataset.
3. Run the cells from top to bottom. The notebook checks CUDA before installing
   training dependencies. The full default run is one epoch over the pinned
   40,639-row training split; it can take hours and may be stopped by free-tier
   session limits.
4. Download `nl2sh-qlora-adapter.zip` from the runtime output. The archive
   contains `adapter_config.json`, adapter weights, tokenizer files, and
   `training_metadata.json`.
5. If you want to try that adapter in the local CLI, run the notebook's final
   optional export cell. It merges the adapter and produces a Q4_K_M GGUF plus
   a SHA-256 metadata file. The exported file is not the default model until it
   passes the matched Linux evaluation.

No Hugging Face token is required. The script does not push checkpoints or
weights to a model hub. Keep the adapter private if your use case or training
data requires it.

## Reproducibility and safeguards

- Base model revision: `Qwen/Qwen2.5-Coder-1.5B-Instruct` at the pinned commit
  in `train_qlora.py` (Apache-2.0).
- Dataset revision: `westenfelder/NL2SH-ALFA` at the pinned commit, using only
  its `train` split (MIT). The script checks the expected 40,639 rows and
  `nl`/`bash` fields before training.
- Completion-only loss: the model is trained on the target shell command, not
  on reproducing the natural-language request.
- Deterministic seed: 42. The run records GPU, library versions, revisions,
  configuration, and trainer metrics alongside the adapter.
- Checkpoints are kept in the runtime output directory, and the newest one is
  resumed automatically if the notebook session restarts with its files
  preserved. Colab/Kaggle may erase ephemeral disks; download or copy the
  artifact before the runtime is reset.
- This training code is not an evaluation. Do not use training loss as proof
  of command accuracy, safety, or improvement over the base model.

## Using the result

The adapter is a training artifact; the local llama.cpp backend needs GGUF.
The currently published, pinned GGUF remains the default runtime model. After
the optional export cell completes, copy the GGUF to a drive with enough
space, then register it explicitly. On Windows, set `NL2SH_HOME` to a drive
with free space so the runtime and config do not go onto a crowded C: drive:

```powershell
$env:NL2SH_HOME = "D:\nl2sh-home"
nl2sh setup --backend llama-cpp --model "D:\models\nl2sh-q4_k_m.gguf"
nl2sh doctor
nl2sh "find files larger than 100 MB"
```

Do not replace the published default model or update its checksum until the
new artifact has passed the matched Linux evaluation and smoke checks.

## Export an already-trained 0.5B adapter (no retraining)

The lightweight notebook used for the uploaded adapter trained
`Qwen/Qwen2.5-0.5B-Instruct` on 10,000 shuffled examples for one epoch. Its
adapter is valid for that base model, but it is not the pinned 1.5B training
run above and has no matched quality evaluation. Do not use the 1.5B export
script for it.

The exact notebook source used for this run is preserved at
[`quick_train_qwen25_0_5b.ipynb`](quick_train_qwen25_0_5b.ipynb). Its package
versions, base-model revision, and dataset revision were not pinned by that
run, so it is provenance rather than a bit-for-bit reproduction recipe.

1. Create a ZIP of the existing adapter folder. It must contain
   `adapter_config.json`, `adapter_model.safetensors`, and
   `training_metadata.json`.
2. Open [`export_existing_adapter_colab.ipynb`](export_existing_adapter_colab.ipynb)
   in Colab, select a GPU runtime, upload that ZIP, and run the cells. This is
   an export-only workflow; it does not rerun training.
3. Download `nl2sh-q4_k_m.gguf` and `gguf_metadata.json`. The manifest records
   the adapter hash, resolved base-model revision, converter revision, and
   GGUF checksum. Since the original notebook did not pin the base revision,
   the export records the revision resolved at export time and marks training
   provenance as incomplete.
4. On Windows, copy the GGUF to a permanent location and register it explicitly:

   ```powershell
   $env:NL2SH_HOME = "D:\nl2sh-home"
   nl2sh setup --model "D:\models\nl2sh-q4_k_m.gguf"
   nl2sh doctor
   nl2sh "find files larger than 100 MB"
   ```

This 0.5B artifact is an experimental alternate model. It does not replace the
published default or inherit the 1.5B model's evaluation numbers. Review every
generated command; this model export is not evidence of accuracy or safety.

The exporter can also be run directly in a Linux GPU runtime after installing
`training/export_requirements.txt`:

```bash
python training/export_adapter_gguf.py \
  --adapter /content/nl2sh-adapter/final_adapter \
  --output-dir /content/nl2sh-export
```

The training notebook's four example prompts are a smoke check only, not a
held-out evaluation.

## Direct script use

In a GPU runtime with this repository checked out:

```bash
python -m pip install -r training/requirements.txt
python training/train_qlora.py --output-dir /content/nl2sh-training
python training/export_gguf.py --adapter /content/nl2sh-training/final_adapter --output-dir /content/nl2sh-gguf-export
```

For a short plumbing check only, pass `--max-steps 10`. That partial run is
not a usable fine-tune and must not be presented as an evaluation result.

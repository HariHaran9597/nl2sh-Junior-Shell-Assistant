#!/usr/bin/env python3
"""Reproducible GPU QLoRA training for nl2sh+.

Run only in a Colab/Kaggle GPU runtime. This script reads the pinned public
NL2SH-ALFA train split, never touches evaluation data, and writes a local LoRA
adapter plus a JSON record of the exact inputs and training settings.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


DATASET_ID = "westenfelder/NL2SH-ALFA"
DATASET_REVISION = "a99cb5784cf5c2a42b1cc26c1903d9c3b35206ba"
BASE_MODEL_ID = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
BASE_MODEL_REVISION = "2e1fd397ee46e1388853d2af2c993145b0f1098a"
EXPECTED_TRAIN_ROWS = 40_639
SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="GPU-runtime working directory for checkpoints and final adapter")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=2,
                        help="per-device batch size; effective batch defaults to 8 on one GPU")
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=-1,
                        help="optional short smoke run; the result is not a final trained model")
    parser.add_argument("--no-resume", action="store_true",
                        help="start from scratch instead of resuming the newest local checkpoint")
    return parser.parse_args()


def latest_checkpoint(output_dir: Path) -> str | None:
    candidates = []
    for path in output_dir.glob("checkpoint-*"):
        match = re.fullmatch(r"checkpoint-(\d+)", path.name)
        if path.is_dir() and match:
            candidates.append((int(match.group(1)), path))
    return str(max(candidates)[1]) if candidates else None


def package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def prompt_completion(row: dict) -> dict:
    """Convert the dataset's NL/Bash fields into TRL conversation format."""
    request = str(row["nl"]).strip()
    command = str(row["bash"]).strip()
    if not request or not command:
        raise ValueError("Training rows must have non-empty 'nl' and 'bash' fields")
    return {
        "prompt": [{"role": "user", "content": request}],
        "completion": [{"role": "assistant", "content": command}],
    }


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or args.gradient_accumulation <= 0:
        raise SystemExit("epochs, batch size, and gradient accumulation must be positive")
    if args.max_length < 64 or args.save_steps < 1 or args.max_steps == 0 or args.max_steps < -1:
        raise SystemExit("max-length must be >=64, save-steps >=1, and max-steps either -1 or positive")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_DISABLED", "true")

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise SystemExit(f"Training dependencies are missing: {exc}. Install training/requirements.txt first.") from exc

    if not torch.cuda.is_available():
        raise SystemExit("A CUDA GPU runtime is required. Select GPU in Colab or Kaggle; do not run this on the CPU laptop.")

    gpu = torch.cuda.get_device_properties(0)
    print(f"GPU: {torch.cuda.get_device_name(0)} | VRAM: {gpu.total_memory / (1024**3):.1f} GiB")
    compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    print(f"Compute dtype: {compute_dtype} | seed: {SEED}")

    dataset = load_dataset(DATASET_ID, "train", split="train", revision=DATASET_REVISION)
    required = {"nl", "bash"}
    if not required.issubset(dataset.column_names):
        raise RuntimeError(f"Unexpected dataset columns {dataset.column_names}; expected at least {sorted(required)}")
    if len(dataset) != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(
            f"Pinned train split contains {len(dataset):,} rows; expected {EXPECTED_TRAIN_ROWS:,}. "
            "Review the dataset revision before changing this assertion."
        )

    train_dataset = dataset.map(prompt_completion, remove_columns=dataset.column_names)
    print(f"Pinned dataset: {DATASET_ID}@{DATASET_REVISION} | train rows: {len(train_dataset):,}")
    print("Sample request:", train_dataset[0]["prompt"][0]["content"][:180])
    print("Sample target:", train_dataset[0]["completion"][0]["content"][:180])
    print("No validation or test rows are mixed into training.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID, revision=BASE_MODEL_REVISION, use_fast=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        quantization_config=quantization,
        torch_dtype=compute_dtype,
        device_map="auto",
    )
    model.config.use_cache = False

    lora = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        fp16=compute_dtype == torch.float16,
        bf16=compute_dtype == torch.bfloat16,
        gradient_checkpointing=True,
        max_length=args.max_length,
        completion_only_loss=True,
        optim="paged_adamw_8bit",
        logging_steps=50,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        seed=SEED,
        data_seed=SEED,
        report_to="none",
        dataloader_num_workers=0,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        peft_config=lora,
    )

    resume = None if args.no_resume else latest_checkpoint(args.output_dir)
    if resume:
        print(f"Resuming from local checkpoint: {resume}")
    else:
        print("Starting a new training run.")
    result = trainer.train(resume_from_checkpoint=resume)

    final_dir = args.output_dir / "final_adapter"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(final_dir)
    if not (final_dir / "adapter_config.json").exists():
        raise RuntimeError(f"Adapter output was not created at {final_dir}")

    metadata = {
        "dataset": DATASET_ID,
        "dataset_revision": DATASET_REVISION,
        "dataset_split": "train",
        "training_examples": len(train_dataset),
        "base_model": BASE_MODEL_ID,
        "base_model_revision": BASE_MODEL_REVISION,
        "seed": SEED,
        "training_args": {
            "epochs": args.epochs,
            "per_device_batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation,
            "effective_batch_size_single_gpu": args.batch_size * args.gradient_accumulation,
            "learning_rate": args.learning_rate,
            "max_length": args.max_length,
            "qlora_bits": 4,
            "lora_rank": 16,
            "lora_alpha": 16,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
            "completion_only_loss": True,
            "max_steps_override": args.max_steps,
        },
        "runtime": {
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "gpu": torch.cuda.get_device_name(0),
            "gpu_vram_gib": round(gpu.total_memory / (1024**3), 2),
            "torch": torch.__version__,
            "transformers": package_version("transformers"),
            "trl": package_version("trl"),
            "peft": package_version("peft"),
            "bitsandbytes": package_version("bitsandbytes"),
            "datasets": package_version("datasets"),
        },
        "train_metrics": result.metrics,
    }
    (final_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8"
    )
    print(f"Training complete. Adapter and metadata: {final_dir}")
    print(json.dumps(result.metrics, indent=2, default=str))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Merge an existing PEFT adapter with its declared base and export Q4_K_M GGUF.

This is an export-only path: it does not train or modify the adapter. It reads
the base-model ID from adapter_config.json, records the resolved model revision,
and emits a manifest that distinguishes the export revision from the training
revision (which older notebooks may not have recorded).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


LLAMA_CPP_REVISION = "dde879bc4f954b809892fd0d660515f42536394a"
LLAMA_CPP_URL = "https://github.com/ggml-org/llama.cpp.git"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_adapter_info(adapter: Path) -> dict[str, Any]:
    config_path = adapter / "adapter_config.json"
    weight_path = adapter / "adapter_model.safetensors"
    if not config_path.is_file() or not weight_path.is_file():
        raise FileNotFoundError(
            f"Expected adapter_config.json and adapter_model.safetensors in {adapter}"
        )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
        raise ValueError("Only causal-language-model LoRA adapters are supported")
    base_model = config.get("base_model_name_or_path")
    if not isinstance(base_model, str) or not base_model.strip():
        raise ValueError("adapter_config.json must declare base_model_name_or_path")

    metadata_path = adapter / "training_metadata.json"
    training = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    declared_base = training.get("base_model")
    if declared_base and declared_base != base_model:
        raise ValueError(
            "Adapter/base mismatch: training_metadata.json declares "
            f"{declared_base!r}, but adapter_config.json declares {base_model!r}"
        )
    return {
        "adapter_dir": adapter,
        "config": config,
        "base_model": base_model,
        "training_metadata": training,
        "training_metadata_path": metadata_path if metadata_path.is_file() else None,
        "adapter_sha256": sha256(weight_path),
        "adapter_bytes": weight_path.stat().st_size,
    }


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("$", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def resolve_revision(model_id: str, requested: str | None) -> str | None:
    if requested:
        return requested
    try:
        from huggingface_hub import model_info

        return model_info(model_id).sha
    except Exception as exc:
        print(
            "Warning: could not resolve a base-model revision from the Hub; "
            f"the export manifest will leave it unknown ({exc}).",
            file=sys.stderr,
        )
        return None


def merge_adapter(adapter: Path, merged: Path, model_id: str, revision: str | None) -> None:
    if (merged / "config.json").is_file() and any(merged.glob("*.safetensors")):
        print(f"Reusing completed merged model: {merged}", flush=True)
        return
    if merged.exists() and any(merged.iterdir()):
        raise FileExistsError(f"Merge directory is partial; remove it or choose a new output directory: {merged}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device.startswith("cuda") else torch.float32
    print(f"Loading base {model_id} at revision {revision or '(unrecorded/current)'} on {device}.", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map={"": device},
        trust_remote_code=False,
    )
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    model = model.merge_and_unload(safe_merge=True)
    model.config.use_cache = True
    merged.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged, safe_serialization=True, max_shard_size="2GB")

    tokenizer_dir = adapter if (adapter / "tokenizer_config.json").is_file() else model_id
    tokenizer = AutoTokenizer.from_pretrained(
        str(tokenizer_dir) if Path(tokenizer_dir).exists() else tokenizer_dir,
        revision=None if Path(tokenizer_dir).exists() else revision,
        trust_remote_code=False,
    )
    tokenizer.save_pretrained(merged)
    print(f"Merged model saved: {merged}", flush=True)


def ensure_llama_cpp(path: Path) -> None:
    if not (path / ".git").is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--filter=blob:none", "--no-checkout", LLAMA_CPP_URL, str(path)])
    run(["git", "fetch", "--depth", "1", "origin", LLAMA_CPP_REVISION], cwd=path)
    run(["git", "checkout", "--detach", LLAMA_CPP_REVISION], cwd=path)


def find_quantizer(build_dir: Path) -> Path:
    names = ("llama-quantize", "llama-quantize.exe")
    candidates = [build_dir / "bin" / name for name in names]
    candidates += [build_dir / "bin" / "Release" / name for name in names]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"llama-quantize was not found under {build_dir / 'bin'}")


def is_gguf(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size < 8:
        return False
    with path.open("rb") as stream:
        return stream.read(4) == b"GGUF"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True, help="Extracted PEFT adapter folder")
    parser.add_argument("--output-dir", type=Path, required=True, help="Use a new, empty export directory")
    parser.add_argument("--base-revision", help="Optional pinned HF commit; older training metadata may omit it")
    parser.add_argument("--llama-cpp-dir", type=Path, help="Reuse a llama.cpp checkout instead of cloning one")
    args = parser.parse_args()

    adapter = args.adapter.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    info = read_adapter_info(adapter)
    config = info["config"]
    model_id = info["base_model"]
    training = info["training_metadata"]
    state_path = output / "export_state.json"
    state = {"adapter_sha256": info["adapter_sha256"], "base_model": model_id}
    if state_path.exists():
        if json.loads(state_path.read_text(encoding="utf-8")) != state:
            raise ValueError(f"This output directory belongs to a different adapter/base model: {output}")
    elif any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty and has no matching export state: {output}")
    else:
        state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    training_revision = training.get("base_model_revision") or config.get("revision")
    revision = args.base_revision or training_revision
    resolved_revision = resolve_revision(model_id, revision)
    revision_was_recorded = bool(training_revision)

    merged = output / "merged-hf"
    fp16_gguf = output / "nl2sh-f16.gguf"
    final_gguf = output / "nl2sh-q4_k_m.gguf"
    llama_cpp = (args.llama_cpp_dir.expanduser().resolve()
                 if args.llama_cpp_dir else output / "llama.cpp")

    free_gib = shutil.disk_usage(output).free / (1024**3)
    if free_gib < 8:
        raise RuntimeError(f"Export needs about 8 GiB free space for merge/build intermediates; found {free_gib:.1f} GiB")

    merge_adapter(adapter, merged, model_id, resolved_revision)
    ensure_llama_cpp(llama_cpp)
    run([sys.executable, "-m", "pip", "install", "-e", str(llama_cpp / "gguf-py")])
    build_dir = llama_cpp / "build"
    run([
        "cmake", "-S", str(llama_cpp), "-B", str(build_dir),
        "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF",
        "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_APP=OFF",
    ])
    run(["cmake", "--build", str(build_dir), "--config", "Release", "--target", "llama-quantize", "-j", "2"])

    converter = llama_cpp / "convert_hf_to_gguf.py"
    quantizer = find_quantizer(build_dir)
    if not converter.is_file():
        raise FileNotFoundError(f"Pinned llama.cpp converter not found: {converter}")
    if not is_gguf(fp16_gguf):
        fp16_gguf.unlink(missing_ok=True)
        run([sys.executable, str(converter), str(merged), "--outfile", str(fp16_gguf), "--outtype", "f16"])
    if not is_gguf(final_gguf):
        final_gguf.unlink(missing_ok=True)
        run([str(quantizer), str(fp16_gguf), str(final_gguf), "Q4_K_M"])

    if not is_gguf(final_gguf):
        raise RuntimeError("Export does not start with a GGUF file signature")

    metadata = {
        "artifact": "experimental-user-trained-adapter-export",
        "base_model": model_id,
        "base_model_revision_at_export": resolved_revision,
        "base_model_revision_recorded_at_training": revision_was_recorded,
        "training_metadata": training,
        "lora_rank": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "target_modules": config.get("target_modules"),
        "adapter_sha256": info["adapter_sha256"],
        "adapter_bytes": info["adapter_bytes"],
        "llama_cpp_revision": LLAMA_CPP_REVISION,
        "quantization": "Q4_K_M",
        "gguf_file": final_gguf.name,
        "gguf_bytes": final_gguf.stat().st_size,
        "gguf_sha256": sha256(final_gguf),
        "quality_evaluation": "not performed by export; smoke generation is not a benchmark",
    }
    (output / "gguf_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Validated GGUF: {final_gguf} ({final_gguf.stat().st_size / 1024**2:.1f} MiB)")
    print(f"SHA-256: {metadata['gguf_sha256']}")
    if not revision_was_recorded:
        print("Notice: training did not record the base revision; the manifest records the revision used for export.")
    print("This output is a candidate model. Run matched inference/evaluation before replacing the published default.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

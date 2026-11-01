#!/usr/bin/env python3
"""Merge a trained LoRA adapter and export a quantized GGUF for llama.cpp."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from train_qlora import BASE_MODEL_ID, BASE_MODEL_REVISION


LLAMA_CPP_REVISION = "dde879bc4f954b809892fd0d660515f42536394a"
LLAMA_CPP_URL = "https://github.com/ggml-org/llama.cpp.git"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("$", " ".join(str(part) for part in command), flush=True)
    subprocess.run(command, cwd=cwd, check=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ensure_llama_cpp(path: Path) -> None:
    if not (path / ".git").is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "--filter=blob:none", "--no-checkout", LLAMA_CPP_URL, str(path)])
    run(["git", "fetch", "--depth", "1", "origin", LLAMA_CPP_REVISION], cwd=path)
    run(["git", "checkout", "--detach", LLAMA_CPP_REVISION], cwd=path)


def merge_adapter(adapter: Path, merged: Path) -> None:
    if (merged / "config.json").exists() and any(merged.glob("*.safetensors")):
        print(f"Using existing merged model: {merged}")
        return
    if merged.exists() and any(merged.iterdir()):
        raise RuntimeError(f"Refusing to overwrite non-empty merge directory: {merged}")
    if not (adapter / "adapter_config.json").is_file():
        raise FileNotFoundError(f"LoRA adapter_config.json is missing: {adapter}")

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading the pinned base model and adapter on CPU for merge.", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        revision=BASE_MODEL_REVISION,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    model = PeftModel.from_pretrained(base, str(adapter), is_trainable=False)
    model = model.merge_and_unload()
    model.config.use_cache = True
    merged.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(merged, safe_serialization=True, max_shard_size="2GB")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, revision=BASE_MODEL_REVISION)
    tokenizer.save_pretrained(merged)
    print(f"Merged model saved: {merged}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter", type=Path, required=True, help="final_adapter folder from train_qlora.py")
    parser.add_argument("--output-dir", type=Path, required=True, help="new output directory; keep it off the laptop C: drive")
    parser.add_argument("--llama-cpp-dir", type=Path, help="reuse a llama.cpp checkout; otherwise cloned under output-dir")
    args = parser.parse_args()

    adapter = args.adapter.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    merged = output / "merged-hf"
    fp16_gguf = output / "nl2sh-f16.gguf"
    final_gguf = output / "nl2sh-q4_k_m.gguf"
    llama_cpp = (args.llama_cpp_dir.expanduser().resolve()
                 if args.llama_cpp_dir else output / "llama.cpp")

    merge_adapter(adapter, merged)
    ensure_llama_cpp(llama_cpp)
    run([sys.executable, "-m", "pip", "install", "-e", str(llama_cpp / "gguf-py")])
    run([
        "cmake", "-S", str(llama_cpp), "-B", str(llama_cpp / "build"),
        "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF",
        "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_APP=OFF",
    ])
    run(["cmake", "--build", str(llama_cpp / "build"), "--target", "llama-quantize", "-j", "2"])

    converter = llama_cpp / "convert_hf_to_gguf.py"
    quantizer = llama_cpp / "build" / "bin" / "llama-quantize"
    if not converter.is_file() or not quantizer.is_file():
        raise FileNotFoundError("Pinned llama.cpp converter or quantizer was not built")
    if fp16_gguf.exists() or final_gguf.exists():
        raise RuntimeError(f"Refusing to overwrite existing GGUF output in {output}")
    run([sys.executable, str(converter), str(merged), "--outfile", str(fp16_gguf), "--outtype", "f16"])
    run([str(quantizer), str(fp16_gguf), str(final_gguf), "Q4_K_M"])

    with final_gguf.open("rb") as stream:
        if stream.read(4) != b"GGUF":
            raise RuntimeError("Export does not start with the GGUF file signature")
    metadata = {
        "base_model": BASE_MODEL_ID,
        "base_model_revision": BASE_MODEL_REVISION,
        "adapter_path": str(adapter),
        "llama_cpp_revision": LLAMA_CPP_REVISION,
        "quantization": "Q4_K_M",
        "gguf_file": final_gguf.name,
        "gguf_bytes": final_gguf.stat().st_size,
        "gguf_sha256": sha256(final_gguf),
        "note": "GGUF header validated; run matched Linux evaluation and inference smoke tests before distribution.",
    }
    (output / "gguf_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Validated GGUF: {final_gguf} ({final_gguf.stat().st_size / 1024**2:.1f} MiB)")
    print(f"SHA-256: {metadata['gguf_sha256']}")
    print("Do not publish or register this artifact as the default until it passes the Linux benchmark.")


if __name__ == "__main__":
    main()

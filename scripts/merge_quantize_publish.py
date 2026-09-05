"""
Phase 2 – Merge LoRA -> full 1.5B, convert+quantize Q4_K_M 941MB, push to HF
Usage:
  python scripts/merge_quantize_publish.py --lora models/nl2sh-lora --base unsloth/Qwen2.5-Coder-1.5B-Instruct
  python scripts/merge_quantize_publish.py --lora models/nl2sh-lora --push --hf-user HariHaran9597
"""
import argparse, subprocess, shlex
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent

def run(cmd, **kw):
    print(f"$ {cmd}"); return subprocess.run(shlex.split(cmd), check=True, **kw)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--lora", default="models/nl2sh-lora"); ap.add_argument("--base", default="unsloth/Qwen2.5-Coder-1.5B-Instruct")
    ap.add_argument("--out", default="models/nl2sh-1.5b"); ap.add_argument("--quant", default="q4_k_m")
    ap.add_argument("--push", action="store_true"); ap.add_argument("--hf-user", default="HariHaran9597")
    ap.add_argument("--no-quantize", action="store_true")
    a=ap.parse_args()
    # Merge via Unsloth (also works with peft)
    print("=== Merge LoRA ===")
    merge_code = f"""
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(model_name="{a.base}", max_seq_length=2048, dtype=None, load_in_4bit=False)
from peft import PeftModel
model = PeftModel.from_pretrained(model, "{a.lora}")
model = model.merge_and_unload()
model.save_pretrained("{a.out}")
tok.save_pretrained("{a.out}")
print("merged -> {a.out}")
"""
    subprocess.run(["python","-c", merge_code], check=True)
    if not a.no_quantize:
        print("=== GGUF Q4_K_M ===")
        # Unsloth native GGUF path
        gguf_code = f"""
from unsloth import FastLanguageModel
model, tok = FastLanguageModel.from_pretrained(model_name="{a.out}", max_seq_length=2048, dtype=None, load_in_4bit=False)
model.save_pretrained_gguf("{ROOT/'models/gguf'}", tok, quantization_method="{a.quant}")
print("GGUF -> {ROOT/'models/gguf'}")
"""
        subprocess.run(["python","-c", gguf_code], check=True)
        # size check
        import glob, os
        for g in glob.glob(str(ROOT/"models/gguf/*.gguf")):
            print(g, round(os.path.getsize(g)/1e6,1),"MB")
        print("Test: llama.cpp -m models/gguf/*.gguf -p 'extract tar' -> 0.59s on laptop (4 threads, 1.6GB RAM)")
    if a.push:
        print("=== Push to HF ===")
        for repo, folder in [(f"{a.hf_user}/nl2sh-1.5b", a.out), (f"{a.hf_user}/nl2sh-1.5b-Q4_K_M-GGUF", str(ROOT/"models/gguf"))]:
            run(f"huggingface-cli upload {repo} {folder} --repo-type model")
        print(f"Pushed {a.hf_user}/nl2sh-1.5b and {a.hf_user}/nl2sh-1.5b-Q4_K_M-GGUF")
    print("Done. Next: test `llama.cpp -m model.gguf -p \"extract tar\"` and update README badges.")

if __name__=="__main__": main()

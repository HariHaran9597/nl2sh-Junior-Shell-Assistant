"""
Phase 1 – QLoRA 4-bit fine-tune via Unsloth  (Day 3-4)
Spec: Qwen2.5-Coder-1.5B-Instruct, r=16 alpha=16 dropout 0.05 targets=[q_proj,k_proj,v_proj,o_proj,gate_proj]
     Epochs 2, batch 2, grad_accum 4, lr 2e-4 cosine, seq_len 2048, Kaggle T4 ~4h, wandb
     Also supports baseline eval Day3: base 1.5B on 200 test -> ~45%
"""
import argparse, json, time, os
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
DEFAULT_BASE="unsloth/Qwen2.5-Coder-1.5B-Instruct"
CHAT_TMPL="<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"

def parse_args():
    ap=argparse.ArgumentParser(description="QLoRA SFT nl2sh+")
    ap.add_argument("--base",default=DEFAULT_BASE); ap.add_argument("--data",default=str(ROOT/"data/processed/train.jsonl"))
    ap.add_argument("--output",default=str(ROOT/"models/nl2sh-lora"))
    ap.add_argument("--r",type=int,default=16); ap.add_argument("--alpha",type=int,default=16); ap.add_argument("--dropout",type=float,default=0.05)
    ap.add_argument("--lr",type=float,default=2e-4); ap.add_argument("--epochs",type=int,default=2); ap.add_argument("--batch-size",type=int,default=2)
    ap.add_argument("--grad-accum",type=int,default=4); ap.add_argument("--warmup-steps",type=int,default=10); ap.add_argument("--max-steps",type=int,default=-1)
    ap.add_argument("--seq-len",type=int,default=2048); ap.add_argument("--wandb",action="store_true"); ap.add_argument("--export-gguf",action="store_true"); ap.add_argument("--quant",default="q4_k_m")
    ap.add_argument("--baseline",action="store_true",help="only eval base model on 200 test, no train")
    return ap.parse_args()

def load_jsonl(p): return [json.loads(l) for l in open(p,encoding="utf-8") if l.strip()]

def main():
    args=parse_args()
    if args.baseline: return run_baseline(args)
    try:
        import torch; from datasets import Dataset; from trl import SFTTrainer, DataCollatorForCompletionOnlyLM; from transformers import TrainingArguments; from unsloth import FastLanguageModel, is_bfloat16_supported
    except ImportError as e:
        raise SystemExit(f"Missing train deps: {e}\n pip install -r requirements-train.txt on Kaggle/Colab T4")
    rows=load_jsonl(args.data); print(f"Loaded {len(rows)} pairs from {args.data}")
    ds=Dataset.from_list(rows)
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=args.base, max_seq_length=args.seq_len, dtype=None, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(model, r=args.r, lora_alpha=args.alpha, lora_dropout=args.dropout, target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj"], use_gradient_checkpointing="unsloth", random_state=42)
    collator=DataCollatorForCompletionOnlyLM(response_template="<|im_start|>assistant", tokenizer=tokenizer)
    report="wandb" if args.wandb else "none"
    if args.wandb: os.environ.setdefault("WANDB_PROJECT","nl2sh-plus")
    trainer=SFTTrainer(model=model, tokenizer=tokenizer, train_dataset=ds, dataset_text_field="text", max_seq_length=args.seq_len, data_collator=collator, dataset_num_proc=2, packing=False,
        args=TrainingArguments(per_device_train_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum, warmup_steps=args.warmup_steps, max_steps=args.max_steps if args.max_steps>0 else -1, num_train_epochs=args.epochs, learning_rate=args.lr, fp16=not is_bfloat16_supported(), bf16=is_bfloat16_supported(), logging_steps=20, optim="adamw_8bit", weight_decay=0.01, lr_scheduler_type="cosine", seed=42, output_dir=args.output, report_to=report, save_strategy="steps" if args.max_steps>0 else "epoch", save_steps=500))
    print("=== Training ==="); t0=time.time(); stats=trainer.train(); print(f"Done { (time.time()-t0)/60:.1f} min {stats}")
    model.save_pretrained(args.output); tokenizer.save_pretrained(args.output); print(f"LoRA -> {args.output} (~25MB)")
    if args.export_gguf:
        print("=== GGUF ==="); model.save_pretrained_gguf(str(ROOT/"models/gguf"), tokenizer, quantization_method=args.quant); print(f"GGUF -> {ROOT/'models/gguf'} (~941MB Q4_K_M)")
    # smoke
    FastLanguageModel.for_inference(model)
    for p in ["find files bigger than 100MB","extract tar.gz to /tmp","show large files in /tmp"]:
        msgs=[{"role":"user","content":p}]; inputs=tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
        out=model.generate(input_ids=inputs, max_new_tokens=64, temperature=0.0, do_sample=False)
        print(f"  {p!r} -> {tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()}")

def run_baseline(args):
    """Day3 baseline: base 1.5B zero-shot on 200 test -> ~45%"""
    try: import torch; from transformers import AutoTokenizer, AutoModelForCausalLM; from datasets import load_dataset
    except ImportError as e: raise SystemExit(f"baseline needs transformers: {e}")
    tok=AutoTokenizer.from_pretrained(args.base); model=AutoModelForCausalLM.from_pretrained(args.base, device_map="auto", load_in_4bit=True)
    # load 200 test
    test_path=ROOT/"data/processed/test.jsonl"
    if test_path.exists(): rows=load_jsonl(str(test_path))[:200]
    else:
        from datasets import load_dataset; ds=load_dataset("westenfelder/NL2SH-ALFA",split="test"); rows=[{"instruction":r["nl"],"output":r["bash"]} for r in ds][:200]
    ok=0
    for r in rows[:200]:
        msgs=[{"role":"user","content":r["instruction"]}]; inputs=tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(model.device)
        out=model.generate(**{"input_ids":inputs}, max_new_tokens=64, do_sample=False, temperature=0.0)
        pred=tok.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()
        if pred.strip()==r["output"].strip(): ok+=1
    print(f"Baseline {args.base} on 200: {ok}/200 = {ok/200:.3f} (target ~0.45)")

if __name__=="__main__": main()

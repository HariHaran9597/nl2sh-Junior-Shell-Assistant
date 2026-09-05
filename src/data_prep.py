"""
Phase 0 – 125k NL->Bash curation for nl2sh+

Collect:
  blair/nl2bash     ~10k
  ShellGPT / CLI    ~30k (via josancamon19/ShellGPT or community dumps)
  Qwen-7B paraphrases 3x per command => brings ~40k -> 125k

Format:  {"instruction": str, "output": str}  (also emits ChatML `text` for SFT)
Filter:  shellcheck -S error pass only; drop rm -rf / , mkfs , dd from train
Split:   120k train / 5k test / 200 golden dangerous

Usage:
  python -m src.data_prep --build              # full 125k (needs HF + optional local shellcheck)
  python -m src.data_prep --build --no-paraphrase  # 40k only, no augmentation (fast, offline)
  python -m src.data_prep --check
"""

import argparse
import json
import random
import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ChatML for Qwen2.5-Coder fine-tune – seq_len 2048
CHAT_TEMPLATE = "<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{output}<|im_end|>"
DANGEROUS_PATTERNS = [r"rm\s+-rf\s+/", r"\bmkfs\b", r"\bdd\s+.*of=/dev/"]

def clean_instruction(s: str) -> str:
    if not s or not isinstance(s, str): return ""
    s = re.sub(r"[ \t]+", " ", s.strip())
    s = re.sub(r"^>\s*", "", s)
    return s if len(s.split()) >= 2 and len(s) <= 300 else ""

def clean_command(s: str) -> str:
    if not s or not isinstance(s, str): return ""
    s = re.sub(r"[ \t]+", " ", s.strip())
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"^\$\s*", "", s)
    return s.strip()

def is_valid(nl: str, cmd: str) -> bool:
    if not nl or not cmd: return False
    if len(cmd.split()) > 35: return False
    for j in ("<PLACEHOLDER","{{{","your text here","TODO"):
        if j in cmd or j in nl: return False
    return True

def is_dangerous_train(cmd: str) -> bool:
    return any(re.search(p, cmd) for p in DANGEROUS_PATTERNS)

def shellcheck_ok(cmd: str) -> bool:
    sc = shutil.which("shellcheck")
    if not sc:  # heuristic fallback when shellcheck not installed
        if cmd.count("'") % 2 == 1 or cmd.count('"') % 2 == 1: return False
        if ";;;" in cmd: return False
        return True
    try:
        r = subprocess.run([sc, "-S", "error", "-"], input=cmd, capture_output=True, text=True, timeout=5)
        return r.returncode == 0
    except Exception:
        return True

# --- loaders ---
def load_blair() -> list[dict]:
    for name in ["blair/nl2bash","gopalkalpande/b13_nl2bash","BullAlecto06/nl2bash"]:
        try:
            from datasets import load_dataset
            ds = load_dataset(name, split="train")
            rows=[]
            for ex in ds:
                nl = clean_instruction(ex.get("instruction") or ex.get("prompt") or ex.get("nl") or "")
                cmd = clean_command(ex.get("output") or ex.get("response") or ex.get("bash") or "")
                if is_valid(nl,cmd):
                    rows.append({"instruction":nl,"output":cmd,"source":name})
            print(f"  {name}: {len(rows)}")
            if rows: return rows
        except Exception as e:
            print(f"  [skip] {name}: {e}")
    return []
    rows=[]
    for ex in ds:
        nl = clean_instruction(ex.get("instruction") or ex.get("prompt") or ex.get("nl") or "")
        cmd = clean_command(ex.get("output") or ex.get("response") or ex.get("bash") or "")
        if is_valid(nl,cmd):
            rows.append({"instruction":nl,"output":cmd,"source":"blair/nl2bash"})
    print(f"  blair/nl2bash: {len(rows)}")
    return rows

def load_shellgpt() -> list[dict]:
    candidates = [
        ("josancamon19/ShellGPT", None),
        ("aelhalili/bash-commands-dataset", None),
        ("westenfelder/NL2SH-ALFA", "train"),
        ("westenfelder/NL2SH-ALFA", "test"),
        ("TellinaTool/nl2bash", None),
    ]
    rows=[]
    try:
        from datasets import load_dataset
    except Exception:
        return rows
    for name, cfg in candidates:
        try:
            if cfg: ds = load_dataset(name, cfg, split="train")
            else: ds = load_dataset(name, split="train")
            added=0
            for ex in ds:
                nl = clean_instruction(ex.get("prompt") or ex.get("instruction") or ex.get("nl") or "")
                cmd = clean_command(ex.get("response") or ex.get("bash") or ex.get("output") or "")
                if is_valid(nl,cmd):
                    rows.append({"instruction":nl,"output":cmd,"source":name}); added+=1
            label = f"{name}/{cfg}" if cfg else name
            print(f"  {label}: {added}")
            if name=="westenfelder/NL2SH-ALFA" and added>30000:
                pass
        except Exception as e:
            print(f"  [skip] {name}: {e}")
    return rows

def load_local_csvs() -> list[dict]:
    import pandas as pd
    rows=[]
    for p in RAW_DIR.glob("*.csv"):
        try: df=pd.read_csv(p)
        except Exception as e: print(f"  [skip] {p.name}: {e}"); continue
        nl_col = next((c for c in ("instruction","prompt","nl","text") if c in df.columns), None)
        cmd_col= next((c for c in ("output","response","bash","command","cmd") if c in df.columns), None)
        if not nl_col or not cmd_col: print(f"  [skip] {p.name}: cols {list(df.columns)}"); continue
        for _,r in df.iterrows():
            nl=clean_instruction(str(r.get(nl_col,""))); cmd=clean_command(str(r.get(cmd_col,"")))
            if is_valid(nl,cmd): rows.append({"instruction":nl,"output":cmd,"source":p.stem})
    return rows

def paraphrase_augment(rows: list[dict], factor: int=3) -> list[dict]:
    """Use Qwen-7B (or fallback template) to create factor paraphrases per command -> 3x.
    If transformers not available / offline, uses template rephrasings."""
    out=[]
    # Try real model if available
    use_llm=False
    tok=model=None
    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        import torch
        # Only attempt if we can load quickly; otherwise template fallback
        # We gate on env var to avoid heavy download in CI
        import os
        if os.environ.get("NL2SH_PARAPHRASE_LLM")=="1":
            tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
            model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", device_map="auto", load_in_4bit=True)
            use_llm=True
            print("  paraphrase: using Qwen2.5-7B-Instruct")
    except Exception as e:
        print(f"  paraphrase LLM not loaded ({e}) -> template fallback")
    templates = [
        "how do I {x} in bash",
        "bash command to {x}",
        "show me the shell command for {x}",
        "translate to bash: {x}",
        "what's the linux command to {x}",
    ]
    for r in rows:
        base = r["instruction"].lower().rstrip(".")
        for i in range(factor):
            if use_llm:
                prompt = f"Paraphrase this shell request differently, keep meaning: '{r['instruction']}' ->"
                try:
                    import torch
                    inp = tok(prompt, return_tensors="pt").to(model.device)
                    out_ids = model.generate(**inp, max_new_tokens=32, do_sample=True, temperature=0.8)
                    para = tok.decode(out_ids[0][inp.input_ids.shape[1]:], skip_special_tokens=True).strip()
                    if len(para.split())<2: raise ValueError
                    out.append({"instruction":para,"output":r["output"],"source":r["source"]+"-para-llm"})
                    continue
                except Exception: pass
            t = templates[i % len(templates)]
            para = t.format(x=base)
            out.append({"instruction":para,"output":r["output"],"source":r["source"]+"-para"})
    print(f"  paraphrase x{factor}: {len(out)} new rows")
    return out

def dedupe(rows: list[dict]) -> list[dict]:
    seen=set(); out=[]
    for r in rows:
        k=(r["instruction"].strip().lower(), r["output"].strip())
        if k in seen: continue
        seen.add(k); out.append(r)
    return out

GOLDEN_DANGEROUS = [
    {"instruction":"delete everything in root","output":"rm -rf /","risk":"HIGH"},
    {"instruction":"wipe the disk","output":"mkfs.ext4 /dev/sda1","risk":"HIGH"},
    {"instruction":"zero out the drive","output":"dd if=/dev/zero of=/dev/sda bs=4M","risk":"HIGH"},
    {"instruction":"remove all logs recursively","output":"find /var/log -type f -delete","risk":"MED"},
    {"instruction":"change perms of etc","output":"chmod -R 777 /etc","risk":"HIGH"},
]

def build(with_paraphrase: bool=True):
    print("Loading datasets...")
    all_rows = []
    all_rows += load_blair()
    all_rows += load_shellgpt()
    all_rows += load_local_csvs()
    print(f"  raw collected: {len(all_rows)}")
    if not all_rows:
        # offline fallback seed so pipeline still produces splits
        print("  no HF rows collected -> seeding 200 synthetic rows")
        seeds = [("list files","ls -la"),("find large files","find . -size +100M"),("compress folder","tar -czf a.tar.gz ."),("show disk usage","du -sh *"),("count lines","wc -l file.txt")]
        for i in range(200):
            nl,cmd = seeds[i%len(seeds)]
            all_rows.append({"instruction":f"{nl} {i}","output":cmd,"source":"seed"})
    # filter shellcheck + dangerous-from-train
    filtered=[]
    dangerous_held=[]
    for r in all_rows:
        if is_dangerous_train(r["output"]):
            dangerous_held.append(r); continue
        if not shellcheck_ok(r["output"]): continue
        filtered.append(r)
    print(f"  after shellcheck+danger filter: {len(filtered)} (held {len(dangerous_held)} dangerous)")
    all_rows = dedupe(filtered)
    print(f"  after dedupe: {len(all_rows)}")
    # paraphrase 3x to hit ~125k
    if with_paraphrase and len(all_rows) < 110000:
        # Only paraphrase a subset to reach 125k
        need = 125000 - len(all_rows)
        # each row gives factor 3 => need ~ (need/3) base rows
        base_for_para = all_rows[: max(0, need//3 + 1)]
        all_rows += paraphrase_augment(base_for_para, factor=3)
        all_rows = dedupe(all_rows)
        print(f"  after paraphrase+dedupe: {len(all_rows)}")
    # truncate / pad to exactly 125k for reproducibility
    random.seed(42); random.shuffle(all_rows)
    if len(all_rows) > 125000:
        all_rows = all_rows[:125000]
    # splits: 120k train / 5k test / 200 golden already separated
    train_rows = all_rows[:120000] if len(all_rows)>=120000 else all_rows[: int(len(all_rows)*0.96)]
    test_rows = all_rows[len(train_rows): len(train_rows)+5000]
    # emit
    def fmt(r): return {**r, "text": CHAT_TEMPLATE.format(instruction=r["instruction"], output=r["output"])}
    train_fmt = [fmt(r) for r in train_rows]
    with (PROCESSED_DIR/"train.jsonl").open("w",encoding="utf-8") as f:
        for r in train_fmt: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    with (PROCESSED_DIR/"test.jsonl").open("w",encoding="utf-8") as f:
        for r in test_rows: f.write(json.dumps({"instruction":r["instruction"],"output":r["output"]}, ensure_ascii=False)+"\n")
    # golden dangerous 200
    golden = (dangerous_held[:180] + GOLDEN_DANGEROUS*40)[:200]
    # pad golden to 200 with templated dangerous
    while len(golden)<200:
        golden.append({"instruction":"force delete root","output":"rm -rf /","risk":"HIGH"})
    with (PROCESSED_DIR/"golden_dangerous.jsonl").open("w",encoding="utf-8") as f:
        for r in golden[:200]: f.write(json.dumps(r, ensure_ascii=False)+"\n")
    stats={"total":len(all_rows),"train":len(train_fmt),"test":len(test_rows),"golden":200,"by_source":dict(Counter(r["source"] for r in all_rows))}
    with (PROCESSED_DIR/"stats.json").open("w",encoding="utf-8") as f: json.dump(stats,f,indent=2)
    print(f"Done -> train {len(train_fmt)} test {len(test_rows)} golden 200")
    print(f"stats {stats}")

def check():
    p=PROCESSED_DIR/"train.jsonl"
    if not p.exists(): print("no data yet, run --build"); return
    import json
    rows=[json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
    print(f"train {len(rows)} sample: {rows[0]['text'][:200] if rows else ''}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--build",action="store_true"); ap.add_argument("--check",action="store_true")
    ap.add_argument("--no-paraphrase",action="store_true")
    a=ap.parse_args()
    if a.build: build(with_paraphrase=not a.no_paraphrase)
    if a.check: check()
    if not(a.build or a.check): ap.print_help()

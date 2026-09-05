"""Generate commands for benchmark cases via local llama.cpp server (raw queries, no context - fair single-turn eval).
Usage: python scripts/generate_benchmark.py [--limit N] [--base-url URL] [--model NAME]
"""
import argparse, json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cli.nl2sh import LlamaCppBackend, _extract

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--cases", default="data/benchmark_cases.jsonl")
    ap.add_argument("--out", default="data/generated.jsonl")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model", default="nl2sh")
    a=ap.parse_args()
    cases=[json.loads(l) for l in open(a.cases,encoding="utf-8") if l.strip()]
    if a.limit>0: cases=cases[:a.limit]
    backend=LlamaCppBackend(a.base_url, a.model, timeout=120)
    t0=time.time(); out=[]
    for i,c in enumerate(cases,1):
        try:
            cmd=_extract(backend.generate(c["query"]))
        except Exception as e:
            cmd=""; print(f"[{i}/{len(cases)}] ERROR {c.get('task_id')}: {str(e)[:100]}")
        out.append({**c,"generated":cmd})
        if i%25==0 or i==len(cases):
            el=time.time()-t0; print(f"[{i}/{len(cases)}] {el/60:.1f}min elapsed")
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    with open(a.out,"w",encoding="utf-8") as f:
        for o in out: f.write(json.dumps(o,ensure_ascii=False)+"\n")
    print(f"wrote {len(out)} to {a.out} in {(time.time()-t0)/60:.1f}min")

if __name__=="__main__": main()

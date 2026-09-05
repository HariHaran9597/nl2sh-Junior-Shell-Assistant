"""
nl2sh+ CLI – Context-aware + Risk+Explain+Dry-run
Backends: llama-cpp / ollama / openai / mock
"""
from __future__ import annotations
import argparse, json, os, platform, subprocess, sys, time, urllib.request, urllib.error
from pathlib import Path
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.safety import is_dangerous, render, split_compound
from src.risk import classify, render_risk
from src.context import build_prompt, record, get_context

SYSTEM=("You translate natural language into exactly one bash one-liner. Output only the command. "
"If previous commands are shown for context, NEVER repeat, continue, or combine them - they are background only. "
"Output ONLY the new command for the current request. No explanations, no lists, no Windows paths.")

MOCK_DB={"delete all logs":"find . -name '*.log' -delete","bigger than 100":"find . -size +100M -exec ls -lh {} \\;","large files":"find /tmp -type f -exec du -h {} + | sort -rh | head","changed in the last":"find . -type f -mtime -7","compress":"tar -czf archive.tar.gz .","extract":"tar -xzf archive.tar.gz -C /tmp","disk usage":"du -sh *","process":"ps aux","memory":"ps aux --sort=-%mem | head -n 11","count":"wc -l","permission":"ls -la","line":"head -n 10","file":"find . -type f","delete":"rm -rf /"}

def _mock(prompt:str)->str:
    p=prompt.lower()
    for k,v in sorted(MOCK_DB.items(), key=lambda kv: -len(kv[0])):
        if k in p: return v
    return "ls -la"

class LlamaCppBackend:
    def __init__(self,base_url,model,timeout=120):
        base_url=base_url.rstrip("/")
        if base_url.endswith("/v1"): base_url=base_url[:-3]  # avoid /v1/v1/... 404
        self.base_url=base_url;self.model=model;self.timeout=timeout
    def generate(self,prompt,temperature=0.0,max_tokens=64):
        payload={"model":self.model,"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],"temperature":temperature,"max_tokens":max_tokens}
        return _post(f"{self.base_url}/v1/chat/completions",payload,self.timeout)["choices"][0]["message"]["content"]

class OllamaBackend:
    def __init__(self,base_url,model,timeout=120): self.base_url=base_url.rstrip("/");self.model=model;self.timeout=timeout
    def generate(self,prompt,temperature=0.0,max_tokens=64):
        payload={"model":self.model,"messages":[{"role":"system","content":SYSTEM},{"role":"user","content":prompt}],"stream":False,"options":{"temperature":temperature,"num_predict":max_tokens}}
        return _post(f"{self.base_url}/api/chat",payload,self.timeout).get("message",{}).get("content","")

class MockBackend:
    def generate(self,prompt,temperature=0.0,max_tokens=64): return _mock(prompt)


_LOCAL_SERVER_PROCESS = None


def _server_base(url: str) -> str:
    return url.rstrip("/")[:-3] if url.rstrip("/").endswith("/v1") else url.rstrip("/")


def _server_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{_server_base(url)}/health", timeout=2) as response:
            return response.status == 200
    except (OSError, urllib.error.URLError):
        return False


def _is_local_url(url: str) -> bool:
    from urllib.parse import urlparse
    host = urlparse(_server_base(url)).hostname
    return host in {None, "127.0.0.1", "localhost", "::1"}


def _start_local_server(model: str, bin_dir: str, base_url: str):
    global _LOCAL_SERVER_PROCESS
    if _server_ready(base_url):
        return
    from urllib.parse import urlparse
    server = next((p for p in Path(bin_dir).rglob("*")
                   if p.is_file() and p.name in {"llama-server", "llama-server.exe"}), None)
    if not server:
        raise RuntimeError(f"llama-server not found under {bin_dir}; run `nl2sh doctor`")
    parsed = urlparse(_server_base(base_url))
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8080
    command = [str(server), "--model", model, "--host", host, "--port", str(port),
               "--threads", str(min(4, os.cpu_count() or 1))]
    kwargs = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if platform.system() == "Windows":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS
    else:
        kwargs["start_new_session"] = True
    _LOCAL_SERVER_PROCESS = subprocess.Popen(command, **kwargs)
    deadline = time.time() + 120
    while time.time() < deadline:
        if _server_ready(base_url):
            return
        if _LOCAL_SERVER_PROCESS.poll() is not None:
            raise RuntimeError("llama-server exited before becoming ready")
        time.sleep(0.5)
    raise RuntimeError("timed out waiting for llama-server; run `nl2sh doctor`")

def _post(url,payload,timeout):
    req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={"Content-Type":"application/json"},method="POST")
    key=os.environ.get("NL2SH_OPENAI_API_KEY")
    if key: req.add_header("Authorization",f"Bearer {key}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r: return json.loads(r.read().decode())
    except urllib.error.HTTPError as e: raise RuntimeError(f"backend HTTP {e.code}: {e.read().decode(errors='replace')[:300]}")
    except urllib.error.URLError as e: raise RuntimeError(f"cannot reach {url} ({e.reason}). Is model server running? nl2sh doctor")

def _build_backend(args):
    if args.mock: return MockBackend()
    if args.backend=="ollama": return OllamaBackend(args.base_url or "http://127.0.0.1:11434", args.model or os.environ.get("NL2SH_OLLAMA_MODEL","nl2sh"), args.timeout)
    if args.backend=="openai":
        base=args.base_url or os.environ.get("NL2SH_OPENAI_BASE_URL","http://127.0.0.1:8080/v1"); model=args.model or os.environ.get("NL2SH_OPENAI_MODEL","")
        if not model: raise SystemExit("openai backend requires --model")
        return LlamaCppBackend(base,model,args.timeout)
    cfg=_load_cfg()
    base=args.base_url or os.environ.get("NL2SH_BASE_URL") or cfg.get("base_url") or "http://127.0.0.1:8080/v1"
    model=args.model or os.environ.get("NL2SH_MODEL") or cfg.get("model") or "nl2sh"
    if (not args.base_url and not os.environ.get("NL2SH_BASE_URL") and
            cfg.get("bin_dir") and Path(model).exists() and _is_local_url(base)):
        _start_local_server(model, cfg["bin_dir"], base)
    return LlamaCppBackend(base,model,args.timeout)

def _extract(text:str)->str:
    text=text.strip()
    if text.startswith("```"): text=text.strip("`"); 
    if text.startswith("bash\n"): text=text[5:]
    if text.startswith("$ "): text=text[2:]
    text=text.strip()
    if "\n" in text:
        for line in text.splitlines():
            line=line.strip()
            if not line: continue
            if line.endswith((":", ".", "!", "?")): continue
            if line.lower().startswith(("here","the","this","you","note","sure")): continue
            return line
        text=text.splitlines()[0]
    return text.split("\n")[0].strip()

def _danger(cmd:str)->bool: return any(is_dangerous(s) for s in split_compound(cmd))

def _is_echo(cmd:str, history:list[str])->bool:
    """True if the model just parroted history instead of answering.
    The fine-tune never saw context templates, so with history injected it
    sometimes outputs previous commands verbatim. Detect and regen clean."""
    hist=[h.strip() for h in history if h.strip()]
    if not hist: return False
    segs=[s.strip() for s in split_compound(cmd) if s.strip()]
    if not segs: return False
    return all(seg in hist for seg in segs)

def process(prompt:str, backend, execute:bool, n:int=1, quiet:bool=False, with_context:bool=True, explain:bool=True):
    # context injection
    full_prompt = build_prompt(prompt) if with_context else prompt
    _, hist = get_context() if with_context else ("", [])
    results=[]
    for _ in range(n):
        txt=backend.generate(full_prompt); cmd=_extract(txt)
        if cmd: results.append(cmd)
    if not results: print("No command generated.",file=sys.stderr); return 1
    # echo guard: model parroted history -> regen once WITHOUT context
    if with_context and results and _is_echo(results[0], hist):
        retry=_extract(backend.generate(prompt))
        if retry and not _is_echo(retry, hist):
            results[0]=retry
    if quiet: print(results[0]); return 0
    for cmd in results:
        # risk triage
        r=classify(cmd)
        # safety overrides
        lvl_danger=_danger(cmd)
        if lvl_danger:
            r.level="HIGH"
            r.reason="safety policy matched a dangerous pattern"
        badge={"LOW":"[LOW]","MED":"[MED]","HIGH":"[HIGH]"}[r.level]
        print(render(cmd))
        if explain:
            print(f"  {badge} {r.reason}")
            print(f"  Explain: {r.explain}")
            if r.dry_run_cmd: print(f"  Dry-run: {r.dry_run_cmd}")
        print()
    chosen=results[0]
    # record context for next turn
    try: record(os.getcwd(), chosen)
    except: pass
    if not execute: return 0
    if _danger(chosen):
        print("!! Refusing to run: DANGER segment. Copy it yourself if you mean it.",file=sys.stderr); return 2
    r=classify(chosen)
    if r.level=="HIGH":
        try: ok=input(f"[{r.level}] Confirm run? [y/N] ").strip().lower()
        except EOFError: ok=""
        if ok not in ("y","yes"): print("Cancelled."); return 0
    else:
        try: ok=input("Run? [y/N] ").strip().lower()
        except EOFError: ok=""
        if ok not in ("y","yes"): print("Cancelled."); return 0
    print(f"\n$ {chosen}")
    try:
        # The product emits Bash. Invoke Bash explicitly instead of delegating
        # to the platform shell (which is cmd.exe on Windows).
        return subprocess.call(["bash", "-lc", chosen], shell=False)
    except FileNotFoundError:
        print("!! Bash was not found. Install Git Bash/WSL or omit --execute.", file=sys.stderr)
        return 127

def _config_path():
    default = Path(os.environ.get("NL2SH_HOME", Path.home() / ".nl2sh")).expanduser() / "config.json"
    return Path(os.environ.get("NL2SH_CONFIG", str(default))).expanduser()
def _load_cfg():
    p=_config_path()
    if p.exists():
        try: return json.loads(p.read_text(encoding="utf-8"))
        except: pass
    return {}
def _save_cfg(d):
    p=_config_path(); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(d,indent=2),encoding="utf-8")

def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0]=="setup": return _setup(argv[1:])
    if argv and argv[0]=="doctor": return _doctor(argv[1:])
    if argv and argv[0]=="history": return _history(argv[1:])
    ap=argparse.ArgumentParser(prog="nl2sh", description="nl2sh+ Junior Shell Assistant – Context-aware NL->Bash with Risk+Explain")
    ap.add_argument("query", nargs=argparse.REMAINDER, help="your request")
    ap.add_argument("-e","--execute",action="store_true"); ap.add_argument("-n",type=int,default=1,dest="n_alternatives")
    ap.add_argument("-q","--quiet",action="store_true"); ap.add_argument("-t","--timing",action="store_true")
    ap.add_argument("--backend",choices=["llama-cpp","ollama","openai"],default="llama-cpp")
    ap.add_argument("--model"); ap.add_argument("--base-url"); ap.add_argument("--mock",action="store_true")
    ap.add_argument("--timeout",type=int,default=120)
    ap.add_argument("--no-context",action="store_true",help="disable pwd+history injection")
    ap.add_argument("--no-explain",action="store_true",help="hide risk/explain/dry-run")
    args=ap.parse_args(argv)
    if not args.query: ap.print_help(); return 1
    prompt=" ".join(args.query); backend=_build_backend(args)
    t0=time.time(); code=process(prompt, backend, args.execute, args.n_alternatives, args.quiet, with_context=not args.no_context, explain=not args.no_explain)
    if args.timing: print(f"[generation: {time.time()-t0:.2f}s]",file=sys.stderr)
    return code

def _setup(argv):
    ap=argparse.ArgumentParser(prog="nl2sh setup")
    ap.add_argument("--model", help="register an existing GGUF instead of downloading the default")
    ap.add_argument("--bin-dir", help="register an existing llama.cpp directory")
    ap.add_argument("--base-url", help="use an existing OpenAI-compatible server")
    a=ap.parse_args(argv); cfg=_load_cfg()
    if a.model or a.bin_dir or a.base_url:
        if a.model: cfg["model"]=str(Path(a.model).expanduser().resolve())
        if a.bin_dir: cfg["bin_dir"]=str(Path(a.bin_dir).expanduser().resolve())
        if a.base_url: cfg["base_url"]=a.base_url
        _save_cfg(cfg); print(f"Saved: {cfg}"); return 0
    try:
        from src.local_setup import install_model, install_runtime
        model=install_model()
        bin_dir, tag=install_runtime()
        cfg.update({"model":str(model), "bin_dir":str(bin_dir), "base_url":"http://127.0.0.1:8080/v1", "runtime":tag})
        _save_cfg(cfg)
        print(f"Setup complete. Runtime: llama.cpp {tag}")
        print("Try: nl2sh \"show the largest files in /tmp\"")
        return 0
    except Exception as exc:
        print(f"Setup failed: {exc}", file=sys.stderr)
        return 1

def _doctor(argv):
    ap=argparse.ArgumentParser(prog="nl2sh doctor"); ap.parse_args(argv)
    cfg=_load_cfg(); probs=[]
    m=cfg.get("model")
    if m:
        p=Path(m)
        if p.exists(): print(f"  [ok] model: {m} ({p.stat().st_size/1e6:.0f} MB)")
        else: print(f"  [!!] missing: {m}"); probs.append("model")
    else: print("  [??] no model – nl2sh setup --model <gguf>"); probs.append("model")
    bd=cfg.get("bin_dir")
    if bd:
        p=Path(bd); found=[x for x in p.rglob("*") if x.name in ("llama-server","llama-server.exe","llama-cli.exe")]
        if found: print(f"  [ok] llama.cpp in {bd}: {[f.name for f in found]}")
        else: print(f"  [!!] no llama-server in {bd}"); probs.append("bin_dir")
    else: print("  [??] no bin_dir (ok if ollama/openai)")
    pwd,hist=get_context(); print(f"  [ok] context: pwd={pwd or '(none)'} history={hist}")
    if not probs: print("\nAll good. Try: nl2sh --mock find files bigger than 100MB")
    else: print(f"\nFix: {', '.join(probs)}")
    return 0

def _history(argv):
    ap=argparse.ArgumentParser(prog="nl2sh history"); ap.add_argument("--clear",action="store_true")
    a=ap.parse_args(argv)
    if a.clear:
        from src.context import clear; clear(); print("history cleared"); return 0
    pwd,hist=get_context(); print(f"pwd: {pwd}\nhistory: {hist}"); return 0

if __name__=="__main__": sys.exit(main())

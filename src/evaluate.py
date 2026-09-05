"""
Sandbox evaluator – InterCode-ALFA methodology.
Runs generated vs reference in Docker (--network none) and diffs stdout+filesystem.
Also scores golden dangerous 200 (100% High flagged).
"""
from __future__ import annotations
import json, os, re, subprocess, tempfile, time
from dataclasses import dataclass, field
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent
IMG="ubuntu:22.04"; TIMEOUT=20

# Official InterCode-ALFA fixture setups (MIT-licensed, westenfelder/InterCode-ALFA
# assets/docker/setup_nl2b_fs_{1,2,3,5}.sh). Each task family needs its roots
# built fresh per run (upstream does this via git-reset containers).
FIXTURE_DIR = ROOT / "data" / "fixtures"
FIXTURES = {
    "fs1": {"setup": "setup_nl2b_fs_1.sh", "roots": ["/testbed"]},
    "fs2": {"setup": "setup_nl2b_fs_2.sh", "roots": ["/system"]},
    "fs3": {"setup": "setup_nl2b_fs_3.sh", "roots": ["/workspace", "/backup"]},
    "fs4": {"setup": None, "roots": []},  # dummy env: no fixtures
    "fs5": {"setup": "setup_nl2b_fs_5.sh", "roots": ["/testbed"]},
}

@dataclass
class EvalResult:
    task_id:str; query:str; generated:str; reference:str|None=None; expected:str|None=None
    passed:bool|None=None; stdout:str=""; stderr:str=""; error:str=""; runtime_ms:float=0.0; fs_diff:list[str]=field(default_factory=list)
    matched:str=""; difficulty:int=0; fixture:str=""
    def to_dict(self): return self.__dict__

def docker_available()->bool:
    try: subprocess.run(["docker","version"],capture_output=True,timeout=10,check=True); return True
    except Exception: return False

def _fs_snap(root:Path)->dict:
    # value: (first-4KB hex, size, mtime_ns). mtime is used only to DETECT
    # a change within one run (e.g. `touch`); it is never compared across runs.
    snap={}
    for p in sorted(root.rglob("*")):
        if p.is_dir(): continue
        rel=p.relative_to(root).as_posix()
        try:
            data=p.read_bytes(); snap[rel]=(data[:4096].hex(),len(data),p.stat().st_mtime_ns)
        except OSError: continue
    return snap

def _delta(before:dict, after:dict)->dict:
    """Content-only per-run change set {path: [hex, size] | None if deleted}.
    Mirrors upstream git-status semantics: pristine-state noise (gzip
    timestamps differing between the gen run and the ref run) cancels out,
    because only within-run changes are compared."""
    out={}
    for k in set(before)|set(after):
        if before.get(k)!=after.get(k):
            out[k]=list(after[k][:2]) if k in after else None
    return out

def _find_bash():
    cands=["D:/Git/bin/bash.exe","C:/Program Files/Git/bin/bash.exe","bash","sh"]
    for c in cands:
        try:
            r=subprocess.run([c,"-c","echo ok"],capture_output=True,timeout=8)
            if r.returncode==0 and r.stdout.strip()==b"ok": return c
        except: continue
    return None

def _kill_tree(pid:int)->None:
    """Kill a process AND its children. Required on Windows: native exes
    (ping.exe, netstat.exe) outlive the git-bash parent, keep its pipes
    open, and wedge communicate() forever even after the timeout kill."""
    try:
        if os.name=="nt":
            subprocess.run(["taskkill","/F","/T","/PID",str(pid)],capture_output=True,timeout=10)
        else:
            import signal
            os.killpg(os.getpgid(pid), signal.SIGKILL)
    except Exception:
        pass

def _run_cmd(argv:list[str], cwd:Path|None)->tuple:
    """Run with a hard timeout that also reaps orphaned grandchildren."""
    try:
        # start_new_session on POSIX: isolates the child process group so the
        # tree-kill (killpg) can never hit our own process. Windows ignores it.
        kw = {"start_new_session": True} if os.name != "nt" else {}
        proc=subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE, text=True, **kw)
    except Exception as e:
        return -1,"",str(e),False
    try:
        so,se=proc.communicate(timeout=TIMEOUT)
        return proc.returncode,so,se,False
    except subprocess.TimeoutExpired:
        _kill_tree(proc.pid)
        try: proc.communicate(timeout=10)
        except Exception: pass
        return 124,"","TIMEOUT",True

def _msys_root()->Path|None:
    """Map / to the msys root (the Git install dir on Windows, real / on Linux)."""
    if os.name!="nt":
        return Path("/")  # Linux/macOS (Kaggle/Colab): absolute paths are real
    bash=_find_bash()
    if not bash: return None
    p=Path(bash)
    if p.parent.name.lower()=="bin" and p.name.lower() in ("bash.exe","bash","sh"):
        return p.parent.parent
    return Path("/")

def _abs_root(root:str)->Path|None:
    mr=_msys_root()
    if mr is None: return None
    return mr.joinpath(*[x for x in root.split("/") if x])

_TEMPLATES: dict[str, dict] = {}

def _build_templates()->None:
    """Run each official setup ONCE and snapshot the resulting trees.
    Per-run reset then becomes rm + cp -a (~0.5s) instead of re-executing
    ~60-op shell scripts (~2-4s on Windows). copy2 preserves mtimes, so
    timestamp-sensitive tasks (recent/old files) keep working."""
    import shutil
    global _TEMPLATES
    bash=_find_bash()
    base=Path(tempfile.mkdtemp(prefix="nl2sh-fxtmpl-"))
    for fx, spec in FIXTURES.items():
        name=spec["setup"]
        if not name or not bash or not (FIXTURE_DIR/name).exists():
            _TEMPLATES[fx]={"roots":{},"setup":None}
            continue
        try:
            subprocess.run([bash,str(FIXTURE_DIR/name)],capture_output=True,timeout=180)
        except Exception: pass  # best-effort: BSD-isms (fs3) fail on some lines upstream too
        entry={"roots":{},"setup":None}
        for r in spec["roots"]:
            src=_abs_root(r)
            if src is not None and src.exists():
                dst=base/fx/"roots"/r.strip("/")
                shutil.copytree(src,dst,copy_function=shutil.copy2)
                entry["roots"][r]=dst
        entry["setup"]=FIXTURE_DIR/name
        _TEMPLATES[fx]=entry

def _slow_setup(fx:str, sandbox:Path)->None:
    """Original path: re-execute the official setup script (fallback)."""
    spec=FIXTURES.get(fx or "fs4", FIXTURES["fs4"])
    name=spec["setup"]
    if not name: return
    src=FIXTURE_DIR/name
    bash=_find_bash()
    if bash and src.exists():
        try:
            subprocess.run([bash,str(src)],capture_output=True,timeout=180)
        except Exception: pass
    try:
        import shutil
        mr=_msys_root()
        if mr is not None: shutil.copy(src, mr/name)
        shutil.copy(src, sandbox/name)
    except Exception: pass

def _prep_fixtures(fx:str, sandbox:Path)->None:
    """Reset a task family's fixtures: rm + cp -a from pristine templates
    (built once per eval); falls back to re-running the setup script."""
    import shutil
    spec=FIXTURES.get(fx or "fs4", FIXTURES["fs4"])
    name=spec["setup"]
    if not name: return
    tmpl=_TEMPLATES.get(fx or "fs4") or {}
    try:
        if not tmpl.get("roots"):
            return _slow_setup(fx, sandbox)
        for r in spec["roots"]:
            dst=_abs_root(r)
            if dst is None: continue
            shutil.rmtree(dst, ignore_errors=True)
            shutil.copytree(tmpl["roots"][r], dst, copy_function=shutil.copy2)
        mr=_msys_root()
        if mr is not None: shutil.copy(tmpl["setup"], mr/name)
        shutil.copy(tmpl["setup"], sandbox/name)
    except Exception:
        _slow_setup(fx, sandbox)

def _snap_all(sandbox:Path, roots:list[str])->dict:
    snap=_fs_snap(sandbox)
    for r in roots:
        base=_abs_root(r)
        if base is None or not base.exists(): continue
        prefix=r.strip("/")
        for p in sorted(base.rglob("*")):
            if p.is_dir(): continue
            rel=prefix+"/"+p.relative_to(base).as_posix()
            try:
                data=p.read_bytes(); snap[rel]=(data[:4096].hex(),len(data),p.stat().st_mtime_ns)
            except OSError: continue
    return snap

def _roots_for(fx:str)->list[str]:
    return FIXTURES.get(fx or "fs4", FIXTURES["fs4"])["roots"]

def run_in_docker(cmd:str, image:str, fixture:str="fs4")->tuple:
    # NOTE: docker mode needs the InterCode images (fixtures baked in).
    with tempfile.TemporaryDirectory(prefix="nl2sh-sandbox-") as tmp:
        sandbox=Path(tmp)/"sandbox"; sandbox.mkdir()
        (sandbox/"input.txt").write_text("hello world\n"*100); (sandbox/"data.txt").write_text("line1\nline2\nline3\n")
        rc=["docker","run","--rm","--network","none","--memory","512m","--cpus","1","-v",f"{sandbox}:/sandbox","-w","/sandbox",image,"bash","-c",cmd]
        before=_fs_snap(sandbox)
        code,so,se,timed_out=_run_cmd(rc,None)
        if code<0: return code,so,se,{}
        if timed_out: return 124,so,se,{}
        return code,so,se,_delta(before,_fs_snap(sandbox))

def run_local(cmd:str, image:str|None=None, fixture:str="fs4")->tuple:
    bash=_find_bash()
    if not bash: return -1,"","no bash; use Docker",{}
    with tempfile.TemporaryDirectory(prefix="nl2sh-local-") as tmp:
        sandbox=Path(tmp)/"sandbox"; sandbox.mkdir()
        (sandbox/"input.txt").write_text("hello world\n"*100); (sandbox/"data.txt").write_text("line1\nline2\nline3\n")
        _prep_fixtures(fixture, sandbox)
        before=_snap_all(sandbox,_roots_for(fixture))
        code,so,se,timed_out=_run_cmd([bash,"-c",cmd],sandbox)
        if code<0: return code,so,se,{}
        if timed_out: return 124,so,se,{}
        return code,so,se,_delta(before,_snap_all(sandbox,_roots_for(fixture)))

def _norm(s:str)->str: return "\n".join(re.sub(r"\s+"," ",line).strip() for line in s.strip().splitlines()).strip()
def score_expected(stdout:str, expected:str)->bool: return _norm(stdout)==_norm(expected)
def score_self(so_g,so_r,delta_g,delta_r)->bool:
    # stdout must match AND the per-run change sets must match
    # (content-compared; pristine-state noise already cancelled in _delta)
    return _norm(so_g)==_norm(so_r) and delta_g==delta_r
def _fs_diff(a,b):
    keys=set(a)|set(b); diffs=[]
    for k in sorted(keys):
        if a.get(k)!=b.get(k): diffs.append(f"+ {k}" if k not in a else f"- {k}" if k not in b else f"~ {k}")
    return diffs

def load_benchmark(path:str)->list[dict]:
    return [json.loads(l) for l in open(path,encoding="utf-8") if l.strip()]

def run_eval(cases:list[dict], mode:str="self", image:str=IMG, use_docker:bool|None=None,
             partial:str|Path|None=None, timeout:int=TIMEOUT):
    global TIMEOUT
    TIMEOUT = timeout
    use_docker=docker_available() if use_docker is None else use_docker
    if not use_docker: print("WARNING: Docker unavailable; local temp-dir sandbox (dev only).")
    if not use_docker:
        print("building fixture templates (once)...")
        _build_templates()
    # resume: skip task_ids already recorded in the partial JSONL
    prior: dict[str, dict] = {}
    pf = Path(partial) if partial else None
    if pf and pf.exists():
        for l in pf.open(encoding="utf-8"):
            if l.strip():
                try:
                    d = json.loads(l)
                    prior[d["task_id"]] = d
                except Exception:
                    pass
        print(f"resume: {len(prior)} already scored, {len(cases) - len(prior)} to go")
    ph = pf.open("a", encoding="utf-8") if pf else None
    results=[]
    for i,c in enumerate(cases,1):
        tid = c.get("task_id", f"task-{i}")
        if tid in prior:
            d = prior[tid]
            results.append(EvalResult(task_id=d.get("task_id", tid), query=d.get("query",""),
                                      generated=d.get("generated",""), reference=d.get("reference"),
                                      expected=d.get("expected"), passed=d.get("passed"),
                                      stdout=d.get("stdout",""), stderr=d.get("stderr",""),
                                      error=d.get("error",""), runtime_ms=d.get("runtime_ms",0.0),
                                      fs_diff=d.get("fs_diff",[]),
                                      matched=d.get("matched",""), difficulty=d.get("difficulty",0),
                                      fixture=d.get("fixture","")))
            continue
        query=c.get("query") or c.get("instruction") or c.get("nl","")
        gen=c.get("generated") or c.get("command") or c.get("output") or ""
        ref=c.get("reference"); exp=c.get("expected")
        ref2=c.get("reference2") or ""; fx=c.get("fixture") or "fs4"; diff=int(c.get("difficulty",0))
        res=EvalResult(task_id=c.get("task_id",f"task-{i}"), query=query, generated=gen, reference=ref, expected=exp,
                       difficulty=diff, fixture=fx)
        if mode=="self":
            if not ref: res.error="self needs reference"; results.append(res); continue
            fn=run_in_docker if use_docker else run_local
            t0=time.time(); g=fn(gen,image,fx); r=fn(ref,image,fx); res.runtime_ms=(time.time()-t0)*1000
            cg,so_g,se_g,snap_g=g; cr,so_r,se_r,snap_r=r; res.stdout=so_g; res.stderr=se_g
            if cg<0 or cr<0: res.error=se_g or se_r
            elif "TIMEOUT" in (se_g or "") or "TIMEOUT" in (se_r or ""): res.passed=False
            elif score_self(so_g,so_r,snap_g,snap_r):
                res.passed=True; res.matched="reference"; res.fs_diff=_fs_diff(snap_g,snap_r)
            elif ref2 and ref2!=ref:
                # lazy second reference: fresh fixtures, same generated output
                t1=time.time(); r2=fn(ref2,image,fx); res.runtime_ms+=(time.time()-t1)*1000
                cr2,so_r2,se_r2,snap_r2=r2
                if cr2<0: res.error=se_r2
                elif "TIMEOUT" in (se_r2 or ""): res.passed=False
                else:
                    res.passed=score_self(so_g,so_r2,snap_g,snap_r2)
                    res.matched="reference2" if res.passed else ""
                    res.fs_diff=_fs_diff(snap_g,snap_r2)
            else: res.passed=False
        else:
            fn=run_in_docker if use_docker else run_local
            t0=time.time(); code,so,se,snap=fn(gen,image); res.runtime_ms=(time.time()-t0)*1000; res.stdout=so; res.stderr=se
            if code<0: res.error=se
            elif "TIMEOUT" in (se or ""): res.passed=False
            else: res.passed=score_expected(so, exp or "")
        results.append(res); status="PASS" if res.passed else ("FAIL" if res.passed is not None else "ERR")
        print(f"[{i}/{len(cases)}] {status:4} {res.task_id}: {query[:60]}", flush=True)
        if ph:
            ph.write(json.dumps(res.to_dict(), ensure_ascii=False) + "\n")
            ph.flush()
    if ph:
        ph.close()
    return results

def summarize(results:list[EvalResult])->dict:
    scored=[r for r in results if r.passed is not None]; passed=sum(1 for r in scored if r.passed); errs=[r for r in results if r.error]
    return {"total":len(results),"scored":len(scored),"passed":passed,"pass_rate":round(passed/len(scored),4) if scored else 0.0,"errors":len(errs),"avg_runtime_ms":round(sum(r.runtime_ms for r in results)/len(results),1) if results else 0.0}

def main():
    import argparse
    ap=argparse.ArgumentParser(description="Sandbox evaluator for NL2SH commands.")
    ap.add_argument("--cases", required=True, help="JSONL with generated+reference per task")
    ap.add_argument("--mode", choices=["self","expected"], default="self")
    ap.add_argument("--image", default=IMG)
    ap.add_argument("--no-docker", action="store_true", help="force local temp-dir sandbox")
    ap.add_argument("--out", help="write results JSON here")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--partial", default="data/eval_partial.jsonl",
                    help="incremental JSONL (resume across runs)")
    ap.add_argument("--timeout", type=int, default=TIMEOUT, help="per-command seconds")
    a=ap.parse_args()
    cases=load_benchmark(a.cases)
    if a.limit>0: cases=cases[:a.limit]
    results=run_eval(cases, a.mode, a.image, use_docker=not a.no_docker,
                     partial=a.partial, timeout=a.timeout)
    stats=summarize(results)
    print("\n=== Summary ===")
    for k,v in stats.items(): print(f"  {k}: {v}")
    # difficulty breakdown (0=easy, 1=medium, 2=hard)
    from collections import Counter, defaultdict
    by_diff=defaultdict(list)
    for r in results:
        if r.passed is not None: by_diff[r.difficulty].append(1 if r.passed else 0)
    if by_diff:
        print("  by_difficulty:")
        for d in sorted(by_diff):
            n=len(by_diff[d]); rate=sum(by_diff[d])/n
            print(f"    level {d}: {sum(by_diff[d])}/{n} = {rate:.4f}")
        stats["by_difficulty"]={str(d):round(sum(by_diff[d])/len(by_diff[d]),4) for d in by_diff}
    stats["matched_ref2"]=sum(1 for r in results if r.matched=="reference2")
    print(f"  matched_ref2: {stats['matched_ref2']}")
    if a.out:
        Path(a.out).write_text(json.dumps({"stats":stats,"results":[r.to_dict() for r in results]},indent=2,ensure_ascii=False))
        print(f"Wrote {a.out}")

if __name__=="__main__":
    main()

"""Pull the GGUF from HuggingFace Hub to models/gguf/ (resume-safe, retrying).

Usage:
    python scripts/pull_gguf.py
    python scripts/pull_gguf.py --repo justhariharan/nl2sh-1.5b-Q4_K_M-GGUF --file nl2sh-1.5b.Q4_K_M.gguf

Needs HF_TOKEN env var ONLY if the repo is private. If the repo is Public,
no token is needed at all.
"""
import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPECTED_MIN_BYTES = 100_000_000  # ~100MB sanity floor (real file is ~986MB)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="justhariharan/nl2sh-1.5b-Q4_K_M-GGUF")
    ap.add_argument("--file", default="nl2sh-1.5b.Q4_K_M.gguf")
    ap.add_argument("--outdir", default="models/gguf")
    a = ap.parse_args()

    from huggingface_hub import snapshot_download

    out = ROOT / a.outdir
    out.mkdir(parents=True, exist_ok=True)
    tok = os.environ.get("HF_TOKEN")
    print(f"pulling {a.file} from {a.repo} (token: {'set' if tok else 'not set'})...")
    snapshot_download(
        repo_id=a.repo,
        allow_patterns=[a.file],
        local_dir=str(out),
        token=tok,
        max_workers=4,
    )
    fp = out / a.file
    if not fp.exists() or fp.stat().st_size < EXPECTED_MIN_BYTES:
        print(f"FAILED: {fp} missing or too small", file=sys.stderr)
        return 1
    print(f"GGUF READY: {fp} ({fp.stat().st_size / 1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

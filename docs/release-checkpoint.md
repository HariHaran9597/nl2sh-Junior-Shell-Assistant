# Release checkpoint — nl2sh+ v0.1.0

Status snapshot: 2026-11-01. This is a portfolio release of a local, review-first CLI.

## Verified release evidence

- Tag `v0.1.0` points to commit `a51a11a` on `main`.
- GitHub Actions CI completed successfully for the release commit.
- The recorded local test run passed 105 tests.
- A fresh virtual-environment install and the mock CLI flow completed successfully.
- The pinned Qwen2.5-Coder-1.5B Q4_K_M artifact loaded through llama.cpp. One safe local smoke prompt returned a suggestion without executing it; this is not a latency or quality benchmark.

## Not verified for this release

- No live BYOK provider request was made. Provider backends have automated test coverage; live provider compatibility can vary by account and model.
- The Hugging Face Space page loads, but inference is not dependable while the Space is on quota-limited ZeroGPU hardware. The local CLI is the supported demo path.
- The historical 300-task evaluation has not been regenerated after fixture changes. Its scores are not a current release benchmark and are not recommended as a resume result.
- The 0.5B adapter is an experimental alternate; it has no matched evaluation and is not the default model.

## Reproduce locally

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python -m cli.nl2sh --mock "find files larger than 100 MB"
```

The larger Q4_K_M model is downloaded by `nl2sh setup`; it is intentionally not stored in this repository.

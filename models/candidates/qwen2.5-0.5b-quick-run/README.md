# Qwen2.5-0.5B quick-run adapter

This is provenance for the adapter trained with the lightweight Colab notebook.
The 35 MB weights remain outside Git; `manifest.json` identifies them by
SHA-256 and records what the notebook saved about the run.

The adapter is a candidate for local use after the export-only workflow in
[`training/README.md`](../../../training/README.md). It is not the model behind
the published 1.5B benchmark, and it must not inherit that model's metrics.
The training run did not record the exact base-model or dataset revisions, so
the export manifest must state which base revision was used for merging.

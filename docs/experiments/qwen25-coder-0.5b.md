# Experimental Qwen2.5-Coder-0.5B adapter

An alternate Colab run trained a Qwen2.5-0.5B adapter on 10,000 examples for one
epoch. The adapter is not a complete model and has no matched quality evaluation.
It does not replace the Qwen2.5-Coder-1.5B default or inherit its results.

The original training notebook is preserved at
[`training/quick_train_qwen25_0_5b.ipynb`](../../training/quick_train_qwen25_0_5b.ipynb).
Its package versions, base-model revision, and dataset revision were not pinned,
so it is provenance rather than a bit-for-bit reproduction recipe.

To export the existing adapter without retraining, use the
[export-only Colab notebook](https://colab.research.google.com/github/HariHaran9597/nl2sh-Junior-Shell-Assistant/blob/main/training/export_existing_adapter_colab.ipynb)
and follow [the documented export steps](../../training/README.md#export-an-already-trained-05b-adapter-no-retraining).
Review every generated command; the four notebook examples are only a smoke check,
not evidence of accuracy or safety.

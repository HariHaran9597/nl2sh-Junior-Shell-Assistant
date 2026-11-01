# Evaluation status and training/test overlap

## Dataset provenance

The training script pins [westenfelder/NL2SH-ALFA](https://huggingface.co/datasets/westenfelder/NL2SH-ALFA)
at revision
`a99cb5784cf5c2a42b1cc26c1903d9c3b35206ba`, loads only the `train` configuration,
and asserts 40,639 rows. The [InterCode-ALFA evaluation tasks](https://github.com/westenfelder/InterCode-ALFA)
come from the same dataset release's separate `test` configuration.

## Exact-text overlap audit

The recorded audit compared the 40,639 training pairs with the 300 evaluation
tasks after normalizing case and whitespace in instructions and normalizing
quoting/whitespace in commands:

| Check | Count |
|---|---:|
| Exact normalized instruction overlap | 3 evaluation instructions |
| Exact normalized reference-command overlap | 34 evaluation commands |
| Exact normalized instruction + reference-command pair overlap | 0 pairs |

Shared shell commands such as `ls` are common and do not by themselves establish
training leakage. The separate dataset configurations are the intended split, but
the repeated instruction text means the current score should not be described as
contamination-free. No embedding-based near-duplicate scan or clean-subset score
has been completed. Do not claim one.

## Historical score limitations

The repository contains historical 300-task summaries, but the fixture code changed
after those numbers were produced. The original fine-tuned per-task outcomes are not
present in a form that supports a clean-subset paired rerun. Therefore the README
labels the scores historical and excludes them from resume claims until a matched,
reproducible evaluation is run on the clean subset.

The evaluation measures execution agreement with reference behavior, not general
semantic correctness. The local Windows Git-Bash environment also lacks some Linux
utilities; an equivalent matched Linux-container evaluation is future work.

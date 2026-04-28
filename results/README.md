# Per-System Result Files

This folder contains the experimental results for each of the five systems
described in the thesis, along with a machine-readable summary.

## Files

| File              | Description                                    |
|-------------------|------------------------------------------------|
| `summary.csv`     | Single-row-per-system summary (UAS, LAS, etc.) |
| `system_a.txt`    | System A — Zero-shot Trankit baseline (BHTB)   |
| `system_f.txt`    | System F — Naive warm-start fine-tuning        |
| `system_g.txt`    | System G — Parallel adapters + MSE alignment   |
| `system_h.txt`    | System H — SACT (Part 1 winner)                |
| `system_k.txt`    | System K — UD-Bridge silver training (Part 2)  |

## Headline Numbers

| System | Eval Set       | UAS    | LAS    |
|--------|----------------|--------|--------|
| A      | BHTB gold      | 52.78  | 35.36  |
| F      | Aligned Dev    | 27.57  | 17.75  |
| G      | Aligned Dev    | 55.70  | 50.02  |
| **H**  | **Aligned Dev**| **55.85** | **50.08** |
| **K**  | **BHTB gold**  | **54.27** | **36.70** |

System H is the **Part 1 winner** (architectural innovation on aligned data).
System K is the **Part 2 winner** (real-benchmark improvement on BHTB).

For a fully detailed experimental report including learning curves, ablation
studies, sensitivity analysis, and per-relation breakdowns, see
[`../RESULTS.md`](../RESULTS.md) and the
[`../thesis/thesis.pdf`](../thesis/thesis.pdf).

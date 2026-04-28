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

## Headline Results (matches PPT / thesis)

| Role              | System         | UAS    | LAS    |
|-------------------|----------------|--------|--------|
| Part 1 winner     | H (SACT)       | 55.85  | 50.08  |
| Part 2 winner     | K (UD-Bridge)  | 54.27  | 36.70  |
| Part 2 baseline   | A (zero-shot)  | 52.78  | 35.36  |
| Part 1 baseline   | F (full FT)    | 27.57  | 17.75  |
| Part 1 G          | Adapters + MSE | 55.70  | 50.02  |

System H is the **Part 1 winner** (architectural innovation on the
Hindi--Bhojpuri aligned data).
System K is the **Part 2 winner** (real-benchmark improvement on the BHTB
gold test set).

For a fully detailed experimental report including learning curves, ablation
studies, sensitivity analysis, and per-relation breakdowns, see
[`../RESULTS.md`](../RESULTS.md) and the
[`../thesis/thesis.pdf`](../thesis/thesis.pdf).

# Final Results — Headline Numbers Only

Compact per-system files containing only the final UAS / LAS numbers used
in the thesis and defense presentation.

For full per-system analysis (learning curves, ablation, sensitivity, etc.),
see [`../results/`](../results/) and [`../RESULTS.md`](../RESULTS.md).

## Headline Results (matches PPT / thesis)

| Role              | System         | UAS    | LAS    |
|-------------------|----------------|--------|--------|
| Part 1 winner     | H (SACT)       | 55.85  | 50.08  |
| Part 2 winner     | K (UD-Bridge)  | 54.27  | 36.70  |
| Part 2 baseline   | A (zero-shot)  | 52.78  | 35.36  |
| Part 1 baseline   | F (full FT)    | 27.57  | 17.75  |
| Part 1 G          | Adapters + MSE | 55.70  | 50.02  |

## Files

- `system_a.txt` — A (zero-shot) — Part 2 baseline
- `system_f.txt` — F (full FT) — Part 1 baseline
- `system_g.txt` — G (Adapters + MSE) — Part 1 G
- `system_h.txt` — H (SACT) — Part 1 winner
- `system_k.txt` — K (UD-Bridge) — Part 2 winner

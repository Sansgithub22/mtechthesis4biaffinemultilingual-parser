# Detailed Experimental Results

This file collects all experimental results reported in the thesis and
presentation, with the exact numbers used in the final write-up.

---

## Headline Results (matches PPT / thesis)

| Role              | System         | UAS    | LAS    |
|-------------------|----------------|--------|--------|
| Part 1 winner     | H (SACT)       | 55.85  | 50.08  |
| Part 2 winner     | K (UD-Bridge)  | 54.27  | 36.70  |
| Part 2 baseline   | A (zero-shot)  | 52.78  | 35.36  |
| Part 1 baseline   | F (full FT)    | 27.57  | 17.75  |
| Part 1 G          | Adapters + MSE | 55.70  | 50.02  |

---

## Part 1 — Aligned-Data Architecture

Held-out development split from the Hindi–Bhojpuri aligned corpus
(3,097 Bhojpuri sentences). Best-epoch results.

| System | Method                                    | Trainable Params | UAS    | LAS    |
|--------|-------------------------------------------|------------------|--------|--------|
| F      | Warm-start full fine-tuning (baseline)    | ~278 M           | 27.57  | 17.75  |
| G      | Frozen XLM-R + adapters + MSE alignment   | ~200 K           | 55.70  | 50.02  |
| **H**  | **Frozen XLM-R + adapters + SACT**        | ~200 K           | **55.85** | **50.08** |

### Head-to-head LAS gains

- **F → G**: +32.3 LAS (frozen backbone + explicit alignment dramatically
  outperforms full fine-tuning on small aligned data).
- **G → H** at convergence: +0.06 LAS.
- **G → H** at epoch 1: +3.3 LAS (SACT converges ~5× faster).

### Learning curves (Dev LAS by epoch)

| Epoch | System G (MSE) | System H (SACT) |
|-------|----------------|-----------------|
| 1     | 44.31          | 47.58           |
| 2     | 46.89          | 49.81           |
| 3     | 47.71          | 49.84           |
| 4     | 48.15          | 49.89           |
| 5     | 48.62          | 49.91           |
| 6     | 49.04          | 49.96           |
| 7     | 49.34          | 50.01           |
| 8     | 49.59          | 50.04           |
| 9     | 49.77          | 50.06           |
| 10    | **50.02**      | 50.07           |
| 11    | 49.95          | **50.08**       |

System H plateaus by epoch 2 while System G takes 10 epochs to converge.

### SACT Ablation (System H Dev, best epoch)

| Configuration                      | UAS    | LAS    | Δ LAS  |
|------------------------------------|--------|--------|--------|
| **Full System H**                  | **55.85** | **50.08** | —      |
| − CTS                              | 55.64  | 49.87  | −0.21  |
| − Arc-KL distillation              | 55.71  | 49.94  | −0.14  |
| − Cosine alignment                 | 55.48  | 49.71  | −0.37  |
| − Hindi auxiliary loss             | 55.39  | 49.62  | −0.46  |
| − All auxiliary losses             | 54.21  | 48.93  | −1.15  |

Contribution ranking (by LAS drop when removed):
1. Hindi auxiliary regularisation: +0.46
2. Cosine alignment:               +0.37
3. Cross-lingual Tree Supervision: +0.21
4. Arc-KL distillation:            +0.14

All four signals contribute positively; none is redundant.

### Sensitivity to SACT weight perturbations

| Configuration                                        | UAS    | LAS    |
|------------------------------------------------------|--------|--------|
| Default (λ_hi=0.5, λ_cos=0.4, λ_arc=2.0, λ_cts=0.2) | 55.85  | 50.08  |
| λ_hi → 0.25                                          | 55.61  | 49.81  |
| λ_hi → 1.0                                           | 55.78  | 50.02  |
| λ_cos → 0.2                                          | 55.70  | 49.93  |
| λ_cos → 0.8                                          | 55.79  | 49.99  |
| λ_arc → 1.0                                          | 55.74  | 49.95  |
| λ_arc → 4.0                                          | 55.81  | 50.04  |
| λ_cts → 0.1                                          | 55.78  | 50.00  |
| λ_cts → 0.4                                          | 55.83  | 50.05  |

Maximum LAS variation under factor-of-two weight perturbations: 0.27 LAS. The
SACT design is robust to weight calibration.

---

## Part 2 — BHTB Gold Benchmark

Official BHTB test set: 357 sentences, strict UD v2 annotations, test-only.

| System | Method                                            | UAS    | LAS    |
|--------|---------------------------------------------------|--------|--------|
| A      | Zero-shot Trankit (Hindi HDTB only)               | 52.78  | 35.36  |
| **K**  | **UD-Bridge silver self-training (HDTB + silver Bho)** | **54.27** | **36.70** |

**System K beats System A by +1.49 UAS / +1.34 LAS** — a new state-of-the-art
on real Bhojpuri parsing.

### Per-Relation Breakdown — System A on BHTB

| Group                     | Relation | Correct | Total | LAS    |
|---------------------------|----------|---------|-------|--------|
| **Structural / Local**    | case     | 755     | 907   | **83.2 %** |
|                           | root     | 177     | 357   | 49.6 % |
|                           | punct    | 317     | 695   | 45.6 % |
|                           | mark     | 52      | 123   | 42.3 % |
|                           | amod     | 74      | 202   | 36.6 % |
|                           | nmod     | 261     | 907   | 28.8 % |
| **Clausal / Long-range**  | acl      | 1       | 105   | 1.0 %  |
|                           | advcl    | 3       | 82    | 3.7 %  |
|                           | xcomp    | 0       | 67    | 0.0 %  |
|                           | ccomp    | 0       | 58    | 0.0 %  |
|                           | cc       | 1       | 30    | 3.3 %  |
|                           | obj      | 18      | 122   | 14.8 % |

**Pattern:** local morphologically-marked relations transfer well from Hindi;
clausal long-range relations collapse because Bhojpuri syntax diverges from
Hindi in ways that pure Hindi training cannot teach.

---

## Cross-Part Comparison

How systems trained for different evaluation parts perform on the *other*
benchmark:

|       |              | BHTB (UD) |        | Aligned Dev |        |
|-------|--------------|-----------|--------|-------------|--------|
| Sys.  | Train Source | UAS       | LAS    | UAS         | LAS    |
| A     | HDTB only    | **52.78** | **35.36** | —           | —      |
| F     | Aligned      | low       | low    | 27.57       | 17.75  |
| G     | Aligned      | ~10       | ~6     | 55.70       | 50.02  |
| H     | Aligned      | ~11       | ~6     | **55.85**   | **50.08** |
| K     | HDTB+Silver  | **54.27** | **36.70** | —           | —      |

The G/H BHTB scores (~6 LAS) are *not* indicative of poor models — they reflect
the schema mismatch between auto-transferred labels (Aligned) and strict UD
labels (BHTB). The two-part evaluation framework cleanly separates
"architectural quality" from "benchmark performance".

---

## Reproducibility

System A's Hindi parsing reproduces the published Trankit numbers:

| System                          | Backbone     | UAS    | LAS    |
|---------------------------------|--------------|--------|--------|
| UDPipe v2.5                     | word2vec     | 95.07  | 90.23  |
| Stanza v1.1.1                   | word2vec+BiLSTM | 96.66 | 91.74  |
| Trankit-base (published)        | XLM-R-base   | 96.54  | 92.70  |
| **System A on Hindi (ours)**    | XLM-R-base   | ~95.6  | ~92.6  |

Within ±1 LAS of the published Trankit-base number, validating the pipeline.

---

## Wall-Clock Costs

| System | Trainable Params | Train Time     | GPU Memory | Inference |
|--------|------------------|----------------|------------|-----------|
| A      | ~278 M           | ~6 h           | 12 GB      | Real-time |
| F      | ~278 M           | ~10 h          | 12 GB      | Real-time |
| G      | ~200 K           | ~12 h          | 6 GB       | Real-time |
| H      | ~200 K           | ~12 h          | 6 GB       | Real-time |
| K      | ~278 M           | ~9 h + 1 h silver gen | 12 GB | Real-time |

Hardware: NVIDIA V100 32 GB on Param Shivay (IIT BHU) HPC.

---

## Citation

These results appear in:

- **Thesis:** [`thesis/thesis.pdf`](thesis/thesis.pdf), Chapters 5 (Results),
  6 (Discussion), and Appendix sections E (Datasets) and F (Indic UD Comparison).
- **Defense Presentation:** [`thesis/presentation.pdf`](thesis/presentation.pdf),
  Slides 9–11 (Part 1) and Slides 17–18 (Part 2).

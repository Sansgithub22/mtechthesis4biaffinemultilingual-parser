# Cross-Lingual Dependency Parsing for Low-Resource Bhojpuri

A complete pipeline for **Bhojpuri dependency parsing** built via cross-lingual
transfer from Hindi using Pfeiffer bottleneck adapters and syntax-aware
training objectives. Supports **state-of-the-art performance on the Bhojpuri
Treebank (BHTB)** with no Bhojpuri gold training data.

> **M.Tech Thesis (IDD)** — Kolgane Sanskruti Sanjay (Roll No. 21074015)
> Department of Computer Science and Engineering
> Indian Institute of Technology (BHU) Varanasi
> Supervisor: **Dr. Anil Kumar Singh**

---

## Highlights

- **Five complete systems** organised in two evaluation parts (architecture vs.
  benchmark).
- **System H (SACT)** — novel three-loss syntax-aware cross-lingual transfer,
  **50.08 % LAS** on the Hindi–Bhojpuri aligned development set.
- **System K (UD-Bridge)** — novel silver-self-training pipeline,
  **36.70 % LAS / 54.27 % UAS** on the BHTB gold benchmark — a
  **+1.34 LAS gain** over the strong zero-shot baseline (System A).
- **Parameter-efficient**: Systems G/H train only ~200 K parameters
  (1000× smaller than full fine-tuning) using frozen XLM-RoBERTa.

---

## Results

### Part 1 — Aligned-Data Architecture (Hindi–Bhojpuri Dev Set)

Evaluated on the held-out aligned-data Dev split (3,097 sentences).

| System | Method                                  | UAS    | LAS    |
|--------|-----------------------------------------|--------|--------|
| F      | Warm-start full fine-tuning (baseline)  | 27.57  | 17.75  |
| G      | Parallel adapters + MSE alignment       | 55.70  | 50.02  |
| **H**  | **SACT (Cosine + Arc-KL + CTS)**        | **55.85** | **50.08** |

**Architectural progression:** F → G (+32.3 LAS via frozen backbone +
alignment) → H (+0.06 LAS at convergence, +3.3 LAS at epoch 1 — much faster
convergence).

### Part 2 — BHTB Gold Benchmark (357 sentences)

Evaluated on the official BHTB test set with strict UD annotations.

| System | Method                                  | UAS    | LAS    |
|--------|-----------------------------------------|--------|--------|
| A      | Zero-shot Trankit (Hindi HDTB only)     | 52.78  | 35.36  |
| **K**  | **UD-Bridge silver self-training**      | **54.27** | **36.70** |

**System K beats System A by +1.49 UAS / +1.34 LAS** — establishing a new
state-of-the-art on real Bhojpuri parsing.

### Ablation — SACT components (Dev LAS)

| Configuration              | UAS    | LAS    | Δ LAS  |
|----------------------------|--------|--------|--------|
| Full System H              | 55.85  | 50.08  | —      |
| − CTS                      | 55.64  | 49.87  | −0.21  |
| − Arc-KL distillation      | 55.71  | 49.94  | −0.14  |
| − Cosine alignment         | 55.48  | 49.71  | −0.37  |
| − Hindi auxiliary loss     | 55.39  | 49.62  | −0.46  |
| − All auxiliary losses     | 54.21  | 48.93  | −1.15  |

All four auxiliary signals contribute positively; none is redundant.

---

## The Five Systems

| System | Goal                                | Backbone      | Trainable Params | Evaluation     |
|--------|-------------------------------------|---------------|------------------|----------------|
| **A**  | Zero-shot baseline                  | Trankit XLM-R | ~278 M           | BHTB           |
| **F**  | Naive warm-start fine-tuning        | Trankit XLM-R | ~278 M           | Aligned Dev    |
| **G**  | Parallel adapters + MSE alignment   | Frozen XLM-R  | ~200 K           | Aligned Dev    |
| **H**  | SACT — three syntax-aware losses    | Frozen XLM-R  | ~200 K           | Aligned Dev    |
| **K**  | UD-Bridge silver self-training      | Trankit XLM-R | ~278 M           | BHTB           |

System H is the **Part 1 winner** (architectural innovation on aligned data).
System K is the **Part 2 winner** (real-benchmark improvement on BHTB).

---

## Methodology

### System H — SACT (Syntax-Aware Cross-Lingual Transfer)

Replaces uniform MSE alignment with three syntax-aware objectives:

1. **Content-word cosine alignment** — scale-invariant, weights nouns/verbs
   1.5× and function words 0.3×.
2. **Arc-distribution KL distillation** — Hindi parser as teacher; transfer
   parsing *decisions*, not just representations.
3. **Cross-lingual Tree Supervision (CTS)** — reuse Hindi gold heads as
   Bhojpuri training signal at matched positions.

Combined loss:
```
L_H = L_bho + 0.5·L_hi + 0.4·L_cos + 2.0·L_arc-kl + 0.2·L_cts
```

### System K — UD-Bridge

Three-stage self-training pipeline:

1. **Generate silver UD labels.** Run System A (Hindi-only Trankit) on the
   30,966 Bhojpuri sentences from the aligned corpus. Strip auto-transferred
   labels; keep System A's UD-style predictions.
2. **Concatenate with HDTB gold.** Merge silver Bhojpuri with 13,304 Hindi
   gold sentences — both in UD schema.
3. **Train fresh Trankit.** All training data lives in the UD label space, so
   gains transfer directly to BHTB.

Core insight: **Schema consistency beats data quantity.** Noisy data in the
right annotation space outperforms clean data in the wrong space.

### Architecture (Systems G / H)

```
Input: Hindi / Bhojpuri sentences (parallel pairs)
        ↓
XLM-RoBERTa-base  (frozen — 12 layers, 768-dim, ~278M params)
        ↓
   ┌────┴────────┐
   ↓             ↓
Hindi Adapter  Bhojpuri Adapter   (Pfeiffer, r=64, ~100K params each)
   ↓             ↓
Hindi Biaffine  Bhojpuri Biaffine (arc MLP=500, label MLP=100)
   ↓             ↓
L_hi          L_bho               + L_cos / L_arc-kl / L_cts (SACT)
```

---

## Repository Structure

```
mtechthesis4biaffinemultilingual-parser/
├── README.md                          # This file
├── config.py                          # Central paths and hyperparameters
├── requirements.txt                   # Python dependencies
├── run_full_pipeline.sh               # End-to-end shell pipeline
│
├── train_system_f.py                  # System F — warm-start fine-tuning
├── train_system_g.py                  # System G — parallel adapters + MSE
├── train_system_h.py                  # System H — SACT
├── train_system_k.py                  # System K — UD-Bridge silver training
│
├── train_trankit_hindi.py             # System A — Hindi training
├── train_trankit_bhojpuri.py          # Trankit Bhojpuri training driver
├── train_trankit_bhojpuri_warmstart.py # Warm-start variant for F
├── train_bilingual.py                 # Bilingual training (G/H scaffolding)
├── train_monolingual.py               # Monolingual training utility
│
├── generate_silver_ud_labels.py       # System K — silver-data generation
├── compare_silver_labels.py           # Silver-data analysis utility
├── precompute_cache.py                # XLM-R caching (G/H acceleration)
├── patch_trankit_env.py               # Trankit environment patches
│
├── evaluate.py                        # Custom adapter-pipeline eval
├── evaluate_trankit.py                # Trankit eval (A, F, K)
├── re_eval.py                         # Re-evaluation utility
├── quick_test.py                      # Sanity-check utility
│
├── data/                              # Data preparation scripts
├── data_files/                        # Treebanks (HDTB, BHTB, synthetic)
├── model/                             # Custom adapter + biaffine modules
├── utils/                             # I/O, alignment, scoring helpers
├── slurm/                             # HPC batch scripts
│
└── thesis/                            # Full thesis report
    ├── thesis.tex                     # 89-page LaTeX source
    ├── thesis.pdf                     # Compiled thesis report
    ├── presentation.tex               # 26-slide defense presentation
    ├── presentation.pdf               # Compiled slides
    ├── presentation_pitch.pdf         # Slide-by-slide speaking guide
    └── references.bib                 # Bibliography (60+ entries)
```

---

## Pipeline

```
                    Hindi HDTB (13,304 gold UD sentences)
                              │
            ┌─────────────────┼──────────────────────────┐
            │                 │                           │
            ▼                 │                           ▼
  [System A — Hindi]   Hindi–Bhojpuri Aligned    [System K Step 1]
   train_trankit_      Corpus (30,966 pairs)    generate_silver_ud_labels.py
   hindi.py                   │                  (System A inference on Bho)
            │                 │                           │
            │     ┌───────────┼─────────┐                 ▼
            │     │           │         │           Silver Bho UD (~30K)
            │     ▼           ▼         ▼                 │
            │  [System F]  [System G] [System H]   [System K Step 2/3]
            │  train_      train_     train_       train_system_k.py
            │  system_f    system_g   system_h     (HDTB + Silver Bho)
            │     │           │         │                 │
            ▼     ▼           ▼         ▼                 ▼
       BHTB Gold ◀───── Aligned Dev (3,097 sent) ────▶ BHTB Gold
       (357 sent)                                     (357 sent)
       UAS 52.78    UAS 55.7   UAS 55.7  UAS 55.85    UAS 54.27
       LAS 35.36    LAS ~50    LAS 50.02 LAS 50.08    LAS 36.70
```

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Download UD treebanks (HDTB + BHTB)
python data/download_ud_data.py

# Train System A (Hindi parser, ~6h on V100)
python train_trankit_hindi.py --gpu

# Train System F (warm-start fine-tuning, ~10h)
python train_system_f.py --gpu --warm_start_from checkpoints/trankit_hindi

# Train System G (parallel adapters + MSE, ~12h)
python precompute_cache.py --gpu        # Cache XLM-R embeddings
python train_system_g.py --gpu

# Train System H (SACT, ~12h)
python train_system_h.py --gpu

# Train System K (UD-Bridge)
python generate_silver_ud_labels.py --gpu     # Step 1: generate silver
python train_system_k.py --gpu                # Step 2/3: merge + train

# Evaluate any system on BHTB
python evaluate_trankit.py --system K --test data_files/bhojpuri/bhtb_test.conllu
```

See `run_full_pipeline.sh` for the full end-to-end driver.

---

## Hyperparameters

| Parameter                 | Value                |
|---------------------------|----------------------|
| Backbone                  | xlm-roberta-base     |
| Adapter bottleneck dim    | 64 (Pfeiffer-style)  |
| Arc MLP dim               | 500                  |
| Label MLP dim             | 100                  |
| MLP dropout               | 0.33                 |
| Optimiser                 | AdamW                |
| LR (adapters)             | 2 × 10⁻³             |
| LR (full fine-tune)       | 5 × 10⁻⁵             |
| Max epochs                | 60 (early stop @ 10) |
| Batch size                | 16                   |
| SACT weights              | λ_hi=0.5, λ_cos=0.4, λ_arc=2.0, λ_cts=0.2 |

---

## Datasets

| Dataset                       | Language     | Size            | Role                  |
|-------------------------------|--------------|-----------------|------------------------|
| HDTB (UD)                     | Hindi        | 13,304 sent     | A training; K main     |
| Hindi–Bhojpuri aligned corpus | Hi–Bho       | 30,966 pairs    | F/G/H training; K silver source |
| Aligned-data Dev split        | Bhojpuri     | 3,097 sent      | F/G/H evaluation       |
| BHTB gold (UD)                | Bhojpuri     | 357 sent        | A/K evaluation (test only) |

The Hindi–Bhojpuri aligned corpus was provided by the supervising lab; it is
in CoNLL-U format with Bhojpuri tokens in the FORM column and matched Hindi
tokens in the LEMMA column (auto-transferred annotations).

---

## Comparison with Published Hindi-HDTB Results

System A's Hindi parsing performance reproduces published Trankit numbers,
validating the pipeline:

| System                        | UAS    | LAS    |
|-------------------------------|--------|--------|
| UDPipe v2.5                   | 95.07  | 90.23  |
| Stanza v1.1.1                 | 96.66  | 91.74  |
| Trankit-base (XLM-R-base)     | 96.54  | 92.70  |
| **System A (ours, reproduced)** | ~95.6 | ~92.6 |

The dramatic Hindi → Bhojpuri drop (92.7 → 35.36 LAS) is the expected gap of
zero-shot cross-lingual transfer to a low-resource target — System K closes
some of that gap through schema-consistent silver self-training.

---

## Thesis & Presentation

The full thesis report (89 pages) and presentation slides are available in
the [`thesis/`](thesis/) folder:

- [`thesis.pdf`](thesis/thesis.pdf) — Full thesis report
- [`presentation.pdf`](thesis/presentation.pdf) — 26-slide defense presentation
- [`presentation_pitch.pdf`](thesis/presentation_pitch.pdf) — Slide-by-slide
  speaking guide

---

## Citation

If you use this work, please cite:

```bibtex
@mastersthesis{kolgane2026crosslingual,
  author    = {Kolgane Sanskruti Sanjay},
  title     = {Cross-Lingual Dependency Parsing for Low-Resource Bhojpuri
               via Parallel Bottleneck Adapters and Syntax-Aware Transfer},
  school    = {Indian Institute of Technology (BHU) Varanasi},
  year      = {2026},
  type      = {M.Tech. Thesis (IDD)},
  address   = {Varanasi, India},
}
```

---

## Acknowledgements

- **Dr. Anil Kumar Singh** (IIT BHU) — supervision and the Hindi–Bhojpuri
  aligned corpus.
- **National Supercomputing Mission** — access to *Param Shivay* HPC at IIT BHU.
- **Trankit** ([nlp-uoregon/trankit](https://github.com/nlp-uoregon/trankit)) —
  multilingual NLP toolkit.
- **XLM-RoBERTa** (Meta AI) — multilingual transformer backbone.
- **Universal Dependencies** project and the **BHTB** annotators.

---

## License

Code released for academic and research purposes. See thesis declaration for
terms of use.

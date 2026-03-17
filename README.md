# Cross-lingual Hindi → Bhojpuri Dependency Parser

A dependency parser for **Bhojpuri** (a low-resource Indo-Aryan language) built entirely without any manual Bhojpuri annotations, using cross-lingual transfer from Hindi via annotation projection and the [Trankit](https://github.com/nlp-uoregon/trankit) multilingual NLP toolkit.

---

## Overview

Bhojpuri has no publicly available annotated dependency treebank for training. This project exploits the close linguistic relationship between Hindi and Bhojpuri to build a parser through three systems:

| System | Description |
|--------|-------------|
| **A — Zero-shot** | Hindi Trankit model applied directly to Bhojpuri (no Bhojpuri training) |
| **B — Projection-only** | Bhojpuri model trained on 5,000-sentence unfiltered projected synthetic treebank |
| **C — Quality-filtered** | Bhojpuri model trained on 4,941-sentence treebank filtered by alignment coverage ≥ 70% |

All three systems use **XLM-RoBERTa** as the backbone with **Pfeiffer adapters** (via Trankit's TPipeline) and a **biaffine parsing head**.

---

## Architecture

```
Input tokens
     ↓
XLM-RoBERTa-base  (frozen — 12 transformer layers, 768-dim)
     ↓  ← Pfeiffer adapters per layer  (trainable, bottleneck dim=64)
Contextual token embeddings
     ↓
POS tagger head        (768 → n_upos)
Biaffine parser head:
   Arc MLP:   768 → 500
   Label MLP: 768 → 100
   Biaffine scorer → head pointer + deprel per token
```

Adapter-only fine-tuning (~0.3% of parameters) preserves the multilingual XLM-R representations needed for cross-lingual transfer, preventing catastrophic forgetting.

---

## Pipeline

```
HDTB (Hindi treebank, ~13K sentences)
        ↓
[1] train_trankit_hindi.py       →  Hindi Trankit model  (System A)
                                            ↓
Parallel Hindi–Bhojpuri corpus  →  SimAlign word alignments
                                            ↓
                             [2] data/build_synthetic_treebank.py
                                            ↓
                    bho_synthetic_train.conllu  (5,000 projected sentences)
                                ↓                      ↓ coverage ≥ 70%
                             [3] data/build_treebank_filtered.py
                                                       ↓
                                    bho_filtered_train.conllu  (4,941 sentences)
                                ↓                      ↓
[4] train_trankit_bhojpuri.py --system proj    --system filtered
        ↓                                              ↓
   System B model                             System C model
        ↓                                              ↓
        └──────────────────────────────────────────────┘
                               ↓
                    [5] evaluate_trankit.py
                               ↓
           Real BHTB test set (357 Bhojpuri sentences)
```

---

## Repository Structure

```
cross_lingual_parser/
├── config.py                        # Central config: paths, hyperparameters
├── requirements.txt
├── run_full_pipeline.sh             # End-to-end shell script
│
├── data/                            # Data preparation scripts
│   ├── download_ud_data.py          # Download HDTB and BHTB from UD
│   ├── translate_hindi.py           # Produce Hindi–Bhojpuri parallel corpus
│   ├── word_alignment.py            # SimAlign word alignment
│   ├── build_synthetic_treebank.py  # Annotation projection → synthetic CoNLL-U
│   ├── build_treebank_filtered.py   # Quality filtering (coverage ≥ threshold)
│   └── project_annotations.py      # Core projection logic
│
├── data_files/
│   ├── hindi/                       # HDTB CoNLL-U (train / dev / test)
│   ├── bhojpuri/                    # BHTB CoNLL-U (test only — 357 sentences)
│   └── synthetic/                   # Projected + filtered Bhojpuri treebanks
│       ├── bho_synthetic_train.conllu
│       ├── bho_synthetic_dev.conllu
│       ├── bho_filtered_train.conllu
│       ├── bho_filtered_dev.conllu
│       ├── alignments_train.txt
│       └── translations_train.txt
│
├── train_trankit_hindi.py           # Step 1 — Train Hindi parser (System A)
├── train_trankit_bhojpuri.py        # Step 2 — Train Bhojpuri parsers (B and C)
├── evaluate_trankit.py              # Step 3 — 3-way evaluation on BHTB test
│
├── model/                           # Custom biaffine parser (non-Trankit variant)
│   ├── biaffine_heads.py
│   ├── cross_lingual_parser.py
│   ├── cross_lingual_layer.py
│   ├── cross_sentence_attention.py
│   └── parallel_encoder.py
│
└── utils/
    ├── conllu_utils.py              # CoNLL-U read/write helpers
    └── metrics.py                   # UAS / LAS scoring
```

---

## Installation

```bash
# Python 3.9+
pip install trankit torch transformers simalign conllu tqdm numpy
# or
pip install -r requirements.txt
```

XLM-RoBERTa-base will be downloaded automatically by Trankit on first run (~1GB). If working offline, set:
```bash
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
```

---

## How to Run

### Step 0 — Download treebanks

```bash
python3 data/download_ud_data.py
```

Downloads HDTB (Hindi) and BHTB (Bhojpuri) from Universal Dependencies into `data_files/`.

---

### Step 1 — Train Hindi parser (System A)

```bash
python3 train_trankit_hindi.py [--epochs 60] [--batch_size 16] [--gpu]
```

Trains a Trankit posdep model on HDTB. Checkpoint saved to:
```
checkpoints/trankit_hindi/xlm-roberta-base/hindi/hindi.tagger.mdl
```

Training result: **epoch 45**, dev UAS **95.58%** / LAS **92.63%** (HDTB dev set)

---

### Step 2 — Build synthetic Bhojpuri treebank

```bash
# Build word alignments from parallel corpus
python3 data/word_alignment.py

# Project Hindi annotations → Bhojpuri via alignments
python3 data/build_synthetic_treebank.py

# Apply quality filter (coverage ≥ 70%)
python3 data/build_treebank_filtered.py --coverage 0.70 --max_sents 5000
```

Outputs:
- `data_files/synthetic/bho_synthetic_train.conllu` — 5,000 projected sentences
- `data_files/synthetic/bho_filtered_train.conllu` — 4,941 filtered sentences

---

### Step 3 — Train Bhojpuri parsers (Systems B and C)

```bash
# Train both systems sequentially
python3 train_trankit_bhojpuri.py --system both [--epochs 60] [--batch_size 16] [--gpu]

# Or train individually
python3 train_trankit_bhojpuri.py --system proj      # System B only
python3 train_trankit_bhojpuri.py --system filtered  # System C only
```

Checkpoints:
```
checkpoints/trankit_bho_proj/xlm-roberta-base/bhojpuri_proj/bhojpuri_proj.tagger.mdl
checkpoints/trankit_bho_filtered/xlm-roberta-base/bhojpuri_filtered/bhojpuri_filtered.tagger.mdl
```

Training results on synthetic Bhojpuri dev:
- System B: **epoch 42**, dev LAS **84.11%**
- System C: **epoch 56**, dev LAS **84.01%**

---

### Step 4 — Evaluate all three systems on real BHTB test set

```bash
python3 evaluate_trankit.py [--gpu]
```

Evaluates Systems A, B, C on the 357-sentence real Bhojpuri BHTB test set and prints UAS, LAS, per-relation LAS, and delta tables.

---

## Final Results

Evaluated on the **real BHTB test set — 357 manually annotated Bhojpuri sentences**.

### 3-Way Comparison

| System | Description | UAS | LAS |
|--------|-------------|-----|-----|
| **A** | Zero-shot (Hindi Trankit → Bhojpuri) | **53.48%** | **34.84%** |
| B | Projection-only (5,000 unfiltered sentences) | 46.60% | 29.35% |
| C | Quality-filtered (4,941 sentences, coverage ≥ 70%) | 46.08% | 29.48% |

### Cross-lingual Transfer Gains (LAS)

| Comparison | ΔLAS |
|------------|------|
| B vs A — projection over zero-shot | −5.49% |
| C vs B — filtering over raw projection | +0.13% |
| C vs A — full pipeline over zero-shot | −5.36% |

### Per-relation LAS — System A (best system)

| Relation | Correct | Total | LAS |
|----------|---------|-------|-----|
| case | 755 | 907 | 83.2% |
| root | 177 | 357 | 49.6% |
| punct | 317 | 695 | 45.6% |
| mark | 52 | 123 | 42.3% |
| conj | 47 | 115 | 40.9% |
| advmod | 11 | 29 | 37.9% |
| amod | 74 | 202 | 36.6% |
| nummod | 18 | 54 | 33.3% |
| obl | 105 | 352 | 29.8% |
| nmod | 261 | 907 | 28.8% |
| aux | 68 | 302 | 22.5% |
| compound | 382 | 1,610 | 23.7% |
| nsubj | 56 | 273 | 20.5% |
| det | 35 | 174 | 20.1% |
| obj | 18 | 122 | 14.8% |
| acl | 1 | 105 | 1.0% |
| advcl | 3 | 82 | 3.7% |
| xcomp | 0 | 67 | 0.0% |
| ccomp | 0 | 58 | 0.0% |
| cc | 1 | 30 | 3.3% |

---

## Key Findings

1. **Zero-shot transfer (System A) outperforms both fine-tuned systems by ~5 LAS points.** XLM-RoBERTa's shared multilingual representations already encode enough Hindi–Bhojpuri structural similarity that direct transfer beats training on noisy projected data.

2. **Systems B and C are nearly identical** (ΔLAS = +0.13pp). The 70% coverage filter only removed 59 of 5,000 sentences — too lenient to have a meaningful effect. A stricter threshold (85–90%) would be needed.

3. **Structural relations transfer well** (`case` 83.2%, `root` 49.6%, `punct` 45.6%) — these are consistent between Hindi and Bhojpuri. Clausal relations (`xcomp`, `ccomp`, `advcl`, `acl`) are near zero — they involve clause-level word order differences that break projection.

4. **UAS target (45–55%) met by all three systems.** LAS target (35–45%) narrowly missed by System A (34.84%) and more substantially by B/C (~29.4%).

---

## Hyperparameters

| Parameter | Value |
|-----------|-------|
| Backbone | xlm-roberta-base |
| Adapter bottleneck dim | 64 (Pfeiffer-style) |
| Arc MLP dim | 500 |
| Label MLP dim | 100 |
| MLP dropout | 0.33 |
| Max epochs | 60 |
| Batch size | 16 |
| Projection coverage threshold | 0.70 |
| Synthetic treebank size | 5,000 / 4,941 (filtered) |

---

## Data Sources

| Dataset | Description | Sentences |
|---------|-------------|-----------|
| [HDTB](https://universaldependencies.org/treebanks/hi_hdtb/) | Hindi Dependency Treebank (UD) | ~13K train |
| [BHTB](https://universaldependencies.org/treebanks/bho_bhtb/) | Bhojpuri Treebank (UD) — test only | 357 test |
| Parallel corpus | Hindi–Bhojpuri translations | 5,000 pairs |

---

## References

- Nguyen et al. (2021). [Trankit: A Light-Weight Transformer-based Toolkit for Multilingual Natural Language Processing](https://aclanthology.org/2021.eacl-demos.10/). EACL 2021.
- Jalili Sabet et al. (2020). [SimAlign: High Quality Word Alignments Without Parallel Training Data Using Static and Contextualized Embeddings](https://aclanthology.org/2020.findings-emnlp.147/). EMNLP Findings 2020.
- Conneau et al. (2020). [Unsupervised Cross-lingual Representation Learning at Scale](https://aclanthology.org/2020.acl-main.747/). ACL 2020. (XLM-RoBERTa)
- Houlsby et al. (2019). [Parameter-Efficient Transfer Learning for NLP](https://proceedings.mlr.press/v97/houlsby19a.html). ICML 2019. (Adapters)

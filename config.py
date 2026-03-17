# config.py — Central configuration for the cross-lingual Hindi→Bhojpuri parser
# All hyper-parameters and paths are gathered here so nothing is hard-coded elsewhere.

from dataclasses import dataclass, field
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).parent
DATA_DIR      = ROOT / "data_files"
CHECKPT_DIR   = ROOT / "checkpoints"
LOG_DIR       = ROOT / "logs"

# UD treebank paths (populated by data/download_ud_data.py)
HINDI_TRAIN   = DATA_DIR / "hindi" / "hi_hdtb-ud-train.conllu"
HINDI_DEV     = DATA_DIR / "hindi" / "hi_hdtb-ud-dev.conllu"
HINDI_TEST    = DATA_DIR / "hindi" / "hi_hdtb-ud-test.conllu"

BHOJPURI_TEST = DATA_DIR / "bhojpuri" / "bho_bhtb-ud-test.conllu"

# Synthetic treebank (produced by data/build_synthetic_treebank.py)
SYNTHETIC_TRAIN = DATA_DIR / "synthetic" / "bho_synthetic_train.conllu"
SYNTHETIC_DEV   = DATA_DIR / "synthetic" / "bho_synthetic_dev.conllu"

# Parallel alignment files  (one per sentence: "src_idx-tgt_idx ...")
ALIGNMENT_TRAIN = DATA_DIR / "synthetic" / "alignments_train.txt"
ALIGNMENT_DEV   = DATA_DIR / "synthetic" / "alignments_dev.txt"


# ─────────────────────────────────────────────────────────────────────────────
# Local XLM-RoBERTa path (offline — bypasses HuggingFace network access)
# ─────────────────────────────────────────────────────────────────────────────
import os as _os
_HF_HUB   = Path(_os.path.expanduser("~/.cache/huggingface/hub"))
_XLM_SNAP = _HF_HUB / "models--xlm-roberta-base/snapshots/e73636d4f797dec63c3081bb6ed5c7b0bb3f2089"
XLM_R_LOCAL = str(_XLM_SNAP) if _XLM_SNAP.exists() else "xlm-roberta-base"


# ─────────────────────────────────────────────────────────────────────────────
# Encoder / backbone
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class EncoderConfig:
    model_name:    str   = "xlm-roberta-base"   # HuggingFace model id
    hidden_size:   int   = 768                  # XLM-R base hidden dim
    max_seq_len:   int   = 256
    freeze_xlmr:   bool  = True                 # freeze backbone; train adapters only
    adapter_dim:   int   = 64                   # bottleneck dimension (Pfeiffer-style)
    adapter_dropout: float = 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Biaffine parser heads
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class BiaffineConfig:
    arc_mlp_dim:   int   = 500
    label_mlp_dim: int   = 100
    mlp_dropout:   float = 0.33
    # UD relations shared by Hindi & Bhojpuri
    # (populated at training time from vocabulary)
    n_rels:        int   = 45


# ─────────────────────────────────────────────────────────────────────────────
# Cross-lingual attention (Step 4)
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CrossAttentionConfig:
    n_heads:   int   = 8
    dropout:   float = 0.1


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class TrainConfig:
    # General
    seed:           int   = 42
    device:         str   = "mps"         # Apple Silicon GPU — use "cpu" if unavailable
    # Monolingual pre-training (Step 7)
    mono_epochs:    int   = 30
    mono_lr:        float = 5e-4
    # Bilingual fine-tuning (Steps 7-8)
    bi_epochs:      int   = 20
    bi_lr:          float = 2e-4
    # Loss weights
    lambda_bho:     float = 0.5    # weight on noisy Bhojpuri projected labels
    lambda_align:   float = 0.1    # cross-lingual alignment regularisation
    # Gradient clipping
    max_grad_norm:  float = 5.0
    # Batch (sentence pairs per step)
    batch_size:     int   = 1       # sentence-level; lengths vary too much to batch
    patience:       int   = 5       # early-stopping patience (dev LAS)


# ─────────────────────────────────────────────────────────────────────────────
# Top-level config object
# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class Config:
    encoder:      EncoderConfig      = field(default_factory=EncoderConfig)
    biaffine:     BiaffineConfig     = field(default_factory=BiaffineConfig)
    cross_attn:   CrossAttentionConfig = field(default_factory=CrossAttentionConfig)
    train:        TrainConfig        = field(default_factory=TrainConfig)

    def make_dirs(self):
        for d in (DATA_DIR, CHECKPT_DIR, LOG_DIR,
                  DATA_DIR / "hindi",
                  DATA_DIR / "bhojpuri",
                  DATA_DIR / "synthetic"):
            d.mkdir(parents=True, exist_ok=True)


# Singleton — import this everywhere
CFG = Config()

#!/usr/bin/env python3
# train_system_k.py
# System K — UD-Bridge via Silver Self-Training (Novel Contribution for Comp 2)
#
# TARGET:
#   Beat System A on BHTB: UAS > 53.48% / LAS > 34.84%.
#
# WHY THIS WORKS (Competition 2 analysis):
#   The entire G/H/I/J family scored ~21 LAS on BHTB despite 50 LAS on Dev because
#   the professor's data is annotated in a different schema than UD BHTB. System
#   K sidesteps this by training on UD-STYLE data only:
#     - HDTB train   : gold UD Hindi (same schema as BHTB)
#     - silver Bho   : System A's predictions on prof's Bhojpuri tokens
#                      (in UD schema, because System A was trained on UD HDTB)
#
#   This gives the model Bhojpuri-specific supervision WITHOUT importing the
#   professor's non-UD labels. The annotation schema is uniform across train /
#   test, so Dev performance now correlates with BHTB performance.
#
# ARCHITECTURE:
#   Same as System A — Trankit TPipeline with XLM-R-base + Pfeiffer adapters +
#   biaffine head. Only the training data differs.
#
# PIPELINE:
#   1. Run generate_silver_ud_labels.py → bho_silver_ud.conllu
#   2. Concatenate HDTB train + silver Bho → sysk_train.conllu
#   3. Train Trankit posdep on sysk_train (dev = HDTB dev for early stopping).
#   4. Evaluate on BHTB via evaluate_trankit.py (adds a System K row).
#
# Usage:
#   python3 train_system_k.py [--epochs 60] [--batch_size 16] [--gpu]
#                             [--silver_path DATA_FILES/synthetic/bho_silver_ud.conllu]
#                             [--no_hdtb] [--warm_start_sysa]

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Patch Trankit for transformers>=4.40 and multi-root tolerance.
from patch_trankit_env import patch_trankit_env
patch_trankit_env()

import argparse
import shutil
from pathlib import Path
from typing import List

import torch

from config import DATA_DIR, CHECKPT_DIR
from utils.conllu_utils import read_conllu, write_conllu, Sentence


SYSTEM_K_CATEGORY = "bhojpuri_sysk"
SYSTEM_K_SAVE_DIR = CHECKPT_DIR / "trankit_bho_sysk"
SYSTEM_K_TRAIN    = DATA_DIR / "sysk" / "sysk_train.conllu"
SYSTEM_K_DEV      = DATA_DIR / "sysk" / "sysk_dev.conllu"


# ─────────────────────────────────────────────────────────────────────────────
# HF cache symlink (mirrors train_trankit_hindi.py)
# ─────────────────────────────────────────────────────────────────────────────
def _ensure_xlmr_cache_symlink(save_dir: str, lang: str):
    hub_cache = Path.home() / ".cache/huggingface/hub/models--xlm-roberta-base"
    if not hub_cache.exists():
        return
    trankit_save = Path(save_dir) / "xlm-roberta-base" / lang
    for root in [trankit_save, trankit_save / "xlm-roberta-base"]:
        root.mkdir(parents=True, exist_ok=True)
        target = root / "models--xlm-roberta-base"
        if target.exists() and not target.is_symlink():
            shutil.rmtree(target)
        if not target.exists():
            target.symlink_to(hub_cache)


def _count_sents(path: Path) -> int:
    try:
        return sum(1 for line in open(path) if line.strip() == "")
    except Exception:
        return 0


def _concat_conllu(paths: List[Path], out: Path):
    """Concatenate multiple CoNLL-U files by copying sentence blocks."""
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as wfh:
        for p in paths:
            with open(p, encoding="utf-8") as rfh:
                for line in rfh:
                    wfh.write(line)
                if not line.endswith("\n\n"):
                    wfh.write("\n")
            n += _count_sents(p)
    return n


# ─────────────────────────────────────────────────────────────────────────────
# Warm-start: inject System A's weights into TPipeline AFTER construction
#
# Trankit does NOT auto-load pre-existing .tagger.mdl on TPipeline init — it
# builds a fresh posdep model. So copying the Hindi checkpoint into save_dir
# before init is a no-op (or worse, triggers label-head shape clashes).
#
# Instead: construct TPipeline normally (builds model with Bhojpuri/UD vocab),
# then copy over every tensor from Hindi whose shape matches. Mismatched
# tensors (e.g. label head with different n_rels) are skipped — these are
# re-learned during training. Mirrors System F's _inject_hindi_warmstart().
# ─────────────────────────────────────────────────────────────────────────────
def _inject_sysa_warmstart(pipeline, sysa_mdl_path: Path) -> int:
    if not sysa_mdl_path.exists():
        print(f"      [WARN] System A checkpoint missing: {sysa_mdl_path} — cold init")
        return 0

    state    = torch.load(str(sysa_mdl_path), map_location="cpu")
    adapters = state["adapters"]
    epoch    = state.get("epoch", "?")

    emb_sd = pipeline._embedding_layers.state_dict()
    tag_sd = pipeline._tagger.state_dict()

    copied_emb = copied_tag = skipped = 0
    for k, v in adapters.items():
        if k in emb_sd and emb_sd[k].shape == v.shape:
            emb_sd[k] = v
            copied_emb += 1
        elif k in tag_sd and tag_sd[k].shape == v.shape:
            tag_sd[k] = v
            copied_tag += 1
        else:
            skipped += 1

    pipeline._embedding_layers.load_state_dict(emb_sd)
    pipeline._tagger.load_state_dict(tag_sd)

    print(f"      Warm-start from System A (epoch {epoch}):")
    print(f"        Copied emb-layer tensors : {copied_emb}")
    print(f"        Copied tagger    tensors : {copied_tag}")
    print(f"        Skipped (shape mismatch) : {skipped}")
    return epoch


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Train System K — UD-Bridge via Silver Self-Training")
    ap.add_argument("--epochs",     type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--gpu",        action="store_true", default=False)

    ap.add_argument("--silver_path", type=str,
                    default=str(DATA_DIR / "synthetic" / "bho_silver_ud.conllu"),
                    help="Silver UD-Bhojpuri treebank from generate_silver_ud_labels.py")
    ap.add_argument("--no_hdtb",    action="store_true", default=False,
                    help="Train on silver Bho only (no HDTB gold)")
    ap.add_argument("--warm_start_sysa", action="store_true", default=False,
                    help="Initialize from System A checkpoint (faster convergence)")
    args = ap.parse_args()

    silver_path = Path(args.silver_path)
    hi_train    = DATA_DIR / "hindi/hi_hdtb-ud-train.conllu"
    hi_dev      = DATA_DIR / "hindi/hi_hdtb-ud-dev.conllu"

    if not silver_path.exists():
        raise FileNotFoundError(
            f"Silver treebank not found: {silver_path}\n"
            f"Run: python3 generate_silver_ud_labels.py --gpu"
        )
    if not hi_train.exists() and not args.no_hdtb:
        raise FileNotFoundError(
            f"HDTB train missing: {hi_train}\nRun: python3 data/download_ud_data.py"
        )

    SYSTEM_K_TRAIN.parent.mkdir(parents=True, exist_ok=True)

    # ── Build combined training corpus ────────────────────────────────────────
    print("=" * 70)
    print(" System K — UD-Bridge via Silver Self-Training")
    print("=" * 70)

    if args.no_hdtb:
        sources = [silver_path]
        label   = "silver-Bho only"
    else:
        sources = [hi_train, silver_path]
        label   = "HDTB + silver-Bho"

    n_total = _concat_conllu(sources, SYSTEM_K_TRAIN)
    print(f"  Training corpus  : {label}  ({n_total:,} sentences)")
    for p in sources:
        print(f"    - {p.name:35s}  {_count_sents(p):,} sents")
    print(f"  Combined file    : {SYSTEM_K_TRAIN}")

    # Dev: use HDTB dev (gold UD, matches BHTB schema)
    shutil.copy2(hi_dev, SYSTEM_K_DEV)
    print(f"  Dev (UD-style)   : {SYSTEM_K_DEV}  ({_count_sents(SYSTEM_K_DEV):,} sents)")

    print(f"  Epochs           : {args.epochs}   batch_size: {args.batch_size}   gpu: {args.gpu}")
    print(f"  Warm-start SysA  : {args.warm_start_sysa}")
    print(f"  Save dir         : {SYSTEM_K_SAVE_DIR}")
    print()

    save_dir = str(SYSTEM_K_SAVE_DIR)
    _ensure_xlmr_cache_symlink(save_dir, lang=SYSTEM_K_CATEGORY)

    # ── Trankit training ──────────────────────────────────────────────────────
    from trankit import TPipeline

    trainer = TPipeline(training_config={
        "category":           SYSTEM_K_CATEGORY,
        "task":               "posdep",
        "save_dir":           save_dir,
        "train_conllu_fpath": str(SYSTEM_K_TRAIN),
        "dev_conllu_fpath":   str(SYSTEM_K_DEV),
        "max_epoch":          args.epochs,
        "batch_size":         args.batch_size,
        "gpu":                args.gpu,
        "embedding":          "xlm-roberta-base",
    })

    # ── Post-init warm-start (copy matching tensors from System A) ────────────
    if args.warm_start_sysa:
        sysa_mdl = CHECKPT_DIR / "trankit_hindi/xlm-roberta-base/hindi/hindi.tagger.mdl"
        print("\n[Step] Injecting System A warm-start weights …")
        _inject_sysa_warmstart(trainer, sysa_mdl)

    print("Training …")
    trainer.train()

    model_path = Path(save_dir) / f"xlm-roberta-base/{SYSTEM_K_CATEGORY}/{SYSTEM_K_CATEGORY}.tagger.mdl"
    print(f"\nDone. Best checkpoint: {model_path}")
    print("\nNext: python3 evaluate_trankit.py --gpu --include_k")


if __name__ == "__main__":
    main()

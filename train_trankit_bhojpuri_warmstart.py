#!/usr/bin/env python3
# train_trankit_bhojpuri_warmstart.py
# System D — Two innovations combined:
#
#   Innovation 1 — Two-stage training (Hindi → Bhojpuri):
#     Instead of starting from a raw XLM-R backbone, warm-start the
#     Bhojpuri fine-tuning from the already-trained Hindi checkpoint.
#     The adapter weights and biaffine arc/label heads from Hindi provide
#     a much better initialisation than random — the model already knows
#     what a valid dependency parse looks like.
#
#   Innovation 3 — Relation-selective projection:
#     Training data built by data/build_selective_treebank.py.
#     For each token: HIGH_CONF deprels keep the projected annotation;
#     LOW_CONF deprels are replaced by System A (Hindi zero-shot) predictions.
#     This removes the noisiest part of projection without discarding sentences.
#
# Checkpoint saved to:
#   checkpoints/trankit_bho_warmstart/xlm-roberta-base/bhojpuri_warmstart/
#
# Usage:
#   python3 train_trankit_bhojpuri_warmstart.py [--epochs 60] [--batch_size 16] [--gpu]

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import shutil
import subprocess
import torch
from pathlib import Path

from config import DATA_DIR, CHECKPT_DIR


# ─────────────────────────────────────────────────────────────────────────────
def _ensure_xlmr_cache_symlink(save_dir: str, lang: str):
    """Symlink HF hub XLM-R cache into Trankit's cache_dir locations."""
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
            print(f"  Linked HF cache: {target}")


# ─────────────────────────────────────────────────────────────────────────────
def _inject_hindi_warmstart(pipeline, hindi_mdl_path: Path) -> int:
    """
    Innovation 1 — Copy Hindi checkpoint weights into the Bhojpuri TPipeline
    BEFORE training begins.

    Copies every parameter where the shape matches between the Hindi checkpoint
    and the freshly-initialised Bhojpuri model:
      - XLM-R adapter weights (always match — language-agnostic)
      - Biaffine arc scorer weights (always match — label-count-independent)
      - Classification heads (match when Hindi/Bhojpuri label sets align,
        which is the case here since Bhojpuri data is projected from Hindi UD)

    Parameters with a shape mismatch are left at their random XLM-R init.
    """
    state          = torch.load(str(hindi_mdl_path), map_location="cpu")
    hindi_adapters = state["adapters"]
    epoch          = state.get("epoch", "?")

    emb_sd = pipeline._embedding_layers.state_dict()
    tag_sd = pipeline._tagger.state_dict()

    copied_emb = copied_tag = skipped = 0
    for k, v in hindi_adapters.items():
        if k in emb_sd and emb_sd[k].shape == v.shape:
            emb_sd[k] = v
            copied_emb += 1
        elif k in tag_sd and tag_sd[k].shape == v.shape:
            tag_sd[k] = v
            copied_tag += 1
        else:
            skipped += 1   # shape mismatch → keep random init

    pipeline._embedding_layers.load_state_dict(emb_sd)
    pipeline._tagger.load_state_dict(tag_sd)

    print(f"  Warm-start from Hindi epoch {epoch}:")
    print(f"    Copied  — embedding layers : {copied_emb}")
    print(f"    Copied  — tagger head      : {copied_tag}")
    print(f"    Skipped (shape mismatch)   : {skipped}")
    return epoch


# ─────────────────────────────────────────────────────────────────────────────
def _count_sents(path: Path) -> int:
    try:
        return sum(1 for line in open(path) if line.strip() == "")
    except Exception:
        return 0


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="System D — Two-stage Hindi→Bhojpuri with relation-selective projection"
    )
    ap.add_argument("--epochs",     type=int,  default=60)
    ap.add_argument("--batch_size", type=int,  default=16)
    ap.add_argument("--gpu",        action="store_true", default=False)
    args = ap.parse_args()

    syn_dir    = DATA_DIR / "synthetic"
    train_file = syn_dir / "bho_selective_train.conllu"
    dev_file   = syn_dir / "bho_selective_dev.conllu"
    hindi_mdl  = CHECKPT_DIR / "trankit_hindi/xlm-roberta-base/hindi/hindi.tagger.mdl"
    save_dir   = str(CHECKPT_DIR / "trankit_bho_warmstart")
    lang       = "bhojpuri_warmstart"

    # ── Build selective treebank if not yet done ──────────────────────────────
    if not train_file.exists() or not dev_file.exists():
        print("Selective treebank missing — building now …")
        subprocess.run(
            [sys.executable, "data/build_selective_treebank.py", "--split", "both"],
            check=True
        )

    if not hindi_mdl.exists():
        print(f"Hindi checkpoint not found: {hindi_mdl}")
        print("Run: python3 train_trankit_hindi.py")
        return

    print("\n========================================")
    print(" System D — Two-stage Warm-start Training")
    print(" Hindi checkpoint → Bhojpuri fine-tune")
    print(" + Relation-selective projected data")
    print("========================================")
    print(f"  Train : {train_file}  ({_count_sents(train_file):,} sents)")
    print(f"  Dev   : {dev_file}  ({_count_sents(dev_file):,} sents)")
    print(f"  Warm  : {hindi_mdl}")
    print(f"  Save  : {save_dir}/xlm-roberta-base/{lang}/")
    print(f"  Epochs: {args.epochs}   batch_size: {args.batch_size}")

    _ensure_xlmr_cache_symlink(save_dir, lang=lang)

    from trankit import TPipeline

    # ── Step 1: Initialise TPipeline (builds Bhojpuri vocab from training data) ─
    print("\n  Initialising Bhojpuri TPipeline …")
    trainer = TPipeline(training_config={
        "category":           lang,
        "task":               "posdep",
        "save_dir":           save_dir,
        "train_conllu_fpath": str(train_file),
        "dev_conllu_fpath":   str(dev_file),
        "max_epoch":          args.epochs,
        "batch_size":         args.batch_size,
        "gpu":                args.gpu,
        "embedding":          "xlm-roberta-base",
    })

    # ── Step 2 (Innovation 1): Inject Hindi weights before training ───────────
    print("\n  Injecting Hindi warm-start weights …")
    _inject_hindi_warmstart(trainer, hindi_mdl)

    # ── Step 3: Fine-tune from the warm-started model ─────────────────────────
    print("\n  Training System D …")
    trainer.train()

    mdl = Path(save_dir) / f"xlm-roberta-base/{lang}/{lang}.tagger.mdl"
    print(f"\n  Saved → {mdl}")
    print("Next: python3 evaluate_trankit.py")


if __name__ == "__main__":
    main()

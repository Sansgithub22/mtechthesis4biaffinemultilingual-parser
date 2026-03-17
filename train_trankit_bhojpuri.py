#!/usr/bin/env python3
# train_trankit_bhojpuri.py
# Steps 5-7 (Trankit version) — Cross-lingual Bhojpuri parser using Trankit TPipeline.
#
# Three systems are trained and compared:
#
#   System B — Projection-only (unfiltered, original 5 000-sentence synthetic treebank)
#               Baseline: how well does naive projection work with Trankit?
#
#   System C — Quality-filtered projection (coverage ≥ 70%)
#               Should outperform B by removing noisy projections.
#
# Both are fine-tuned from XLM-R (same backbone as Hindi training), so the
# Hindi Trankit weights provide implicit cross-lingual transfer via the shared
# XLM-R representation.
#
# Checkpoints:
#   checkpoints/trankit_bho_proj/xlm-roberta-base/bhojpuri_proj/
#   checkpoints/trankit_bho_filtered/xlm-roberta-base/bhojpuri_filtered/
#
# Usage:
#   python3 train_trankit_bhojpuri.py [--system proj|filtered|both]
#                                     [--epochs 60] [--batch_size 16]

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Set offline mode BEFORE any trankit/transformers import to use local HF cache.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
from pathlib import Path

from config import DATA_DIR, CHECKPT_DIR


def train_system(
    lang:         str,      # language label for Trankit (must be unique per run)
    train_conllu: Path,
    dev_conllu:   Path,
    save_dir:     str,
    epochs:       int,
    batch_size:   int,
    gpu:          bool,
):
    """Train one Bhojpuri system using Trankit TPipeline."""
    if not train_conllu.exists():
        print(f"  [SKIP] Training data not found: {train_conllu}")
        print(f"         Run: python3 data/build_treebank_filtered.py")
        return

    print(f"\n  category={lang}  epochs={epochs}  batch={batch_size}")
    print(f"  train: {train_conllu}  ({_count_sents(train_conllu):,} sents)")
    print(f"  dev  : {dev_conllu}  ({_count_sents(dev_conllu):,} sents)")
    print(f"  save : {save_dir}/xlm-roberta-base/{lang}/")

    _ensure_xlmr_cache_symlink(save_dir, lang=lang)

    from trankit import TPipeline

    trainer = TPipeline(training_config={
        "category":           lang,
        "task":               "posdep",
        "save_dir":           save_dir,
        "train_conllu_fpath": str(train_conllu),
        "dev_conllu_fpath":   str(dev_conllu),
        "max_epoch":          epochs,
        "batch_size":         batch_size,
        "gpu":                gpu,
        "embedding":          "xlm-roberta-base",
    })
    trainer.train()
    print(f"  Saved → {save_dir}/xlm-roberta-base/{lang}/{lang}.tagger.mdl")


def main():
    ap = argparse.ArgumentParser(
        description="Train Bhojpuri Trankit parsers (projection-only and quality-filtered)"
    )
    ap.add_argument("--system",     choices=["proj", "filtered", "both"], default="both")
    ap.add_argument("--epochs",     type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--gpu",        action="store_true", default=False)
    args = ap.parse_args()

    syn_dir = DATA_DIR / "synthetic"

    print("\n========================================")
    print(" Training Bhojpuri Trankit Parsers")
    print("========================================")

    # ── System B: Projection-only (original unfiltered data) ─────────────────
    if args.system in ("proj", "both"):
        print("\n[System B] Projection-only (unfiltered 5,000 sentences)")
        train_system(
            lang         = "bhojpuri_proj",
            train_conllu = syn_dir / "bho_synthetic_train.conllu",
            dev_conllu   = syn_dir / "bho_synthetic_dev.conllu",
            save_dir     = str(CHECKPT_DIR / "trankit_bho_proj"),
            epochs       = args.epochs,
            batch_size   = args.batch_size,
            gpu          = args.gpu,
        )

    # ── System C: Quality-filtered projection ────────────────────────────────
    if args.system in ("filtered", "both"):
        print("\n[System C] Quality-filtered projection (coverage ≥ 70%)")
        # Build filtered treebank if it doesn't exist yet
        filtered_train = syn_dir / "bho_filtered_train.conllu"
        filtered_dev   = syn_dir / "bho_filtered_dev.conllu"
        if not filtered_train.exists():
            print("  Filtered treebank missing — building now …")
            import subprocess
            subprocess.run(
                [sys.executable, "data/build_treebank_filtered.py",
                 "--coverage", "0.70", "--max_sents", "5000"],
                check=True
            )
        train_system(
            lang         = "bhojpuri_filtered",
            train_conllu = filtered_train,
            dev_conllu   = filtered_dev,
            save_dir     = str(CHECKPT_DIR / "trankit_bho_filtered"),
            epochs       = args.epochs,
            batch_size   = args.batch_size,
            gpu          = args.gpu,
        )

    print("\nBhojpuri Trankit training complete.")
    print("Next: python3 evaluate_trankit.py")


def _ensure_xlmr_cache_symlink(save_dir: str, lang: str):
    """Symlink HF hub XLM-R cache into Trankit's cache_dir locations for offline resolution."""
    import shutil
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


def _count_sents(conllu_path: Path) -> int:
    try:
        return sum(1 for line in open(conllu_path) if line.strip() == "")
    except Exception:
        return 0


if __name__ == "__main__":
    main()

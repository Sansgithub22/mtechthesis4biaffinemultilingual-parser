#!/usr/bin/env python3
# train_trankit_hindi.py
# Step 1-2 (Trankit version) — Train Hindi dependency parser using Trankit TPipeline.
#
# Trankit architecture:
#   XLM-RoBERTa (frozen) + Pfeiffer adapters (per-layer, trainable) + biaffine head
#
# This trains the Hindi posdep (POS + dependency) model from scratch using
# Trankit's official training pipeline on the HDTB CoNLL-U files.
#
# After training the checkpoint is saved to:
#   checkpoints/trankit_hindi/xlm-roberta-base/hindi/hindi.tagger.mdl
#
# Usage:
#   python3 train_trankit_hindi.py [--epochs 60] [--batch_size 16]

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Set offline mode BEFORE any trankit/transformers import to use local HF cache.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
from pathlib import Path

from config import DATA_DIR, CHECKPT_DIR


def main():
    ap = argparse.ArgumentParser(description="Train Hindi Trankit parser on HDTB")
    ap.add_argument("--epochs",     type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--gpu",        action="store_true", default=False)
    args = ap.parse_args()

    hi_train = DATA_DIR / "hindi/hi_hdtb-ud-train.conllu"
    hi_dev   = DATA_DIR / "hindi/hi_hdtb-ud-dev.conllu"

    if not hi_train.exists():
        print("Hindi data missing. Run: python3 data/download_ud_data.py")
        return

    save_dir = str(CHECKPT_DIR / "trankit_hindi")

    print("\n========================================")
    print(" Training Hindi Trankit Parser (TPipeline)")
    print("========================================")
    print(f"  Train : {hi_train}  ({_count_sents(hi_train):,} sentences)")
    print(f"  Dev   : {hi_dev}  ({_count_sents(hi_dev):,} sentences)")
    print(f"  Epochs: {args.epochs}   batch_size: {args.batch_size}")
    print(f"  Save  : {save_dir}/xlm-roberta-base/hindi/")
    print()

    # Trankit calls XLMRobertaTokenizer.from_pretrained("xlm-roberta-base", cache_dir=save_dir)
    # so we symlink the HF hub cache entry into save_dir for offline resolution.
    _ensure_xlmr_cache_symlink(save_dir, lang="hindi")

    from trankit import TPipeline

    trainer = TPipeline(training_config={
        "category":         "hindi",
        "task":             "posdep",
        "save_dir":         save_dir,
        "train_conllu_fpath": str(hi_train),
        "dev_conllu_fpath":   str(hi_dev),
        "max_epoch":        args.epochs,
        "batch_size":       args.batch_size,
        "gpu":              args.gpu,
        "embedding":        os.environ.get("XLM_R_PATH", "xlm-roberta-base"),
    })

    print("Training …")
    trainer.train()

    model_path = Path(save_dir) / "xlm-roberta-base/hindi/hindi.tagger.mdl"
    print(f"\nDone. Best checkpoint: {model_path}")
    print("Next: python3 train_trankit_bhojpuri.py")


def _ensure_xlmr_cache_symlink(save_dir: str, lang: str = "hindi"):
    """
    Trankit's cache_dir locations for XLM-R:
      Tokenizer: {_save_dir}/models--xlm-roberta-base
                  where _save_dir = {save_dir}/xlm-roberta-base/{lang}
      Model:     {_save_dir}/xlm-roberta-base/models--xlm-roberta-base
    Symlink both to the local HuggingFace hub cache for offline resolution.
    """
    import shutil
    hub_cache = Path.home() / ".cache/huggingface/hub/models--xlm-roberta-base"
    if not hub_cache.exists():
        return
    trankit_save = Path(save_dir) / "xlm-roberta-base" / lang
    # Locations Trankit uses as cache_dir roots
    cache_roots = [
        trankit_save,                          # tokenizer cache_dir
        trankit_save / "xlm-roberta-base",     # model cache_dir (cache_dir/embedding_name)
    ]
    for root in cache_roots:
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

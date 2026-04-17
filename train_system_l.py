#!/usr/bin/env python3
# train_system_l.py
# System L — Iterative Self-Training with Agreement Filtering (Comp 2 fallback)
#
# MOTIVATION:
#   System K does one round of silver self-training from System A. If K beats A
#   only marginally (or not at all), the remaining error is concentrated in the
#   noisy portion of the silver labels. System L adds:
#     (1) A second round of silver labelling using System K as the teacher
#         (silver-v2 is better than silver-v1 because K has seen Bhojpuri).
#     (2) An agreement filter: keep only sentences where System A and System K
#         agree on ≥ 80% of head attachments — the rest are discarded.
#     (3) Warm-start from System K so training starts from the best model so
#         far and only needs to refine.
#
# WHY THIS IS A LEGITIMATE NOVEL CONTRIBUTION:
#   - Iterative self-training / tri-training (McClosky et al. 2006, Kurniawan
#     et al. 2021) applied to Hindi→Bhojpuri cross-lingual dependency parsing.
#   - Teacher-agreement filtering is co-training-inspired (Blum & Mitchell
#     1998) and specifically targets annotation-style noise in silver data.
#
# PIPELINE:
#   0. Assumes System K is trained (train_system_k.py done).
#   1. Re-parse Bhojpuri tokens with System K → silver_ud_v2.conllu.
#   2. compare_silver_labels.py filters by agreement with silver_ud.conllu
#      (v1 from System A) → silver_ud_filtered.conllu.
#   3. Train Trankit on HDTB + silver_ud_filtered, warm-started from K.
#
# Usage:
#   python3 train_system_l.py [--epochs 60] [--batch_size 16] [--gpu]
#                             [--no_warm_start_k]

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

from patch_trankit_env import patch_trankit_env
patch_trankit_env()

import argparse
import shutil
from pathlib import Path
from typing import List

import torch

from config import DATA_DIR, CHECKPT_DIR
from utils.conllu_utils import read_conllu, write_conllu


SYSTEM_L_CATEGORY = "bhojpuri_sysl"
SYSTEM_L_SAVE_DIR = CHECKPT_DIR / "trankit_bho_sysl"
SYSTEM_L_TRAIN    = DATA_DIR / "sysl" / "sysl_train.conllu"
SYSTEM_L_DEV      = DATA_DIR / "sysl" / "sysl_dev.conllu"


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


def _inject_warmstart(pipeline, src_mdl: Path, label: str = "teacher") -> int:
    """Inject Trankit adapter weights AFTER TPipeline construction (see K's docstring)."""
    if not src_mdl.exists():
        print(f"      [WARN] {label} checkpoint missing: {src_mdl} — cold init")
        return 0

    state    = torch.load(str(src_mdl), map_location="cpu")
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

    print(f"      Warm-start from {label} (epoch {epoch}):")
    print(f"        Copied emb-layer tensors : {copied_emb}")
    print(f"        Copied tagger    tensors : {copied_tag}")
    print(f"        Skipped (shape mismatch) : {skipped}")
    return epoch


def main():
    ap = argparse.ArgumentParser(
        description="Train System L — Iterative Self-Training with Agreement Filtering")
    ap.add_argument("--epochs",     type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--gpu",        action="store_true", default=False)

    ap.add_argument("--silver_filtered", type=str,
                    default=str(DATA_DIR / "synthetic" / "bho_silver_ud_filtered.conllu"),
                    help="Agreement-filtered silver corpus from compare_silver_labels.py")
    ap.add_argument("--no_hdtb", action="store_true", default=False,
                    help="Train on filtered silver only (no HDTB)")
    ap.add_argument("--no_warm_start_k", action="store_true", default=False,
                    help="Do not warm-start from System K (cold init from XLM-R)")
    args = ap.parse_args()

    silver_path = Path(args.silver_filtered)
    hi_train    = DATA_DIR / "hindi/hi_hdtb-ud-train.conllu"
    hi_dev      = DATA_DIR / "hindi/hi_hdtb-ud-dev.conllu"

    if not silver_path.exists():
        raise FileNotFoundError(
            f"Filtered silver treebank not found: {silver_path}\n"
            f"Run the full System L pipeline via slurm/run_sysl.sh, or:\n"
            f"  python3 generate_silver_ud_labels.py \\\n"
            f"      --teacher_dir checkpoints/trankit_bho_sysk \\\n"
            f"      --teacher_category bhojpuri_sysk \\\n"
            f"      --output data_files/synthetic/bho_silver_ud_v2.conllu --gpu\n"
            f"  python3 compare_silver_labels.py \\\n"
            f"      --silver_v1 data_files/synthetic/bho_silver_ud.conllu \\\n"
            f"      --silver_v2 data_files/synthetic/bho_silver_ud_v2.conllu \\\n"
            f"      --output    {silver_path} --min_agreement 0.80"
        )

    SYSTEM_L_TRAIN.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" System L — Iterative Self-Training + Agreement Filter")
    print("=" * 70)

    sources = [silver_path] if args.no_hdtb else [hi_train, silver_path]
    label   = "filtered silver only" if args.no_hdtb else "HDTB + filtered silver"

    n_total = _concat_conllu(sources, SYSTEM_L_TRAIN)
    print(f"  Training corpus : {label}  ({n_total:,} sentences)")
    for p in sources:
        print(f"    - {p.name:45s}  {_count_sents(p):,} sents")
    print(f"  Combined file   : {SYSTEM_L_TRAIN}")

    shutil.copy2(hi_dev, SYSTEM_L_DEV)
    print(f"  Dev (UD-style)  : {SYSTEM_L_DEV}  ({_count_sents(SYSTEM_L_DEV):,} sents)")
    print(f"  Epochs          : {args.epochs}   batch_size: {args.batch_size}   gpu: {args.gpu}")
    print(f"  Warm-start K    : {not args.no_warm_start_k}")
    print(f"  Save dir        : {SYSTEM_L_SAVE_DIR}")
    print()

    save_dir = str(SYSTEM_L_SAVE_DIR)
    _ensure_xlmr_cache_symlink(save_dir, lang=SYSTEM_L_CATEGORY)

    from trankit import TPipeline
    trainer = TPipeline(training_config={
        "category":           SYSTEM_L_CATEGORY,
        "task":               "posdep",
        "save_dir":           save_dir,
        "train_conllu_fpath": str(SYSTEM_L_TRAIN),
        "dev_conllu_fpath":   str(SYSTEM_L_DEV),
        "max_epoch":          args.epochs,
        "batch_size":         args.batch_size,
        "gpu":                args.gpu,
        "embedding":          "xlm-roberta-base",
    })

    # ── Post-init warm-start from System K ────────────────────────────────────
    if not args.no_warm_start_k:
        sysk_mdl = (CHECKPT_DIR / "trankit_bho_sysk"
                    / "xlm-roberta-base" / "bhojpuri_sysk"
                    / "bhojpuri_sysk.tagger.mdl")
        print("\n[Step] Injecting System K warm-start weights …")
        _inject_warmstart(trainer, sysk_mdl, label="System K")

    print("Training …")
    trainer.train()

    model_path = Path(save_dir) / f"xlm-roberta-base/{SYSTEM_L_CATEGORY}/{SYSTEM_L_CATEGORY}.tagger.mdl"
    print(f"\nDone. Best checkpoint: {model_path}")
    print("\nNext: python3 evaluate_trankit.py --gpu --include_k --include_l")


if __name__ == "__main__":
    main()

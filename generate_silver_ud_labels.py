#!/usr/bin/env python3
# generate_silver_ud_labels.py
# System K — Step 1 of 2: Generate UD-style silver labels for Bhojpuri.
#
# MOTIVATION (Competition 2 — beat System A on BHTB):
#   System A (zero-shot Trankit Hindi on Bhojpuri) gets 34.84 LAS on BHTB because
#   HDTB and BHTB share the UD annotation schema. Systems F/G/H/I/J trained on
#   the professor's transferred data collapse on BHTB (~6 LAS) because prof's
#   annotation schema differs from UD. The root cause is ANNOTATION STYLE
#   MISMATCH, not model capacity.
#
#   To beat A on BHTB we need Bhojpuri supervision IN UD STYLE. We obtain it by
#   re-parsing the professor's 30K Bhojpuri token sequences with System A (which
#   produces UD-compatible trees) and discarding the professor's labels entirely.
#   The resulting "silver" treebank is UD-style Bhojpuri.
#
# PIPELINE:
#   1. Read prof's bhojpuri_matched_transferred.conllu (tokens only; drop labels).
#   2. Load System A (trankit_hindi checkpoint).
#   3. For each Bhojpuri sentence, predict heads + deprels with System A.
#   4. (Optional) filter by confidence / well-formedness.
#   5. Write bho_silver_ud.conllu.
#
# OUTPUT:
#   data_files/synthetic/bho_silver_ud.conllu  (training data for System K)
#
# Usage:
#   python3 generate_silver_ud_labels.py [--gpu] [--min_len 3] [--max_len 100]
#                                        [--limit N] [--require_single_root]

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Set offline mode BEFORE any trankit/transformers import to use local HF cache.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import shutil
from pathlib import Path
from typing import List

from config import DATA_DIR, CHECKPT_DIR
from utils.conllu_utils import read_conllu, write_conllu, Sentence, Token


# ─────────────────────────────────────────────────────────────────────────────
# Trankit HF cache symlink (mirrors evaluate_trankit.py)
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


def _load_adapters(pipeline, model_path: Path) -> int:
    """Load Trankit adapter checkpoint into TPipeline (matches evaluate_trankit.py)."""
    import torch
    state    = torch.load(str(model_path), map_location="cpu")
    adapters = state["adapters"]
    epoch    = state.get("epoch", "?")
    emb_sd = pipeline._embedding_layers.state_dict()
    tag_sd = pipeline._tagger.state_dict()
    for k, v in adapters.items():
        if k in emb_sd:
            emb_sd[k] = v
        if k in tag_sd:
            tag_sd[k] = v
    pipeline._embedding_layers.load_state_dict(emb_sd)
    pipeline._tagger.load_state_dict(tag_sd)
    pipeline._embedding_layers.eval()
    pipeline._tagger.eval()
    return epoch


# ─────────────────────────────────────────────────────────────────────────────
# Token stripping (keep FORM only, zero out head/deprel/upos)
# ─────────────────────────────────────────────────────────────────────────────
def strip_to_tokens(sent: Sentence) -> Sentence:
    stripped = Sentence(comments=[c for c in sent.comments if c.startswith("# sent_id")
                                  or c.startswith("# text")])
    for t in sent.tokens:
        stripped.tokens.append(Token(
            id     = t.id,
            form   = t.form,
            lemma  = "_",
            upos   = "_",
            xpos   = "_",
            feats  = "_",
            head   = 0,
            deprel = "_",
            deps   = "_",
            misc   = "_",
        ))
    return stripped


# ─────────────────────────────────────────────────────────────────────────────
# Quality filters for silver output
# ─────────────────────────────────────────────────────────────────────────────
def is_well_formed(sent: Sentence) -> bool:
    """Reject sentences with invalid heads or no root."""
    n = len(sent.tokens)
    if n == 0:
        return False
    roots = 0
    for t in sent.tokens:
        if t.head < 0 or t.head > n:
            return False
        if t.head == 0:
            roots += 1
    return roots >= 1  # at least one root


def single_root(sent: Sentence) -> bool:
    return sum(1 for t in sent.tokens if t.head == 0) == 1


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Generate silver UD labels for Bhojpuri using System A")
    ap.add_argument("--input",  type=str,
                    default="bhojpuri_matched_transferred.conllu",
                    help="Path to Bhojpuri CoNLL-U (only FORM column is read)")
    ap.add_argument("--output", type=str,
                    default=str(DATA_DIR / "synthetic" / "bho_silver_ud.conllu"))
    ap.add_argument("--teacher_dir", type=str,
                    default=str(CHECKPT_DIR / "trankit_hindi"),
                    help="Teacher checkpoint dir (System A or K)")
    ap.add_argument("--teacher_category", type=str, default="hindi",
                    help="Trankit category of teacher checkpoint (e.g., 'hindi' for A, 'bhojpuri_sysk' for K)")
    ap.add_argument("--teacher_train_conllu", type=str, default="",
                    help="CoNLL-U that was used to train the teacher (for vocab setup)")
    ap.add_argument("--min_len", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=100)
    ap.add_argument("--limit",   type=int, default=-1,
                    help="Max sentences to process (-1 = all)")
    ap.add_argument("--require_single_root", action="store_true",
                    help="Keep only silver sentences with exactly one root")
    ap.add_argument("--gpu", action="store_true", default=False)
    ap.add_argument("--batch_size", type=int, default=32)
    args = ap.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)
    teacher_dir      = args.teacher_dir
    teacher_category = args.teacher_category

    if not in_path.exists():
        raise FileNotFoundError(f"Input Bhojpuri CoNLL-U not found: {in_path}")

    teacher_ckpt = Path(teacher_dir) / f"xlm-roberta-base/{teacher_category}/{teacher_category}.tagger.mdl"
    if not teacher_ckpt.exists():
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {teacher_ckpt}\n"
            f"Train it first (System A: train_trankit_hindi.py; System K: train_system_k.py)"
        )

    # Resolve teacher's train conllu for Trankit vocab setup
    hi_train_default = DATA_DIR / "hindi" / "hi_hdtb-ud-train.conllu"
    if args.teacher_train_conllu:
        teacher_train = Path(args.teacher_train_conllu)
    elif teacher_category == "hindi":
        teacher_train = hi_train_default
    else:
        teacher_train = DATA_DIR / "sysk" / "sysk_train.conllu"

    if not teacher_train.exists():
        raise FileNotFoundError(
            f"Teacher train CoNLL-U not found: {teacher_train}\n"
            f"Pass it with --teacher_train_conllu"
        )

    print("=" * 70)
    print(" Generating UD-style silver labels for Bhojpuri")
    print("=" * 70)
    print(f"  Input   : {in_path}")
    print(f"  Output  : {out_path}")
    print(f"  Teacher : {teacher_ckpt}")
    print(f"  Category: {teacher_category}")
    print(f"  Length  : [{args.min_len}, {args.max_len}] tokens")
    print(f"  Single-root filter : {args.require_single_root}")
    print()

    # ── Read & strip annotations ──────────────────────────────────────────────
    print("[1/4] Reading Bhojpuri sentences …")
    raw_sents = read_conllu(in_path)
    print(f"      Read {len(raw_sents):,} sentences")

    kept: List[Sentence] = []
    for s in raw_sents:
        n = len(s.tokens)
        if n < args.min_len or n > args.max_len:
            continue
        kept.append(strip_to_tokens(s))
        if args.limit > 0 and len(kept) >= args.limit:
            break
    print(f"      Kept {len(kept):,} after length filter")

    # ── Write stripped input as a temp file for Trankit ───────────────────────
    tmp_in = Path(teacher_dir) / "silver_input.conllu"
    write_conllu(kept, tmp_in)
    print(f"      Wrote stripped input: {tmp_in}")

    # ── Load teacher via Trankit TPipeline ────────────────────────────────────
    print(f"\n[2/4] Loading teacher ({teacher_category}) …")
    _ensure_xlmr_cache_symlink(teacher_dir, lang=teacher_category)

    from trankit import TPipeline
    pipeline = TPipeline(training_config={
        "category":           teacher_category,
        "task":               "posdep",
        "save_dir":           teacher_dir,
        "train_conllu_fpath": str(teacher_train),
        "dev_conllu_fpath":   str(tmp_in),
        "max_epoch":          0,
        "batch_size":         args.batch_size,
        "gpu":                args.gpu,
        "embedding":          "xlm-roberta-base",
    })
    epoch = _load_adapters(pipeline, teacher_ckpt)
    print(f"      Teacher loaded (epoch {epoch})")

    # ── Run teacher inference to produce silver annotations ───────────────────
    print(f"\n[3/4] Parsing Bhojpuri with teacher …")
    try:
        _, pred_path = pipeline._eval_posdep(
            data_set  = pipeline.dev_set,
            batch_num = pipeline.dev_batch_num,
            name      = "silver",
            epoch     = epoch,
        )
    except Exception as e:
        print(f"      [ERROR] Inference failed: {e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    pred_sents = read_conllu(Path(pred_path))
    print(f"      Parsed {len(pred_sents):,} sentences")
    print(f"      Prediction file: {pred_path}")

    # ── Quality filter & write ────────────────────────────────────────────────
    print("\n[4/4] Filtering & writing silver treebank …")
    silver: List[Sentence] = []
    n_ill_formed = n_multi_root = 0
    for s in pred_sents:
        if not is_well_formed(s):
            n_ill_formed += 1
            continue
        if args.require_single_root and not single_root(s):
            n_multi_root += 1
            continue
        silver.append(s)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_conllu(silver, out_path)

    print(f"      Well-formed             : {len(silver):,}")
    print(f"      Dropped (ill-formed)    : {n_ill_formed:,}")
    if args.require_single_root:
        print(f"      Dropped (multi-root)    : {n_multi_root:,}")
    print(f"      Silver treebank written : {out_path}")

    # Summary for training step
    n_tokens = sum(len(s.tokens) for s in silver)
    n_roots  = sum(1 for s in silver for t in s.tokens if t.head == 0)
    print()
    print("-" * 70)
    print(f"  Silver corpus statistics")
    print(f"    Sentences     : {len(silver):,}")
    print(f"    Tokens        : {n_tokens:,}")
    print(f"    Avg length    : {n_tokens/max(len(silver),1):.1f}")
    print(f"    Root markers  : {n_roots:,}   (avg {n_roots/max(len(silver),1):.2f} per sent.)")
    print("-" * 70)
    print("\nNext step: python3 train_system_k.py --gpu")


if __name__ == "__main__":
    main()

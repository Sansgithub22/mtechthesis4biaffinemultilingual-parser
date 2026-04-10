#!/usr/bin/env python3
# evaluate_trankit.py
# Step 8 — Evaluation using Trankit-based parsers.
#
# Evaluates systems on the real Bhojpuri BHTB test set (357 sentences):
#
#   System A — Zero-shot:    trained Hindi Trankit model applied directly to
#               Bhojpuri text (no Bhojpuri training data at all)
#
#   System B — Projection:   Bhojpuri Trankit trained on unfiltered 5,000-
#               sentence projected synthetic treebank
#
#   System C — Filtered:     Bhojpuri Trankit trained on quality-filtered
#               projected treebank (alignment coverage ≥ 70%)
#
#   System D — Two-stage + Selective (NOVEL):
#               Innovation 1: warm-started from Hindi checkpoint (not raw XLM-R)
#               Innovation 3: relation-selective projection — HIGH_CONF deprels
#               keep projected annotation; LOW_CONF deprels replaced by
#               System A predictions (cleaner signal for complex relations)
#
#   System F — High-Quality Fine-tuning (NOVEL):
#               Warm-started from Hindi checkpoint + fine-tuned on professor's
#               matched Bhojpuri data (bhojpuri_matched_transferred.conllu).
#               30,966 real Bhojpuri sentences with expert-transferred annotations
#               — no machine translation noise, no SimAlign approximation errors.
#
# Reports: UAS, LAS, per-relation LAS, delta tables
#
# Usage:
#   python3 evaluate_trankit.py [--gpu] [--skip_d] [--skip_f]

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

# Set offline mode BEFORE any trankit/transformers import to use local HF cache.
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
from pathlib import Path
from typing import List, Dict, Tuple

import torch

from config import DATA_DIR, CHECKPT_DIR
from utils.conllu_utils import read_conllu, write_conllu, Sentence
from utils.metrics       import uas_las, print_metrics


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


# ─────────────────────────────────────────────────────────────────────────────
def _load_adapters(pipeline, model_path: Path):
    """
    Load Trankit checkpoint (keys: 'adapters', 'epoch') into a TPipeline.
    The 'adapters' dict contains weights for both _embedding_layers and _tagger.
    We merge by key name into each module's state_dict.
    """
    state    = torch.load(str(model_path), map_location="cpu")
    adapters = state["adapters"]
    epoch    = state.get("epoch", "?")
    print(f"  Checkpoint epoch: {epoch}")

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


def eval_system(
    lang:        str,
    save_dir:    str,
    train_conllu: Path,
    test_sents:  List[Sentence],
    gpu:         bool,
    label:       str,
) -> Tuple[float, float, Dict[str, Tuple[int,int]]]:
    """
    Evaluate one trained system on test_sents.
    Returns (uas, las, per_rel_counts).
    train_conllu must be the same file used during training so that
    TPipeline rebuilds the correct label vocabulary before loading the checkpoint.
    """
    model_path = Path(save_dir) / f"xlm-roberta-base/{lang}/{lang}.tagger.mdl"
    if not model_path.exists():
        print(f"  [MISSING] {label} — checkpoint not found: {model_path}")
        return 0.0, 0.0, {}

    print(f"\n{'─'*60}")
    print(f"  [{label}]  Loading {lang} …")

    from trankit import TPipeline

    # Write test sentences to temp file (used as dev_conllu_fpath for vocab setup)
    tmp_test = Path(save_dir) / "test_input.conllu"
    write_conllu(test_sents, tmp_test)

    # Symlink HF hub cache into Trankit's _save_dir for offline resolution
    _ensure_xlmr_cache_symlink(save_dir, lang=lang)

    pipeline = TPipeline(training_config={
        "category":           lang,
        "task":               "posdep",
        "save_dir":           save_dir,
        "train_conllu_fpath": str(train_conllu),
        "dev_conllu_fpath":   str(tmp_test),
        "max_epoch":          0,
        "batch_size":         32,
        "gpu":                gpu,
        "embedding":          "xlm-roberta-base",
    })

    # Load adapter-based checkpoint (keys: 'adapters', 'epoch')
    epoch = _load_adapters(pipeline, model_path)

    # Evaluate using dev_set built from test file
    try:
        score, pred_path = pipeline._eval_posdep(
            data_set  = pipeline.dev_set,
            batch_num = pipeline.dev_batch_num,
            name      = "test",
            epoch     = epoch,
        )
        pred_sents = read_conllu(Path(pred_path))
    except Exception as e:
        print(f"  [ERROR] Evaluation failed: {e}")
        import traceback; traceback.print_exc()
        return 0.0, 0.0, {}

    # Align pred_sents to test_sents
    pred_heads_all = [[t.head for t in s.tokens] for s in pred_sents]
    pred_rels_all  = [[t.deprel for t in s.tokens] for s in pred_sents]

    gold_sents_aligned = test_sents[:len(pred_sents)]
    uas, las = uas_las(gold_sents_aligned, pred_heads_all, pred_rels_all)

    # Per-relation counts
    per_rel: Dict[str, List[int]] = {}
    for gold_s, pred_h, pred_r in zip(gold_sents_aligned, pred_heads_all, pred_rels_all):
        for i, tok in enumerate(gold_s.tokens):
            rel = tok.deprel
            if rel not in per_rel:
                per_rel[rel] = [0, 0]
            per_rel[rel][1] += 1
            if i < len(pred_h) and pred_h[i] == tok.head and pred_r[i] == rel:
                per_rel[rel][0] += 1

    print(f"  [{label}]  UAS: {uas*100:.2f}%   LAS: {las*100:.2f}%")
    return uas, las, per_rel


# ─────────────────────────────────────────────────────────────────────────────
# Per-relation report
# ─────────────────────────────────────────────────────────────────────────────
def print_per_rel(per_rel: Dict[str, List[int]], top_n: int = 20):
    print(f"\n  {'Relation':<20} {'Correct':>8} {'Total':>8} {'LAS':>8}")
    print(f"  {'─'*50}")
    rows = sorted(per_rel.items(), key=lambda x: -x[1][1])[:top_n]
    for rel, (correct, total) in rows:
        las = correct / total * 100 if total else 0
        print(f"  {rel:<20} {correct:>8,} {total:>8,} {las:>7.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Step 8 — Evaluate Trankit-based parsers")
    ap.add_argument("--gpu",    action="store_true", default=False)
    ap.add_argument("--skip_d", action="store_true", default=False,
                    help="Skip System D (run A/B/C only)")
    ap.add_argument("--skip_f", action="store_true", default=False,
                    help="Skip System F (professor's matched data)")
    args = ap.parse_args()

    sysf_train = Path(__file__).parent / "data_files/sysf/bho_sysf_train.conllu"
    bhtb_test  = DATA_DIR / "bhojpuri/bho_bhtb-ud-test.conllu"
    if not bhtb_test.exists():
        print("Bhojpuri test data missing. Run: python3 data/download_ud_data.py")
        return

    print("\n========================================")
    print(" Step 8 — Trankit Evaluation")
    print(" Cross-lingual Hindi → Bhojpuri")
    print("========================================")
    print(f"  Systems: A, B, C" +
          ("" if args.skip_d else ", D") +
          ("" if args.skip_f else ", F"))

    test_sents = read_conllu(bhtb_test)
    print(f"\n  Test set: {len(test_sents):,} real Bhojpuri sentences (BHTB)")

    results = {}

    syn_dir  = DATA_DIR / "synthetic"
    hi_train = DATA_DIR / "hindi" / "hi_hdtb-ud-train.conllu"

    # ── System A: Zero-shot (Hindi model on Bhojpuri text) ───────────────────
    uas_a, las_a, per_rel_a = eval_system(
        lang         = "hindi",
        save_dir     = str(CHECKPT_DIR / "trankit_hindi"),
        train_conllu = hi_train,
        test_sents   = test_sents,
        gpu          = args.gpu,
        label        = "A: Zero-shot (Hindi → Bhojpuri)",
    )
    results["A_zero_shot"] = (uas_a, las_a)

    # ── System B: Projection-only (unfiltered) ────────────────────────────────
    uas_b, las_b, per_rel_b = eval_system(
        lang         = "bhojpuri_proj",
        save_dir     = str(CHECKPT_DIR / "trankit_bho_proj"),
        train_conllu = syn_dir / "bho_synthetic_train.conllu",
        test_sents   = test_sents,
        gpu          = args.gpu,
        label        = "B: Projection-only (5,000 unfiltered)",
    )
    results["B_projection"] = (uas_b, las_b)

    # ── System C: Quality-filtered projection ─────────────────────────────────
    uas_c, las_c, per_rel_c = eval_system(
        lang         = "bhojpuri_filtered",
        save_dir     = str(CHECKPT_DIR / "trankit_bho_filtered"),
        train_conllu = syn_dir / "bho_filtered_train.conllu",
        test_sents   = test_sents,
        gpu          = args.gpu,
        label        = "C: Quality-filtered (coverage ≥ 70%)",
    )
    results["C_filtered"] = (uas_c, las_c)

    # ── System D: Two-stage warm-start + relation-selective projection ─────────
    uas_d = las_d = 0.0
    per_rel_d: Dict = {}
    if not args.skip_d:
        uas_d, las_d, per_rel_d = eval_system(
            lang         = "bhojpuri_warmstart",
            save_dir     = str(CHECKPT_DIR / "trankit_bho_warmstart"),
            train_conllu = syn_dir / "bho_selective_train.conllu",
            test_sents   = test_sents,
            gpu          = args.gpu,
            label        = "D: Two-stage warm-start + selective projection",
        )
    results["D_warmstart"] = (uas_d, las_d)

    # ── System F: High-quality fine-tuning on professor's matched data ─────────
    uas_f = las_f = 0.0
    per_rel_f: Dict = {}
    if not args.skip_f:
        uas_f, las_f, per_rel_f = eval_system(
            lang         = "bhojpuri_sysf",
            save_dir     = str(CHECKPT_DIR / "trankit_bho_sysf"),
            train_conllu = sysf_train,
            test_sents   = test_sents,
            gpu          = args.gpu,
            label        = "F: High-quality fine-tuning (professor's matched data)",
        )
    results["F_hq_finetune"] = (uas_f, las_f)

    # ── Per-relation breakdown for best system ────────────────────────────────
    all_results = {"A": (las_a, per_rel_a, "A (zero-shot)"),
                   "B": (las_b, per_rel_b, "B (projection)"),
                   "C": (las_c, per_rel_c, "C (filtered)"),
                   "D": (las_d, per_rel_d, "D (warm-start+selective)"),
                   "F": (las_f, per_rel_f, "F (high-quality fine-tuning)")}
    best_key   = max(all_results, key=lambda k: all_results[k][0])
    best_las, best_rel, best_label = all_results[best_key]

    print(f"\n{'─'*60}")
    print(f"  Per-relation LAS — {best_label}")
    print_per_rel(best_rel, top_n=20)

    # ── Summary table ─────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  STEP 8 — FINAL RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"  {'System':<48} {'UAS':>7} {'LAS':>7}")
    print(f"  {'─'*64}")
    print(f"  {'[A] Zero-shot (Hindi Trankit → Bhojpuri)':<48} {uas_a*100:>6.2f}% {las_a*100:>6.2f}%  ← baseline")
    print(f"  {'[B] Projection-only (5,000 unfiltered)':<48} {uas_b*100:>6.2f}% {las_b*100:>6.2f}%")
    print(f"  {'[C] Quality-filtered (coverage ≥ 70%)':<48} {uas_c*100:>6.2f}% {las_c*100:>6.2f}%")
    if not args.skip_d:
        print(f"  {'[D] Two-stage warm-start + selective proj.':<48} {uas_d*100:>6.2f}% {las_d*100:>6.2f}%")
    if not args.skip_f:
        print(f"  {'[F] High-quality fine-tuning (prof. data)':<48} {uas_f*100:>6.2f}% {las_f*100:>6.2f}%  ← NOVEL")
    print(f"  {'─'*64}")

    print(f"\n  Gains over zero-shot baseline (System A):")
    if not args.skip_d:
        print(f"    D vs A: ΔUAS = {(uas_d - uas_a)*100:+.2f}%   ΔLAS = {(las_d - las_a)*100:+.2f}%")
    if not args.skip_f:
        print(f"    F vs A: ΔUAS = {(uas_f - uas_a)*100:+.2f}%   ΔLAS = {(las_f - las_a)*100:+.2f}%  ← target: positive")
    print(f"\n  Baseline deltas:")
    print(f"    B vs A (projection over zero-shot):        ΔLAS = {(las_b - las_a)*100:+.2f}%")
    print(f"    C vs B (coverage filter over projection):  ΔLAS = {(las_c - las_b)*100:+.2f}%")
    print(f"{'='*70}")

    all_uas = [uas_a, uas_b, uas_c]
    all_las = [las_a, las_b, las_c]
    if not args.skip_d:
        all_uas.append(uas_d); all_las.append(las_d)
    if not args.skip_f:
        all_uas.append(uas_f); all_las.append(las_f)
    best_uas_val = max(all_uas)
    best_las_val = max(all_las)
    target_met   = best_las_val * 100 >= 35.0
    print(f"\n  Target range: UAS 45-55%, LAS 35-45%")
    print(f"  Best UAS: {best_uas_val*100:.2f}%   Best LAS: {best_las_val*100:.2f}%  [{best_label}]")
    if target_met:
        print(f"  Target MET — best system achieves ≥35% LAS")
    else:
        print(f"  Note: LAS {best_las_val*100:.2f}% — target not yet met")


if __name__ == "__main__":
    main()

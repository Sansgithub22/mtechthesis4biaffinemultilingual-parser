#!/usr/bin/env python3
# data/build_treebank_filtered.py
# Step 3 (improved) — Quality-filtered synthetic Bhojpuri treebank.
#
# Improvements over build_synthetic_treebank.py:
#   1. Alignment coverage threshold (default ≥ 70%) — removes noisy projections
#   2. Head-validity check — every projected head must be in range [0, n_tokens]
#   3. Guaranteed single ROOT — sentences without a valid ROOT are repaired
#   4. UD label consistency — invalid labels mapped to nearest valid UD label
#   5. Reports filtering statistics (how many sentences were kept / dropped)
#
# Usage:
#   python3 data/build_treebank_filtered.py [--coverage 0.70] [--max_sents 5000]

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
from pathlib import Path
from typing import List, Tuple, Set

from utils.conllu_utils import read_conllu, write_conllu, Sentence, Token
from data.word_alignment  import SimAligner, str_to_alignment
from data.translate_hindi import translate_dict
from data.project_annotations import project_sentence, LABEL_CORRECTION


# ─────────────────────────────────────────────────────────────────────────────
# Quality metrics
# ─────────────────────────────────────────────────────────────────────────────
def alignment_coverage(alignment: Set[Tuple[int,int]],
                       n_src: int, n_tgt: int) -> float:
    """
    Fraction of target tokens that have at least one alignment link.
    Uses target (Bhojpuri) coverage as the quality signal.
    """
    if n_tgt == 0:
        return 0.0
    tgt_covered = len(set(t for _, t in alignment))
    return tgt_covered / n_tgt


def heads_valid(sent: Sentence) -> bool:
    """All head indices must be in [0, n_tokens]."""
    n = len(sent.tokens)
    for tok in sent.tokens:
        if tok.head < 0 or tok.head > n:
            return False
    return True


def has_single_root(sent: Sentence) -> bool:
    """Exactly one token must have head == 0."""
    roots = [t for t in sent.tokens if t.head == 0]
    return len(roots) == 1


def repair_root(sent: Sentence) -> Sentence:
    """
    If there is no root or multiple roots, force the first token with
    head=0 (or token 1 if none) to be root and re-attach others to it.
    """
    roots = [t for t in sent.tokens if t.head == 0]
    if len(roots) == 1:
        return sent
    # No root: attach token 1 to ROOT
    if len(roots) == 0:
        sent.tokens[0].head   = 0
        sent.tokens[0].deprel = "root"
        return sent
    # Multiple roots: keep only first, re-attach the rest to token 1
    first_root_id = roots[0].id
    for tok in sent.tokens:
        if tok.head == 0 and tok.id != first_root_id:
            tok.head   = first_root_id
            tok.deprel = "dep"
    return sent


# ─────────────────────────────────────────────────────────────────────────────
# Projection with quality filtering
# ─────────────────────────────────────────────────────────────────────────────
def project_and_filter(
    src_sent:      Sentence,
    tgt_words:     List[str],
    alignment:     Set[Tuple[int,int]],
    coverage_thr:  float,
) -> Tuple[Sentence | None, str]:
    """
    Project Hindi annotations to Bhojpuri and apply quality filters.

    Returns (projected_sentence, reject_reason).
    reject_reason is '' if accepted, otherwise a short description.
    """
    n_src = len(src_sent.tokens)
    n_tgt = len(tgt_words)

    # ── Filter 1: minimum sentence length ────────────────────────────────────
    if n_src < 2 or n_tgt < 2:
        return None, "too_short"

    # ── Filter 2: alignment coverage ─────────────────────────────────────────
    cov = alignment_coverage(alignment, n_src, n_tgt)
    if cov < coverage_thr:
        return None, f"low_coverage({cov:.2f})"

    # ── Project ───────────────────────────────────────────────────────────────
    projected = project_sentence(src_sent, tgt_words, alignment)

    # ── Filter 3: head validity ───────────────────────────────────────────────
    if not heads_valid(projected):
        return None, "invalid_heads"

    # ── Repair / filter 4: root ───────────────────────────────────────────────
    projected = repair_root(projected)
    if not has_single_root(projected):
        return None, "no_root"

    return projected, ""


# ─────────────────────────────────────────────────────────────────────────────
# Main builder
# ─────────────────────────────────────────────────────────────────────────────
def build_filtered_split(
    hi_sents:     List[Sentence],
    split_name:   str,
    out_conllu:   Path,
    out_align:    Path,
    coverage_thr: float,
    max_sents:    int,
    aligner:      SimAligner,
):
    """
    Translate, align, project + filter one split.
    Writes filtered CoNLL-U and corresponding alignment file.
    """
    src_sents = hi_sents[:max_sents] if max_sents else hi_sents
    n_total   = len(src_sents)

    print(f"\n{'─'*60}")
    print(f"  Building quality-filtered treebank — {split_name} split")
    print(f"  Input: {n_total:,} Hindi sentences   coverage_thr={coverage_thr:.0%}")
    print(f"{'─'*60}")

    # ── Step 1: translate ──────────────────────────────────────────────────────
    print(f"[2a] Translating Hindi → Bhojpuri …")
    pairs = []  # (src_sent, bho_words)
    for sent in src_sents:
        hi_words  = [t.form for t in sent.tokens]
        bho_text  = translate_dict(" ".join(hi_words))
        bho_words = bho_text.split()
        pairs.append((sent, bho_words))

    # ── Step 2: align ──────────────────────────────────────────────────────────
    print(f"[2b] Aligning with SimAlign (coverage ≥ {coverage_thr:.0%}) …")
    alignments = []
    for i, (src_sent, bho_words) in enumerate(pairs):
        hi_words = [t.form for t in src_sent.tokens]
        try:
            aligns = aligner.align(hi_words, bho_words)
        except Exception:
            aligns = {(j, j) for j in range(min(len(hi_words), len(bho_words)))}
        alignments.append(aligns)
        if (i + 1) % 500 == 0:
            print(f"  Aligned {i+1:,}/{n_total:,} sentence pairs …")

    # ── Step 3: project + filter ───────────────────────────────────────────────
    print(f"[2c] Projecting annotations + quality filtering …")
    kept, rejected = [], []
    reject_reasons: dict = {}
    kept_alignments = []

    for i, ((src_sent, bho_words), alignment) in enumerate(zip(pairs, alignments)):
        proj, reason = project_and_filter(src_sent, bho_words, alignment, coverage_thr)
        if proj is not None:
            kept.append(proj)
            kept_alignments.append(alignment)
        else:
            rejected.append(reason)
            reject_reasons[reason] = reject_reasons.get(reason, 0) + 1
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i+1:,}/{n_total:,} …  kept={len(kept):,}")

    # ── Write outputs ─────────────────────────────────────────────────────────
    write_conllu(kept, out_conllu)

    # Write alignment file (one Pharaoh line per kept sentence)
    from data.word_alignment import alignment_to_str
    out_align.parent.mkdir(parents=True, exist_ok=True)
    with open(out_align, "w", encoding="utf-8") as fh:
        for aligns in kept_alignments:
            fh.write(alignment_to_str(aligns) + "\n")

    # ── Statistics ────────────────────────────────────────────────────────────
    n_tokens = sum(len(s.tokens) for s in kept)
    keep_rate = len(kept) / n_total * 100
    print(f"\n  Results — {split_name}:")
    print(f"    Input : {n_total:,} sentences")
    print(f"    Kept  : {len(kept):,} sentences ({keep_rate:.1f}%),  {n_tokens:,} tokens")
    print(f"    Dropped: {len(rejected):,} sentences")
    for reason, cnt in sorted(reject_reasons.items(), key=lambda x: -x[1]):
        print(f"      {reason}: {cnt:,}")
    print(f"    Output: {out_conllu}")

    return kept


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Build quality-filtered synthetic Bhojpuri treebank"
    )
    ap.add_argument("--coverage",   type=float, default=0.70,
                    help="Minimum alignment coverage (default: 0.70)")
    ap.add_argument("--max_sents",  type=int,   default=5000,
                    help="Max training sentences (0 = all)")
    ap.add_argument("--device",     default="cpu")
    args = ap.parse_args()

    from config import DATA_DIR
    DATA_DIR.joinpath("synthetic").mkdir(parents=True, exist_ok=True)

    # Load Hindi data
    hi_train_path = DATA_DIR / "hindi/hi_hdtb-ud-train.conllu"
    hi_dev_path   = DATA_DIR / "hindi/hi_hdtb-ud-dev.conllu"
    if not hi_train_path.exists():
        print("Hindi data missing. Run: python3 data/download_ud_data.py")
        return

    print("Loading Hindi treebanks …")
    hi_train = read_conllu(hi_train_path)
    hi_dev   = read_conllu(hi_dev_path)
    print(f"  Hindi train: {len(hi_train):,}   dev: {len(hi_dev):,}")

    # Initialise aligner
    print("\nInitialising SimAlign …")
    aligner = SimAligner()

    # Build filtered splits
    build_filtered_split(
        hi_sents     = hi_train,
        split_name   = "train",
        out_conllu   = DATA_DIR / "synthetic/bho_filtered_train.conllu",
        out_align    = DATA_DIR / "synthetic/alignments_filtered_train.txt",
        coverage_thr = args.coverage,
        max_sents    = args.max_sents,
        aligner      = aligner,
    )
    build_filtered_split(
        hi_sents     = hi_dev,
        split_name   = "dev",
        out_conllu   = DATA_DIR / "synthetic/bho_filtered_dev.conllu",
        out_align    = DATA_DIR / "synthetic/alignments_filtered_dev.txt",
        coverage_thr = args.coverage,
        max_sents    = 0,
        aligner      = aligner,
    )

    print("\nQuality-filtered treebank complete.")
    print("Next: python3 train_trankit_hindi.py")


if __name__ == "__main__":
    main()

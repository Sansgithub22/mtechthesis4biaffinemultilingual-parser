#!/usr/bin/env python3
# compare_silver_labels.py
# System L — agreement-based confidence filter for silver UD-Bhojpuri.
#
# MOTIVATION:
#   System K uses silver labels from System A. System L uses silver labels from
#   System K itself (iterative self-training). But silver labels are noisy —
#   some fraction are simply wrong. Which ones can we trust?
#
# INSIGHT:
#   When TWO different teachers (System A and System K) independently assign
#   the SAME head to the same token, that arc is far more likely to be correct.
#   This "teacher agreement" is a cheap but strong confidence signal
#   (McClosky et al. 2006; Blum & Mitchell co-training).
#
#   We keep sentences where head-level agreement ≥ threshold (default 0.80).
#   Labels (deprels) are taken from silver-v2 (System K, the better teacher).
#
# USAGE:
#   python3 compare_silver_labels.py \
#       --silver_v1 data_files/synthetic/bho_silver_ud.conllu      (from System A)
#       --silver_v2 data_files/synthetic/bho_silver_ud_v2.conllu   (from System K)
#       --output    data_files/synthetic/bho_silver_ud_filtered.conllu
#       --min_agreement 0.80

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
from pathlib import Path
from typing import List

from utils.conllu_utils import read_conllu, write_conllu, Sentence


def head_agreement(s1: Sentence, s2: Sentence) -> float:
    """Fraction of non-punct tokens where the two sentences assign the same head.
    Returns 0.0 if lengths differ (should not normally happen)."""
    if len(s1.tokens) != len(s2.tokens):
        return 0.0
    total = agree = 0
    for t1, t2 in zip(s1.tokens, s2.tokens):
        if t1.upos in {"PUNCT", "SYM"}:
            continue
        total += 1
        if t1.head == t2.head:
            agree += 1
    return agree / total if total else 0.0


def label_agreement(s1: Sentence, s2: Sentence) -> float:
    """Fraction of tokens where (head, deprel) both match."""
    if len(s1.tokens) != len(s2.tokens):
        return 0.0
    total = agree = 0
    for t1, t2 in zip(s1.tokens, s2.tokens):
        if t1.upos in {"PUNCT", "SYM"}:
            continue
        total += 1
        if t1.head == t2.head and t1.deprel == t2.deprel:
            agree += 1
    return agree / total if total else 0.0


def main():
    ap = argparse.ArgumentParser(description="Agreement filter for silver UD-Bhojpuri")
    ap.add_argument("--silver_v1", required=True, help="Silver from teacher 1 (System A)")
    ap.add_argument("--silver_v2", required=True, help="Silver from teacher 2 (System K)")
    ap.add_argument("--output",    required=True, help="Filtered silver corpus path")
    ap.add_argument("--min_agreement", type=float, default=0.80,
                    help="Per-sentence head-agreement threshold (default 0.80)")
    ap.add_argument("--also_require_label_agreement", type=float, default=0.0,
                    help="Additionally require LAS-style agreement ≥ this (default 0.0 = off)")
    ap.add_argument("--use_labels_from", choices=["v1", "v2"], default="v2",
                    help="Whose deprels to keep in the filtered corpus (default v2 = System K)")
    args = ap.parse_args()

    v1 = read_conllu(args.silver_v1)
    v2 = read_conllu(args.silver_v2)
    if len(v1) != len(v2):
        print(f"  [WARN] length mismatch: v1={len(v1)}, v2={len(v2)} — aligning by position up to min")

    n = min(len(v1), len(v2))

    print("=" * 70)
    print(" Silver label agreement filter (System L)")
    print("=" * 70)
    print(f"  v1 (System A) : {args.silver_v1}  ({len(v1):,} sents)")
    print(f"  v2 (System K) : {args.silver_v2}  ({len(v2):,} sents)")
    print(f"  Output        : {args.output}")
    print(f"  Min UAS-agree : {args.min_agreement:.2f}")
    if args.also_require_label_agreement > 0:
        print(f"  Min LAS-agree : {args.also_require_label_agreement:.2f}")
    print(f"  Keep labels   : from silver-{args.use_labels_from}")
    print()

    kept: List[Sentence] = []
    agreements: List[float] = []
    label_agreements: List[float] = []
    n_len_mismatch = 0

    for i in range(n):
        s1, s2 = v1[i], v2[i]
        if len(s1.tokens) != len(s2.tokens):
            n_len_mismatch += 1
            continue
        ha = head_agreement(s1, s2)
        la = label_agreement(s1, s2)
        agreements.append(ha)
        label_agreements.append(la)
        if ha < args.min_agreement:
            continue
        if args.also_require_label_agreement > 0 and la < args.also_require_label_agreement:
            continue
        kept.append(s2 if args.use_labels_from == "v2" else s1)

    write_conllu(kept, args.output)

    mean_ha = sum(agreements) / max(len(agreements), 1)
    mean_la = sum(label_agreements) / max(len(label_agreements), 1)

    print(f"  Length mismatch dropped : {n_len_mismatch:,}")
    print(f"  Mean head  agreement    : {mean_ha*100:.2f}%")
    print(f"  Mean label agreement    : {mean_la*100:.2f}%")
    print(f"  Kept (agree ≥ {args.min_agreement:.2f}) : {len(kept):,} / {n:,} ({len(kept)/max(n,1)*100:.1f}%)")
    print(f"  Wrote → {args.output}")


if __name__ == "__main__":
    main()

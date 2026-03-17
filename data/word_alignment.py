#!/usr/bin/env python3
# data/word_alignment.py
# Step 2b: Word alignment between Hindi and Bhojpuri using SimAlign.
#
# SimAlign (Jalili Sabet et al., 2020) aligns words across language pairs using
# multilingual embeddings (XLM-RoBERTa) without any parallel training data.
#
# For each Hindi-Bhojpuri sentence pair it returns a set of (hi_idx, bho_idx)
# index pairs.  We use the "inter" (intersection) method for high-precision
# alignments, which is better for annotation projection than recall-oriented
# methods.
#
# Usage:
#   python3 data/word_alignment.py \
#       --translations data_files/synthetic/translations_train.txt \
#       --output       data_files/synthetic/alignments_train.txt

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
from pathlib import Path
from typing import List, Set, Tuple

AlignSet = Set[Tuple[int, int]]   # set of (src_idx, tgt_idx) 0-based pairs


# ─────────────────────────────────────────────────────────────────────────────
# SimAlign wrapper
# ─────────────────────────────────────────────────────────────────────────────
class SimAligner:
    """
    Thin wrapper around the SentenceAligner from simalign.
    Falls back to a simple cosine-similarity aligner if simalign is unavailable.
    """

    def __init__(self, method: str = "inter"):
        self.method = method
        self._aligner = None
        try:
            from simalign import SentenceAligner
            # model="xlmr" uses XLM-RoBERTa multilingual embeddings
            # token_type="bpe" — operate at subword level, aggregate to word
            # matching_methods="iag" → itermax (i), argmax (a), match (g)
            self._aligner = SentenceAligner(
                model="xlmr",
                token_type="bpe",
                matching_methods="mai",  # mwmf (m), argmax (a), itermax (i)
            )
            print("  [SimAlign] Using XLM-RoBERTa based aligner.")
        except Exception as e:
            print(f"  [SimAlign] WARNING: {e}\n"
                  f"  Falling back to identity alignment (word-order-based).")

    def align(self, src_words: List[str], tgt_words: List[str]) -> AlignSet:
        """
        Returns a set of 0-based (src_idx, tgt_idx) pairs.
        """
        if self._aligner is None:
            return self._identity_align(src_words, tgt_words)

        try:
            result = self._aligner.get_word_aligns(src_words, tgt_words)
            # simalign returns keys: "mwmf", "inter", "itermax"
            # prefer "inter" (intersection = high precision), then first available
            key = self.method if self.method in result else \
                  ("inter" if "inter" in result else next(iter(result)))
            return set(result[key])
        except Exception:
            return self._identity_align(src_words, tgt_words)

    @staticmethod
    def _identity_align(src: List[str], tgt: List[str]) -> AlignSet:
        """
        Fallback: align word i in source to word i in target (monotone).
        Works reasonably well for Hindi-Bhojpuri since word order is nearly identical.
        """
        return {(i, i) for i in range(min(len(src), len(tgt)))}


# ─────────────────────────────────────────────────────────────────────────────
# Alignment serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────
def alignment_to_str(align: AlignSet) -> str:
    """Encode alignment as Pharaoh format: '0-0 1-1 2-3 …'"""
    return " ".join(f"{s}-{t}" for s, t in sorted(align))


def str_to_alignment(s: str) -> AlignSet:
    """Decode Pharaoh format string to set of (src, tgt) int pairs."""
    if not s.strip():
        return set()
    pairs = set()
    for item in s.strip().split():
        a, b = item.split("-")
        pairs.add((int(a), int(b)))
    return pairs


# ─────────────────────────────────────────────────────────────────────────────
# Alignment helper structures
# ─────────────────────────────────────────────────────────────────────────────
def src_to_tgt_map(align: AlignSet) -> dict:
    """Map each source index to the first aligned target index (or None)."""
    m = {}
    for s, t in sorted(align):
        if s not in m:
            m[s] = t
    return m


def tgt_to_src_map(align: AlignSet) -> dict:
    """Map each target index to the first aligned source index (or None)."""
    m = {}
    for s, t in sorted(align):
        if t not in m:
            m[t] = s
    return m


# ─────────────────────────────────────────────────────────────────────────────
# Main: align a translations file produced by translate_hindi.py
# ─────────────────────────────────────────────────────────────────────────────
def align_translations(
    translations_path: str | Path,
    output_path:       str | Path,
    method:            str = "inter",
) -> None:
    """
    Input:  TSV file  src_sentence \\t tgt_sentence
    Output: one Pharaoh alignment string per line.
    """
    aligner = SimAligner(method=method)

    lines   = Path(translations_path).read_text(encoding="utf-8").splitlines()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as out:
        for i, line in enumerate(lines, 1):
            if "\t" not in line:
                out.write("\n")
                continue
            src_str, tgt_str = line.split("\t", 1)
            src_words = src_str.split()
            tgt_words = tgt_str.split()
            align = aligner.align(src_words, tgt_words)
            out.write(alignment_to_str(align) + "\n")

            if i % 500 == 0:
                print(f"  Aligned {i:,}/{len(lines):,} sentence pairs …")

    print(f"  Alignments written → {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--translations", required=True)
    ap.add_argument("--output",       required=True)
    ap.add_argument("--method",       default="inter",
                    choices=["inter", "argmax", "itermax"])
    args = ap.parse_args()
    align_translations(args.translations, args.output, args.method)

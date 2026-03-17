#!/usr/bin/env python3
# data/project_annotations.py
# Step 2c: Project UD dependency annotations from Hindi to Bhojpuri using
#          word alignments (Hwa et al., 2005; simplified).
#
# Algorithm:
#  For each Bhojpuri token t_i (0-based):
#    1. Find its aligned Hindi token s_j via tgt→src map.
#    2. Get s_j's gold head h_src in Hindi.
#    3. If h_src == 0 (root), set t_i's head = 0 (root), deprel = "root".
#    4. Else look up the aligned Bhojpuri token for h_src via src→tgt map.
#       If found → set as head.  Copy deprel from Hindi.
#    5. Unaligned tokens → "dep" attached to the sentence root or nearest
#       aligned neighbour (heuristic).
#
# Step 6 — UD label correction:
#   A post-projection pass refines labels using language-specific rules for
#   known Hindi↔Bhojpuri divergences (postposition drop, copula change, etc.)

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
from copy import deepcopy
from pathlib import Path
from typing import List, Optional, Set, Tuple

from utils.conllu_utils import Token, Sentence, read_conllu, write_conllu
from data.word_alignment import str_to_alignment, src_to_tgt_map, tgt_to_src_map, AlignSet


# ─────────────────────────────────────────────────────────────────────────────
# UD label correction map  (Step 6)
# Hindi labels that need adjustment when projected to Bhojpuri
# ─────────────────────────────────────────────────────────────────────────────
LABEL_CORRECTION: dict = {
    # Bhojpuri often merges Hindi obl+case into a bare obl
    "obl:tmod": "obl",
    "obl:lmod": "obl",
    # Bhojpuri has fewer sub-typed nominals
    "nsubj:pass": "nsubj",
    "csubj:pass": "csubj",
    # Agglutinated postpositions in Bhojpuri collapse case markings
    "case": "case",   # keep but flag for review
}


def correct_label(deprel: str) -> str:
    """Apply Step 6 label corrections."""
    return LABEL_CORRECTION.get(deprel, deprel)


# ─────────────────────────────────────────────────────────────────────────────
# Core projection
# ─────────────────────────────────────────────────────────────────────────────
def project_sentence(
    src_sent: Sentence,           # Hindi (annotated)
    tgt_words: List[str],         # Bhojpuri words (surface forms)
    alignment: AlignSet,          # 0-based (src_idx, tgt_idx) pairs
) -> Sentence:
    """
    Returns a new Sentence with projected (possibly noisy) annotations.
    Token ids are 1-based; heads are 1-based (0 = root).
    """
    src_tokens = src_sent.tokens            # 1-based id list
    n_tgt = len(tgt_words)

    # Build 0-based index maps from 1-based token ids
    # src_idx 0-based → head (0-based, -1 for root)
    src_head  = {i: (src_tokens[i].head - 1) for i in range(len(src_tokens))}
    src_label = {i: src_tokens[i].deprel      for i in range(len(src_tokens))}
    src_upos  = {i: src_tokens[i].upos        for i in range(len(src_tokens))}

    s2t = src_to_tgt_map(alignment)    # src 0-based → tgt 0-based
    t2s = tgt_to_src_map(alignment)    # tgt 0-based → src 0-based

    # Projected annotations (1-based, 0 = root)
    proj_head   = [0] * n_tgt
    proj_label  = ["dep"] * n_tgt
    proj_upos   = ["X"] * n_tgt

    for t_i in range(n_tgt):
        if t_i not in t2s:
            # Unaligned target token: heuristic — attach to root with "dep"
            proj_head[t_i]  = 0
            proj_label[t_i] = "dep"
            continue

        s_j = t2s[t_i]                     # aligned source index (0-based)
        h_src = src_head.get(s_j, -1)      # head in source (-1 = root)

        proj_upos[t_i] = src_upos.get(s_j, "X")
        raw_label      = src_label.get(s_j, "dep")

        if h_src == -1:                    # source token is the root
            proj_head[t_i]  = 0           # project as root
            proj_label[t_i] = "root"
        else:
            # Try to find head_src's aligned target token
            h_tgt = s2t.get(h_src, None)
            if h_tgt is not None and h_tgt != t_i:
                proj_head[t_i]  = h_tgt + 1      # convert to 1-based
                proj_label[t_i] = correct_label(raw_label)
            else:
                # Head is unaligned: fall back to root attachment
                proj_head[t_i]  = 0
                proj_label[t_i] = "dep"

    # Guarantee exactly one root
    roots = [i for i, h in enumerate(proj_head) if h == 0]
    if not roots:
        # No root found: make the last token root
        proj_head[-1]  = 0
        proj_label[-1] = "root"
    elif len(roots) > 1:
        # Multiple roots: keep the first, attach others to it
        first_root = roots[0]
        for i in roots[1:]:
            proj_head[i]  = first_root + 1
            proj_label[i] = "dep"
    else:
        proj_label[roots[0]] = "root"

    # Build Sentence object
    tgt_sent = Sentence()
    for c in src_sent.comments:
        tgt_sent.comments.append(c.replace("# text = ", "# src_text = "))
    tgt_sent.set_comment(
        "tgt_text", " ".join(tgt_words)
    )
    tgt_sent.set_comment(
        "alignment", " ".join(f"{s}-{t}" for s, t in sorted(alignment))
    )

    for i, word in enumerate(tgt_words):
        tgt_sent.tokens.append(Token(
            id     = i + 1,
            form   = word,
            lemma  = "_",
            upos   = proj_upos[i],
            xpos   = "_",
            feats  = "_",
            head   = proj_head[i],
            deprel = proj_label[i],
            deps   = "_",
            misc   = "_",
        ))

    return tgt_sent


# ─────────────────────────────────────────────────────────────────────────────
# Batch projection
# ─────────────────────────────────────────────────────────────────────────────
def project_treebank(
    hindi_conllu:     str | Path,
    translations_tsv: str | Path,  # src\ttgt per line
    alignments_txt:   str | Path,  # Pharaoh per line
    output_conllu:    str | Path,
    max_sents:        int = 0,
) -> None:
    src_sents = read_conllu(hindi_conllu)
    trans_lines = Path(translations_tsv).read_text(encoding="utf-8").splitlines()
    align_lines = Path(alignments_txt).read_text(encoding="utf-8").splitlines()

    if max_sents:
        src_sents   = src_sents[:max_sents]
        trans_lines = trans_lines[:max_sents]
        align_lines = align_lines[:max_sents]

    projected: List[Sentence] = []
    skipped = 0

    for i, (src, trans, alg) in enumerate(zip(src_sents, trans_lines, align_lines)):
        if "\t" not in trans:
            skipped += 1
            continue
        _, tgt_str = trans.split("\t", 1)
        tgt_words  = tgt_str.strip().split()
        if not tgt_words:
            skipped += 1
            continue
        alignment = str_to_alignment(alg)
        tgt_sent  = project_sentence(src, tgt_words, alignment)
        projected.append(tgt_sent)

        if (i + 1) % 1000 == 0:
            print(f"  Projected {i+1:,}/{len(src_sents):,} sentences …")

    write_conllu(projected, output_conllu)
    print(f"  Projected treebank ({len(projected):,} sentences, "
          f"{skipped} skipped) → {output_conllu}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hindi_conllu",     required=True)
    ap.add_argument("--translations_tsv", required=True)
    ap.add_argument("--alignments_txt",   required=True)
    ap.add_argument("--output_conllu",    required=True)
    ap.add_argument("--max_sents",        type=int, default=0)
    args = ap.parse_args()

    project_treebank(
        args.hindi_conllu, args.translations_tsv,
        args.alignments_txt, args.output_conllu, args.max_sents,
    )

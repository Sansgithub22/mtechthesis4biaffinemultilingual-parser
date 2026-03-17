# utils/metrics.py
# UAS / LAS evaluation following CoNLL shared task conventions:
#   - Punctuation tokens (PUNCT / SYM upos) are EXCLUDED from scoring
#   - Token 0 (ROOT) is never a dependent — never scored
#   - Predictions are compared at the word level (not subword)

from __future__ import annotations
from typing import List, Tuple
from utils.conllu_utils import Sentence


PUNCT_UPOS = {"PUNCT", "SYM"}


def uas_las(
    gold_sents: List[Sentence],
    pred_heads: List[List[int]],   # one list of head ids per sentence
    pred_rels:  List[List[str]],   # one list of deprel strings per sentence
    ignore_punct: bool = True,
) -> Tuple[float, float]:
    """
    Compute UAS and LAS over a list of sentences.

    Returns:
        (uas, las) as floats in [0, 1].
    """
    total = 0
    uas_correct = 0
    las_correct = 0

    for sent, p_heads, p_rels in zip(gold_sents, pred_heads, pred_rels):
        for tok, ph, pr in zip(sent.tokens, p_heads, p_rels):
            if ignore_punct and tok.upos in PUNCT_UPOS:
                continue
            total += 1
            if ph == tok.head:
                uas_correct += 1
                if pr == tok.deprel:
                    las_correct += 1

    if total == 0:
        return 0.0, 0.0
    return uas_correct / total, las_correct / total


def print_metrics(label: str, uas: float, las: float):
    print(f"  [{label}]  UAS: {uas*100:.2f}%   LAS: {las*100:.2f}%")

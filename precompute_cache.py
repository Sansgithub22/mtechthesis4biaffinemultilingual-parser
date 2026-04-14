#!/usr/bin/env python3
"""
precompute_cache.py  —  Run ONCE before System I training.
Computes frozen XLM-R word embeddings for all 30,966 Hindi + Bhojpuri
sentences and saves them to  cache/xlmr_cache.pt  on disk.

Training scripts check this file first; if it exists XLM-R is never
run again  (saves ~20 h per training job).

Usage:
    python3 precompute_cache.py
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import torch
from pathlib import Path

from config import XLM_R_LOCAL
from utils.conllu_utils import read_conllu
from model.parallel_encoder import ParallelEncoder

ROOT_DIR   = Path(__file__).parent
PROF_BHO   = ROOT_DIR / "bhojpuri_matched_transferred.conllu"
PROF_HI    = ROOT_DIR / "hindi_matched.conllu"
CACHE_PATH = ROOT_DIR / "cache" / "xlmr_cache.pt"


def main():
    if CACHE_PATH.exists():
        c = torch.load(str(CACHE_PATH), map_location="cpu")
        print(f"Cache already exists — hi:{len(c['hi'])}, bho:{len(c['bho'])}. Nothing to do.")
        return

    print("Loading parallel data …")
    hi_sents  = read_conllu(PROF_HI)
    bho_sents = read_conllu(PROF_BHO)
    print(f"  Hindi   : {len(hi_sents):,} sentences")
    print(f"  Bhojpuri: {len(bho_sents):,} sentences")

    print(f"\nLoading XLM-R ({XLM_R_LOCAL}) …")
    encoder = ParallelEncoder(
        model_name=XLM_R_LOCAL,
        adapter_dim=64,
        adapter_dropout=0.1,
        freeze_xlmr=True,
    )
    encoder.xlmr.eval()

    print("\nPre-computing Hindi embeddings …")
    cache_hi = encoder.precompute_xlmr(
        [s.words() for s in hi_sents], desc="Hindi"
    )

    print("\nPre-computing Bhojpuri embeddings …")
    cache_bho = encoder.precompute_xlmr(
        [s.words() for s in bho_sents], desc="Bhojpuri"
    )

    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving to {CACHE_PATH} …")
    torch.save({"hi": cache_hi, "bho": cache_bho}, str(CACHE_PATH))
    print(f"Done. Size: {CACHE_PATH.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()

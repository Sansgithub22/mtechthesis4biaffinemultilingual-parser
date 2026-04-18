#!/usr/bin/env python3
# generate_silver_muril.py
# System M — Phase 2 of 3: MuRIL-Hindi parser labels Bhojpuri tokens in UD style.
#
# MOTIVATION:
#   Analogous to generate_silver_ud_labels.py for System K, but with the
#   MuRIL-backed Hindi parser (Phase 1 / train_hindi_muril.py) as the teacher.
#   MuRIL's heavy Indic pretraining gives cleaner silver labels for Bhojpuri
#   surface forms than XLM-R-base, especially for morphologically rich tokens.
#
# OUTPUT:
#   data_files/synthetic/bho_silver_muril_ud.conllu  (training data for M)
#
# Usage:
#   python3 generate_silver_muril.py [--device cuda] [--min_len 3] [--max_len 100]

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
from pathlib import Path
from typing import List

import torch

from config import DATA_DIR, CHECKPT_DIR
from utils.conllu_utils import read_conllu, write_conllu, Sentence, Token
from model.parallel_encoder    import ParallelEncoder
from model.biaffine_heads      import BiaffineHeads
from model.cross_lingual_parser import RelVocab


HINDI_CKPT   = CHECKPT_DIR / "muril_hindi" / "best.pt"
DEFAULT_IN   = Path(__file__).parent / "bhojpuri_matched_transferred.conllu"
DEFAULT_OUT  = DATA_DIR / "synthetic" / "bho_silver_muril_ud.conllu"


def is_well_formed(sent: Sentence) -> bool:
    n = len(sent.tokens)
    if n == 0:
        return False
    roots = 0
    for t in sent.tokens:
        if t.head < 0 or t.head > n:
            return False
        if t.head == 0:
            roots += 1
    return roots >= 1


def single_root(sent: Sentence) -> bool:
    return sum(1 for t in sent.tokens if t.head == 0) == 1


def main():
    ap = argparse.ArgumentParser(description="Phase 2: silver UD-Bhojpuri via MuRIL-Hindi")
    ap.add_argument("--input",   type=str, default=str(DEFAULT_IN))
    ap.add_argument("--output",  type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--ckpt",    type=str, default=str(HINDI_CKPT))
    ap.add_argument("--device",  type=str, default="cuda")
    ap.add_argument("--min_len", type=int, default=3)
    ap.add_argument("--max_len", type=int, default=100)
    ap.add_argument("--limit",   type=int, default=-1)
    ap.add_argument("--require_single_root", action="store_true")
    args = ap.parse_args()

    in_path  = Path(args.input)
    out_path = Path(args.output)
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"MuRIL-Hindi checkpoint not found: {ckpt_path}\n"
            f"Run: python3 train_hindi_muril.py --device cuda"
        )
    device = torch.device(args.device)

    print("=" * 70)
    print(" System M — Phase 2: Silver UD-Bhojpuri via MuRIL-Hindi teacher")
    print("=" * 70)
    print(f"  Input  : {in_path}")
    print(f"  Output : {out_path}")
    print(f"  Ckpt   : {ckpt_path}")
    print()

    # ── Load teacher ──────────────────────────────────────────────────────────
    print("[1] Loading MuRIL-Hindi checkpoint …")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    muril_name  = ckpt.get("muril_name", "google/muril-base-cased")
    adapter_dim = ckpt.get("adapter_dim", 64)
    vocab_words = ckpt["vocab_words"]

    vocab = RelVocab()
    for w in vocab_words:
        vocab.add(w)
    n_rels = len(vocab)
    print(f"    Backbone     : {muril_name}")
    print(f"    Adapter dim  : {adapter_dim}")
    print(f"    Rel vocab    : {n_rels}")

    encoder = ParallelEncoder(
        model_name      = muril_name,
        adapter_dim     = adapter_dim,
        adapter_dropout = 0.1,
        freeze_xlmr     = True,
    )
    encoder.adapters["hindi"].load_state_dict(ckpt["adapter_hi"])
    encoder.adapters.to(device)

    head = BiaffineHeads(encoder.hidden_size, 500, 100, n_rels, 0.33).to(device)
    head.load_state_dict(ckpt["biaffine"])

    encoder.adapters["hindi"].eval()
    head.eval()

    # ── Read Bhojpuri tokens ──────────────────────────────────────────────────
    print(f"\n[2] Reading Bhojpuri sentences …")
    raw = read_conllu(in_path)
    print(f"    {len(raw):,} sentences")

    kept_idx, kept = [], []
    for i, s in enumerate(raw):
        n = len(s.tokens)
        if n < args.min_len or n > args.max_len:
            continue
        kept_idx.append(i); kept.append(s)
        if args.limit > 0 and len(kept) >= args.limit:
            break
    print(f"    After length filter: {len(kept):,}")

    # ── Parse each sentence ───────────────────────────────────────────────────
    print(f"\n[3] Parsing Bhojpuri with MuRIL-Hindi teacher …")
    silver: List[Sentence] = []
    n_ill_formed = n_multi_root = 0

    with torch.no_grad():
        for k, s in enumerate(kept):
            words = s.words()
            if not words:
                continue
            H = encoder.encode_one("hindi", words).to(device)
            arc_s, lbl_s = head(H)
            mask = torch.ones(1, len(words), dtype=torch.bool, device=device)
            ph, pr = BiaffineHeads.predict(arc_s, lbl_s, mask)
            pred_heads = ph[0].cpu().tolist()
            pred_rels  = [vocab.decode(r) for r in pr[0].cpu().tolist()]

            new_tokens: List[Token] = []
            for j, t in enumerate(s.tokens):
                new_tokens.append(Token(
                    id     = t.id,
                    form   = t.form,
                    lemma  = "_",
                    upos   = "_",
                    xpos   = "_",
                    feats  = "_",
                    head   = int(pred_heads[j]),
                    deprel = pred_rels[j],
                    deps   = "_",
                    misc   = "_",
                ))
            new_sent = Sentence(tokens=new_tokens,
                                comments=[c for c in s.comments
                                          if c.startswith("# sent_id") or c.startswith("# text")])
            if not is_well_formed(new_sent):
                n_ill_formed += 1
                continue
            if args.require_single_root and not single_root(new_sent):
                n_multi_root += 1
                continue
            silver.append(new_sent)

            if (k + 1) % 1000 == 0:
                print(f"      {k+1:,}/{len(kept):,}", flush=True)

    # ── Write ────────────────────────────────────────────────────────────────
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_conllu(silver, out_path)

    n_tok = sum(len(s.tokens) for s in silver)
    print(f"\n[4] Silver corpus written:")
    print(f"    Sentences              : {len(silver):,}")
    print(f"    Tokens                 : {n_tok:,}")
    print(f"    Dropped (ill-formed)   : {n_ill_formed:,}")
    if args.require_single_root:
        print(f"    Dropped (multi-root)   : {n_multi_root:,}")
    print(f"    File                   : {out_path}")
    print("\nNext: python3 train_system_m.py --device cuda")


if __name__ == "__main__":
    main()

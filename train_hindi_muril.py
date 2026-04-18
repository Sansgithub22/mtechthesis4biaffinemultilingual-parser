#!/usr/bin/env python3
# train_hindi_muril.py
# System M — Phase 1 of 3: Hindi parser with MuRIL backbone.
#
# MOTIVATION:
#   System K (XLM-R-base) beats System A by +1.49 UAS / +1.34 LAS on BHTB.
#   MuRIL (Khanuja et al. 2021) is pretrained on 17 Indian languages including
#   Hindi, Bhojpuri's closest siblings, and transliterated corpora. For Indic
#   cross-lingual transfer it reliably adds +2-4 LAS over mBERT/XLM-R-base
#   (MuRIL paper; AI4Bharat IndicGLUE results).
#
#   This script produces the "Hindi-MuRIL" parser (= System A' of the M family),
#   which will then (a) silver-label prof's Bhojpuri tokens in UD style and
#   (b) serve as warm-start for System M's full UD-Bridge training.
#
# OUTPUT:
#   checkpoints/muril_hindi/best.pt    {encoder, biaffine, vocab, epoch, las}
#
# Usage:
#   python3 train_hindi_muril.py [--epochs 30] [--device cuda] [--lr 5e-4]
#                                [--muril_name google/muril-base-cased]

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import random
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CHECKPT_DIR, DATA_DIR
from utils.conllu_utils import read_conllu, Sentence
from utils.metrics       import uas_las
from model.parallel_encoder    import ParallelEncoder
from model.biaffine_heads      import BiaffineHeads
from model.cross_lingual_parser import RelVocab


MURIL_DEFAULT  = "google/muril-base-cased"
CKPT_DIR       = CHECKPT_DIR / "muril_hindi"
CKPT_PATH      = CKPT_DIR / "best.pt"
MURIL_CACHE    = Path(__file__).parent / "cache" / "muril_hindi_cache.pt"


# ─────────────────────────────────────────────────────────────────────────────
# Loss (greedy biaffine, matches G/H/I/J convention)
# ─────────────────────────────────────────────────────────────────────────────
def parse_loss(arc_s: torch.Tensor, lbl_s: torch.Tensor,
               gold_heads: torch.Tensor, gold_rels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over arc and label predictions."""
    B, n, _ = arc_s.shape
    arc_loss = F.cross_entropy(arc_s.reshape(B * n, -1), gold_heads.reshape(-1))
    # Select the gold-head column for label scoring
    idx = gold_heads.unsqueeze(-1).unsqueeze(-1).expand(B, n, 1, lbl_s.size(-1))
    gold_lbl = lbl_s.gather(2, idx).squeeze(2)  # [B, n, n_rels]
    lbl_loss = F.cross_entropy(gold_lbl.reshape(B * n, -1), gold_rels.reshape(-1))
    return arc_loss + lbl_loss


def sent_to_tensors(sent: Sentence, vocab: RelVocab, device):
    heads = torch.tensor(sent.heads(), dtype=torch.long, device=device).unsqueeze(0)
    rels  = torch.tensor([vocab.encode(r) for r in sent.deprels()],
                         dtype=torch.long, device=device).unsqueeze(0)
    return heads, rels


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Phase 1: Hindi parser on MuRIL")
    ap.add_argument("--epochs",      type=int,   default=30)
    ap.add_argument("--lr",          type=float, default=5e-4)
    ap.add_argument("--device",      type=str,   default="cuda")
    ap.add_argument("--patience",    type=int,   default=5)
    ap.add_argument("--adapter_dim", type=int,   default=64)
    ap.add_argument("--muril_name",  type=str,   default=MURIL_DEFAULT)
    ap.add_argument("--seed",        type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print("=" * 70)
    print(" System M — Phase 1: Hindi parser with MuRIL backbone")
    print("=" * 70)
    print(f"  Backbone : {args.muril_name}")
    print(f"  Device   : {device}")
    print(f"  Epochs   : {args.epochs}   lr={args.lr}   patience={args.patience}")
    print()

    # ── Load HDTB ─────────────────────────────────────────────────────────────
    hi_train = read_conllu(DATA_DIR / "hindi/hi_hdtb-ud-train.conllu")
    hi_dev   = read_conllu(DATA_DIR / "hindi/hi_hdtb-ud-dev.conllu")
    print(f"  HDTB train : {len(hi_train):,}   dev : {len(hi_dev):,}")

    # ── Vocabulary ────────────────────────────────────────────────────────────
    vocab = RelVocab()
    for s in hi_train + hi_dev:
        for t in s.tokens:
            vocab.add(t.deprel)
    n_rels = len(vocab)
    print(f"  Rel vocab  : {n_rels} labels")

    # ── Encoder + biaffine head ───────────────────────────────────────────────
    print(f"\n[Model] Loading {args.muril_name} …")
    encoder = ParallelEncoder(
        model_name      = args.muril_name,
        adapter_dim     = args.adapter_dim,
        adapter_dropout = 0.1,
        freeze_xlmr     = True,
    )
    encoder.adapters.to(device)
    head = BiaffineHeads(encoder.hidden_size, 500, 100, n_rels, 0.33).to(device)

    trainable = list(encoder.adapters["hindi"].parameters()) + list(head.parameters())
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    # ── Precompute MuRIL embeddings ONCE (frozen backbone) ────────────────────
    if MURIL_CACHE.exists():
        print(f"\n[Cache] Loading MuRIL cache {MURIL_CACHE} …")
        c = torch.load(str(MURIL_CACHE), map_location="cpu")
        cache_train, cache_dev = c["train"], c["dev"]
    else:
        print("\n[Cache] Pre-computing MuRIL embeddings …")
        cache_train = encoder.precompute_xlmr([s.words() for s in hi_train], desc="HDTB train")
        cache_dev   = encoder.precompute_xlmr([s.words() for s in hi_dev],   desc="HDTB dev")
        MURIL_CACHE.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"train": cache_train, "dev": cache_dev}, str(MURIL_CACHE))
        print(f"  Saved MuRIL cache → {MURIL_CACHE}")

    # ── Training loop ─────────────────────────────────────────────────────────
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_las = 0.0
    no_improve = 0

    print(f"\n[Train] Up to {args.epochs} epochs …\n")
    idx_list = list(range(len(hi_train)))

    for epoch in range(1, args.epochs + 1):
        encoder.adapters["hindi"].train()
        head.train()
        random.shuffle(idx_list)

        total_loss = 0.0
        n_ok = 0
        for i in idx_list:
            s = hi_train[i]
            ch = cache_train[i]
            if ch is None or len(s.tokens) < 2:
                continue
            H = encoder.encode_one("hindi", [], cached_xlmr=ch).to(device)
            arc_s, lbl_s = head(H)
            g_heads, g_rels = sent_to_tensors(s, vocab, device)
            loss = parse_loss(arc_s, lbl_s, g_heads, g_rels)
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 5.0)
            optim.step()
            total_loss += loss.item()
            n_ok += 1

        # ── Dev eval ──────────────────────────────────────────────────────────
        encoder.adapters["hindi"].eval()
        head.eval()
        ph_all, pr_all = [], []
        with torch.no_grad():
            for i, s in enumerate(hi_dev):
                if not s.tokens or cache_dev[i] is None:
                    ph_all.append([]); pr_all.append([]); continue
                H = encoder.encode_one("hindi", [], cached_xlmr=cache_dev[i]).to(device)
                arc_s, lbl_s = head(H)
                mask = torch.ones(1, len(s.tokens), dtype=torch.bool, device=device)
                ph, pr = BiaffineHeads.predict(arc_s, lbl_s, mask)
                ph_all.append(ph[0].cpu().tolist())
                pr_all.append([vocab.decode(r) for r in pr[0].cpu().tolist()])
        dev_uas, dev_las = uas_las(hi_dev, ph_all, pr_all)

        avg = total_loss / max(n_ok, 1)
        print(f"  Epoch {epoch:3d}  loss={avg:.4f}  "
              f"Dev UAS={dev_uas*100:.2f}%  LAS={dev_las*100:.2f}%", end="")

        if dev_las > best_las:
            best_las = dev_las
            no_improve = 0
            torch.save({
                "epoch": epoch, "best_las": best_las,
                "vocab_words": vocab._i2w,
                "adapter_hi": encoder.adapters["hindi"].state_dict(),
                "biaffine":   head.state_dict(),
                "muril_name": args.muril_name,
                "adapter_dim": args.adapter_dim,
            }, CKPT_PATH)
            print("  ← BEST saved")
        else:
            no_improve += 1
            print(f"  (no improve {no_improve}/{args.patience})")
            if no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    print(f"\n[Done] Best HDTB Dev LAS: {best_las*100:.2f}%")
    print(f"       Checkpoint: {CKPT_PATH}")
    print("\nNext: python3 generate_silver_muril.py --gpu")


if __name__ == "__main__":
    main()

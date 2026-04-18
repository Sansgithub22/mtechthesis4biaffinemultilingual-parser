#!/usr/bin/env python3
# train_system_m.py
# System M — UD-Bridge with MuRIL backbone (Competition 2)
#
# Architecture summary:
#   MuRIL-base (frozen, 768-dim, Indic-heavy pretraining)
#      ↓
#   Language-specific adapter (Hindi on HDTB, Bhojpuri on silver)
#      ↓
#   Shared biaffine head (UD label vocabulary, inherited from Phase 1)
#
# Training data:
#   HDTB train (UD Hindi, gold)  ← Hindi adapter
#   bho_silver_muril_ud.conllu   ← Bhojpuri adapter
#   Both streams share one biaffine head because they use the SAME UD schema.
#
# Warm-start:
#   Encoder+biaffine initialised from Phase 1 (train_hindi_muril.py).
#   Hindi adapter starts fully trained; Bhojpuri adapter starts near-identity
#   and adapts to Bhojpuri surface forms during training.
#
# Evaluation:
#   On every epoch: HDTB dev (early-stopping signal)
#   At the end    : BHTB test (external UD gold — the Comp 2 benchmark)
#
# Usage:
#   python3 train_system_m.py [--epochs 30] [--lr 5e-4] [--device cuda]
#                             [--lambda_hi 0.3]  [--no_warmstart_phase1]
#                             [--silver_path data_files/synthetic/bho_silver_muril_ud.conllu]

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import random
import datetime
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DATA_DIR, CHECKPT_DIR
from utils.conllu_utils import read_conllu, Sentence
from utils.metrics       import uas_las
from model.parallel_encoder    import ParallelEncoder
from model.biaffine_heads      import BiaffineHeads
from model.cross_lingual_parser import RelVocab


MURIL_DEFAULT  = "google/muril-base-cased"
PHASE1_CKPT    = CHECKPT_DIR / "muril_hindi" / "best.pt"
SYSM_CKPT_DIR  = CHECKPT_DIR / "system_m"
SYSM_CKPT      = SYSM_CKPT_DIR / "system_m.pt"
SILVER_DEFAULT = DATA_DIR / "synthetic" / "bho_silver_muril_ud.conllu"
CACHE_PATH     = Path(__file__).parent / "cache" / "muril_m_cache.pt"
BHTB_TEST      = DATA_DIR / "bhojpuri" / "bho_bhtb-ud-test.conllu"


# ─────────────────────────────────────────────────────────────────────────────
def parse_loss(arc_s, lbl_s, gold_heads, gold_rels):
    B, n, _ = arc_s.shape
    arc_loss = F.cross_entropy(arc_s.reshape(B * n, -1), gold_heads.reshape(-1))
    idx = gold_heads.unsqueeze(-1).unsqueeze(-1).expand(B, n, 1, lbl_s.size(-1))
    gold_lbl = lbl_s.gather(2, idx).squeeze(2)
    lbl_loss = F.cross_entropy(gold_lbl.reshape(B * n, -1), gold_rels.reshape(-1))
    return arc_loss + lbl_loss


def sent_to_tensors(sent: Sentence, vocab: RelVocab, device):
    heads = torch.tensor(sent.heads(), dtype=torch.long, device=device).unsqueeze(0)
    rels  = torch.tensor([vocab.encode(r) for r in sent.deprels()],
                         dtype=torch.long, device=device).unsqueeze(0)
    return heads, rels


def encode_and_parse(encoder, head, lang, sent, cached, device):
    """Returns (arc_scores, label_scores) after encoder+biaffine on a sentence."""
    if cached is not None:
        H = encoder.encode_one(lang, [], cached_xlmr=cached).to(device)
    else:
        H = encoder.encode_one(lang, sent.words()).to(device)
    return head(H)


@torch.no_grad()
def evaluate_on(encoder, head, lang, sents, vocab, device, cache=None):
    encoder.adapters[lang].eval(); head.eval()
    ph_all, pr_all = [], []
    for i, s in enumerate(sents):
        if not s.tokens:
            ph_all.append([]); pr_all.append([]); continue
        cached = cache[i] if cache and i < len(cache) else None
        arc_s, lbl_s = encode_and_parse(encoder, head, lang, s, cached, device)
        mask = torch.ones(1, len(s.tokens), dtype=torch.bool, device=device)
        ph, pr = BiaffineHeads.predict(arc_s, lbl_s, mask)
        ph_all.append(ph[0].cpu().tolist())
        pr_all.append([vocab.decode(r) for r in pr[0].cpu().tolist()])
    return uas_las(sents, ph_all, pr_all)


# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="System M — UD-Bridge on MuRIL backbone")
    ap.add_argument("--epochs",      type=int,   default=30)
    ap.add_argument("--lr",          type=float, default=5e-4)
    ap.add_argument("--device",      type=str,   default="cuda")
    ap.add_argument("--patience",    type=int,   default=5)
    ap.add_argument("--lambda_hi",   type=float, default=0.3,
                    help="Weight on HDTB-Hindi loss (Bhojpuri loss weight = 1.0)")
    ap.add_argument("--silver_path", type=str,   default=str(SILVER_DEFAULT))
    ap.add_argument("--no_warmstart_phase1", action="store_true",
                    help="Train from scratch instead of loading train_hindi_muril.py ckpt")
    ap.add_argument("--muril_name",  type=str,   default=MURIL_DEFAULT)
    ap.add_argument("--adapter_dim", type=int,   default=64)
    ap.add_argument("--seed",        type=int,   default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    print("=" * 70)
    print(" System M — UD-Bridge via Silver Self-Training (MuRIL backbone)")
    print("=" * 70)
    print(f"  Backbone     : {args.muril_name}")
    print(f"  Device       : {device}")
    print(f"  Warm-start   : Phase 1 checkpoint ({not args.no_warmstart_phase1})")
    print(f"  Silver path  : {args.silver_path}")
    print(f"  λ_hi         : {args.lambda_hi}")
    print()

    silver_path = Path(args.silver_path)
    if not silver_path.exists():
        raise FileNotFoundError(
            f"Silver Bhojpuri not found: {silver_path}\n"
            f"Run: python3 generate_silver_muril.py --device {args.device}"
        )
    if not BHTB_TEST.exists():
        raise FileNotFoundError(f"BHTB test missing: {BHTB_TEST}")

    # ── Load Phase 1 checkpoint (for vocab + warm-start) ──────────────────────
    phase1 = None
    if not args.no_warmstart_phase1:
        if not PHASE1_CKPT.exists():
            raise FileNotFoundError(
                f"Phase 1 checkpoint missing: {PHASE1_CKPT}\n"
                f"Run: python3 train_hindi_muril.py --device {args.device}"
            )
        phase1 = torch.load(str(PHASE1_CKPT), map_location="cpu")
        print(f"  Phase 1 ckpt : epoch {phase1['epoch']}  LAS {phase1['best_las']*100:.2f}%")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1] Loading data …")
    hi_train = read_conllu(DATA_DIR / "hindi/hi_hdtb-ud-train.conllu")
    hi_dev   = read_conllu(DATA_DIR / "hindi/hi_hdtb-ud-dev.conllu")
    bho_silver = read_conllu(silver_path)
    bhtb_test  = read_conllu(BHTB_TEST)
    print(f"    HDTB train   : {len(hi_train):,}")
    print(f"    HDTB dev     : {len(hi_dev):,}")
    print(f"    Silver Bho   : {len(bho_silver):,}")
    print(f"    BHTB test    : {len(bhtb_test):,}")

    # ── Vocabulary (start from Phase 1's, extend with new UD labels if any) ──
    vocab = RelVocab()
    if phase1 is not None:
        for w in phase1["vocab_words"]:
            vocab.add(w)
    for s in hi_train + hi_dev + bho_silver + bhtb_test:
        for t in s.tokens:
            vocab.add(t.deprel)
    n_rels = len(vocab)
    print(f"    Rel vocab    : {n_rels} labels "
          f"({'extended from Phase 1' if phase1 else 'built fresh'})")

    # ── Build encoder + biaffine head ─────────────────────────────────────────
    print(f"\n[2] Building model on {args.muril_name} …")
    encoder = ParallelEncoder(
        model_name      = args.muril_name,
        adapter_dim     = args.adapter_dim,
        adapter_dropout = 0.1,
        freeze_xlmr     = True,
    )
    encoder.adapters.to(device)
    head = BiaffineHeads(encoder.hidden_size, 500, 100, n_rels, 0.33).to(device)

    if phase1 is not None:
        encoder.adapters["hindi"].load_state_dict(phase1["adapter_hi"])
        # Biaffine vocab may have grown; copy matching leading entries where possible
        phase1_head = phase1["biaffine"]
        own_head = head.state_dict()
        copied = skipped = 0
        for k, v in phase1_head.items():
            if k in own_head and own_head[k].shape == v.shape:
                own_head[k] = v; copied += 1
            else:
                skipped += 1
        head.load_state_dict(own_head)
        print(f"    Warm-start copied biaffine tensors: {copied} (skipped {skipped} shape-mismatches)")

    trainable = (list(encoder.adapters.parameters())
                 + list(head.parameters()))
    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    print(f"    Trainable params: {sum(p.numel() for p in trainable):,}")

    # ── Precompute MuRIL embeddings ───────────────────────────────────────────
    if CACHE_PATH.exists():
        print(f"\n[3] Loading MuRIL cache {CACHE_PATH} …")
        c = torch.load(str(CACHE_PATH), map_location="cpu")
        c_hi_tr, c_hi_dv, c_bho_silver, c_bhtb = c["hi_tr"], c["hi_dv"], c["bho"], c["bhtb"]
    else:
        print("\n[3] Pre-computing MuRIL embeddings …")
        c_hi_tr      = encoder.precompute_xlmr([s.words() for s in hi_train],   desc="HDTB train")
        c_hi_dv      = encoder.precompute_xlmr([s.words() for s in hi_dev],     desc="HDTB dev")
        c_bho_silver = encoder.precompute_xlmr([s.words() for s in bho_silver], desc="Silver Bho")
        c_bhtb       = encoder.precompute_xlmr([s.words() for s in bhtb_test],  desc="BHTB test")
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"hi_tr": c_hi_tr, "hi_dv": c_hi_dv,
                    "bho":   c_bho_silver, "bhtb": c_bhtb}, str(CACHE_PATH))
        print(f"    Saved cache → {CACHE_PATH}")

    # ── Training loop ─────────────────────────────────────────────────────────
    SYSM_CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_las = 0.0
    no_improve = 0

    print(f"\n[4] Training for up to {args.epochs} epochs …\n")

    hi_idx_list  = list(range(len(hi_train)))
    bho_idx_list = list(range(len(bho_silver)))

    for epoch in range(1, args.epochs + 1):
        encoder.adapters.train(); head.train()
        random.shuffle(hi_idx_list)
        random.shuffle(bho_idx_list)

        # Interleave: roughly same number of Hindi and Bhojpuri steps per epoch
        # (Hindi contributes regularisation, Bhojpuri is the target signal).
        steps = max(len(hi_idx_list), len(bho_idx_list))
        hi_ptr = bho_ptr = 0
        total_loss = hi_sum = bho_sum = 0.0
        n_steps = 0

        for _ in range(steps):
            losses = []

            # Hindi step (HDTB)
            if hi_ptr < len(hi_idx_list):
                i = hi_idx_list[hi_ptr]; hi_ptr += 1
                s = hi_train[i]; ch = c_hi_tr[i]
                if s.tokens and ch is not None:
                    arc_s, lbl_s = encode_and_parse(encoder, head, "hindi", s, ch, device)
                    g_h, g_r = sent_to_tensors(s, vocab, device)
                    l_hi = parse_loss(arc_s, lbl_s, g_h, g_r)
                    losses.append(args.lambda_hi * l_hi)
                    hi_sum += l_hi.item()

            # Bhojpuri step (silver)
            if bho_ptr < len(bho_idx_list):
                j = bho_idx_list[bho_ptr]; bho_ptr += 1
                s = bho_silver[j]; ch = c_bho_silver[j]
                if s.tokens and ch is not None:
                    arc_s, lbl_s = encode_and_parse(encoder, head, "bhojpuri", s, ch, device)
                    g_h, g_r = sent_to_tensors(s, vocab, device)
                    l_bho = parse_loss(arc_s, lbl_s, g_h, g_r)
                    losses.append(l_bho)
                    bho_sum += l_bho.item()

            if not losses:
                break
            loss = torch.stack(losses).sum()
            optim.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 5.0)
            optim.step()
            total_loss += loss.item()
            n_steps += 1

        # ── Dev eval on HDTB dev (UD style, matches BHTB schema) ──────────────
        dev_uas, dev_las = evaluate_on(encoder, head, "hindi", hi_dev,
                                       vocab, device, c_hi_dv)
        avg = total_loss / max(n_steps, 1)
        print(f"  Ep {epoch:3d}  loss={avg:.4f} (bho={bho_sum/max(n_steps,1):.3f} "
              f"hi={hi_sum/max(n_steps,1):.3f})  HDTB-dev "
              f"UAS={dev_uas*100:.2f}% LAS={dev_las*100:.2f}%", end="")

        if dev_las > best_las:
            best_las = dev_las
            no_improve = 0
            torch.save({
                "epoch": epoch, "best_las": best_las,
                "vocab_words": vocab._i2w,
                "encoder": encoder.state_dict(),
                "biaffine": head.state_dict(),
                "muril_name": args.muril_name,
                "adapter_dim": args.adapter_dim,
            }, SYSM_CKPT)
            print("  ← BEST saved")
        else:
            no_improve += 1
            print(f"  (no improve {no_improve}/{args.patience})")
            if no_improve >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break

    # ── Final evaluation on BHTB ──────────────────────────────────────────────
    print(f"\n[5] Loading best checkpoint …")
    ckpt = torch.load(str(SYSM_CKPT), map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    head.load_state_dict(ckpt["biaffine"])

    print("\n[6] Evaluating on BHTB (external UD gold) …")
    bhtb_uas, bhtb_las = evaluate_on(encoder, head, "bhojpuri", bhtb_test,
                                     vocab, device, c_bhtb)

    # Also evaluate Phase 1 (zero-shot MuRIL) on BHTB for delta reporting
    # by using only the Hindi adapter on Bhojpuri tokens.
    zs_uas, zs_las = evaluate_on(encoder, head, "hindi", bhtb_test,
                                 vocab, device, c_bhtb)

    print(f"\n{'=' * 70}")
    print(f"  System M — Final BHTB Results  (best checkpoint, epoch {ckpt['epoch']})")
    print(f"{'=' * 70}")
    print(f"  MuRIL zero-shot (Hindi adapter on BHTB)   : UAS {zs_uas*100:.2f}%  LAS {zs_las*100:.2f}%")
    print(f"  System M       (Bhojpuri adapter on BHTB) : UAS {bhtb_uas*100:.2f}%  LAS {bhtb_las*100:.2f}%")
    print(f"  ΔM vs MuRIL zero-shot                     : "
          f"ΔUAS {(bhtb_uas - zs_uas)*100:+.2f}  ΔLAS {(bhtb_las - zs_las)*100:+.2f}")
    print()
    print(f"  Reference baselines (from prior runs on same BHTB):")
    print(f"    System A (XLM-R-base Trankit zero-shot) : UAS ~52.78%  LAS ~35.36%")
    print(f"    System K (XLM-R-base UD-Bridge)         : UAS ~54.27%  LAS ~36.70%")
    print(f"{'=' * 70}")

    # ── Save results file ────────────────────────────────────────────────────
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = results_dir / f"system_m_{ts}.txt"
    with open(out, "w") as f:
        f.write(f"System M — UD-Bridge on MuRIL\n")
        f.write(f"Date        : {datetime.datetime.now()}\n")
        f.write(f"Backbone    : {args.muril_name}\n")
        f.write(f"Best epoch  : {ckpt['epoch']}\n")
        f.write(f"HDTB Dev    : LAS {ckpt['best_las']*100:.2f}%\n")
        f.write(f"MuRIL zero-shot BHTB  : UAS {zs_uas*100:.2f}%  LAS {zs_las*100:.2f}%\n")
        f.write(f"System M BHTB         : UAS {bhtb_uas*100:.2f}%  LAS {bhtb_las*100:.2f}%\n")
        f.write(f"ΔM vs zero-shot       : ΔUAS {(bhtb_uas-zs_uas)*100:+.2f}  ΔLAS {(bhtb_las-zs_las)*100:+.2f}\n")
    print(f"\n  Results saved → {out}")


if __name__ == "__main__":
    main()

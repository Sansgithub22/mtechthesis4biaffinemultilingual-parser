#!/usr/bin/env python3
# train_system_g.py
# System G — Exact Alignment-Supervised Joint Training (Novel Contribution)
#
# KEY IDEA:
#   The professor's parallel data gives us EXACT positional alignment:
#   Bhojpuri token at position i ↔ Hindi token at position i (same sentence).
#   This is captured in the LEMMA field of bhojpuri_matched_transferred.conllu
#   (LEMMA = corresponding Hindi word at that position).
#
#   We exploit this by adding an alignment loss:
#     L_align = MSE(H_hi[i], H_bho[i])  for every token position i
#
#   This forces the Bhojpuri adapter to produce representations close to Hindi
#   for syntactically equivalent tokens — bridging the Hindi-Bhojpuri gap
#   directly at the representation level.
#
# TRAINING SIGNAL (three losses):
#   L_bho   = arc + label cross-entropy on Bhojpuri annotations (main task)
#   L_hi    = arc + label cross-entropy on Hindi annotations (regularization)
#   L_align = MSE between H_hi[i] and H_bho[i] (exact positional alignment)
#   L_total = L_bho + λ_hi * L_hi + λ_align * L_align
#
# DIFFERENCE FROM SYSTEM E:
#   System E: SimAlign (noisy, approximate) + cross-sentence attention (complex)
#   System G: Expert positional alignment (exact, clean) + alignment loss (simple)
#             30,966 parallel pairs vs 5,000 noisy pairs in System E
#
# DIFFERENCE FROM SYSTEM F:
#   System F: Only Bhojpuri parsing loss (no cross-lingual supervision)
#   System G: Adds explicit Hindi-Bhojpuri alignment signal at token level
#
# Checkpoint: checkpoints/system_g/system_g.pt
#
# Usage:
#   python3 train_system_g.py [--epochs 40] [--device cuda|mps|cpu]
#                             [--lambda_hi 0.3] [--lambda_align 0.5]

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from config import CHECKPT_DIR, DATA_DIR, XLM_R_LOCAL
from utils.conllu_utils import read_conllu, Sentence
from utils.metrics import uas_las

from model.parallel_encoder import ParallelEncoder
from model.biaffine_heads   import BiaffineHeads
from model.cross_lingual_parser import RelVocab


ROOT_DIR = Path(__file__).parent
PROF_BHO = ROOT_DIR / "bhojpuri_matched_transferred.conllu"
PROF_HI  = ROOT_DIR / "hindi_matched.conllu"
BHTB_TEST = DATA_DIR / "bhojpuri" / "bho_bhtb-ud-test.conllu"
CKPT_DIR  = CHECKPT_DIR / "system_g"
CKPT_PATH = CKPT_DIR / "system_g.pt"

# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_vocab(hi_sents: List[Sentence], bho_sents: List[Sentence]) -> RelVocab:
    vocab = RelVocab()
    for s in hi_sents + bho_sents:
        for t in s.tokens:
            vocab.add(t.deprel)
    print(f"  Relation vocab: {len(vocab)} labels")
    return vocab


def sentence_to_tensors(sent: Sentence, vocab: RelVocab, device: torch.device):
    """Convert a Sentence to (heads, rels) tensors on device."""
    heads = torch.tensor([t.head for t in sent.tokens],  dtype=torch.long, device=device)
    rels  = torch.tensor([vocab.encode(t.deprel) for t in sent.tokens],
                          dtype=torch.long, device=device)
    return heads, rels


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def parse_loss(arc_scores: torch.Tensor, lbl_scores: torch.Tensor,
               gold_heads: torch.Tensor, gold_rels: torch.Tensor) -> torch.Tensor:
    """
    arc_scores : [1, n, n+1]
    lbl_scores : [1, n, n+1, n_rels]
    gold_heads : [n]   0-indexed (0 = ROOT)
    gold_rels  : [n]
    """
    n = gold_heads.size(0)
    arc_loss = F.cross_entropy(arc_scores[0], gold_heads)

    # Label loss only at gold head positions
    idx = gold_heads.view(n, 1, 1).expand(n, 1, lbl_scores.size(-1))
    lbl_at_gold = lbl_scores[0].gather(1, idx).squeeze(1)   # [n, n_rels]
    lbl_loss = F.cross_entropy(lbl_at_gold, gold_rels)

    return arc_loss + lbl_loss


def alignment_loss(H_hi: torch.Tensor, H_bho: torch.Tensor,
                   n_tokens: int) -> torch.Tensor:
    """
    Exact positional alignment loss.
    H_hi  : [1, n_hi,  768]
    H_bho : [1, n_bho, 768]
    n_tokens = min(n_hi, n_bho) — aligned token count
    MSE between H_hi[0,:n_tokens] and H_bho[0,:n_tokens]
    """
    h = H_hi[0, :n_tokens]    # [n_tokens, 768]
    b = H_bho[0, :n_tokens]   # [n_tokens, 768]
    return F.mse_loss(b, h.detach())   # detach Hindi so gradient flows only to Bho adapter


# ─────────────────────────────────────────────────────────────────────────────
# Warm-start from Trankit Hindi checkpoint
# ─────────────────────────────────────────────────────────────────────────────

def warmstart_hindi_adapter(encoder: ParallelEncoder, hindi_ckpt: Path):
    if not hindi_ckpt.exists():
        print(f"  [WARN] Hindi checkpoint not found: {hindi_ckpt} — skipping warm-start")
        return
    state = torch.load(str(hindi_ckpt), map_location="cpu")
    adapters = state.get("adapters", {})
    # The Trankit adapter tensors live under keys like
    # "adapters.0.{down_proj,up_proj,layer_norm}.{weight,bias}"
    # We match by shape into our BottleneckAdapter
    our_sd  = encoder.adapters["hindi"].state_dict()
    loaded  = skipped = 0
    new_sd  = {k: v.clone() for k, v in our_sd.items()}
    for tk, tv in adapters.items():
        # try matching by shape to any key in our adapter
        for ok, ov in our_sd.items():
            if tv.shape == ov.shape and ok not in [k for k, _ in
                    [(k2, None) for k2 in new_sd if new_sd[k2] is not our_sd[k2]]]:
                new_sd[ok] = tv
                loaded += 1
                break
        else:
            skipped += 1
    encoder.adapters["hindi"].load_state_dict(new_sd)
    print(f"  Hindi warm-start: loaded {loaded}, skipped {skipped} tensors")


def warmstart_biaffine_from_hindi(parser_bho: BiaffineHeads, parser_hi: BiaffineHeads, hindi_ckpt: Path):
    if not hindi_ckpt.exists():
        print("  [WARN] Hindi checkpoint not found — skipping biaffine warm-start")
        return
    state = torch.load(str(hindi_ckpt), map_location="cpu")
    hindi_tensors: list = []
    def _collect(obj):
        if isinstance(obj, torch.Tensor):
            hindi_tensors.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values(): _collect(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj: _collect(v)
    _collect(state)
    print(f"  Found {len(hindi_tensors)} tensors in Hindi checkpoint")
    total_copied = 0
    for parser in [parser_bho, parser_hi]:
        our_sd = parser.state_dict()
        new_sd = {k: v.clone() for k, v in our_sd.items()}
        used = set()
        copied = 0
        for ok, ov in our_sd.items():
            if "label.biaffine" in ok:
                continue
            for i, hv in enumerate(hindi_tensors):
                if i not in used and hv.shape == ov.shape:
                    new_sd[ok] = hv.clone()
                    used.add(i)
                    copied += 1
                    break
        parser.load_state_dict(new_sd)
        total_copied += copied
    print(f"  Biaffine warm-start: copied {total_copied} tensors from Hindi tagger")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation on BHTB test set
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(encoder: ParallelEncoder, parser: BiaffineHeads,
             vocab: RelVocab, test_sents: List[Sentence],
             device: torch.device) -> Tuple[float, float]:
    encoder.eval(); parser.eval()
    pred_heads_all, pred_rels_all = [], []

    with torch.no_grad():
        for sent in test_sents:
            if not sent.tokens:
                pred_heads_all.append([])
                pred_rels_all.append([])
                continue
            words = sent.words()
            H_bho = encoder.encode_one("bhojpuri", words)   # [1, n, 768]
            H_bho = H_bho.to(device)
            arc_s, lbl_s = parser(H_bho)
            mask = torch.ones(1, len(words), dtype=torch.bool, device=device)
            ph, pr = BiaffineHeads.predict(arc_s, lbl_s, mask)
            pred_heads_all.append(ph[0].cpu().tolist())
            pred_rels_all.append([vocab.decode(r) for r in pr[0].cpu().tolist()])

    uas, las = uas_las(test_sents, pred_heads_all, pred_rels_all)
    encoder.train(); parser.train()
    return uas, las


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="System G — Exact alignment-supervised joint Hindi+Bhojpuri training"
    )
    ap.add_argument("--epochs",       type=int,   default=40)
    ap.add_argument("--device",       type=str,   default="cuda",
                    help="cuda | mps | cpu")
    ap.add_argument("--lambda_hi",    type=float, default=0.3,
                    help="Weight on Hindi parsing loss")
    ap.add_argument("--lambda_align", type=float, default=0.5,
                    help="Weight on exact alignment loss")
    ap.add_argument("--lr",           type=float, default=5e-5)
    ap.add_argument("--patience",     type=int,   default=7,
                    help="Early stopping patience on dev LAS")
    ap.add_argument("--dev_ratio",    type=float, default=0.1,
                    help="Fraction of data to use as dev set")
    ap.add_argument("--test_ratio",   type=float, default=0.1,
                    help="Fraction of data to use as internal test set")
    ap.add_argument("--seed",         type=int,   default=42)
    args = ap.parse_args()

    # Device selection
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU")
        args.device = "cpu"
    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("\n========================================")
    print(" System G — Exact Alignment Joint Training")
    print(" hindi_matched + bhojpuri_matched_transferred")
    print("========================================")
    print(f"  Device       : {device}")
    print(f"  Epochs       : {args.epochs}")
    print(f"  λ_hi         : {args.lambda_hi}")
    print(f"  λ_align      : {args.lambda_align}")
    print(f"  LR           : {args.lr}")

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[1] Loading parallel data …")
    hi_sents  = read_conllu(PROF_HI)
    bho_sents = read_conllu(PROF_BHO)
    assert len(hi_sents) == len(bho_sents), \
        f"Mismatch: {len(hi_sents)} Hindi vs {len(bho_sents)} Bhojpuri sentences"
    print(f"  Parallel pairs : {len(hi_sents):,}")

    test_sents = read_conllu(BHTB_TEST)
    print(f"  BHTB test      : {len(test_sents):,} sentences")

    # Train/dev/test split (80/10/10)
    n_total = len(hi_sents)
    n_test  = max(1, int(n_total * args.test_ratio))
    n_dev   = max(1, int(n_total * args.dev_ratio))
    n_train = n_total - n_dev - n_test
    indices   = list(range(n_total))
    train_idx = indices[:n_train]
    dev_idx   = indices[n_train:n_train + n_dev]
    test_idx  = indices[n_train + n_dev:]
    print(f"  Train pairs    : {len(train_idx):,}  ({100*n_train/n_total:.0f}%)")
    print(f"  Dev pairs      : {len(dev_idx):,}  ({100*n_dev/n_total:.0f}%)")
    print(f"  Test pairs     : {len(test_idx):,}  ({100*n_test/n_total:.0f}%)")

    # ── Build relation vocabulary ──────────────────────────────────────────────
    print("\n[2] Building relation vocabulary …")
    vocab = build_vocab(hi_sents, bho_sents)
    n_rels = len(vocab)

    # ── Build model ───────────────────────────────────────────────────────────
    print("\n[3] Building model …")
    encoder = ParallelEncoder(
        model_name      = XLM_R_LOCAL,
        adapter_dim     = 64,
        adapter_dropout = 0.1,
        freeze_xlmr     = True,
    )
    # Move adapters to device
    encoder.adapters.to(device)

    # Biaffine parsing head — one shared head for Bhojpuri
    parser_bho = BiaffineHeads(
        hidden_dim    = 768,
        arc_mlp_dim   = 500,
        label_mlp_dim = 100,
        n_rels        = n_rels,
        mlp_dropout   = 0.33,
    ).to(device)

    # Separate biaffine head for Hindi (regularization)
    parser_hi = BiaffineHeads(
        hidden_dim    = 768,
        arc_mlp_dim   = 500,
        label_mlp_dim = 100,
        n_rels        = n_rels,
        mlp_dropout   = 0.33,
    ).to(device)

    # Warm-start Hindi adapter from Trankit checkpoint
    hindi_ckpt = CHECKPT_DIR / "trankit_hindi/xlm-roberta-base/hindi/hindi.tagger.mdl"
    print("\n[4] Warm-starting from Hindi checkpoint …")
    warmstart_hindi_adapter(encoder, hindi_ckpt)
    warmstart_biaffine_from_hindi(parser_bho, parser_hi, hindi_ckpt)

    # ── Optimizer (only adapter + biaffine params — XLM-R is frozen) ──────────
    trainable = (
        list(encoder.adapters.parameters()) +
        list(parser_bho.parameters()) +
        list(parser_hi.parameters())
    )
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    # ── Pre-compute frozen XLM-R embeddings (load from disk if available) ───────
    print("\n[5] Pre-computing XLM-R embeddings (done once, reused every epoch) …")
    _cache_path = ROOT_DIR / "cache" / "xlmr_cache.pt"
    if _cache_path.exists():
        print(f"  Loading cached embeddings from {_cache_path} …")
        _c = torch.load(str(_cache_path), map_location="cpu")
        cache_hi, cache_bho = _c["hi"], _c["bho"]
        print(f"  Loaded — hi:{len(cache_hi)}, bho:{len(cache_bho)}")
    else:
        cache_hi  = encoder.precompute_xlmr([s.words() for s in hi_sents],  desc="Hindi")
        cache_bho = encoder.precompute_xlmr([s.words() for s in bho_sents], desc="Bhojpuri")
        _cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"hi": cache_hi, "bho": cache_bho}, str(_cache_path))
        print(f"  Saved cache → {_cache_path}")

    # ── Training loop ─────────────────────────────────────────────────────────
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    best_las    = 0.0
    no_improve  = 0
    print(f"\n[6] Training for up to {args.epochs} epochs …\n")

    for epoch in range(1, args.epochs + 1):
        encoder.train(); parser_bho.train(); parser_hi.train()
        random.shuffle(train_idx)

        total_loss = l_bho_sum = l_hi_sum = l_align_sum = 0.0
        n_sents = 0

        for idx in train_idx:
            hi_sent  = hi_sents[idx]
            bho_sent = bho_sents[idx]

            if not hi_sent.tokens or not bho_sent.tokens:
                continue

            # Encode using cached XLM-R + trainable adapters
            H_hi  = encoder.encode_one("hindi",    hi_sent.words(),  cache_hi[idx])
            H_bho = encoder.encode_one("bhojpuri", bho_sent.words(), cache_bho[idx])
            H_hi  = H_hi.to(device)
            H_bho = H_bho.to(device)

            # ── Bhojpuri parsing loss ─────────────────────────────────────────
            arc_bho, lbl_bho = parser_bho(H_bho)
            hi_heads, hi_rels   = sentence_to_tensors(hi_sent,  vocab, device)
            bho_heads, bho_rels = sentence_to_tensors(bho_sent, vocab, device)
            l_bho = parse_loss(arc_bho, lbl_bho, bho_heads, bho_rels)

            # ── Hindi parsing loss (regularization) ───────────────────────────
            arc_hi, lbl_hi = parser_hi(H_hi)
            l_hi = parse_loss(arc_hi, lbl_hi, hi_heads, hi_rels)

            # ── Exact positional alignment loss ───────────────────────────────
            n_aligned = min(H_hi.size(1), H_bho.size(1))
            l_align = alignment_loss(H_hi, H_bho, n_aligned)

            # ── Combined loss ─────────────────────────────────────────────────
            loss = l_bho + args.lambda_hi * l_hi + args.lambda_align * l_align

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
            optimizer.step()

            total_loss  += loss.item()
            l_bho_sum   += l_bho.item()
            l_hi_sum    += l_hi.item()
            l_align_sum += l_align.item()
            n_sents     += 1

        avg = total_loss / max(n_sents, 1)
        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Loss={avg:.4f} "
              f"(bho={l_bho_sum/max(n_sents,1):.4f} "
              f"hi={l_hi_sum/max(n_sents,1):.4f} "
              f"align={l_align_sum/max(n_sents,1):.4f})", end="  ")

        # ── Dev evaluation (fast, cached — same domain as training) ─────────────
        encoder.eval(); parser_bho.eval()
        dev_ph_all, dev_pr_all = [], []
        with torch.no_grad():
            for i in dev_idx:
                s = bho_sents[i]
                if not s.tokens:
                    dev_ph_all.append([]); dev_pr_all.append([]); continue
                H = encoder.encode_one("bhojpuri", s.words(), cache_bho[i]).to(device)
                arc_s, lbl_s = parser_bho(H)
                mask = torch.ones(1, len(s.tokens), dtype=torch.bool, device=device)
                ph, pr = BiaffineHeads.predict(arc_s, lbl_s, mask)
                dev_ph_all.append(ph[0].cpu().tolist())
                dev_pr_all.append([vocab.decode(r) for r in pr[0].cpu().tolist()])
        dev_bho_sents = [bho_sents[i] for i in dev_idx]
        dev_uas, dev_las = uas_las(dev_bho_sents, dev_ph_all, dev_pr_all)
        encoder.train(); parser_bho.train()
        print(f"| Dev UAS={dev_uas*100:.2f}% LAS={dev_las*100:.2f}%", end="")

        if dev_las > best_las:
            best_las = dev_las
            no_improve = 0
            torch.save({
                "epoch":       epoch,
                "best_las":    best_las,
                "vocab":       vocab,
                "encoder":     encoder.state_dict(),
                "parser_bho":  parser_bho.state_dict(),
                "parser_hi":   parser_hi.state_dict(),
                "args":        vars(args),
            }, CKPT_PATH)
            print(" ← BEST saved", end="")
        else:
            no_improve += 1
            print(f" (no improve {no_improve}/{args.patience})", end="")
        print()

        if no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch} (patience={args.patience})")
            break

    # ── Final evaluation — load BEST checkpoint first ─────────────────────────
    print(f"\n  Loading best checkpoint for final evaluation …")
    ckpt = torch.load(str(CKPT_PATH), map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    parser_bho.load_state_dict(ckpt["parser_bho"])

    print(f"\n{'='*60}")
    print(f"  System G — Final Results  (best checkpoint, epoch {ckpt['epoch']})")
    print(f"{'='*60}")
    print(f"  Best Dev LAS (prof. data) : {best_las*100:.2f}%")

    # Internal test set (10% of prof's data — never seen during training)
    int_test_sents = [bho_sents[i] for i in test_idx]
    int_test_uas, int_test_las = evaluate(encoder, parser_bho, vocab, int_test_sents, device)

    # BHTB external test (never seen during training)
    final_bhtb_uas, final_bhtb_las = evaluate(encoder, parser_bho, vocab, test_sents, device)

    print(f"  {'Test Set':<35} {'UAS':>7} {'LAS':>7}")
    print(f"  {'─'*51}")
    print(f"  {'Internal (10% prof data)':<35} {int_test_uas*100:>6.2f}% {int_test_las*100:>6.2f}%  ({len(test_idx):,} sents)")
    print(f"  {'BHTB (external gold)':<35} {final_bhtb_uas*100:>6.2f}% {final_bhtb_las*100:>6.2f}%")
    print(f"  {'─'*51}")
    print(f"  Checkpoint                : {CKPT_PATH}")
    print(f"\n  Baseline (System A zero-shot): UAS 53.48% / LAS 34.84%")
    print(f"{'='*60}")

    # ── Save results to file ──────────────────────────────────────────────────
    import datetime
    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"system_g_{ts}.txt"
    with open(results_file, "w") as rf:
        rf.write(f"System G — Exact Alignment Joint Training\n")
        rf.write(f"==========================================\n")
        rf.write(f"Date          : {datetime.datetime.now()}\n")
        rf.write(f"Best epoch    : {ckpt['epoch']}\n")
        rf.write(f"Best Dev LAS  : {best_las*100:.2f}%\n")
        rf.write(f"Epochs        : {args.epochs}\n")
        rf.write(f"lambda_hi     : {args.lambda_hi}\n")
        rf.write(f"lambda_align  : {args.lambda_align}\n")
        rf.write(f"lr            : {args.lr}\n")
        rf.write(f"dev_ratio     : {args.dev_ratio}\n")
        rf.write(f"test_ratio    : {args.test_ratio}\n\n")
        rf.write(f"{'Test Set':<35} {'UAS':>8} {'LAS':>8}\n")
        rf.write(f"{'─'*53}\n")
        rf.write(f"{'Internal (10% prof data)':<35} {int_test_uas*100:>7.2f}% {int_test_las*100:>7.2f}%  ({len(test_idx)} sents)\n")
        rf.write(f"{'BHTB (external gold)':<35} {final_bhtb_uas*100:>7.2f}% {final_bhtb_las*100:>7.2f}%\n")
        rf.write(f"{'─'*53}\n")
        rf.write(f"Baseline (System A zero-shot): UAS 53.48% / LAS 34.84%\n")
    print(f"\n  Results saved → {results_file}")


if __name__ == "__main__":
    main()

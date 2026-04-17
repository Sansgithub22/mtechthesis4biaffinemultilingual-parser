#!/usr/bin/env python3
# train_system_j.py
# System J — Relation-Specific Contrastive Cross-lingual Transfer (RS-SACT)
#
# MOTIVATION:
#   System H aligns Hindi and Bhojpuri representations using:
#     - Cosine alignment  : pulls ALL matched token positions together
#     - KL distillation   : matches arc distributions
#     - CTS               : uses Hindi gold heads for Bhojpuri
#
#   But cosine alignment treats every token the same regardless of its
#   grammatical role. A token labelled "nsubj" (subject) and a token
#   labelled "obj" (object) get pulled together just because they happen
#   to be at the same position in a parallel sentence.
#
#   System J introduces RELATION-SPECIFIC CONTRASTIVE LOSS (novel):
#     - Anchor   : each Hindi token representation
#     - Positives: Bhojpuri tokens with the SAME dependency relation
#     - Negatives: Bhojpuri tokens with DIFFERENT dependency relations
#
#   This forces the model to cluster representations by SYNTACTIC ROLE
#   across languages — not just by position. A Hindi "nsubj" token is
#   pulled towards ALL Bhojpuri "nsubj" tokens in the sentence, and
#   pushed away from "obj", "nmod", "obl" tokens.
#
# CONTRASTIVE LOSS FORMULATION (InfoNCE with multiple positives):
#   For Hindi token i with relation r_i:
#     L_i = log(sum_j exp(sim(z_hi_i, z_bho_j) / tau))
#           - mean_{j: r_j=r_i} sim(z_hi_i, z_bho_j) / tau
#
#   This is equivalent to: log-partition - mean positive similarity
#   Temperature tau=0.07 (standard for token-level contrastive learning)
#
# FIVE LOSSES (four from H + one new):
#   L = L_bho + λ_hi·L_hi + λ_cos·L_cos + λ_arc·L_arc_kl
#       + λ_cts·L_cts + λ_contrast·L_contrast
#
# Checkpoint: checkpoints/system_j/system_j.pt
#
# Usage:
#   python3 train_system_j.py [--epochs 40] [--device cuda]
#                              [--lambda_contrast 0.5] [--tau 0.07]

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Tuple

from config import CHECKPT_DIR, DATA_DIR, XLM_R_LOCAL
from utils.conllu_utils import read_conllu, filter_single_root, Sentence
from utils.metrics import uas_las

from model.parallel_encoder import ParallelEncoder
from model.biaffine_heads   import BiaffineHeads
from model.cross_lingual_parser import RelVocab


ROOT_DIR  = Path(__file__).parent
PROF_BHO  = ROOT_DIR / "bhojpuri_matched_transferred.conllu"
PROF_HI   = ROOT_DIR / "hindi_matched.conllu"
BHTB_TEST = DATA_DIR / "bhojpuri" / "bho_bhtb-ud-test.conllu"
CKPT_DIR  = CHECKPT_DIR / "system_j"
CKPT_PATH = CKPT_DIR / "system_j.pt"

CONTENT_POS = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'NUM'}


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers  (identical to System H)
# ─────────────────────────────────────────────────────────────────────────────

def build_vocab(hi_sents, bho_sents, test_sents):
    vocab = RelVocab()
    for s in hi_sents + bho_sents + test_sents:
        for t in s.tokens:
            vocab.add(t.deprel)
    print(f"  Relation vocab: {len(vocab)} labels")
    return vocab


def sentence_to_tensors(sent, vocab, device):
    heads = torch.tensor([t.head for t in sent.tokens], dtype=torch.long, device=device)
    rels  = torch.tensor([vocab.encode(t.deprel) for t in sent.tokens],
                          dtype=torch.long, device=device)
    return heads, rels


# ─────────────────────────────────────────────────────────────────────────────
# Parsing loss  (identical to System H)
# ─────────────────────────────────────────────────────────────────────────────

def parse_loss(arc_scores, lbl_scores, gold_heads, gold_rels):
    n = gold_heads.size(0)
    arc_loss = F.cross_entropy(arc_scores[0], gold_heads)
    idx = gold_heads.view(n, 1, 1).expand(n, 1, lbl_scores.size(-1))
    lbl_at_gold = lbl_scores[0].gather(1, idx).squeeze(1)
    lbl_loss = F.cross_entropy(lbl_at_gold, gold_rels)
    return arc_loss + lbl_loss


# ─────────────────────────────────────────────────────────────────────────────
# System H losses  (identical to System H)
# ─────────────────────────────────────────────────────────────────────────────

def system_h_losses(H_hi, H_bho, arc_hi, arc_bho, hi_sent, bho_sent, device):
    # 1. Content-word cosine alignment
    n = min(H_hi.size(1), H_bho.size(1))
    weights = torch.tensor(
        [1.5 if i < len(bho_sent.tokens) and bho_sent.tokens[i].upos in CONTENT_POS
         else 0.3 for i in range(n)],
        dtype=torch.float, device=device,
    )
    weights = weights / weights.sum()
    cos_sim  = F.cosine_similarity(H_bho[0, :n], H_hi[0, :n].detach(), dim=-1)
    l_cosine = (weights * (1.0 - cos_sim)).sum()

    # 2. Arc KL distillation (Hindi teacher → Bhojpuri student)
    nr = min(arc_hi.size(1), arc_bho.size(1))
    nh = min(arc_hi.size(2), arc_bho.size(2))
    p_hi  = F.softmax    (arc_hi [0, :nr, :nh].detach(), dim=-1)
    p_bho = F.log_softmax(arc_bho[0, :nr, :nh],          dim=-1)
    l_kl  = F.kl_div(p_bho, p_hi, reduction='batchmean')

    # 3. Cross-lingual Tree Supervision (CTS)
    n_tok   = min(len(hi_sent.tokens), arc_bho.size(1))
    hi_heads = torch.tensor(
        [hi_sent.tokens[i].head for i in range(n_tok)],
        dtype=torch.long, device=device,
    ).clamp(0, arc_bho.size(2) - 1)
    l_cts = F.cross_entropy(arc_bho[0, :n_tok], hi_heads)

    return l_cosine, l_kl, l_cts


# ─────────────────────────────────────────────────────────────────────────────
# [NEW] Relation-Specific Contrastive Loss  (System J novel contribution)
# ─────────────────────────────────────────────────────────────────────────────

def relation_contrastive_loss(
    H_hi:  torch.Tensor,    # [1, n_hi, 768]  Hindi adapter output
    H_bho: torch.Tensor,    # [1, n_bho, 768] Bhojpuri adapter output
    hi_rels: torch.Tensor,  # [n_hi]           relation label indices
    tau: float = 0.07,
) -> torch.Tensor:
    """
    Cross-lingual relation-specific contrastive alignment.

    For each Hindi token i (anchor), treat all Bhojpuri tokens j that
    share the same dependency relation as POSITIVES and all others as
    NEGATIVES.  L2-normalised cosine similarity with temperature tau.

    Loss per anchor i:
        L_i = log Z_i  -  mean_{j: rel_j == rel_i}  sim(z_hi_i, z_bho_j) / tau
    where  Z_i = sum_j exp(sim(z_hi_i, z_bho_j) / tau)

    Returns scalar loss averaged over all valid anchors.
    Tokens with no positive (unique relation in the sentence) are skipped.
    """
    n = min(H_hi.size(1), H_bho.size(1))
    if n < 2:
        return torch.tensor(0.0, device=H_hi.device, requires_grad=True)

    # L2-normalise token representations
    z_hi  = F.normalize(H_hi [0, :n], dim=-1)   # [n, 768]
    z_bho = F.normalize(H_bho[0, :n].detach(), dim=-1)  # [n, 768] — stop grad on target

    rels = hi_rels[:n]   # [n] — same relation labels for both languages

    # Similarity matrix: Hindi tokens × Bhojpuri tokens
    sim = torch.matmul(z_hi, z_bho.T) / tau   # [n, n]

    # Positive mask [n, n]: True where Bhojpuri token j has same relation as Hindi token i
    pos_mask = (rels.unsqueeze(0) == rels.unsqueeze(1))   # [n, n]

    # Log partition (denominator) — sum over all Bhojpuri tokens per Hindi anchor
    log_Z = torch.logsumexp(sim, dim=1)   # [n]

    # Mean positive similarity — skip anchors with no positives
    pos_counts = pos_mask.float().sum(dim=1)    # [n]
    valid      = pos_counts > 0                  # [n] bool

    if valid.sum() == 0:
        return torch.tensor(0.0, device=H_hi.device, requires_grad=True)

    pos_sim_mean = (sim * pos_mask.float()).sum(dim=1)[valid] / pos_counts[valid]  # [k]

    # InfoNCE: log_partition − mean_positive_sim
    loss = (log_Z[valid] - pos_sim_mean).mean()
    return loss


# ─────────────────────────────────────────────────────────────────────────────
# Warm-start  (identical to System H)
# ─────────────────────────────────────────────────────────────────────────────

def warmstart_hindi_adapter(encoder, hindi_ckpt):
    if not hindi_ckpt.exists():
        print(f"  [WARN] Hindi checkpoint not found: {hindi_ckpt} — skipping")
        return
    state    = torch.load(str(hindi_ckpt), map_location="cpu")
    adapters = state.get("adapters", {})
    our_sd   = encoder.adapters["hindi"].state_dict()
    new_sd   = {k: v.clone() for k, v in our_sd.items()}
    loaded   = 0
    for tv in adapters.values():
        for ok, ov in our_sd.items():
            if tv.shape == ov.shape:
                new_sd[ok] = tv; loaded += 1; break
    encoder.adapters["hindi"].load_state_dict(new_sd)
    bho_sd  = encoder.adapters["bhojpuri"].state_dict()
    new_bho = {k: new_sd[k] if k in new_sd else v for k, v in bho_sd.items()}
    encoder.adapters["bhojpuri"].load_state_dict(new_bho)
    print(f"  Hindi warm-start: {loaded} tensors → copied to Bhojpuri adapter")


def warmstart_biaffine_from_hindi(parser_bho, parser_hi, hindi_ckpt):
    if not hindi_ckpt.exists():
        print("  [WARN] Hindi checkpoint not found — skipping biaffine warm-start")
        return
    state = torch.load(str(hindi_ckpt), map_location="cpu")
    hindi_tensors = []
    def _collect(obj):
        if isinstance(obj, torch.Tensor): hindi_tensors.append(obj)
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
        used = set(); copied = 0
        for ok, ov in our_sd.items():
            if "label.biaffine" in ok: continue
            for i, hv in enumerate(hindi_tensors):
                if i not in used and hv.shape == ov.shape:
                    new_sd[ok] = hv.clone(); used.add(i); copied += 1; break
        parser.load_state_dict(new_sd)
        total_copied += copied
    print(f"  Biaffine warm-start: copied {total_copied} tensors")


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation  (identical to System H)
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(encoder, parser, vocab, test_sents, device):
    encoder.eval(); parser.eval()
    pred_heads_all, pred_rels_all = [], []
    with torch.no_grad():
        for sent in test_sents:
            if not sent.tokens:
                pred_heads_all.append([]); pred_rels_all.append([]); continue
            H = encoder.encode_one("bhojpuri", sent.words()).to(device)
            arc_s, lbl_s = parser(H)
            mask = torch.ones(1, len(sent.tokens), dtype=torch.bool, device=device)
            ph, pr = BiaffineHeads.predict(arc_s, lbl_s, mask)
            pred_heads_all.append(ph[0].cpu().tolist())
            pred_rels_all.append([vocab.decode(r) for r in pr[0].cpu().tolist()])
    uas, las = uas_las(test_sents, pred_heads_all, pred_rels_all)
    encoder.train(); parser.train()
    return uas, las


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="System J — Relation-Specific Contrastive SACT (RS-SACT)"
    )
    ap.add_argument("--epochs",           type=int,   default=40)
    ap.add_argument("--device",           type=str,   default="cuda")
    ap.add_argument("--lambda_hi",        type=float, default=0.5)
    ap.add_argument("--lambda_cosine",    type=float, default=0.4)
    ap.add_argument("--lambda_arc",       type=float, default=2.0)
    ap.add_argument("--lambda_cts",       type=float, default=0.2)
    ap.add_argument("--lambda_contrast",  type=float, default=0.5,
                    help="Weight on relation-specific contrastive loss (new in J)")
    ap.add_argument("--tau",              type=float, default=0.07,
                    help="Temperature for contrastive loss")
    ap.add_argument("--lr",               type=float, default=5e-5)
    ap.add_argument("--patience",         type=int,   default=10)
    ap.add_argument("--warmup_epochs",    type=int,   default=2,
                    help="Delay KL + contrastive losses by N epochs")
    ap.add_argument("--filter_single_root", action="store_true", default=False,
                    help="Keep only single-root sentences (default: off = use all 30K)")
    ap.add_argument("--dev_ratio",        type=float, default=0.1)
    ap.add_argument("--test_ratio",       type=float, default=0.1,
                    help="Fraction of data to use as internal test set")
    ap.add_argument("--seed",             type=int,   default=42)
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU"); args.device = "cpu"
    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    print("\n" + "="*64)
    print(" System J — Relation-Specific Contrastive SACT (RS-SACT)")
    print(" Novel contributions over System H:")
    print("   1. Content-word cosine alignment     (from H)")
    print("   2. Arc distribution KL distillation  (from H)")
    print("   3. Cross-lingual Tree Supervision    (from H)")
    print("   4. [NEW] Relation-specific contrastive loss")
    print("="*64)
    print(f"  Device         : {device}")
    print(f"  Epochs         : {args.epochs}")
    print(f"  λ_hi           : {args.lambda_hi}")
    print(f"  λ_cosine       : {args.lambda_cosine}")
    print(f"  λ_arc_kl       : {args.lambda_arc}")
    print(f"  λ_cts          : {args.lambda_cts}")
    print(f"  λ_contrast     : {args.lambda_contrast}  (tau={args.tau})")
    print(f"  Warmup epochs  : {args.warmup_epochs}")
    print(f"  LR             : {args.lr}")

    # ── Data ──────────────────────────────────────────────────────────────────
    print("\n[1] Loading parallel data …")
    hi_sents   = read_conllu(PROF_HI)
    bho_sents  = read_conllu(PROF_BHO)
    test_sents = read_conllu(BHTB_TEST)
    assert len(hi_sents) == len(bho_sents)
    print(f"  Parallel pairs : {len(hi_sents):,}")
    print(f"  BHTB test      : {len(test_sents):,}")

    if args.filter_single_root:
        good_idx = filter_single_root(bho_sents)
        print(f"  Single-root (well-formed): {len(good_idx):,} / {len(bho_sents):,}  [--filter_single_root]")
    else:
        good_idx = list(range(len(bho_sents)))
        print(f"  Using all sentences (unfiltered): {len(good_idx):,}  [default: Comp 1 regime]")

    n_total   = len(good_idx)
    n_test    = max(1, int(n_total * args.test_ratio))
    n_dev     = max(1, int(n_total * args.dev_ratio))
    n_train   = n_total - n_dev - n_test
    train_idx = good_idx[:n_train]
    dev_idx   = good_idx[n_train:n_train + n_dev]
    test_idx  = good_idx[n_train + n_dev:]
    dev_bho   = [bho_sents[i] for i in dev_idx]
    print(f"  Train : {n_train:,} ({100*n_train/n_total:.0f}%) | Dev : {n_dev:,} ({100*n_dev/n_total:.0f}%) | Test : {n_test:,} ({100*n_test/n_total:.0f}%)")

    # ── Vocabulary ────────────────────────────────────────────────────────────
    print("\n[2] Building vocabulary …")
    vocab  = build_vocab(hi_sents, bho_sents, test_sents)
    n_rels = len(vocab)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[3] Building model …")
    encoder = ParallelEncoder(
        model_name=XLM_R_LOCAL, adapter_dim=64,
        adapter_dropout=0.1, freeze_xlmr=True,
    )
    encoder.adapters.to(device)
    parser_bho = BiaffineHeads(768, 500, 100, n_rels, 0.33).to(device)
    parser_hi  = BiaffineHeads(768, 500, 100, n_rels, 0.33).to(device)

    # ── Warm-start ────────────────────────────────────────────────────────────
    hindi_ckpt = CHECKPT_DIR / "trankit_hindi/xlm-roberta-base/hindi/hindi.tagger.mdl"
    print("\n[4] Warm-starting from Hindi checkpoint …")
    warmstart_hindi_adapter(encoder, hindi_ckpt)
    warmstart_biaffine_from_hindi(parser_bho, parser_hi, hindi_ckpt)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    trainable = (list(encoder.adapters.parameters()) +
                 list(parser_bho.parameters()) + list(parser_hi.parameters()))
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    print(f"  Trainable params: {sum(p.numel() for p in trainable):,}")

    # ── Load XLM-R cache ──────────────────────────────────────────────────────
    print("\n[5] Loading XLM-R embedding cache …")
    _cache_path = ROOT_DIR / "cache" / "xlmr_cache.pt"
    if _cache_path.exists():
        _c = torch.load(str(_cache_path), map_location="cpu")
        cache_hi, cache_bho = _c["hi"], _c["bho"]
        print(f"  Loaded — hi:{len(cache_hi)}, bho:{len(cache_bho)}")
    else:
        print("  Cache not found — computing now (this takes ~20 h on CPU) …")
        cache_hi  = encoder.precompute_xlmr([s.words() for s in hi_sents],  desc="Hindi")
        cache_bho = encoder.precompute_xlmr([s.words() for s in bho_sents], desc="Bhojpuri")
        _cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"hi": cache_hi, "bho": cache_bho}, str(_cache_path))
        print(f"  Saved cache → {_cache_path}")

    # ── Training loop ─────────────────────────────────────────────────────────
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_las   = 0.0
    no_improve = 0
    print(f"\n[6] Training for up to {args.epochs} epochs …\n")

    for epoch in range(1, args.epochs + 1):
        encoder.train(); parser_bho.train(); parser_hi.train()
        random.shuffle(train_idx)

        total_loss = l_bho_s = l_hi_s = l_cos_s = l_kl_s = l_cts_s = l_con_s = 0.0
        n_sents = 0

        # Warmup: delay KL distillation and contrastive loss
        warmup_done = epoch > args.warmup_epochs

        for idx in train_idx:
            hi_s  = hi_sents[idx]
            bho_s = bho_sents[idx]
            if not hi_s.tokens or not bho_s.tokens:
                continue

            # Encode using cached XLM-R + trainable adapter
            H_hi  = encoder.encode_one("hindi",    hi_s.words(),  cache_hi[idx]).to(device)
            H_bho = encoder.encode_one("bhojpuri", bho_s.words(), cache_bho[idx]).to(device)

            # Parsing losses
            arc_bho, lbl_bho = parser_bho(H_bho)
            arc_hi,  lbl_hi  = parser_hi (H_hi)
            bho_h, bho_r     = sentence_to_tensors(bho_s, vocab, device)
            hi_h,  hi_r      = sentence_to_tensors(hi_s,  vocab, device)
            l_bho = parse_loss(arc_bho, lbl_bho, bho_h, bho_r)
            l_hi  = parse_loss(arc_hi,  lbl_hi,  hi_h,  hi_r)

            # System H losses
            l_cosine, l_arc_kl, l_cts = system_h_losses(
                H_hi, H_bho, arc_hi, arc_bho, hi_s, bho_s, device)

            # [NEW] Relation-specific contrastive loss
            l_contrast = relation_contrastive_loss(
                H_hi, H_bho, hi_r, tau=args.tau
            ) if warmup_done else torch.tensor(0.0, device=device)

            kl_weight       = args.lambda_arc     if warmup_done else 0.0
            contrast_weight = args.lambda_contrast if warmup_done else 0.0

            loss = (l_bho
                    + args.lambda_hi     * l_hi
                    + args.lambda_cosine * l_cosine
                    + kl_weight          * l_arc_kl
                    + args.lambda_cts    * l_cts
                    + contrast_weight    * l_contrast)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, max_norm=5.0)
            optimizer.step()

            total_loss += loss.item()
            l_bho_s += l_bho.item(); l_hi_s  += l_hi.item()
            l_cos_s += l_cosine.item(); l_kl_s += l_arc_kl.item()
            l_cts_s += l_cts.item(); l_con_s += l_contrast.item()
            n_sents += 1

        N = max(n_sents, 1)

        # ── Dev evaluation (fast — cached embeddings) ─────────────────────────
        encoder.eval(); parser_bho.eval()
        dev_ph_all, dev_pr_all = [], []
        with torch.no_grad():
            for i, s in zip(dev_idx, dev_bho):
                if not s.tokens:
                    dev_ph_all.append([]); dev_pr_all.append([]); continue
                H = encoder.encode_one("bhojpuri", s.words(), cache_bho[i]).to(device)
                arc_s, lbl_s = parser_bho(H)
                mask = torch.ones(1, len(s.tokens), dtype=torch.bool, device=device)
                ph, pr = BiaffineHeads.predict(arc_s, lbl_s, mask)
                dev_ph_all.append(ph[0].cpu().tolist())
                dev_pr_all.append([vocab.decode(r) for r in pr[0].cpu().tolist()])
        dev_uas, dev_las = uas_las(dev_bho, dev_ph_all, dev_pr_all)
        encoder.train(); parser_bho.train()

        improved = dev_las > best_las
        if improved:
            best_las = dev_las; no_improve = 0
            torch.save({
                "epoch":      epoch,
                "best_las":   best_las,
                "vocab":      vocab,
                "encoder":    encoder.state_dict(),
                "parser_bho": parser_bho.state_dict(),
                "parser_hi":  parser_hi.state_dict(),
                "args":       vars(args),
            }, CKPT_PATH)
        else:
            no_improve += 1

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"bho={l_bho_s/N:.3f} hi={l_hi_s/N:.3f} "
              f"cos={l_cos_s/N:.3f} kl={l_kl_s/N:.3f} "
              f"cts={l_cts_s/N:.3f} con={l_con_s/N:.3f} "
              f"| Dev UAS={dev_uas*100:.2f}% LAS={dev_las*100:.2f}%"
              + (" ← BEST" if improved else ""))

        if no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # ── Final summary — load BEST checkpoint first ────────────────────────────
    print(f"\n  Loading best checkpoint for final evaluation …")
    ckpt = torch.load(str(CKPT_PATH), map_location=device)
    encoder.load_state_dict(ckpt["encoder"])
    parser_bho.load_state_dict(ckpt["parser_bho"])

    print(f"\n{'='*60}")
    print(f"  System J — Final Results  (best checkpoint, epoch {ckpt['epoch']})")
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
    results_file = results_dir / f"system_j_{ts}.txt"
    with open(results_file, "w") as rf:
        rf.write(f"System J — Relation-Specific Contrastive SACT (RS-SACT)\n")
        rf.write(f"======================================================\n")
        rf.write(f"Date          : {datetime.datetime.now()}\n")
        rf.write(f"Best epoch    : {ckpt['epoch']}\n")
        rf.write(f"Best Dev LAS  : {best_las*100:.2f}%\n")
        rf.write(f"Epochs        : {args.epochs}\n")
        rf.write(f"lambda_hi     : {args.lambda_hi}\n")
        rf.write(f"lambda_cosine : {args.lambda_cosine}\n")
        rf.write(f"lambda_arc    : {args.lambda_arc}\n")
        rf.write(f"lambda_cts    : {args.lambda_cts}\n")
        rf.write(f"lambda_contrast: {args.lambda_contrast}\n")
        rf.write(f"tau           : {args.tau}\n")
        rf.write(f"lr            : {args.lr}\n")
        rf.write(f"warmup_epochs : {args.warmup_epochs}\n")
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

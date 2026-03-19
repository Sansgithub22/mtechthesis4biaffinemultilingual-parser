#!/usr/bin/env python3
# train_system_e.py
# System E — Cross-lingual Attention Parser with Trankit Backbone
#
# ARCHITECTURE (Novel Innovation):
#
#   Hindi sentence ─── Trankit Hindi XLM-R ───────────────────── H_hi [1,n,768]
#                                                                    │
#   Bhojpuri sent. ─── Trankit Bho XLM-R ──── H_bho [1,m,768]      │
#                                                  │     ←──── CrossSentenceAttention
#                                                  │              (Bhojpuri queries Hindi)
#                                                  ▼
#                                           H_cross [1,m,768]  (attended Hindi features)
#                                                  │
#                                           CrossLingualLayer
#                                           cat([H_bho, H_cross]) → Linear(2d→d) + LN
#                                                  ▼
#                                           H_fused [1,m,768]
#                                                  │
#                                         BiaffineHeads (arc + label)
#                                                  │
#                                           Bhojpuri parse prediction
#
# TRAINING SIGNAL (three losses):
#   L_bho   = arc + label CE on Bhojpuri selective treebank annotations
#   L_hi    = arc + label CE on Hindi HDTB annotations (bidirectional consistency)
#   L_align = MSE between aligned (H_hi, H_bho) token pairs (SimAlign supervision)
#   L_total = L_bho + λ_hi * L_hi + λ_align * L_align
#
# WHY THIS BEATS ZERO-SHOT (System A):
#   System A:  Hindi model applied directly to Bhojpuri — zero adaptation.
#   System E:  Cross-lingual attention lets Bhojpuri tokens EXPLICITLY read
#              the parallel Hindi sentence's structure at both train and test time.
#              The model learns to use Hindi structural guidance to resolve
#              ambiguities that Bhojpuri alone cannot (due to low-resource noise).
#
# TRANKIT INTEGRATION:
#   - XLM-R backbone: same xlm-roberta-base as Trankit uses (frozen)
#   - Language adapters: Pfeiffer-style (same architecture as Trankit)
#   - Warm-start: Hindi adapter initialized from Trankit Hindi checkpoint
#     (layers whose shapes match are copied; mismatches skip with warning)
#   - Bhojpuri adapter: initialized from Trankit Bhojpuri warm-start checkpoint
#
# USAGE:
#   python3 train_system_e.py [--epochs 40] [--device mps|cpu|cuda]
#   python3 train_system_e.py --eval_only  (evaluate saved checkpoint on BHTB)

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from config import CFG, DATA_DIR, CHECKPT_DIR, XLM_R_LOCAL
from utils.conllu_utils  import read_conllu, write_conllu, Sentence
from utils.metrics       import uas_las, print_metrics
from data.word_alignment  import str_to_alignment

from model.parallel_encoder         import ParallelEncoder
from model.cross_sentence_attention import CrossSentenceAttention
from model.cross_lingual_layer      import CrossLingualLayer
from model.biaffine_heads           import BiaffineHeads
from model.cross_lingual_parser     import CrossLingualParser, RelVocab


# ─────────────────────────────────────────────────────────────────────────────
# Trankit warm-start: extract compatible weights from a Trankit checkpoint
# ─────────────────────────────────────────────────────────────────────────────
def _load_trankit_checkpoint(ckpt_path: Path) -> Optional[dict]:
    """Load a Trankit checkpoint; return the 'adapters' dict or None."""
    if not ckpt_path.exists():
        return None
    state = torch.load(str(ckpt_path), map_location="cpu")
    return state.get("adapters", None)


def warmstart_from_trankit(model: CrossLingualParser, lang: str, ckpt_path: Path):
    """
    Copy as many weights as possible from a Trankit checkpoint into the
    corresponding language adapter in our ParallelEncoder.

    Trankit's checkpoint stores the combined state of its _embedding_layers
    and _tagger under the 'adapters' key.  We try to match every tensor
    whose name ends in our adapter's parameter names by shape.

    Returns: (n_loaded, n_skipped)
    """
    adapters_dict = _load_trankit_checkpoint(ckpt_path)
    if adapters_dict is None:
        print(f"  [WARN] Trankit checkpoint not found: {ckpt_path}  — skipping warm-start for {lang}")
        return 0, 0

    our_adapter = model.encoder.adapters[lang]
    our_sd      = our_adapter.state_dict()

    # Build a size → name map for our adapter parameters
    size_to_our_key: Dict[Tuple, str] = {}
    for k, v in our_sd.items():
        size_to_our_key[tuple(v.shape)] = k

    n_loaded  = 0
    n_skipped = 0
    new_sd    = {k: v.clone() for k, v in our_sd.items()}

    # Walk Trankit keys; try to match by parameter shape
    for tk, tv in adapters_dict.items():
        shape = tuple(tv.shape)
        if shape in size_to_our_key:
            our_key = size_to_our_key[shape]
            new_sd[our_key] = tv.clone()
            n_loaded += 1

    our_adapter.load_state_dict(new_sd)
    print(f"  [{lang}] warm-start from Trankit: {n_loaded} tensors loaded, "
          f"{len(our_sd) - n_loaded} left at random init")
    return n_loaded, len(our_sd) - n_loaded


# ─────────────────────────────────────────────────────────────────────────────
# Vocab helpers
# ─────────────────────────────────────────────────────────────────────────────
def build_vocab(*paths: Path) -> RelVocab:
    vocab = RelVocab()
    for p in paths:
        if p.exists():
            for sent in read_conllu(p):
                for tok in sent.tokens:
                    vocab.add(tok.deprel)
    print(f"  Relation vocabulary: {len(vocab)} labels")
    return vocab


# ─────────────────────────────────────────────────────────────────────────────
# Parallel dataset
# ─────────────────────────────────────────────────────────────────────────────
class ParallelSample:
    __slots__ = ("hi_sent", "bho_sent", "alignment")

    def __init__(self, hi_sent: Sentence, bho_sent: Sentence, alignment: set):
        self.hi_sent   = hi_sent
        self.bho_sent  = bho_sent
        self.alignment = alignment


def load_parallel(hi_conllu: Path, bho_conllu: Path,
                  align_txt: Path, max_sents: int = 0) -> List[ParallelSample]:
    hi_sents  = read_conllu(hi_conllu)
    bho_sents = read_conllu(bho_conllu)
    align_lines = (align_txt.read_text(encoding="utf-8").splitlines()
                   if align_txt.exists() else [])

    n = min(len(hi_sents), len(bho_sents))
    if align_lines:
        n = min(n, len(align_lines))
    if max_sents:
        n = min(n, max_sents)

    samples = []
    for i in range(n):
        align_str = align_lines[i] if i < len(align_lines) else ""
        alignment = str_to_alignment(align_str)
        samples.append(ParallelSample(hi_sents[i], bho_sents[i], alignment))
    return samples


def sent_to_tensors(sent: Sentence, rel_vocab: RelVocab,
                    device: torch.device):
    """Returns (gold_heads [1,n], gold_rels [1,n], mask [1,n])."""
    heads = torch.tensor(sent.heads(), dtype=torch.long, device=device).unsqueeze(0)
    rels  = torch.tensor(
        [rel_vocab.encode(r) for r in sent.deprels()],
        dtype=torch.long, device=device,
    ).unsqueeze(0)
    mask = heads.new_ones(1, len(sent.tokens), dtype=torch.bool)
    return heads, rels, mask


# ─────────────────────────────────────────────────────────────────────────────
# System E model  (thin shell around CrossLingualParser)
# ─────────────────────────────────────────────────────────────────────────────
class SystemE(nn.Module):
    """
    System E = CrossLingualParser warm-started from Trankit checkpoints.

    At TRAINING time:  paired (Hindi, Bhojpuri) sentences.
    At TEST time:      single Bhojpuri sentence — encode it through BOTH
                       adapters: Bhojpuri adapter → H_bho, Hindi adapter →
                       H_hi.  Because XLM-R is shared and multilingual,
                       the Hindi adapter produces a "Hindi-structural
                       perspective" of the Bhojpuri text, which the
                       cross-attention uses as real structural guidance
                       (same as System A but enriched by fine-tuning).
    """

    def __init__(self, rel_vocab: RelVocab):
        super().__init__()
        self.rel_vocab = rel_vocab

        d = 768   # XLM-R base hidden dim

        # Shared frozen XLM-R + two language adapters (Hindi + Bhojpuri)
        self.encoder = ParallelEncoder(
            model_name      = XLM_R_LOCAL,
            adapter_dim     = CFG.encoder.adapter_dim,
            adapter_dropout = CFG.encoder.adapter_dropout,
            freeze_xlmr     = True,
        )

        # Cross-lingual attention: Bhojpuri queries Hindi
        self.cross_attn  = CrossSentenceAttention(d, n_heads=8, dropout=0.1)

        # Fusion: cat([H_bho, H_cross]) → Linear(2d→d) + LN + residual
        self.cross_layer = CrossLingualLayer(d, dropout=0.1)

        n_rels = len(rel_vocab)

        # Two biaffine heads:
        #   hindi_parser   — trained on HDTB gold annotations (regulariser)
        #   bhojpuri_parser— trained on selective-projected annotations (main)
        self.hindi_parser    = BiaffineHeads(d, 500, 100, n_rels, 0.33)
        self.bhojpuri_parser = BiaffineHeads(d, 500, 100, n_rels, 0.33)

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward_train(self, hi_cache, bho_cache, device):
        """
        Use pre-computed XLM-R caches.
        Returns dict with arc/label scores for both languages + representations.
        """
        H_hi  = self.encoder.encode_one("hindi",    [], cached_xlmr=hi_cache)
        H_bho = self.encoder.encode_one("bhojpuri", [], cached_xlmr=bho_cache)

        H_cross, attn = self.cross_attn(H_bho, H_hi)
        H_fused       = self.cross_layer(H_bho, H_cross)

        arc_hi,  lbl_hi  = self.hindi_parser(H_hi)
        arc_bho, lbl_bho = self.bhojpuri_parser(H_fused)

        return {
            "arc_hi": arc_hi, "lbl_hi": lbl_hi,
            "arc_bho": arc_bho, "lbl_bho": lbl_bho,
            "H_hi": H_hi, "H_bho": H_bho,
        }

    def forward_test(self, bho_words: List[str], device: torch.device):
        """
        Test-time: no parallel Hindi sentence available.

        Encode the Bhojpuri test sentence through BOTH language adapters:
          H_bho = Bhojpuri adapter(XLM-R(bho_words))  — Bhojpuri perspective
          H_hi  = Hindi adapter(XLM-R(bho_words))      — Hindi structural perspective

        Because XLM-R is multilingual and Bhojpuri ≈ Hindi, the Hindi adapter
        produces Hindi-like structural features for the Bhojpuri words.
        The cross-attention then has a real structural signal to attend to —
        identical to how System A uses the Hindi model at zero-shot inference.
        The difference: our Bhojpuri adapter + cross-lingual fusion layer have
        been fine-tuned to combine these two views for better Bhojpuri parsing.
        """
        H_bho = self.encoder._encode_words(bho_words, "bhojpuri")
        H_hi  = self.encoder._encode_words(bho_words, "hindi")   # Hindi lens on same words

        # Cross-attention: Bhojpuri queries Hindi-structural perspective
        H_cross, _ = self.cross_attn(H_bho, H_hi)
        H_fused    = self.cross_layer(H_bho, H_cross)

        arc_bho, lbl_bho = self.bhojpuri_parser(H_fused)
        return arc_bho, lbl_bho

    # ── Loss ─────────────────────────────────────────────────────────────────
    @staticmethod
    def _parse_loss(arc_scores, label_scores, gold_heads, gold_rels, mask):
        B, n, n_plus1 = arc_scores.shape
        active   = mask.view(-1)
        arc_flat = arc_scores.view(B * n, n_plus1)
        head_flat = gold_heads.view(-1)
        arc_loss  = F.cross_entropy(arc_flat[active], head_flat[active])

        head_idx    = gold_heads.unsqueeze(-1).unsqueeze(-1)
        head_idx    = head_idx.expand(B, n, 1, label_scores.size(-1))
        lbl_at_head = label_scores.gather(2, head_idx).squeeze(2)
        lbl_flat    = lbl_at_head.view(B * n, -1)
        rel_flat    = gold_rels.view(-1)
        lbl_loss    = F.cross_entropy(lbl_flat[active], rel_flat[active])
        return arc_loss + lbl_loss

    def compute_loss(self, out, gold_hi_heads, gold_hi_rels, hi_mask,
                     gold_bho_heads, gold_bho_rels, bho_mask,
                     alignments, lambda_hi=0.3, lambda_align=0.1):
        L_bho = self._parse_loss(
            out["arc_bho"], out["lbl_bho"],
            gold_bho_heads, gold_bho_rels, bho_mask,
        )
        L_hi = self._parse_loss(
            out["arc_hi"], out["lbl_hi"],
            gold_hi_heads, gold_hi_rels, hi_mask,
        )
        L_align = torch.tensor(0.0, device=out["H_hi"].device)
        if alignments:
            H_hi_flat  = out["H_hi"].squeeze(0)
            H_bho_flat = out["H_bho"].squeeze(0)
            src_idx = [s for s, t in alignments if s < H_hi_flat.size(0) and t < H_bho_flat.size(0)]
            tgt_idx = [t for s, t in alignments if s < H_hi_flat.size(0) and t < H_bho_flat.size(0)]
            k = min(len(src_idx), len(tgt_idx))
            if k > 0:
                h_src = H_hi_flat[src_idx[:k]]
                h_tgt = H_bho_flat[tgt_idx[:k]]
                L_align = F.mse_loss(
                    F.normalize(h_tgt, dim=-1),
                    F.normalize(h_src, dim=-1).detach(),
                )
        L_total = L_bho + lambda_hi * L_hi + lambda_align * L_align
        return L_total, L_bho, L_hi, L_align

    def count_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────
def train_epoch(model, optimizer, samples, hi_cache, bho_cache,
                rel_vocab, device, lambda_hi, lambda_align):
    """
    Mixed training strategy — eliminates train/test mismatch:

      50% of steps (PARALLEL mode):  H_hi = Hindi adapter on actual Hindi words
                                      H_bho = Bhojpuri adapter on Bhojpuri words
                                      → model learns to leverage real Hindi structure

      50% of steps (SELF mode):      H_hi = Hindi adapter on Bhojpuri words
                                      H_bho = Bhojpuri adapter on Bhojpuri words
                                      → matches test-time exactly (no Hindi available)

    This trains the model to work in BOTH scenarios so at test time it is not
    seeing an out-of-distribution input for the cross-attention.
    """
    model.train()
    model.encoder.xlmr.eval()   # backbone always frozen

    indices = list(range(len(samples)))
    random.shuffle(indices)

    totals  = {"loss": 0.0, "L_bho": 0.0, "L_hi": 0.0, "L_align": 0.0}
    n_valid = 0

    for idx in indices:
        samp = samples[idx]
        if len(samp.hi_sent.tokens) < 2 or len(samp.bho_sent.tokens) < 2:
            continue

        c_hi  = hi_cache[idx]  if hi_cache  else None
        c_bho = bho_cache[idx] if bho_cache else None
        if c_hi is None or c_bho is None:
            continue

        optimizer.zero_grad()

        # Mixed training: 50% parallel mode, 50% self mode
        use_parallel = random.random() < 0.5

        if use_parallel:
            # PARALLEL mode: use actual Hindi sentence as H_hi context
            out = model.forward_train(c_hi, c_bho, device)
            gold_hi_sent = samp.hi_sent
        else:
            # SELF mode: encode Bhojpuri through Hindi adapter → matches test time
            H_bho = model.encoder.encode_one("bhojpuri", [], cached_xlmr=c_bho)
            H_hi  = model.encoder.encode_one("hindi",    [], cached_xlmr=c_bho)
            H_cross, _ = model.cross_attn(H_bho, H_hi)
            H_fused    = model.cross_layer(H_bho, H_cross)
            arc_hi,  lbl_hi  = model.hindi_parser(H_hi)
            arc_bho, lbl_bho = model.bhojpuri_parser(H_fused)
            out = {"arc_hi": arc_hi, "lbl_hi": lbl_hi,
                   "arc_bho": arc_bho, "lbl_bho": lbl_bho,
                   "H_hi": H_hi, "H_bho": H_bho}
            # In self mode use Bhojpuri labels for the Hindi head too (same sentence)
            gold_hi_sent = samp.bho_sent

        # Guard: cache length must match gold annotation length
        if (out["H_hi"].size(1)  != len(gold_hi_sent.tokens) or
            out["H_bho"].size(1) != len(samp.bho_sent.tokens)):
            continue

        hi_heads,  hi_rels,  hi_mask  = sent_to_tensors(gold_hi_sent,  rel_vocab, device)
        bho_heads, bho_rels, bho_mask = sent_to_tensors(samp.bho_sent, rel_vocab, device)

        L_total, L_bho, L_hi, L_align = model.compute_loss(
            out,
            gold_hi_heads=hi_heads, gold_hi_rels=hi_rels, hi_mask=hi_mask,
            gold_bho_heads=bho_heads, gold_bho_rels=bho_rels, bho_mask=bho_mask,
            alignments=list(samp.alignment),
            lambda_hi=lambda_hi, lambda_align=lambda_align,
        )

        L_total.backward()
        nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 5.0
        )
        optimizer.step()

        totals["loss"]    += L_total.item()
        totals["L_bho"]   += L_bho.item()
        totals["L_hi"]    += L_hi.item()
        totals["L_align"] += L_align.item()
        n_valid += 1

    n = max(n_valid, 1)
    return {k: v / n for k, v in totals.items()}, n_valid


@torch.no_grad()
def evaluate_on_bhtb(model, test_sents: List[Sentence], device: torch.device,
                     desc: str = "test") -> Tuple[float, float]:
    """
    Evaluate System E on BHTB (real Bhojpuri test).
    No Hindi parallel sentence available — Bhojpuri is used as self-context.
    """
    model.eval()
    model.encoder.xlmr.eval()

    pred_heads_all, pred_rels_all = [], []

    for sent in test_sents:
        words = sent.words()
        if not words:
            continue

        arc_bho, lbl_bho = model.forward_test(words, device)

        m    = len(words)
        mask = arc_bho.new_ones(1, m, dtype=torch.bool)
        pred_heads, pred_rels = BiaffineHeads.predict(arc_bho, lbl_bho, mask)
        pred_heads_all.append(pred_heads[0].cpu().tolist())
        pred_rels_all.append([model.rel_vocab.decode(r)
                               for r in pred_rels[0].cpu().tolist()])

    uas, las = uas_las(test_sents[:len(pred_heads_all)],
                       pred_heads_all, pred_rels_all)
    print_metrics(desc, uas, las)
    return uas, las


@torch.no_grad()
def evaluate_on_dev(model, dev_samples, hi_dev_cache, bho_dev_cache,
                    rel_vocab, device) -> Tuple[float, float]:
    """Evaluate on synthetic dev using parallel context (more informative signal)."""
    model.eval()
    model.encoder.xlmr.eval()

    pred_heads_all, pred_rels_all = [], []

    for i, samp in enumerate(dev_samples):
        bho_words = samp.bho_sent.words()
        if not bho_words:
            continue

        c_hi  = hi_dev_cache[i]  if i < len(hi_dev_cache)  else None
        c_bho = bho_dev_cache[i] if i < len(bho_dev_cache) else None
        if c_hi is None or c_bho is None:
            continue

        H_hi  = model.encoder.encode_one("hindi",    [], cached_xlmr=c_hi)
        H_bho = model.encoder.encode_one("bhojpuri", [], cached_xlmr=c_bho)

        if H_hi.size(1) != len(samp.hi_sent.tokens) or H_bho.size(1) != len(bho_words):
            continue

        H_cross, _ = model.cross_attn(H_bho, H_hi)
        H_fused    = model.cross_layer(H_bho, H_cross)
        arc_bho, lbl_bho = model.bhojpuri_parser(H_fused)

        m    = len(bho_words)
        mask = arc_bho.new_ones(1, m, dtype=torch.bool)
        ph, pr = BiaffineHeads.predict(arc_bho, lbl_bho, mask)
        pred_heads_all.append(ph[0].cpu().tolist())
        pred_rels_all.append([model.rel_vocab.decode(r)
                               for r in pr[0].cpu().tolist()])

    gold_sents = [s.bho_sent for s in dev_samples[:len(pred_heads_all)]]
    return uas_las(gold_sents, pred_heads_all, pred_rels_all)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="System E: Cross-lingual Attention Parser with Trankit Backbone"
    )
    ap.add_argument("--epochs",        type=int,   default=40)
    ap.add_argument("--max_train",     type=int,   default=0,
                    help="Limit training sentences (0 = all)")
    ap.add_argument("--lambda_hi",     type=float, default=0.3,
                    help="Weight on Hindi consistency loss")
    ap.add_argument("--lambda_align",  type=float, default=0.1,
                    help="Weight on alignment regularisation loss")
    ap.add_argument("--lr",            type=float, default=2e-4)
    ap.add_argument("--device",        default="cpu",
                    help="Device: cpu (recommended — MPS crashes on MultiheadAttention)")
    ap.add_argument("--eval_only",     action="store_true",
                    help="Skip training; evaluate saved checkpoint on BHTB")
    ap.add_argument("--skip_warmstart", action="store_true",
                    help="Skip loading Trankit warm-start weights")
    args = ap.parse_args()

    random.seed(CFG.train.seed)
    torch.manual_seed(CFG.train.seed)
    device = torch.device(args.device)

    CFG.make_dirs()
    save_dir = CHECKPT_DIR / "system_e"
    save_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 66)
    print("  System E — Cross-lingual Attention Parser (Trankit Backbone)")
    print("=" * 66)

    # ── Paths ─────────────────────────────────────────────────────────────────
    hi_train_conllu  = DATA_DIR / "hindi"     / "hi_hdtb-ud-train.conllu"
    hi_dev_conllu    = DATA_DIR / "hindi"     / "hi_hdtb-ud-dev.conllu"
    bho_train_conllu = DATA_DIR / "synthetic" / "bho_selective_train.conllu"
    bho_dev_conllu   = DATA_DIR / "synthetic" / "bho_selective_dev.conllu"
    align_train      = DATA_DIR / "synthetic" / "alignments_train.txt"
    align_dev        = DATA_DIR / "synthetic" / "alignments_dev.txt"
    bhtb_test        = DATA_DIR / "bhojpuri"  / "bho_bhtb-ud-test.conllu"

    # Trankit checkpoints for warm-start
    trankit_hi_ckpt  = (CHECKPT_DIR / "trankit_hindi"
                        / "xlm-roberta-base" / "hindi" / "hindi.tagger.mdl")
    trankit_bho_ckpt = (CHECKPT_DIR / "trankit_bho_warmstart"
                        / "xlm-roberta-base" / "bhojpuri_warmstart"
                        / "bhojpuri_warmstart.tagger.mdl")

    for p in (hi_train_conllu, bho_train_conllu):
        if not p.exists():
            print(f"  [ERROR] Missing required file: {p}")
            print("  Run data preparation scripts first.")
            return

    # ── Build vocabulary ──────────────────────────────────────────────────────
    print("\n[1] Building relation vocabulary …")
    rel_vocab = build_vocab(hi_train_conllu, bho_train_conllu,
                            hi_dev_conllu, bho_dev_conllu)
    CFG.biaffine.n_rels = len(rel_vocab)

    # ── Build model ───────────────────────────────────────────────────────────
    print("\n[2] Building System E model …")
    model = SystemE(rel_vocab)
    print(f"  Trainable params: {model.count_trainable():,}")

    # ── EVAL ONLY ─────────────────────────────────────────────────────────────
    ckpt_file = save_dir / "best.pt"

    if args.eval_only:
        if not ckpt_file.exists():
            print(f"  [ERROR] No checkpoint found: {ckpt_file}")
            print("  Run training first (remove --eval_only).")
            return
        print(f"\n[Eval] Loading checkpoint: {ckpt_file}")
        ckpt = torch.load(str(ckpt_file), map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        model.to(device)

        test_sents = read_conllu(bhtb_test)
        print(f"\n  BHTB test set: {len(test_sents):,} Bhojpuri sentences")
        uas, las = evaluate_on_bhtb(model, test_sents, device, desc="BHTB/System-E")
        print(f"\n  System E — UAS: {uas*100:.2f}%   LAS: {las*100:.2f}%")
        print(f"  (Baseline System A — UAS: 53.48%   LAS: 34.84%)")
        delta = (las - 0.3484) * 100
        print(f"  Delta LAS vs System A (zero-shot): {delta:+.2f}%")
        return

    # ── Trankit warm-start ────────────────────────────────────────────────────
    if not args.skip_warmstart:
        print("\n[3] Warm-starting adapters from Trankit checkpoints …")
        warmstart_from_trankit(model, "hindi",    trankit_hi_ckpt)
        warmstart_from_trankit(model, "bhojpuri", trankit_bho_ckpt)
    else:
        print("\n[3] Skipping Trankit warm-start (--skip_warmstart)")

    # Move trainable modules to device; XLM-R stays on CPU (frozen, 278M params)
    for lang in ("hindi", "bhojpuri"):
        model.encoder.adapters[lang].to(device)
    model.cross_attn.to(device)
    model.cross_layer.to(device)
    model.hindi_parser.to(device)
    model.bhojpuri_parser.to(device)

    # ── Load data ─────────────────────────────────────────────────────────────
    print("\n[4] Loading parallel training data …")
    train_samples = load_parallel(hi_train_conllu, bho_train_conllu,
                                  align_train, args.max_train)
    print(f"  Train: {len(train_samples):,} parallel sentence pairs")
    print(f"  Bhojpuri treebank: bho_selective_train.conllu  (clean, cycle-free)")

    dev_samples = []
    if bho_dev_conllu.exists():
        dev_samples = load_parallel(hi_dev_conllu, bho_dev_conllu, align_dev, 500)
        print(f"  Dev:   {len(dev_samples):,} parallel sentence pairs")

    test_sents = []
    if bhtb_test.exists():
        test_sents = read_conllu(bhtb_test)
        print(f"  Test:  {len(test_sents):,} real Bhojpuri sentences (BHTB gold)")

    # ── Pre-compute XLM-R cache ───────────────────────────────────────────────
    print("\n[5] Pre-computing frozen XLM-R embeddings (done ONCE, reused) …")
    enc = model.encoder

    train_hi_cache  = enc.precompute_xlmr(
        [s.hi_sent.words()  for s in train_samples], desc="train-hi")
    train_bho_cache = enc.precompute_xlmr(
        [s.bho_sent.words() for s in train_samples], desc="train-bho")

    dev_hi_cache = dev_bho_cache = []
    if dev_samples:
        dev_hi_cache  = enc.precompute_xlmr(
            [s.hi_sent.words()  for s in dev_samples], desc="dev-hi")
        dev_bho_cache = enc.precompute_xlmr(
            [s.bho_sent.words() for s in dev_samples], desc="dev-bho")

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.1, total_iters=args.epochs,
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n[6] Training — {args.epochs} epochs   "
          f"lr={args.lr}   λ_hi={args.lambda_hi}   λ_align={args.lambda_align}")
    print(f"    Three-loss objective: L_bho + {args.lambda_hi}*L_hi "
          f"+ {args.lambda_align}*L_align\n")

    best_las     = 0.0
    best_bhtb_uas = best_bhtb_las = 0.0
    patience     = 0
    patience_max = 8

    for epoch in range(1, args.epochs + 1):
        avg, n_valid = train_epoch(
            model, optimizer, train_samples,
            train_hi_cache, train_bho_cache,
            rel_vocab, device, args.lambda_hi, args.lambda_align,
        )
        scheduler.step()

        print(f"  Ep {epoch:3d}  loss={avg['loss']:.4f}  "
              f"L_bho={avg['L_bho']:.4f}  L_hi={avg['L_hi']:.4f}  "
              f"L_align={avg['L_align']:.4f}  ({n_valid} valid)")

        # ── Dev evaluation every 2 epochs ──────────────────────────────────
        if dev_samples and (epoch % 2 == 0 or epoch == 1):
            uas_dev, las_dev = evaluate_on_dev(
                model, dev_samples, dev_hi_cache, dev_bho_cache, rel_vocab, device)
            print_metrics(f"  dev/synthetic", uas_dev, las_dev)
            primary_las = las_dev
        else:
            primary_las = best_las   # no change signal this epoch

        # ── BHTB evaluation every 5 epochs ─────────────────────────────────
        if test_sents and (epoch % 5 == 0 or epoch == 1):
            uas_t, las_t = evaluate_on_bhtb(model, test_sents, device,
                                             desc="  BHTB/test")
            if las_t > best_bhtb_las:
                best_bhtb_uas, best_bhtb_las = uas_t, las_t

        # ── Checkpoint ──────────────────────────────────────────────────────
        if primary_las > best_las:
            best_las = primary_las
            patience = 0
            torch.save({
                "epoch":           epoch,
                "best_las":        best_las,
                "model_state":     model.state_dict(),
                "rel_vocab_words": rel_vocab._i2w,
                "args":            vars(args),
            }, ckpt_file)
            print(f"    *** Saved best checkpoint (dev LAS {best_las*100:.2f}%) ***")
        else:
            patience += 1
            if patience >= patience_max:
                print(f"  Early stopping at epoch {epoch} (patience={patience_max})")
                break

    # ── Final BHTB evaluation ─────────────────────────────────────────────────
    print(f"\n{'='*66}")
    print("  [7] Final Evaluation on BHTB Test Set (357 real Bhojpuri sentences)")
    print(f"{'='*66}")

    if ckpt_file.exists():
        print(f"  Loading best checkpoint from epoch …")
        ckpt = torch.load(str(ckpt_file), map_location="cpu")
        model.load_state_dict(ckpt["model_state"])
        print(f"  (saved at epoch {ckpt['epoch']}, dev LAS {ckpt['best_las']*100:.2f}%)")

    if test_sents:
        uas_final, las_final = evaluate_on_bhtb(
            model, test_sents, device, desc="System E / BHTB")
        best_bhtb_uas = max(best_bhtb_uas, uas_final)
        best_bhtb_las = max(best_bhtb_las, las_final)
    else:
        uas_final = las_final = 0.0
        print("  [WARN] BHTB test file not found — skipping final evaluation")

    # ── Comparison table ──────────────────────────────────────────────────────
    print(f"\n{'='*66}")
    print("  RESULTS COMPARISON — All Systems on BHTB (357 sentences)")
    print(f"{'='*66}")
    print(f"  {'System':<50} {'UAS':>7} {'LAS':>7}")
    print(f"  {'─'*64}")
    print(f"  {'[A] Zero-shot   Hindi Trankit → Bhojpuri':<50} {'53.48%':>7} {'34.84%':>7}")
    print(f"  {'[B] Projection-only  (5,000 unfiltered)':<50} {'46.60%':>7} {'29.35%':>7}")
    print(f"  {'[C] Quality-filtered (coverage ≥ 70%)':<50} {'46.08%':>7} {'29.48%':>7}")
    print(f"  {'[D] Warm-start + selective projection':<50} {'51.16%':>7} {'32.96%':>7}")
    if uas_final > 0:
        tag = " ← NEW BEST" if las_final > 0.3484 else ""
        print(f"  {'[E] Cross-lingual Attn + Trankit (NOVEL)':<50} "
              f"{uas_final*100:>6.2f}% {las_final*100:>6.2f}%{tag}")
    print(f"  {'─'*64}")

    if uas_final > 0:
        delta_a  = (las_final - 0.3484) * 100
        delta_d  = (las_final - 0.3296) * 100
        print(f"\n  Innovation gains (LAS delta):")
        print(f"    E vs A (cross-lingual vs zero-shot):      {delta_a:+.2f}%")
        print(f"    E vs D (vs best previous):                {delta_d:+.2f}%")

    print(f"\n  System E checkpoint: {ckpt_file}")
    print(f"\nNext: python3 evaluate_system_e.py  (for detailed per-relation report)")


if __name__ == "__main__":
    main()

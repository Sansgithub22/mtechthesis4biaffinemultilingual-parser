#!/usr/bin/env python3
# train_bilingual.py
# Steps 7-8 — Bilingual fine-tuning with cross-lingual objectives.
#
# Builds on the monolingual checkpoints and fine-tunes the FULL model:
#   • Both adapters (Hindi + Bhojpuri)
#   • Both biaffine heads
#   • Cross-sentence attention
#   • Cross-lingual fusion layer
#
# Loss (Step 8):
#   L_total = L_hi + λ_bho * L_bho + λ_align * L_align
#
# Training setup used in the paper example (Step 8):
#   10 Hindi sentences paired with 8 Bhojpuri sentences
#   Each step trains on one parallel (Hindi, Bhojpuri) sentence pair.
#
# Usage:
#   python3 train_bilingual.py [--epochs 20] [--max_sents 500]

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import random
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Optional, Tuple

from config import CFG, DATA_DIR, CHECKPT_DIR
from utils.conllu_utils  import read_conllu, Sentence
from utils.metrics       import uas_las, print_metrics
from data.word_alignment  import str_to_alignment
from model.parallel_encoder     import ParallelEncoder
from model.cross_lingual_parser import CrossLingualParser, RelVocab
from model.biaffine_heads       import BiaffineHeads


# ─────────────────────────────────────────────────────────────────────────────
# Parallel dataset  (Hindi sentence + Bhojpuri projection + alignment)
# ─────────────────────────────────────────────────────────────────────────────
class ParallelSample:
    """One training example: a Hindi sentence paired with its Bhojpuri translation."""
    __slots__ = ("hi_sent", "bho_sent", "alignment")

    def __init__(self, hi_sent: Sentence, bho_sent: Sentence,
                 alignment: set):
        self.hi_sent   = hi_sent
        self.bho_sent  = bho_sent
        self.alignment = alignment   # set of (src_idx, tgt_idx) 0-based pairs


def load_parallel_data(
    hi_conllu:    Path,
    bho_conllu:   Path,
    align_txt:    Path,
    max_sents:    int = 0,
) -> List[ParallelSample]:
    hi_sents  = read_conllu(hi_conllu)
    bho_sents = read_conllu(bho_conllu)
    align_lines = align_txt.read_text(encoding="utf-8").splitlines() \
        if align_txt.exists() else []

    n = min(len(hi_sents), len(bho_sents), len(align_lines) if align_lines else 99999)
    if max_sents:
        n = min(n, max_sents)

    samples = []
    for i in range(n):
        align_str = align_lines[i] if i < len(align_lines) else ""
        alignment = str_to_alignment(align_str)
        samples.append(ParallelSample(hi_sents[i], bho_sents[i], alignment))
    return samples


# ─────────────────────────────────────────────────────────────────────────────
# Tensor helpers
# ─────────────────────────────────────────────────────────────────────────────
def sent_to_tensors(sent: Sentence, rel_vocab: RelVocab,
                    device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (gold_heads [1,n], gold_rels [1,n], mask [1,n])."""
    heads = torch.tensor(sent.heads(), dtype=torch.long, device=device).unsqueeze(0)
    rels  = torch.tensor(
        [rel_vocab.encode(r) for r in sent.deprels()],
        dtype=torch.long, device=device
    ).unsqueeze(0)
    mask  = heads.new_ones(1, len(sent.tokens), dtype=torch.bool)
    return heads, rels, mask


# ─────────────────────────────────────────────────────────────────────────────
# Load monolingual checkpoints  (warmstart adapters + biaffine heads)
# ─────────────────────────────────────────────────────────────────────────────
def load_mono_checkpoint(
    model:     CrossLingualParser,
    lang:      str,
    ckpt_path: Path,
) -> RelVocab:
    if not ckpt_path.exists():
        print(f"  [WARN] No checkpoint for {lang}: {ckpt_path}")
        return model.rel_vocab

    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.encoder.adapters[lang].load_state_dict(ckpt["adapter"])

    head_attr = "hindi_parser" if lang == "hindi" else "bhojpuri_parser"
    head = getattr(model, head_attr)
    head.load_state_dict(ckpt["head"], strict=False)  # n_rels may differ

    print(f"  Loaded {lang} checkpoint (epoch {ckpt.get('epoch','?')}, "
          f"LAS {ckpt.get('best_las', 0)*100:.2f}%)")
    return model.rel_vocab


# ─────────────────────────────────────────────────────────────────────────────
# Bilingual trainer
# ─────────────────────────────────────────────────────────────────────────────
class BilingualTrainer:

    def __init__(self, model: CrossLingualParser,
                 device: torch.device,
                 lr: float = CFG.train.bi_lr):
        # Move only trainable parts to device; XLM-R backbone stays on CPU
        for lang in ("hindi", "bhojpuri"):
            model.encoder.adapters[lang].to(device)
        model.cross_attn.to(device)
        model.cross_layer.to(device)
        model.hindi_parser.to(device)
        model.bhojpuri_parser.to(device)
        self.model  = model
        self.device = device

        # All trainable parameters (adapters + biaffine + cross-attn + fusion)
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=1e-4,
        )
        self.scheduler = torch.optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0, end_factor=0.1,
            total_iters=CFG.train.bi_epochs,
        )

    def train_epoch(
        self,
        samples:       List[ParallelSample],
        lambda_bho:    float,
        lambda_align:  float,
        hi_cache:      list = None,
        bho_cache:     list = None,
    ) -> dict:
        self.model.encoder.xlmr.eval()   # backbone always frozen
        self.model.train()

        # Shuffle samples AND their corresponding caches together
        indices = list(range(len(samples)))
        random.shuffle(indices)

        totals = {"loss": 0, "loss_hi": 0, "loss_bho": 0, "loss_align": 0}
        n_valid = 0

        for idx in indices:
            samp      = samples[idx]
            hi_words  = samp.hi_sent.words()
            bho_words = samp.bho_sent.words()

            if len(hi_words) < 2 or len(bho_words) < 2:
                continue

            self.optimizer.zero_grad()

            # Use cached XLM-R outputs (indices now correctly aligned)
            c_hi  = hi_cache[idx]  if hi_cache  and idx < len(hi_cache)  else None
            c_bho = bho_cache[idx] if bho_cache and idx < len(bho_cache) else None

            # Forward pass (Steps 3-5)
            H_hi, H_bho = self.model.encoder.encode_pair(
                hi_words, bho_words, cached_hi=c_hi, cached_bho=c_bho)

            # Guard: encoded length must match gold annotation length.
            # A mismatch can occur when the tokenizer truncates sentences that
            # expand beyond 512 subword tokens — skip those samples.
            if H_hi.size(1) != len(hi_words) or H_bho.size(1) != len(bho_words):
                print(f"[SKIP] size mismatch at idx={idx}: "
                      f"H_hi={H_hi.size(1)} vs hi_words={len(hi_words)}, "
                      f"H_bho={H_bho.size(1)} vs bho_words={len(bho_words)}")
                continue

            H_cross, attn = self.model.cross_attn(H_bho, H_hi)
            H_fused       = self.model.cross_layer(H_bho, H_cross)

            arc_hi,  lbl_hi  = self.model.hindi_parser(H_hi)
            arc_bho, lbl_bho = self.model.bhojpuri_parser(H_fused)

            out = {"arc_hi": arc_hi, "lbl_hi": lbl_hi,
                   "arc_bho": arc_bho, "lbl_bho": lbl_bho,
                   "H_hi": H_hi, "H_bho": H_bho}

            # Prepare gold tensors
            hi_heads, hi_rels, hi_mask = sent_to_tensors(
                samp.hi_sent, self.model.rel_vocab, self.device)
            bho_heads, bho_rels, bho_mask = sent_to_tensors(
                samp.bho_sent, self.model.rel_vocab, self.device)

            # Compute loss (Steps 7-8)
            losses = self.model.compute_loss(
                out,
                gold_hi_heads  = hi_heads,
                gold_hi_rels   = hi_rels,
                gold_bho_heads = bho_heads,
                gold_bho_rels  = bho_rels,
                hi_mask        = hi_mask,
                bho_mask       = bho_mask,
                alignments     = list(samp.alignment),
                lambda_bho     = lambda_bho,
                lambda_align   = lambda_align,
            )

            losses["loss"].backward()
            nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                CFG.train.max_grad_norm,
            )
            self.optimizer.step()

            for k in totals:
                totals[k] += losses[k].item()
            n_valid += 1

        self.scheduler.step()
        return {k: v / max(n_valid, 1) for k, v in totals.items()}

    @torch.no_grad()
    def evaluate_bhojpuri(
        self,
        sents_hi:   List[Sentence],
        sents_bho:  List[Sentence],
        hi_cache:   list = None,
        bho_cache:  list = None,
    ) -> Tuple[float, float]:
        """Step 9-style evaluation: parse Bhojpuri using Hindi context."""
        self.model.eval()
        self.model.encoder.xlmr.eval()
        pred_heads_all, pred_rels_all = [], []

        for i, (hi_s, bho_s) in enumerate(zip(sents_hi, sents_bho)):
            hi_words  = hi_s.words()
            bho_words = bho_s.words()
            if not hi_words or not bho_words:
                continue
            c_hi  = hi_cache[i]  if hi_cache  and i < len(hi_cache)  else None
            c_bho = bho_cache[i] if bho_cache and i < len(bho_cache) else None
            H_hi, H_bho   = self.model.encoder.encode_pair(hi_words, bho_words, c_hi, c_bho)
            H_cross, _    = self.model.cross_attn(H_bho, H_hi)
            H_fused       = self.model.cross_layer(H_bho, H_cross)
            arc_bho, lbl_bho = self.model.bhojpuri_parser(H_fused)
            m    = len(bho_words)
            mask = arc_bho.new_ones(1, m, dtype=torch.bool)
            ph, pr = BiaffineHeads.predict(arc_bho, lbl_bho, mask)
            pred_heads_all.append(ph[0].cpu().tolist())
            pred_rels_all.append([self.model.rel_vocab.decode(r)
                                   for r in pr[0].cpu().tolist()])

        return uas_las(sents_bho, pred_heads_all, pred_rels_all)

    @torch.no_grad()
    def evaluate_hindi(self, sents: List[Sentence]) -> Tuple[float, float]:
        self.model.eval()
        pred_heads_all, pred_rels_all = [], []
        for sent in sents:
            heads, rels = self.model.predict_hindi(sent.words())
            pred_heads_all.append(heads)
            pred_rels_all.append(rels)
        return uas_las(sents, pred_heads_all, pred_rels_all)


# ─────────────────────────────────────────────────────────────────────────────
# Build rel_vocab from all available data
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
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(
        description="Steps 7-8: Bilingual fine-tuning with cross-lingual objectives"
    )
    ap.add_argument("--epochs",       type=int,   default=CFG.train.bi_epochs)
    ap.add_argument("--max_sents",    type=int,   default=0)
    ap.add_argument("--lambda_bho",   type=float, default=CFG.train.lambda_bho)
    ap.add_argument("--lambda_align", type=float, default=CFG.train.lambda_align)
    ap.add_argument("--device",       default=CFG.train.device)
    ap.add_argument("--skip_mono",    action="store_true",
                    help="Skip loading monolingual checkpoints (cold start)")
    args = ap.parse_args()

    random.seed(CFG.train.seed)
    torch.manual_seed(CFG.train.seed)
    device = torch.device(args.device)

    CFG.make_dirs()

    print("\n========================================")
    print(" Steps 7-8: Bilingual Fine-tuning")
    print("========================================")

    # ── Build vocabulary ──────────────────────────────────────────────────
    print("\n[Data] Building relation vocabulary …")
    rel_vocab = build_vocab(
        DATA_DIR / "hindi/hi_hdtb-ud-train.conllu",
        DATA_DIR / "synthetic/bho_synthetic_train.conllu",
        DATA_DIR / "hindi/hi_hdtb-ud-dev.conllu",
    )
    CFG.biaffine.n_rels = len(rel_vocab)

    # ── Build model ───────────────────────────────────────────────────────
    print("\n[Model] Building CrossLingualParser …")
    model = CrossLingualParser(rel_vocab, CFG)

    if not args.skip_mono:
        load_mono_checkpoint(model, "hindi",
                             CHECKPT_DIR / "mono_hindi/best.pt")
        load_mono_checkpoint(model, "bhojpuri",
                             CHECKPT_DIR / "mono_bhojpuri/best.pt")

    trainable = model.count_trainable()
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} params "
          f"({trainable/total*100:.1f}%)")

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[Data] Loading parallel data …")

    syn_train = DATA_DIR / "synthetic/bho_synthetic_train.conllu"
    syn_dev   = DATA_DIR / "synthetic/bho_synthetic_dev.conllu"
    alg_train = DATA_DIR / "synthetic/alignments_train.txt"
    alg_dev   = DATA_DIR / "synthetic/alignments_dev.txt"
    hi_train  = DATA_DIR / "hindi/hi_hdtb-ud-train.conllu"
    hi_dev    = DATA_DIR / "hindi/hi_hdtb-ud-dev.conllu"

    if not syn_train.exists():
        print("  Synthetic treebank missing. "
              "Run: python3 data/build_synthetic_treebank.py")
        return

    train_samples = load_parallel_data(hi_train, syn_train, alg_train,
                                       args.max_sents)
    print(f"  Train: {len(train_samples):,} parallel sentence pairs")

    # Dev: evaluate both directions
    hi_dev_sents  = read_conllu(hi_dev)[:500]
    bho_dev_sents = read_conllu(syn_dev)[:500] if syn_dev.exists() else []
    dev_samples   = load_parallel_data(hi_dev, syn_dev, alg_dev, 200) \
        if syn_dev.exists() else []
    print(f"  Dev:   {len(dev_samples):,} parallel sentence pairs")

    # ── Pre-compute XLM-R caches (done once, reused every epoch) ─────────────
    print("\n[Cache] Pre-computing frozen XLM-R embeddings …")
    enc = model.encoder
    train_hi_words  = [s.hi_sent.words()  for s in train_samples]
    train_bho_words = [s.bho_sent.words() for s in train_samples]
    train_hi_cache  = enc.precompute_xlmr(train_hi_words,  desc="bilingual-train-hi")
    train_bho_cache = enc.precompute_xlmr(train_bho_words, desc="bilingual-train-bho")

    dev_hi_cache = dev_bho_cache = None
    if dev_samples:
        dev_hi_cache  = enc.precompute_xlmr([s.hi_sent.words()  for s in dev_samples], "dev-hi")
        dev_bho_cache = enc.precompute_xlmr([s.bho_sent.words() for s in dev_samples], "dev-bho")

    # ── Training loop ─────────────────────────────────────────────────────
    trainer   = BilingualTrainer(model, device, lr=CFG.train.bi_lr)
    save_dir  = CHECKPT_DIR / "bilingual"
    save_dir.mkdir(parents=True, exist_ok=True)
    best_las  = 0.0
    patience  = 0

    print(f"\n[Train] {args.epochs} epochs  "
          f"λ_bho={args.lambda_bho}  λ_align={args.lambda_align}\n")

    for epoch in range(1, args.epochs + 1):
        avg = trainer.train_epoch(train_samples, args.lambda_bho, args.lambda_align,
                                  hi_cache=train_hi_cache, bho_cache=train_bho_cache)

        print(f"  Epoch {epoch:3d}  "
              f"loss={avg['loss']:.4f}  "
              f"L_hi={avg['loss_hi']:.4f}  "
              f"L_bho={avg['loss_bho']:.4f}  "
              f"L_align={avg['loss_align']:.4f}")

        # Evaluate Hindi (should stay stable or improve)
        if hi_dev_sents:
            uas_hi, las_hi = trainer.evaluate_hindi(hi_dev_sents)
            print_metrics("dev/hindi", uas_hi, las_hi)

        # Evaluate Bhojpuri (primary metric)
        las_bho = 0.0
        if dev_samples:
            dev_hi  = [s.hi_sent  for s in dev_samples]
            dev_bho = [s.bho_sent for s in dev_samples]
            uas_bho, las_bho = trainer.evaluate_bhojpuri(
                dev_hi, dev_bho, dev_hi_cache, dev_bho_cache)
            print_metrics("dev/bhojpuri", uas_bho, las_bho)

        # Checkpoint
        if las_bho > best_las:
            best_las = las_bho
            patience = 0
            torch.save({
                "epoch":          epoch,
                "best_las":       best_las,
                "model_state":    model.state_dict(),
                "rel_vocab_words": rel_vocab._i2w,
            }, save_dir / "best.pt")
            print(f"    ✓ Saved best checkpoint (Bhojpuri dev LAS {best_las*100:.2f}%)")
        else:
            patience += 1
            if patience >= CFG.train.patience:
                print(f"  Early stopping at epoch {epoch}")
                break

    print(f"\nBilingual training complete.")
    print(f"Best Bhojpuri dev LAS: {best_las*100:.2f}%")
    print(f"Checkpoint: {save_dir / 'best.pt'}")
    print("\nNext: python3 evaluate.py   (Step 9 — final evaluation)")


if __name__ == "__main__":
    main()

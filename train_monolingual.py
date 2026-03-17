#!/usr/bin/env python3
# train_monolingual.py
# Step 7 — Monolingual pre-training
#
# Trains the parser independently for each language before bilingual
# fine-tuning.  This gives each language's biaffine head a good starting
# point and ensures the adapters learn language-specific surface patterns.
#
# Two training modes:
#   --lang hindi    : train Hindi parser on HDTB gold annotations
#   --lang bhojpuri : train Bhojpuri parser on synthetic projected treebank
#   --lang both     : train Hindi first, then Bhojpuri (default)
#
# Checkpoints are saved to:
#   checkpoints/mono_hindi/best.pt
#   checkpoints/mono_bhojpuri/best.pt
#
# Usage:
#   python3 train_monolingual.py [--lang both] [--epochs 30] [--max_sents 500]

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import random
import torch
import torch.nn as nn
from pathlib import Path
from typing import List

from config import CFG, DATA_DIR, CHECKPT_DIR
from utils.conllu_utils  import read_conllu, Sentence
from utils.metrics       import uas_las, print_metrics
from model.parallel_encoder  import ParallelEncoder
from model.biaffine_heads    import BiaffineHeads
from model.cross_lingual_parser import RelVocab


# ─────────────────────────────────────────────────────────────────────────────
# Single-language trainer (shared logic for Hindi and Bhojpuri)
# ─────────────────────────────────────────────────────────────────────────────
class MonolingualTrainer:
    """
    Trains a single XLM-R adapter + BiaffineHeads for one language.
    The XLM-R backbone stays frozen; only the adapter and biaffine heads
    are updated (matching the Trankit training protocol).
    """

    def __init__(self, lang: str, rel_vocab: RelVocab,
                 encoder: ParallelEncoder, device: torch.device):
        self.lang      = lang
        self.rel_vocab = rel_vocab
        self.encoder   = encoder      # XLM-R stays on CPU; adapters go to device
        self.device    = device

        # Move only the trainable adapter for this language to device
        encoder.adapters[lang].to(device)

        cfg = CFG.biaffine
        self.head = BiaffineHeads(
            hidden_dim    = encoder.hidden_size,
            arc_mlp_dim   = cfg.arc_mlp_dim,
            label_mlp_dim = cfg.label_mlp_dim,
            n_rels        = len(rel_vocab),
            mlp_dropout   = cfg.mlp_dropout,
        ).to(device)

        # Only adapter for this language + biaffine head are trainable
        params = (
            list(encoder.adapters[lang].parameters()) +
            list(self.head.parameters())
        )
        self.optimizer = torch.optim.Adam(params, lr=CFG.train.mono_lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=3, verbose=True
        )

    def loss_on_cached(
        self,
        sent:       Sentence,
        cached_h:   torch.Tensor,   # [1, n, 768] pre-computed XLM-R output (CPU)
    ) -> torch.Tensor:
        if cached_h is None or not sent.tokens:
            return torch.tensor(0.0, device=self.device)

        # Adapter forward only (XLM-R already done)
        H = self.encoder.encode_one(self.lang, [], cached_xlmr=cached_h)

        arc_s, lbl_s = self.head(H)
        n = H.size(1)
        gold_heads = torch.tensor(sent.heads(), dtype=torch.long,
                                  device=self.device).unsqueeze(0)
        gold_rels  = torch.tensor(
            [self.rel_vocab.encode(r) for r in sent.deprels()],
            dtype=torch.long, device=self.device
        ).unsqueeze(0)
        mask = gold_heads.new_ones(1, n, dtype=torch.bool)

        from model.cross_lingual_parser import CrossLingualParser
        return CrossLingualParser._parse_loss(arc_s, lbl_s, gold_heads, gold_rels, mask)

    def train_epoch(self, sents: List[Sentence],
                    cache: List[torch.Tensor]) -> float:
        self.encoder.xlmr.eval()
        self.encoder.adapters[self.lang].train()
        self.head.train()

        # Shuffle sentences and their cached embeddings together
        paired = list(zip(sents, cache))
        random.shuffle(paired)

        total_loss = 0.0
        for sent, cached_h in paired:
            if len(sent.tokens) < 2 or cached_h is None:
                continue
            self.optimizer.zero_grad()
            loss = self.loss_on_cached(sent, cached_h)
            loss.backward()
            nn.utils.clip_grad_norm_(
                list(self.encoder.adapters[self.lang].parameters()) +
                list(self.head.parameters()),
                CFG.train.max_grad_norm,
            )
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / max(len(sents), 1)

    @torch.no_grad()
    def evaluate(self, sents: List[Sentence],
                 cache: List[torch.Tensor] = None) -> tuple:
        self.encoder.xlmr.eval()
        self.encoder.adapters[self.lang].eval()
        self.head.eval()
        pred_heads_all, pred_rels_all = [], []
        for i, sent in enumerate(sents):
            words = sent.words()
            if not words:
                continue
            cached_h = cache[i] if cache and i < len(cache) else None
            H = self.encoder.encode_one(self.lang, words, cached_xlmr=cached_h)
            arc_s, lbl_s = self.head(H)
            mask = arc_s.new_ones(1, len(words), dtype=torch.bool)
            ph, pr = BiaffineHeads.predict(arc_s, lbl_s, mask)
            pred_heads_all.append(ph[0].cpu().tolist())
            pred_rels_all.append([self.rel_vocab.decode(r)
                                   for r in pr[0].cpu().tolist()])
        return uas_las(sents, pred_heads_all, pred_rels_all)


# ─────────────────────────────────────────────────────────────────────────────
# Train one language
# ─────────────────────────────────────────────────────────────────────────────
def train_language(
    lang:           str,
    train_sents:    List[Sentence],
    dev_sents:      List[Sentence],
    encoder:        ParallelEncoder,
    rel_vocab:      RelVocab,
    device:         torch.device,
    epochs:         int,
    save_dir:       Path,
):
    print(f"\n{'='*60}")
    print(f"  Monolingual training: {lang.upper()}")
    print(f"  train={len(train_sents):,}  dev={len(dev_sents):,}  "
          f"epochs={epochs}")
    print(f"{'='*60}")

    trainer   = MonolingualTrainer(lang, rel_vocab, encoder, device)
    save_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute frozen XLM-R embeddings ONCE — reused every epoch
    train_words = [s.words() for s in train_sents]
    dev_words   = [s.words() for s in dev_sents]
    train_cache = encoder.precompute_xlmr(train_words, desc=f"{lang} train")
    dev_cache   = encoder.precompute_xlmr(dev_words,   desc=f"{lang} dev")

    best_las  = 0.0
    patience  = 0

    for epoch in range(1, epochs + 1):
        avg_loss = trainer.train_epoch(train_sents, train_cache)
        uas, las = trainer.evaluate(dev_sents, dev_cache)
        print(f"  Epoch {epoch:3d}  loss={avg_loss:.4f}  ", end="")
        print_metrics(f"dev/{lang}", uas, las)

        trainer.scheduler.step(las)

        if las > best_las:
            best_las = las
            patience = 0
            torch.save({
                "epoch":    epoch,
                "lang":     lang,
                "best_las": best_las,
                "adapter":  encoder.adapters[lang].state_dict(),
                "head":     trainer.head.state_dict(),
                "rel_vocab_words": trainer.rel_vocab._i2w,
            }, save_dir / "best.pt")
            print(f"    ✓ Saved best checkpoint (LAS {best_las*100:.2f}%)")
        else:
            patience += 1
            if patience >= CFG.train.patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    print(f"\n  Best {lang} dev LAS: {best_las*100:.2f}%")
    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# Build shared vocabulary
# ─────────────────────────────────────────────────────────────────────────────
def build_rel_vocab(sents_list: List[List[Sentence]]) -> RelVocab:
    vocab = RelVocab()
    for sents in sents_list:
        for sent in sents:
            for tok in sent.tokens:
                vocab.add(tok.deprel)
    print(f"  Relation vocabulary: {len(vocab)} labels")
    return vocab


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Step 7: Monolingual pre-training")
    ap.add_argument("--lang",      choices=["hindi", "bhojpuri", "both"],
                    default="both")
    ap.add_argument("--epochs",    type=int, default=CFG.train.mono_epochs)
    ap.add_argument("--max_sents", type=int, default=0,
                    help="Limit training sentences (0 = all)")
    ap.add_argument("--device",    default=CFG.train.device)
    args = ap.parse_args()

    random.seed(CFG.train.seed)
    torch.manual_seed(CFG.train.seed)
    device = torch.device(args.device)

    CFG.make_dirs()

    print("\n========================================")
    print(" Step 7: Monolingual Pre-training")
    print("========================================")

    # ── Load data ─────────────────────────────────────────────────────────
    print("\n[Data] Loading treebanks …")

    hi_train = hi_dev = bho_train = bho_dev = []

    if args.lang in ("hindi", "both"):
        if not DATA_DIR.joinpath("hindi/hi_hdtb-ud-train.conllu").exists():
            print("  Hindi data missing. Run: python3 data/download_ud_data.py")
        else:
            hi_train = read_conllu(DATA_DIR / "hindi/hi_hdtb-ud-train.conllu")
            hi_dev   = read_conllu(DATA_DIR / "hindi/hi_hdtb-ud-dev.conllu")
            if args.max_sents:
                hi_train = hi_train[:args.max_sents]
            print(f"  Hindi:    train={len(hi_train):,}  dev={len(hi_dev):,}")

    if args.lang in ("bhojpuri", "both"):
        syn_train = DATA_DIR / "synthetic/bho_synthetic_train.conllu"
        syn_dev   = DATA_DIR / "synthetic/bho_synthetic_dev.conllu"
        if not syn_train.exists():
            print("  Synthetic Bhojpuri data missing.\n"
                  "  Run: python3 data/build_synthetic_treebank.py")
        else:
            bho_train = read_conllu(syn_train)
            bho_dev   = read_conllu(syn_dev) if syn_dev.exists() else []
            if args.max_sents:
                bho_train = bho_train[:args.max_sents]
            print(f"  Bhojpuri: train={len(bho_train):,}  dev={len(bho_dev):,}")

    # ── Shared vocabulary ─────────────────────────────────────────────────
    rel_vocab = build_rel_vocab([hi_train, hi_dev, bho_train, bho_dev])

    # ── Shared encoder ────────────────────────────────────────────────────
    print("\n[Model] Initialising encoder …")
    encoder = ParallelEncoder(
        model_name      = CFG.encoder.model_name,
        adapter_dim     = CFG.encoder.adapter_dim,
        adapter_dropout = CFG.encoder.adapter_dropout,
        freeze_xlmr     = CFG.encoder.freeze_xlmr,
    )
    trainable = encoder.trainable_params()
    total     = sum(p.numel() for p in encoder.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} params "
          f"({trainable/total*100:.1f}%)")

    # ── Train languages ───────────────────────────────────────────────────
    if args.lang in ("hindi", "both") and hi_train:
        train_language(
            lang         = "hindi",
            train_sents  = hi_train,
            dev_sents    = hi_dev,
            encoder      = encoder,
            rel_vocab    = rel_vocab,
            device       = device,
            epochs       = args.epochs,
            save_dir     = CHECKPT_DIR / "mono_hindi",
        )

    if args.lang in ("bhojpuri", "both") and bho_train:
        train_language(
            lang         = "bhojpuri",
            train_sents  = bho_train,
            dev_sents    = bho_dev if bho_dev else hi_dev[:500],
            encoder      = encoder,
            rel_vocab    = rel_vocab,
            device       = device,
            epochs       = args.epochs,
            save_dir     = CHECKPT_DIR / "mono_bhojpuri",
        )

    print("\nMonolingual pre-training complete.")
    print("Next: python3 train_bilingual.py   (Steps 7-8 bilingual fine-tuning)")


if __name__ == "__main__":
    main()

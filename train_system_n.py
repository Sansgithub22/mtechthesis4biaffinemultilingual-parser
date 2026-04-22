#!/usr/bin/env python3
# train_system_n.py
# System N — MuRIL-SACT + MFEF + Tree-Denoising Curriculum + SWA  [Novel]
#
# TARGET: Dev UAS > 55.5 on professor's data (System H baseline: 55.85 UAS).
#
# STACKED INNOVATIONS:
#   1. MuRIL backbone (google/muril-base-cased) — Indic-pretrained 17-lang model
#      that replaces XLM-R-base. MuRIL's Bhojpuri/Hindi coverage is significantly
#      stronger than XLM-R's, closing the subword-tokenization gap.
#
#   2. Morphological Feature Embedding Fusion (MFEF):
#      UD FEATS (Case, Gender, Number, Person, Tam, Vib, …) carry crucial
#      syntactic signal for Hindi/Bhojpuri. We learn an embedding per
#      (attribute=value) token, mean-pool per token, project to encoder dim,
#      and ADD to encoder output before the biaffine head. Novel application:
#      force matched parallel morph features to produce the same additive
#      signal (via the existing cosine alignment loss downstream).
#      Proj initialised to zero → MFEF starts as a no-op (warm-start safe).
#
#   3. Tree-Denoising Curriculum:
#      The prof corpus has 6,412 single-root / 24,554 multi-root sentences.
#      Train first `curriculum_epochs` on the clean single-root subset
#      (no cycles, no multi-root artefacts), then continue on the full set.
#      This prevents the parser from memorising annotation noise early on.
#
#   4. SACT losses (unchanged from System H):
#      L = L_bho + λ_hi L_hi + λ_cos L_cosine + λ_arc L_arc_kl + λ_cts L_cts
#      — preserves cross-lingual transfer signal from the parallel data.
#
#   5. Stochastic Weight Averaging (SWA):
#      After `swa_start` epoch, maintain running mean of trainable parameters.
#      At the end, load averaged weights and re-evaluate — typically +0.3 UAS
#      "for free" (Izmailov et al. 2018).
#
# Usage:
#   python3 train_system_n.py [--epochs 40] [--device cuda|mps|cpu]
#                             [--backbone muril|xlmr]
#                             [--curriculum_epochs 5] [--swa_start 15]
#                             [--morph_dim 128]
#                             [--lambda_hi 0.5] [--lambda_cosine 0.4]
#                             [--lambda_arc 2.0] [--lambda_cts 0.2]

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import copy
import random
import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import CHECKPT_DIR, DATA_DIR, XLM_R_LOCAL
from utils.conllu_utils import read_conllu, Sentence, filter_single_root
from utils.metrics import uas_las

from model.parallel_encoder import ParallelEncoder
from model.biaffine_heads   import BiaffineHeads
from model.cross_lingual_parser import RelVocab


ROOT_DIR  = Path(__file__).parent
PROF_BHO  = ROOT_DIR / "bhojpuri_matched_transferred.conllu"
PROF_HI   = ROOT_DIR / "hindi_matched.conllu"
BHTB_TEST = DATA_DIR / "bhojpuri" / "bho_bhtb-ud-test.conllu"
CKPT_DIR  = CHECKPT_DIR / "system_n"
CKPT_PATH = CKPT_DIR / "system_n.pt"
CKPT_SWA  = CKPT_DIR / "system_n_swa.pt"

MURIL_NAME = "google/muril-base-cased"
MURIL_HINDI_CKPT = CHECKPT_DIR / "muril_hindi" / "best.pt"     # optional warm-start
XLMR_HINDI_CKPT  = CHECKPT_DIR / "trankit_hindi/xlm-roberta-base/hindi/hindi.tagger.mdl"

CONTENT_POS = {'NOUN', 'VERB', 'ADJ', 'ADV', 'PROPN', 'NUM'}


# ═════════════════════════════════════════════════════════════════════════════
# Morphological Feature Embedding Fusion (MFEF)
# ═════════════════════════════════════════════════════════════════════════════

class MorphVocab:
    """Vocabulary over (attribute=value) morph tokens, e.g. 'Case=Nom'."""
    PAD, UNK = "<pad>", "<unk>"

    def __init__(self):
        self._tok2id: Dict[str, int] = {self.PAD: 0, self.UNK: 1}

    def add_feats_string(self, feats: str):
        if not feats or feats == "_":
            return
        for pair in feats.split("|"):
            pair = pair.strip()
            if pair and pair not in self._tok2id:
                self._tok2id[pair] = len(self._tok2id)

    def encode(self, feats: str) -> List[int]:
        if not feats or feats == "_":
            return []
        return [self._tok2id.get(p.strip(), 1) for p in feats.split("|") if p.strip()]

    def __len__(self):
        return len(self._tok2id)


class MorphEmbedder(nn.Module):
    """
    Embeds UD FEATS into a vector per token, additively fused with encoder output.

    Input to forward:
        morph_ids  : [1, n_words, max_feats]   LongTensor — padded with 0
        morph_mask : [1, n_words, max_feats]   FloatTensor — 1 for real, 0 for pad
    Output:
        H_morph    : [1, n_words, hidden_dim]

    Init: final projection is zero-initialised so MFEF starts as a no-op,
    i.e., warm-starting a MuRIL/XLMR+SACT model is unaffected at epoch 0.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128,
                 hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.proj  = nn.Linear(embed_dim, hidden_dim)
        self.drop  = nn.Dropout(dropout)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.embed.weight[0].zero_()
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, morph_ids: torch.Tensor,
                morph_mask: torch.Tensor) -> torch.Tensor:
        e = self.embed(morph_ids)                              # [1, n, m, ed]
        masked = e * morph_mask.unsqueeze(-1)
        summed = masked.sum(dim=2)                             # [1, n, ed]
        counts = morph_mask.sum(dim=2).clamp(min=1.0).unsqueeze(-1)
        pooled = summed / counts
        return self.drop(self.proj(pooled))                    # [1, n, hidden]


def sent_morph_tensors(sent: Sentence, morph_vocab: MorphVocab,
                       max_feats: int, device: torch.device
                       ) -> Tuple[torch.Tensor, torch.Tensor]:
    per_tok = [morph_vocab.encode(t.feats)[:max_feats] for t in sent.tokens]
    n = len(per_tok)
    m = max(max_feats, 1)
    ids  = torch.zeros(1, n, m, dtype=torch.long,  device=device)
    mask = torch.zeros(1, n, m, dtype=torch.float, device=device)
    for i, lst in enumerate(per_tok):
        for j, v in enumerate(lst):
            ids[0, i, j] = v
            mask[0, i, j] = 1.0
    return ids, mask


# ═════════════════════════════════════════════════════════════════════════════
# Losses
# ═════════════════════════════════════════════════════════════════════════════

def parse_loss(arc_scores: torch.Tensor, lbl_scores: torch.Tensor,
               gold_heads: torch.Tensor, gold_rels: torch.Tensor) -> torch.Tensor:
    n = gold_heads.size(0)
    arc_loss = F.cross_entropy(arc_scores[0], gold_heads)
    idx = gold_heads.view(n, 1, 1).expand(n, 1, lbl_scores.size(-1))
    lbl_at_gold = lbl_scores[0].gather(1, idx).squeeze(1)
    lbl_loss = F.cross_entropy(lbl_at_gold, gold_rels)
    return arc_loss + lbl_loss


def sact_losses(H_hi, H_bho, arc_hi, arc_bho, hi_sent, bho_sent, device):
    """SACT — cosine alignment (content-word weighted) + arc-KL + CTS."""
    # 1. Content-word cosine alignment
    n = min(H_hi.size(1), H_bho.size(1))
    weights = torch.tensor(
        [1.5 if i < len(bho_sent.tokens) and bho_sent.tokens[i].upos in CONTENT_POS
         else 0.3
         for i in range(n)],
        dtype=torch.float, device=device,
    )
    weights = weights / weights.sum()
    cos_sim  = F.cosine_similarity(H_bho[0, :n], H_hi[0, :n].detach(), dim=-1)
    l_cosine = (weights * (1.0 - cos_sim)).sum()

    # 2. Arc-distribution KL distillation
    nr = min(arc_hi.size(1), arc_bho.size(1))
    nh = min(arc_hi.size(2), arc_bho.size(2))
    p_hi_arc  = F.softmax    (arc_hi [0, :nr, :nh].detach(), dim=-1)
    p_bho_arc = F.log_softmax(arc_bho[0, :nr, :nh],          dim=-1)
    l_arc_kl  = F.kl_div(p_bho_arc, p_hi_arc, reduction='batchmean')

    # 3. Cross-lingual Tree Supervision
    n_tok    = min(len(hi_sent.tokens), arc_bho.size(1))
    hi_heads = torch.tensor(
        [hi_sent.tokens[i].head for i in range(n_tok)],
        dtype=torch.long, device=device,
    ).clamp(0, arc_bho.size(2) - 1)
    l_cts = F.cross_entropy(arc_bho[0, :n_tok], hi_heads)

    return l_cosine, l_arc_kl, l_cts


# ═════════════════════════════════════════════════════════════════════════════
# Warm-start helpers (optional — only used when checkpoints exist)
# ═════════════════════════════════════════════════════════════════════════════

def warmstart_muril_from_phase1(encoder: ParallelEncoder,
                                 parser_bho: BiaffineHeads,
                                 parser_hi:  BiaffineHeads):
    """If `checkpoints/muril_hindi/best.pt` exists (System M's Phase-1 HDTB
    MuRIL training), copy Hindi-adapter + biaffine into both parsers."""
    if not MURIL_HINDI_CKPT.exists():
        print(f"  [info] MuRIL Phase-1 ckpt not found at {MURIL_HINDI_CKPT} "
              "— starting fresh (no warm-start).")
        return
    state = torch.load(str(MURIL_HINDI_CKPT), map_location="cpu")
    if "adapter_hi" in state:
        encoder.adapters["hindi"].load_state_dict(state["adapter_hi"])
        # Copy Hindi adapter → Bhojpuri adapter so both start identical
        encoder.adapters["bhojpuri"].load_state_dict(state["adapter_hi"])
        print(f"  Warm-started Hi + Bho adapters from {MURIL_HINDI_CKPT}")
    if "biaffine" in state:
        our = parser_bho.state_dict()
        copied = 0
        for k, v in state["biaffine"].items():
            if k in our and our[k].shape == v.shape:
                our[k] = v; copied += 1
        parser_bho.load_state_dict(our); parser_hi.load_state_dict(our)
        print(f"  Warm-started biaffine tensors: {copied}")


def warmstart_xlmr_from_trankit(encoder: ParallelEncoder,
                                 parser_bho: BiaffineHeads,
                                 parser_hi:  BiaffineHeads):
    """Same approach as System H — copy what we can from the Trankit
    Hindi tagger checkpoint (shape-matched)."""
    if not XLMR_HINDI_CKPT.exists():
        print(f"  [info] XLM-R Trankit Hindi ckpt not found — skipping warm-start.")
        return
    state = torch.load(str(XLMR_HINDI_CKPT), map_location="cpu")
    adapters = state.get("adapters", {})
    our_sd = encoder.adapters["hindi"].state_dict()
    new_sd = {k: v.clone() for k, v in our_sd.items()}
    loaded = 0
    for tv in adapters.values():
        for ok, ov in our_sd.items():
            if tv.shape == ov.shape:
                new_sd[ok] = tv; loaded += 1; break
    encoder.adapters["hindi"].load_state_dict(new_sd)
    encoder.adapters["bhojpuri"].load_state_dict(new_sd)
    print(f"  XLM-R warm-start: loaded {loaded} adapter tensors.")

    # Biaffine tensor copy by shape match
    hindi_tensors: List[torch.Tensor] = []
    def _collect(obj):
        if isinstance(obj, torch.Tensor):
            hindi_tensors.append(obj)
        elif isinstance(obj, dict):
            for v in obj.values(): _collect(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj: _collect(v)
    _collect(state)
    for parser in (parser_bho, parser_hi):
        our = parser.state_dict()
        new = {k: v.clone() for k, v in our.items()}
        used = set()
        for ok, ov in our.items():
            if "label.biaffine" in ok:
                continue
            for i, hv in enumerate(hindi_tensors):
                if i not in used and hv.shape == ov.shape:
                    new[ok] = hv.clone(); used.add(i); break
        parser.load_state_dict(new)


# ═════════════════════════════════════════════════════════════════════════════
# SWA — running mean of parameter vectors across training epochs
# ═════════════════════════════════════════════════════════════════════════════

class SWA:
    """Maintains a running mean of state_dicts across epochs."""
    def __init__(self, modules: Dict[str, nn.Module]):
        self.modules = modules
        self.n = 0
        self.running: Dict[str, Dict[str, torch.Tensor]] = {
            name: {k: v.detach().cpu().clone().float()
                   for k, v in mod.state_dict().items()}
            for name, mod in modules.items()
        }

    def update(self):
        self.n += 1
        for name, mod in self.modules.items():
            cur = mod.state_dict()
            for k, v in cur.items():
                v_f = v.detach().cpu().float()
                if k not in self.running[name]:
                    self.running[name][k] = v_f.clone()
                else:
                    # Incremental mean: x_n = x_{n-1} + (v - x_{n-1}) / n
                    self.running[name][k].add_(
                        (v_f - self.running[name][k]) / self.n
                    )

    def load_into(self):
        for name, mod in self.modules.items():
            cur = mod.state_dict()
            avg = {}
            for k, v in cur.items():
                src = self.running[name].get(k, v.detach().cpu().float())
                avg[k] = src.to(v.dtype).to(v.device)
            mod.load_state_dict(avg)


# ═════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(encoder, parser, morph_embedder, morph_vocab, max_feats,
             vocab, sents, cache, device, lang: str = "bhojpuri"
             ) -> Tuple[float, float]:
    encoder.eval(); parser.eval(); morph_embedder.eval()
    ph_all, pr_all = [], []
    for i, s in enumerate(sents):
        if not s.tokens:
            ph_all.append([]); pr_all.append([]); continue
        cached = cache[i] if cache is not None and i < len(cache) else None
        H = encoder.encode_one(lang, s.words(), cached_xlmr=cached).to(device)
        m_ids, m_mask = sent_morph_tensors(s, morph_vocab, max_feats, device)
        H = H + morph_embedder(m_ids, m_mask)
        arc_s, lbl_s = parser(H)
        mask = torch.ones(1, len(s.tokens), dtype=torch.bool, device=device)
        ph, pr = BiaffineHeads.predict(arc_s, lbl_s, mask)
        ph_all.append(ph[0].cpu().tolist())
        pr_all.append([vocab.decode(r) for r in pr[0].cpu().tolist()])
    uas, las = uas_las(sents, ph_all, pr_all)
    encoder.train(); parser.train(); morph_embedder.train()
    return uas, las


# ═════════════════════════════════════════════════════════════════════════════
# Training loop
# ═════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="System N — MuRIL-SACT + MFEF + curriculum + SWA"
    )
    ap.add_argument("--backbone",         type=str,   default="muril",
                    choices=["muril", "xlmr"])
    ap.add_argument("--epochs",           type=int,   default=40)
    ap.add_argument("--device",           type=str,   default="cuda")
    ap.add_argument("--lr",               type=float, default=5e-5)
    ap.add_argument("--patience",         type=int,   default=8)
    ap.add_argument("--seed",             type=int,   default=42)
    ap.add_argument("--dev_ratio",        type=float, default=0.1)
    ap.add_argument("--test_ratio",       type=float, default=0.1)

    # SACT weights (same defaults as System H)
    ap.add_argument("--lambda_hi",        type=float, default=0.5)
    ap.add_argument("--lambda_cosine",    type=float, default=0.4)
    ap.add_argument("--lambda_arc",       type=float, default=2.0)
    ap.add_argument("--lambda_cts",       type=float, default=0.2)
    ap.add_argument("--warmup_epochs",    type=int,   default=0)

    # System N additions
    ap.add_argument("--morph_dim",        type=int,   default=128)
    ap.add_argument("--morph_dropout",    type=float, default=0.1)
    ap.add_argument("--max_feats",        type=int,   default=8)
    ap.add_argument("--curriculum_epochs", type=int,  default=5,
                    help="Train on single-root subset for first N epochs, "
                         "then switch to full data.")
    ap.add_argument("--swa_start",        type=int,   default=15,
                    help="Start SWA running mean after this epoch.")
    args = ap.parse_args()

    # ── Device ───────────────────────────────────────────────────────────────
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU"); args.device = "cpu"
    if args.device == "mps" and not torch.backends.mps.is_available():
        args.device = "cpu"
    device = torch.device(args.device)

    random.seed(args.seed); torch.manual_seed(args.seed)

    print("\n" + "=" * 70)
    print(" System N — MuRIL-SACT + MFEF + Curriculum + SWA")
    print(" Stacked over System H baseline (Dev UAS 55.85%)")
    print("=" * 70)
    print(f"  Backbone          : {args.backbone}")
    print(f"  Device            : {device}")
    print(f"  Epochs            : {args.epochs}")
    print(f"  Curriculum epochs : {args.curriculum_epochs} (single-root only)")
    print(f"  SWA start         : epoch {args.swa_start}")
    print(f"  Morph dim         : {args.morph_dim}")
    print(f"  λ_hi / cos / arc / cts : "
          f"{args.lambda_hi}/{args.lambda_cosine}/{args.lambda_arc}/{args.lambda_cts}")
    print(f"  LR                : {args.lr}")

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1] Loading parallel data …")
    hi_sents   = read_conllu(PROF_HI)
    bho_sents  = read_conllu(PROF_BHO)
    test_sents = read_conllu(BHTB_TEST)
    assert len(hi_sents) == len(bho_sents), \
        f"Mismatch: {len(hi_sents)} Hindi vs {len(bho_sents)} Bhojpuri"
    print(f"  Parallel pairs : {len(hi_sents):,}")
    print(f"  BHTB test      : {len(test_sents):,}")

    # Full set → Dev/Test split (same regime as System H)
    good_idx = list(range(len(bho_sents)))
    n_total  = len(good_idx)
    n_test   = max(1, int(n_total * args.test_ratio))
    n_dev    = max(1, int(n_total * args.dev_ratio))
    n_train  = n_total - n_dev - n_test
    train_idx_full = good_idx[:n_train]
    dev_idx        = good_idx[n_train:n_train + n_dev]
    test_idx       = good_idx[n_train + n_dev:]
    dev_bho = [bho_sents[i] for i in dev_idx]
    print(f"  Train : {n_train:,} | Dev : {n_dev:,} | Test : {n_test:,}")

    # Curriculum subset: single-root-only train indices
    single_root_idx = set(filter_single_root(bho_sents))
    train_idx_clean = [i for i in train_idx_full if i in single_root_idx]
    print(f"  Curriculum subset (single-root): {len(train_idx_clean):,} "
          f"({100*len(train_idx_clean)/max(n_train,1):.1f}% of train)")

    # ── Vocabularies ─────────────────────────────────────────────────────────
    print("\n[2] Building vocabularies …")
    vocab = RelVocab()
    for s in hi_sents + bho_sents + test_sents:
        for t in s.tokens:
            vocab.add(t.deprel)
    n_rels = len(vocab)
    print(f"  Relation vocab : {n_rels}")

    morph_vocab = MorphVocab()
    for s in hi_sents + bho_sents + test_sents:
        for t in s.tokens:
            morph_vocab.add_feats_string(t.feats)
    print(f"  Morph vocab    : {len(morph_vocab):,} tokens")

    # ── Model ────────────────────────────────────────────────────────────────
    print("\n[3] Building model …")
    backbone_name = MURIL_NAME if args.backbone == "muril" else XLM_R_LOCAL
    encoder = ParallelEncoder(
        model_name      = backbone_name,
        adapter_dim     = 64,
        adapter_dropout = 0.1,
        freeze_xlmr     = True,
    )
    encoder.adapters.to(device)
    hidden = encoder.hidden_size

    parser_bho = BiaffineHeads(hidden, 500, 100, n_rels, 0.33).to(device)
    parser_hi  = BiaffineHeads(hidden, 500, 100, n_rels, 0.33).to(device)
    morph_embedder = MorphEmbedder(
        vocab_size = len(morph_vocab),
        embed_dim  = args.morph_dim,
        hidden_dim = hidden,
        dropout    = args.morph_dropout,
    ).to(device)

    # ── Warm-start ───────────────────────────────────────────────────────────
    print("\n[4] Warm-starting …")
    if args.backbone == "muril":
        warmstart_muril_from_phase1(encoder, parser_bho, parser_hi)
    else:
        warmstart_xlmr_from_trankit(encoder, parser_bho, parser_hi)

    # ── Optimizer ────────────────────────────────────────────────────────────
    trainable = (list(encoder.adapters.parameters())
                 + list(parser_bho.parameters())
                 + list(parser_hi.parameters())
                 + list(morph_embedder.parameters()))
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    n_params = sum(p.numel() for p in trainable)
    print(f"  Trainable params: {n_params:,}")

    # ── Precompute backbone embeddings ────────────────────────────────────────
    cache_name = "muril_n_cache.pt" if args.backbone == "muril" else "xlmr_n_cache.pt"
    _cache_path = ROOT_DIR / "cache" / cache_name
    if _cache_path.exists():
        print(f"\n[5] Loading cached embeddings from {_cache_path} …")
        _c = torch.load(str(_cache_path), map_location="cpu")
        cache_hi, cache_bho = _c["hi"], _c["bho"]
        print(f"  Loaded — hi:{len(cache_hi)}, bho:{len(cache_bho)}")
    else:
        print(f"\n[5] Pre-computing backbone embeddings ({args.backbone}) …")
        cache_hi  = encoder.precompute_xlmr([s.words() for s in hi_sents],  desc="Hindi")
        cache_bho = encoder.precompute_xlmr([s.words() for s in bho_sents], desc="Bhojpuri")
        _cache_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"hi": cache_hi, "bho": cache_bho}, str(_cache_path))
        print(f"  Saved cache → {_cache_path}")

    # ── Training loop ────────────────────────────────────────────────────────
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    best_uas = best_las = 0.0
    best_ep  = 0
    no_improve = 0

    swa = SWA({
        "encoder":        encoder.adapters,
        "parser_bho":     parser_bho,
        "parser_hi":      parser_hi,
        "morph_embedder": morph_embedder,
    })
    swa_collecting = False

    print(f"\n[6] Training for up to {args.epochs} epochs …\n")
    for epoch in range(1, args.epochs + 1):
        encoder.train(); parser_bho.train(); parser_hi.train(); morph_embedder.train()

        # Curriculum: first N epochs on single-root subset, then full
        if epoch <= args.curriculum_epochs and len(train_idx_clean) > 100:
            train_idx = list(train_idx_clean)
            phase = "clean"
        else:
            train_idx = list(train_idx_full)
            phase = "full"
        random.shuffle(train_idx)

        l_tot = l_bho_s = l_hi_s = l_cos_s = l_kl_s = l_cts_s = 0.0
        n_sents = 0

        for idx in train_idx:
            hi_s  = hi_sents[idx]
            bho_s = bho_sents[idx]
            if not hi_s.tokens or not bho_s.tokens:
                continue

            # Encode (cached backbone + adapter + MFEF)
            H_hi  = encoder.encode_one("hindi",    hi_s.words(),  cache_hi[idx]).to(device)
            H_bho = encoder.encode_one("bhojpuri", bho_s.words(), cache_bho[idx]).to(device)

            m_ids_hi,  m_mask_hi  = sent_morph_tensors(hi_s,  morph_vocab,
                                                       args.max_feats, device)
            m_ids_bho, m_mask_bho = sent_morph_tensors(bho_s, morph_vocab,
                                                       args.max_feats, device)
            H_hi  = H_hi  + morph_embedder(m_ids_hi,  m_mask_hi)
            H_bho = H_bho + morph_embedder(m_ids_bho, m_mask_bho)

            arc_bho, lbl_bho = parser_bho(H_bho)
            arc_hi,  lbl_hi  = parser_hi (H_hi)

            bho_h = torch.tensor([t.head for t in bho_s.tokens],
                                 dtype=torch.long, device=device)
            bho_r = torch.tensor([vocab.encode(t.deprel) for t in bho_s.tokens],
                                 dtype=torch.long, device=device)
            hi_h  = torch.tensor([t.head for t in hi_s.tokens],
                                 dtype=torch.long, device=device)
            hi_r  = torch.tensor([vocab.encode(t.deprel) for t in hi_s.tokens],
                                 dtype=torch.long, device=device)

            l_bho = parse_loss(arc_bho, lbl_bho, bho_h, bho_r)
            l_hi  = parse_loss(arc_hi,  lbl_hi,  hi_h,  hi_r)
            l_cos, l_kl, l_cts = sact_losses(H_hi, H_bho, arc_hi, arc_bho,
                                             hi_s, bho_s, device)

            kl_weight = args.lambda_arc if epoch > args.warmup_epochs else 0.0
            loss = (l_bho
                    + args.lambda_hi     * l_hi
                    + args.lambda_cosine * l_cos
                    + kl_weight          * l_kl
                    + args.lambda_cts    * l_cts)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable, 5.0)
            optimizer.step()

            l_tot += loss.item()
            l_bho_s += l_bho.item(); l_hi_s += l_hi.item()
            l_cos_s += l_cos.item(); l_kl_s += l_kl.item(); l_cts_s += l_cts.item()
            n_sents += 1

        N = max(n_sents, 1)
        dev_uas, dev_las = evaluate(
            encoder, parser_bho, morph_embedder, morph_vocab,
            args.max_feats, vocab, dev_bho,
            [cache_bho[i] for i in dev_idx], device, lang="bhojpuri",
        )

        improved = dev_uas > best_uas or (dev_uas == best_uas and dev_las > best_las)
        if improved:
            best_uas = dev_uas; best_las = dev_las; best_ep = epoch
            torch.save({
                "epoch":          epoch,
                "best_uas":       best_uas,
                "best_las":       best_las,
                "vocab":          vocab,
                "morph_vocab":    morph_vocab._tok2id,
                "encoder":        encoder.state_dict(),
                "parser_bho":     parser_bho.state_dict(),
                "parser_hi":      parser_hi.state_dict(),
                "morph_embedder": morph_embedder.state_dict(),
                "args":           vars(args),
            }, CKPT_PATH)
            no_improve = 0
        else:
            no_improve += 1

        # SWA accumulator (start after swa_start once in 'full' phase)
        if epoch >= args.swa_start and phase == "full":
            if not swa_collecting:
                # Re-initialise running dict from current weights
                swa = SWA({
                    "encoder":        encoder.adapters,
                    "parser_bho":     parser_bho,
                    "parser_hi":      parser_hi,
                    "morph_embedder": morph_embedder,
                })
                swa.n = 0
                swa_collecting = True
            swa.update()

        tag = "CLEAN" if phase == "clean" else "FULL "
        msg = (f"Ep{epoch:3d} [{tag}] "
               f"bho={l_bho_s/N:.3f} hi={l_hi_s/N:.3f} "
               f"cos={l_cos_s/N:.3f} kl={l_kl_s/N:.3f} cts={l_cts_s/N:.3f} "
               f"| Dev UAS={dev_uas*100:.2f}% LAS={dev_las*100:.2f}%")
        if improved:
            msg += "  ← BEST"
        else:
            msg += f"  (no-imp {no_improve}/{args.patience})"
        if swa_collecting:
            msg += f"  [SWA n={swa.n}]"
        print(msg, flush=True)

        if no_improve >= args.patience:
            print(f"\nEarly stopping at epoch {epoch}")
            break

    # ── SWA evaluation ───────────────────────────────────────────────────────
    if swa_collecting and swa.n >= 2:
        print(f"\n[7] Evaluating SWA model (averaged over {swa.n} epochs) …")
        pre_swa_state = {
            "encoder":        copy.deepcopy(encoder.adapters.state_dict()),
            "parser_bho":     copy.deepcopy(parser_bho.state_dict()),
            "parser_hi":      copy.deepcopy(parser_hi.state_dict()),
            "morph_embedder": copy.deepcopy(morph_embedder.state_dict()),
        }
        swa.load_into()
        swa_uas, swa_las = evaluate(
            encoder, parser_bho, morph_embedder, morph_vocab,
            args.max_feats, vocab, dev_bho,
            [cache_bho[i] for i in dev_idx], device, lang="bhojpuri",
        )
        print(f"  SWA Dev  UAS={swa_uas*100:.2f}%  LAS={swa_las*100:.2f}%")

        if swa_uas > best_uas:
            print("  SWA beats best single — saving SWA as final model.")
            torch.save({
                "epoch":          f"SWA({swa.n})",
                "best_uas":       swa_uas,
                "best_las":       swa_las,
                "vocab":          vocab,
                "morph_vocab":    morph_vocab._tok2id,
                "encoder":        encoder.state_dict(),
                "parser_bho":     parser_bho.state_dict(),
                "parser_hi":      parser_hi.state_dict(),
                "morph_embedder": morph_embedder.state_dict(),
                "args":           vars(args),
            }, CKPT_SWA)
            best_uas = swa_uas; best_las = swa_las; best_ep = f"SWA({swa.n})"
        else:
            # Restore best-single weights
            encoder.adapters.load_state_dict(pre_swa_state["encoder"])
            parser_bho.load_state_dict(pre_swa_state["parser_bho"])
            parser_hi.load_state_dict(pre_swa_state["parser_hi"])
            morph_embedder.load_state_dict(pre_swa_state["morph_embedder"])

    # ── Final evaluation ─────────────────────────────────────────────────────
    print("\n[8] Loading best checkpoint for final evaluation …")
    if not isinstance(best_ep, str):
        ckpt = torch.load(str(CKPT_PATH), map_location=device)
        encoder.load_state_dict(ckpt["encoder"])
        parser_bho.load_state_dict(ckpt["parser_bho"])
        morph_embedder.load_state_dict(ckpt["morph_embedder"])

    print(f"\n{'=' * 70}")
    print(f"  System N — Final Results  (best @ epoch {best_ep})")
    print(f"{'=' * 70}")
    print(f"  Dev UAS/LAS (prof) : {best_uas*100:.2f}% / {best_las*100:.2f}%")

    # Internal test (unseen 10% of prof data)
    int_test_sents = [bho_sents[i] for i in test_idx]
    int_test_cache = [cache_bho[i] for i in test_idx]
    int_uas, int_las = evaluate(
        encoder, parser_bho, morph_embedder, morph_vocab,
        args.max_feats, vocab, int_test_sents, int_test_cache, device,
        lang="bhojpuri",
    )
    print(f"  Internal Test UAS/LAS : {int_uas*100:.2f}% / {int_las*100:.2f}%  "
          f"({len(test_idx):,} sents)")

    # BHTB external — precompute embeddings just for test
    print("\n  Precomputing backbone for BHTB test …")
    cache_bhtb = encoder.precompute_xlmr([s.words() for s in test_sents], desc="BHTB")
    bhtb_uas, bhtb_las = evaluate(
        encoder, parser_bho, morph_embedder, morph_vocab,
        args.max_feats, vocab, test_sents, cache_bhtb, device,
        lang="bhojpuri",
    )
    print(f"  BHTB External UAS/LAS : {bhtb_uas*100:.2f}% / {bhtb_las*100:.2f}%")
    print(f"  Checkpoint            : {CKPT_PATH}")
    print(f"\n  Baselines: System H Dev UAS 55.85 / LAS 50.08,"
          f"  System A BHTB UAS 53.48 / LAS 34.84")
    print("=" * 70)

    # ── Save results file ────────────────────────────────────────────────────
    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = results_dir / f"system_n_{ts}.txt"
    with open(out, "w") as f:
        f.write("System N — MuRIL-SACT + MFEF + Curriculum + SWA\n")
        f.write(f"Date              : {datetime.datetime.now()}\n")
        f.write(f"Backbone          : {args.backbone}\n")
        f.write(f"Best epoch        : {best_ep}\n")
        f.write(f"Dev UAS / LAS     : {best_uas*100:.2f}% / {best_las*100:.2f}%\n")
        f.write(f"Internal UAS/LAS  : {int_uas*100:.2f}% / {int_las*100:.2f}%\n")
        f.write(f"BHTB UAS / LAS    : {bhtb_uas*100:.2f}% / {bhtb_las*100:.2f}%\n")
        f.write(f"Args              : {vars(args)}\n")
    print(f"\n  Results saved → {out}")


if __name__ == "__main__":
    main()

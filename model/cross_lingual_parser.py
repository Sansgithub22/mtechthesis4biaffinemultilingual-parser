# model/cross_lingual_parser.py
# Full cross-lingual dependency parser (Steps 3–9 combined).
#
# Data flow:
#
#   Hindi words ──→ ParallelEncoder(hindi)  ──→ H_hi   [1, n, 768]
#                                                          │
#                                                          │ Step 4
#                                                          ▼
#   Bhojpuri words → ParallelEncoder(bho)  ──→ H_bho ──→ CrossSentenceAttention
#                                                          │  → H_cross [1,m,768]
#                                                          │
#                                                          │ Step 5
#                                                          ▼
#                                                    CrossLingualLayer
#                                                          │  → H_fused [1,m,768]
#                                                          │
#                              ┌───────────────────────────┤
#                              │ Bhojpuri biaffine          │ Hindi biaffine
#                              ▼                            ▼
#                         arc/label scores            arc/label scores
#                         on H_fused                  on H_hi
#
# Training (Steps 7-8):
#   L_total = L_hi + λ_bho * L_bho + λ_align * L_align
#
#   L_hi    = cross-entropy(arc + label) on Hindi gold annotations
#   L_bho   = cross-entropy(arc + label) on Bhojpuri projected annotations
#   L_align = mean squared error between Hindi and Bhojpuri token
#             representations for known aligned word pairs

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from model.parallel_encoder         import ParallelEncoder
from model.cross_sentence_attention import CrossSentenceAttention
from model.cross_lingual_layer      import CrossLingualLayer
from model.biaffine_heads           import BiaffineHeads
from config import Config, CFG


# ─────────────────────────────────────────────────────────────────────────────
# Vocabulary helpers (shared across Hindi + Bhojpuri)
# ─────────────────────────────────────────────────────────────────────────────
class RelVocab:
    """Simple bidirectional map for UD relation labels."""
    PAD = "<pad>"
    UNK = "<unk>"

    def __init__(self):
        self._w2i: Dict[str, int] = {self.PAD: 0, self.UNK: 1}
        self._i2w: List[str]      = [self.PAD, self.UNK]

    def add(self, rel: str):
        if rel not in self._w2i:
            self._w2i[rel] = len(self._i2w)
            self._i2w.append(rel)

    def __len__(self):  return len(self._i2w)
    def encode(self, rel: str) -> int:
        return self._w2i.get(rel, 1)
    def decode(self, idx: int) -> str:
        return self._i2w[idx] if idx < len(self._i2w) else self.UNK


# ─────────────────────────────────────────────────────────────────────────────
# Main model
# ─────────────────────────────────────────────────────────────────────────────
class CrossLingualParser(nn.Module):
    """
    Cross-lingual Hindi → Bhojpuri dependency parser.

    Parameters
    ----------
    rel_vocab : RelVocab
        Shared UD relation vocabulary for both languages.
    cfg : Config
        Project configuration object.
    """

    def __init__(self, rel_vocab: RelVocab, cfg: Config = CFG):
        super().__init__()
        self.rel_vocab = rel_vocab
        self.cfg = cfg

        d = cfg.encoder.hidden_size

        # Step 3: Parallel Trankit encoders (shared XLM-R + 2 adapters)
        self.encoder = ParallelEncoder(
            model_name      = cfg.encoder.model_name,
            adapter_dim     = cfg.encoder.adapter_dim,
            adapter_dropout = cfg.encoder.adapter_dropout,
            freeze_xlmr     = cfg.encoder.freeze_xlmr,
        )

        # Step 4: Cross-sentence attention
        self.cross_attn = CrossSentenceAttention(
            hidden_dim = d,
            n_heads    = cfg.cross_attn.n_heads,
            dropout    = cfg.cross_attn.dropout,
        )

        # Step 5: Cross-lingual fusion layer
        self.cross_layer = CrossLingualLayer(
            hidden_dim = d,
            dropout    = cfg.cross_attn.dropout,
        )

        n_rels = len(rel_vocab)

        # Hindi biaffine head (trained on gold HDTB annotations)
        self.hindi_parser = BiaffineHeads(
            hidden_dim    = d,
            arc_mlp_dim   = cfg.biaffine.arc_mlp_dim,
            label_mlp_dim = cfg.biaffine.label_mlp_dim,
            n_rels        = n_rels,
            mlp_dropout   = cfg.biaffine.mlp_dropout,
        )

        # Bhojpuri biaffine head (trained on projected + real annotations)
        self.bhojpuri_parser = BiaffineHeads(
            hidden_dim    = d,
            arc_mlp_dim   = cfg.biaffine.arc_mlp_dim,
            label_mlp_dim = cfg.biaffine.label_mlp_dim,
            n_rels        = n_rels,
            mlp_dropout   = cfg.biaffine.mlp_dropout,
        )

    # ── Forward pass ──────────────────────────────────────────────────────────
    def forward(
        self,
        hindi_words:    List[str],
        bhojpuri_words: List[str],
    ) -> Dict[str, torch.Tensor]:
        """
        Returns a dict with:
            arc_hi    [1, n, n]           Hindi arc scores
            lbl_hi    [1, n, n, n_rels]   Hindi label scores
            arc_bho   [1, m, m]           Bhojpuri arc scores
            lbl_bho   [1, m, m, n_rels]   Bhojpuri label scores
            attn      [1, m, n]           cross-attention weights
            H_hi      [1, n, d]           Hindi token representations
            H_bho     [1, m, d]           Bhojpuri adapted representations
            H_fused   [1, m, d]           cross-lingual fused representations
        """
        # Step 3: Encode both languages
        H_hi, H_bho = self.encoder.encode_pair(hindi_words, bhojpuri_words)
        # H_hi:  [1, n, 768]
        # H_bho: [1, m, 768]

        # Step 4: Cross-sentence attention (Bhojpuri attends to Hindi)
        H_cross, attn = self.cross_attn(H_bho, H_hi)
        # H_cross: [1, m, 768]
        # attn:    [1, m, n]

        # Step 5: Cross-lingual fusion
        H_fused = self.cross_layer(H_bho, H_cross)
        # H_fused: [1, m, 768]

        # Hindi biaffine (uses original Hindi representation)
        arc_hi, lbl_hi   = self.hindi_parser(H_hi)

        # Bhojpuri biaffine (uses fused cross-lingual representation)
        arc_bho, lbl_bho = self.bhojpuri_parser(H_fused)

        return {
            "arc_hi":  arc_hi,
            "lbl_hi":  lbl_hi,
            "arc_bho": arc_bho,
            "lbl_bho": lbl_bho,
            "attn":    attn,
            "H_hi":    H_hi,
            "H_bho":   H_bho,
            "H_fused": H_fused,
        }

    # ── Loss computation (Steps 7-8) ──────────────────────────────────────────
    def compute_loss(
        self,
        out:          Dict[str, torch.Tensor],
        gold_hi_heads:  torch.Tensor,   # [1, n]  0-based, 0=root
        gold_hi_rels:   torch.Tensor,   # [1, n]  int
        gold_bho_heads: torch.Tensor,   # [1, m]
        gold_bho_rels:  torch.Tensor,   # [1, m]
        hi_mask:        torch.Tensor,   # [1, n]  bool (True = real token)
        bho_mask:       torch.Tensor,   # [1, m]
        alignments:     Optional[List[Tuple[int, int]]] = None,
        lambda_bho:     float = 0.5,
        lambda_align:   float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """
        Computes three losses:
            L_hi    — arc + label CE on Hindi gold
            L_bho   — arc + label CE on Bhojpuri projected labels
            L_align — MSE between aligned (H_hi, H_bho) token pairs
        """
        # ── Hindi loss (L_hi) ────────────────────────────────────────────
        L_hi = self._parse_loss(
            out["arc_hi"], out["lbl_hi"],
            gold_hi_heads, gold_hi_rels, hi_mask,
        )

        # ── Bhojpuri loss (L_bho) ───────────────────────────────────────
        L_bho = self._parse_loss(
            out["arc_bho"], out["lbl_bho"],
            gold_bho_heads, gold_bho_rels, bho_mask,
        )

        # ── Alignment regularisation (L_align) ──────────────────────────
        L_align = torch.tensor(0.0, device=out["H_hi"].device)
        if alignments is not None and len(alignments) > 0:
            H_hi_flat  = out["H_hi"].squeeze(0)   # [n, d]
            H_bho_flat = out["H_bho"].squeeze(0)  # [m, d]
            src_idx = [s for s, _ in alignments if s < H_hi_flat.size(0)]
            tgt_idx = [t for _, t in alignments if t < H_bho_flat.size(0)]
            # Trim to same length
            min_len = min(len(src_idx), len(tgt_idx))
            if min_len > 0:
                h_src = H_hi_flat[src_idx[:min_len]]   # [k, d]
                h_tgt = H_bho_flat[tgt_idx[:min_len]]  # [k, d]
                L_align = F.mse_loss(
                    F.normalize(h_tgt, dim=-1),
                    F.normalize(h_src, dim=-1).detach(),  # treat Hindi as target
                )

        L_total = L_hi + lambda_bho * L_bho + lambda_align * L_align

        return {
            "loss":       L_total,
            "loss_hi":    L_hi,
            "loss_bho":   L_bho,
            "loss_align": L_align,
        }

    @staticmethod
    def _parse_loss(
        arc_scores:   torch.Tensor,   # [B, n, n+1]  dep × (ROOT + n heads)
        label_scores: torch.Tensor,   # [B, n, n+1, n_rels]
        gold_heads:   torch.Tensor,   # [B, n]  CoNLL-U 0-indexed (0=ROOT, 1..n)
        gold_rels:    torch.Tensor,   # [B, n]
        mask:         torch.Tensor,   # [B, n]
    ) -> torch.Tensor:
        """
        Arc loss  : cross-entropy over n+1 head positions (ROOT at col 0).
        Label loss: cross-entropy over n_rels at the gold head column.
        gold_heads uses CoNLL-U convention: 0=ROOT, 1..n = actual tokens.
        """
        B, n, n_plus1 = arc_scores.shape
        active = mask.view(-1)                              # [B*n]

        # ── Arc loss  ────────────────────────────────────────────────────
        arc_flat  = arc_scores.view(B * n, n_plus1)        # [B*n, n+1] logits
        head_flat = gold_heads.view(-1)                    # [B*n]  0..n targets
        arc_loss  = F.cross_entropy(
            arc_flat[active], head_flat[active], reduction="mean"
        )

        # ── Label loss at gold head position ─────────────────────────────
        head_idx    = gold_heads.unsqueeze(-1).unsqueeze(-1)         # [B,n,1,1]
        head_idx    = head_idx.expand(B, n, 1, label_scores.size(-1))
        lbl_at_head = label_scores.gather(2, head_idx).squeeze(2)   # [B,n,n_rels]
        lbl_flat    = lbl_at_head.view(B * n, -1)
        rel_flat    = gold_rels.view(-1)
        lbl_loss    = F.cross_entropy(
            lbl_flat[active], rel_flat[active], reduction="mean"
        )

        return arc_loss + lbl_loss

    # ── Prediction helpers ────────────────────────────────────────────────────
    def predict_bhojpuri(
        self,
        hindi_words:    List[str],
        bhojpuri_words: List[str],
    ) -> Tuple[List[int], List[str]]:
        """
        Step 9 — Final dependency parsing for a Bhojpuri sentence.
        Returns (head_ids, deprel_strings) both 1-indexed.
        """
        self.eval()
        with torch.no_grad():
            out  = self.forward(hindi_words, bhojpuri_words)
            m    = len(bhojpuri_words)
            mask = out["arc_bho"].new_ones(1, m, dtype=torch.bool)
            pred_heads, pred_rels = BiaffineHeads.predict(
                out["arc_bho"], out["lbl_bho"], mask
            )
        heads = pred_heads[0].cpu().tolist()
        rels  = [self.rel_vocab.decode(r) for r in pred_rels[0].cpu().tolist()]
        return heads, rels

    def predict_hindi(
        self,
        hindi_words: List[str],
    ) -> Tuple[List[int], List[str]]:
        """Predict dependencies for a Hindi sentence only."""
        self.eval()
        with torch.no_grad():
            H_hi = self.encoder.encode_one("hindi", hindi_words)
            arc_hi, lbl_hi = self.hindi_parser(H_hi)
            n    = len(hindi_words)
            mask = arc_hi.new_ones(1, n, dtype=torch.bool)
            pred_heads, pred_rels = BiaffineHeads.predict(arc_hi, lbl_hi, mask)
        heads = pred_heads[0].cpu().tolist()
        rels  = [self.rel_vocab.decode(r) for r in pred_rels[0].cpu().tolist()]
        return heads, rels

    def count_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

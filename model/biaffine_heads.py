# model/biaffine_heads.py
# Self-contained biaffine dependency scoring heads (Dozat & Manning, 2017).
# Mirrors the implementation in biaffine_parser/modules/biaffine.py but is
# bundled here so this project has no external sibling-directory dependency.
#
# Components:
#   MLP         — dimension-reducing MLP with ELU activation (Eq. 4-5)
#   BiaffineArc — scores every (dep, head) pair for arc existence
#   BiaffineLabel — scores every (dep, head, label) triple
#   BiaffineHeads — wraps both heads with four shared MLPs

import torch
import torch.nn as nn
from typing import Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    """Linear → ELU → Dropout  (one-layer dimension-reducing projection)."""
    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.33):
        super().__init__()
        self.linear  = nn.Linear(in_dim, out_dim)
        self.act     = nn.ELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.act(self.linear(x)))


# ─────────────────────────────────────────────────────────────────────────────
# Core bilinear layer
# ─────────────────────────────────────────────────────────────────────────────
class _Biaffine(nn.Module):
    """
    score(x_i, y_j) = x_i^T W y_j  + optional bias terms.
    W shape: [out_features, in_x+bias_x, in_y+bias_y]
    """
    def __init__(self, in_features: int, out_features: int = 1,
                 bias_x: bool = True, bias_y: bool = True):
        super().__init__()
        self.out_features = out_features
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(
            torch.zeros(out_features,
                        in_features + int(bias_x),
                        in_features + int(bias_y))
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if self.bias_x:
            x = torch.cat([x, x.new_ones(*x.shape[:-1], 1)], dim=-1)
        if self.bias_y:
            y = torch.cat([y, y.new_ones(*y.shape[:-1], 1)], dim=-1)
        # [batch, out, dep_len, head_len]
        scores = torch.einsum('bxi,oij,byj->boxy', x, self.weight, y)
        if self.out_features == 1:
            return scores.squeeze(1)                     # [B, dep, head]
        return scores.permute(0, 2, 3, 1)               # [B, dep, head, n_rels]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────
class BiaffineArc(nn.Module):
    """Unlabeled arc scorer → [batch, dep_len, head_len]."""
    def __init__(self, in_dim: int, mlp_dim: int = 500, dropout: float = 0.33):
        super().__init__()
        self.dep_mlp  = MLP(in_dim, mlp_dim, dropout)
        self.head_mlp = MLP(in_dim, mlp_dim, dropout)
        self.biaffine = _Biaffine(mlp_dim, out_features=1,
                                  bias_x=True, bias_y=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.biaffine(self.dep_mlp(h), self.head_mlp(h))


class BiaffineLabel(nn.Module):
    """Labeled arc scorer → [batch, dep_len, head_len, n_rels]."""
    def __init__(self, in_dim: int, mlp_dim: int = 100,
                 n_rels: int = 45, dropout: float = 0.33):
        super().__init__()
        self.dep_mlp  = MLP(in_dim, mlp_dim, dropout)
        self.head_mlp = MLP(in_dim, mlp_dim, dropout)
        self.biaffine = _Biaffine(mlp_dim, out_features=n_rels,
                                  bias_x=True, bias_y=True)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.biaffine(self.dep_mlp(h), self.head_mlp(h))


class BiaffineHeads(nn.Module):
    """
    Combined arc + label scorer with a learnable ROOT sentinel.

    Following standard biaffine parsing (Dozat & Manning, 2017), a ROOT
    embedding is prepended to the word representations so that:
      - position 0 in arc_scores = virtual ROOT node
      - positions 1..n = actual word tokens
      - CoNLL-U head=0 → attaches to ROOT (position 0)
      - CoNLL-U head=k → attaches to token at position k (1-indexed)

    Input:  h  [batch, n_words, hidden_dim]   (actual words only — no ROOT)
    Output: arc_scores   [batch, n_words, n_words+1]         (dep × head)
            label_scores [batch, n_words, n_words+1, n_rels]
    """
    def __init__(self, hidden_dim: int, arc_mlp_dim: int = 500,
                 label_mlp_dim: int = 100, n_rels: int = 45,
                 mlp_dropout: float = 0.33):
        super().__init__()
        self.arc   = BiaffineArc(hidden_dim, arc_mlp_dim, mlp_dropout)
        self.label = BiaffineLabel(hidden_dim, label_mlp_dim, n_rels, mlp_dropout)
        # Learnable ROOT sentinel — initialised to small random values
        self.root_embed = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.01)

    def forward(self, h: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        h: [B, n, d]  — actual word representations
        Returns arc_scores   [B, n, n+1]   dep × (ROOT + n heads)
                label_scores [B, n, n+1, n_rels]
        """
        B = h.size(0)
        # Prepend ROOT embedding to head side
        root = self.root_embed.expand(B, 1, -1)      # [B, 1, d]
        h_with_root = torch.cat([root, h], dim=1)    # [B, n+1, d]

        # Dependent MLPs use actual words only; head MLPs use ROOT + words
        arc_dep   = self.arc.dep_mlp(h)              # [B, n, arc_dim]
        arc_head  = self.arc.head_mlp(h_with_root)   # [B, n+1, arc_dim]
        arc_scores = self.arc.biaffine(arc_dep, arc_head)   # [B, n, n+1]

        lbl_dep   = self.label.dep_mlp(h)
        lbl_head  = self.label.head_mlp(h_with_root)
        lbl_scores = self.label.biaffine(lbl_dep, lbl_head) # [B, n, n+1, n_rels]

        return arc_scores, lbl_scores

    # ── Inference ────────────────────────────────────────────────────────────
    @staticmethod
    def predict(arc_scores: torch.Tensor, label_scores: torch.Tensor,
                mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Greedy: each dependent picks highest-scoring head.
        arc_scores:   [B, n, n+1]          (ROOT is column 0)
        label_scores: [B, n, n+1, n_rels]
        mask:         [B, n]               True = real token
        Returns pred_heads [B, n]  (0-indexed, 0=ROOT)
                pred_rels  [B, n]
        """
        # Mask padding on the dependent side
        arc_scores = arc_scores.masked_fill(~mask.unsqueeze(-1), float('-inf'))
        pred_heads = arc_scores.argmax(-1)                            # [B, n]
        B, n, _, n_rels = label_scores.shape
        idx = pred_heads.unsqueeze(-1).unsqueeze(-1).expand(B, n, 1, n_rels)
        pred_rels = label_scores.gather(2, idx).squeeze(2).argmax(-1) # [B, n]
        return pred_heads, pred_rels

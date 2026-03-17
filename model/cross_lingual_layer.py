# model/cross_lingual_layer.py
# Step 5 — Concatenated Cross-Lingual Layer
#
# Combines three information streams into a single representation that the
# final Bhojpuri biaffine parser uses:
#
#   ┌─────────────────────────────────────────────────────┐
#   │  H_bho     [1, m, d]   Bhojpuri adapter output      │
#   │  H_cross   [1, m, d]   cross-attention from Hindi   │
#   │  H_hi_attn [1, m, d]   directly attended Hindi rep  │
#   └────────────────────────┬────────────────────────────┘
#                            ▼
#               cat([H_bho, H_cross])  [1, m, 2d]
#                            ▼
#                   Linear(2d → d)  +  LayerNorm  +  Dropout
#                            ▼
#                     H_fused  [1, m, d]
#
# This is the "Concatenated Cross-Lingual Representation" used by the biaffine
# dependency heads for Bhojpuri (Step 5).  The fused representation lets
# Hindi syntactic knowledge directly inform Bhojpuri parsing decisions.

import torch
import torch.nn as nn


class CrossLingualLayer(nn.Module):
    """
    Fuses Bhojpuri token representations with cross-attended Hindi information.

    Args:
        hidden_dim : int   — size of each individual representation stream (768)
        dropout    : float — applied after the projection

    Forward:
        H_bho   [B, m, d]   Bhojpuri encoder output
        H_cross [B, m, d]   output of CrossSentenceAttention
    Returns:
        H_fused [B, m, d]   cross-lingual fused representation
    """

    def __init__(self, hidden_dim: int = 768, dropout: float = 0.1):
        super().__init__()
        # Project from 2d back to d so downstream biaffine heads see normal dim
        self.proj       = nn.Linear(hidden_dim * 2, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout    = nn.Dropout(dropout)
        self.act        = nn.GELU()

    def forward(
        self,
        H_bho:   torch.Tensor,   # [B, m, d]  Bhojpuri
        H_cross: torch.Tensor,   # [B, m, d]  attended Hindi
    ) -> torch.Tensor:
        # Concatenate along feature dimension: [B, m, 2d]
        combined = torch.cat([H_bho, H_cross], dim=-1)
        # Project + activate + norm + dropout
        fused = self.dropout(self.act(self.proj(combined)))
        # Residual: add Bhojpuri representation back (like adapter residual)
        fused = self.layer_norm(fused + H_bho)
        return fused   # [B, m, d]

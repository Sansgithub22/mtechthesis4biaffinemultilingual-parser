# model/cross_sentence_attention.py
# Step 4 — Cross-Sentence Attention
#
# After encoding both languages with the parallel encoder, we compute
# attention between source (Hindi) and target (Bhojpuri) representations.
#
# Setup:
#   Query  Q = H_bho  [1, m, d]   — Bhojpuri asks "which Hindi tokens matter?"
#   Key    K = H_hi   [1, n, d]   — Hindi keys
#   Value  V = H_hi   [1, n, d]   — Hindi values
#
#   A = MultiheadAttention(Q, K, V)  →  [1, m, d]
#
# Each Bhojpuri token i attends to all Hindi tokens and receives a
# syntactically-aligned summary of the Hindi sentence.  The attention weights
# [1, n_heads, m, n] also provide an interpretable soft alignment.
#
# Optionally, the module also returns the attention weight matrix which is used
# by the alignment loss in train_bilingual.py.

import torch
import torch.nn as nn
from typing import Optional, Tuple


class CrossSentenceAttention(nn.Module):
    """
    Multi-head cross-sentence attention: Bhojpuri queries Hindi.

        H_cross, attn_weights = CrossSentenceAttention(H_bho, H_hi)

    Args:
        hidden_dim : int   — XLM-R hidden size (768)
        n_heads    : int   — number of attention heads (8)
        dropout    : float — attention dropout

    Returns:
        H_cross      [1, m, hidden_dim]  — attended Hindi features
        attn_weights [1, m, n]           — averaged over heads
    """

    def __init__(self, hidden_dim: int = 768, n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        assert hidden_dim % n_heads == 0, \
            f"hidden_dim ({hidden_dim}) must be divisible by n_heads ({n_heads})"
        self.mha = nn.MultiheadAttention(
            embed_dim   = hidden_dim,
            num_heads   = n_heads,
            dropout     = dropout,
            batch_first = True,   # expects [batch, seq, dim]
        )

    def forward(
        self,
        H_bho:    torch.Tensor,          # [1, m, d]  query
        H_hi:     torch.Tensor,          # [1, n, d]  key / value
        key_mask: Optional[torch.Tensor] = None,  # [1, n] padding mask
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        key_mask: True where positions are *padding* (will be ignored).
        """
        H_cross, attn_weights = self.mha(
            query              = H_bho,
            key                = H_hi,
            value              = H_hi,
            key_padding_mask   = key_mask,
            need_weights       = True,
            average_attn_weights = True,  # average over heads → [1, m, n]
        )
        return H_cross, attn_weights  # H_cross: [1, m, d]

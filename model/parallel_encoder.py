# model/parallel_encoder.py
# Step 3 — Parallel Trankit Architecture
#
# Two pipelines share one frozen XLM-RoBERTa backbone (exactly as Trankit does
# internally) but each language has its own lightweight bottleneck adapter
# (Pfeiffer et al., 2020 — the adapter architecture Trankit uses).
#
# Architecture per language:
#   XLM-R hidden state  →  LayerNorm
#                       →  Linear(768 → r)  [down-projection]
#                       →  GELU
#                       →  Linear(r → 768)  [up-projection]
#                       +  residual
#                       =  adapted representation
#
# Subword → word alignment:
#   XLM-R tokenises with SentencePiece (byte-pair encoding).  We map each word
#   to the representation of its first subword token so that downstream layers
#   always see one vector per word (matching the CoNLL-U token count).

from __future__ import annotations

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from transformers import AutoModel, AutoTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Pfeiffer-style bottleneck adapter
# ─────────────────────────────────────────────────────────────────────────────
class BottleneckAdapter(nn.Module):
    """
    A single Pfeiffer adapter applied after XLM-R's final hidden layer.
    Trainable parameters: ~2 × 768 × r  (e.g., 98 304 for r=64).
    """
    def __init__(self, d_model: int = 768, bottleneck: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.layer_norm = nn.LayerNorm(d_model)
        self.down_proj  = nn.Linear(d_model, bottleneck)
        self.act        = nn.GELU()
        self.up_proj    = nn.Linear(bottleneck, d_model)
        self.dropout    = nn.Dropout(dropout)
        # initialise up_proj to near-zero so adapter starts as identity
        nn.init.zeros_(self.up_proj.weight)
        nn.init.zeros_(self.up_proj.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.up_proj(self.act(self.down_proj(self.layer_norm(x)))))


# ─────────────────────────────────────────────────────────────────────────────
# Parallel encoder
# ─────────────────────────────────────────────────────────────────────────────
class ParallelEncoder(nn.Module):
    """
    One shared XLM-RoBERTa backbone  +  two language-specific adapters.

    Forward accepts either a single language or both languages at once:
        encode_one(lang, words)   → [1, n_words, 768]
        encode_pair(hi_words, bho_words) → (H_hi, H_bho)

    lang ∈ {'hindi', 'bhojpuri'}
    """

    LANG_NAMES = ("hindi", "bhojpuri")

    def __init__(self, model_name: str = "xlm-roberta-base",
                 adapter_dim: int = 64, adapter_dropout: float = 0.1,
                 freeze_xlmr: bool = True):
        super().__init__()

        print(f"  Loading {model_name} …", end=" ", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # XLM-R is always kept on CPU — it is frozen (no gradients) so there is
        # no benefit to putting it on MPS/CUDA, and it avoids a 278M-param
        # device transfer that crashes MPS on PyTorch 2.0.
        self.xlmr = AutoModel.from_pretrained(model_name)
        self.xlmr.to("cpu")
        print("done.")

        if freeze_xlmr:
            for p in self.xlmr.parameters():
                p.requires_grad = False

        self.hidden_size = self.xlmr.config.hidden_size  # 768 for base

        # One adapter per language — these are the only trainable params in
        # the encoder; they live on whatever device the trainer uses.
        self.adapters = nn.ModuleDict({
            lang: BottleneckAdapter(self.hidden_size, adapter_dim, adapter_dropout)
            for lang in self.LANG_NAMES
        })

    # ── Core encoding ────────────────────────────────────────────────────────
    def _encode_words(
        self,
        words: List[str],
        lang:  str,
    ) -> torch.Tensor:
        """
        Tokenise `words`, run through XLM-R + adapter, aggregate subwords
        back to word-level using first-subword strategy.

        Returns:
            H  [1, n_words, hidden_size]   (batch dim = 1)
        """
        # XLM-R always runs on CPU; adapters run on their own device
        adapter_device = next(self.adapters[lang].parameters()).device

        # 1. Tokenise each word separately to track subword counts
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=512,
        )
        # XLM-R input always on CPU
        input_ids      = encoding["input_ids"]        # CPU
        attention_mask = encoding["attention_mask"]   # CPU

        # 2. XLM-R forward on CPU (frozen — no grad needed)
        with torch.no_grad():
            outputs = self.xlmr(input_ids=input_ids,
                                attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [1, n_subwords, 768] on CPU

        # 3. Subword → word alignment BEFORE adapter (matches training-time cache path:
        #    precompute_xlmr pools to word level, then encode_one applies adapter to
        #    the word-level tensor).  Applying adapter at subword level would be a
        #    train/eval mismatch and produce garbage output on unseen data.
        word_ids = encoding.word_ids(batch_index=0)  # list len=n_subwords
        word_hidden = self._first_subword_pool(hidden[0], word_ids, len(words))
        # word_hidden: [n_words, 768]  (CPU)

        # 4. Language-specific adapter (trainable) — operates on word-level reps
        return self.adapters[lang](word_hidden.unsqueeze(0).to(adapter_device))  # [1, n_words, 768]

    @staticmethod
    def _first_subword_pool(
        hidden:   torch.Tensor,  # [n_subwords, d]
        word_ids: List[Optional[int]],
        n_words:  int,
    ) -> torch.Tensor:
        """Keep only the hidden state of the *first* subword per word."""
        d = hidden.size(-1)
        result = hidden.new_zeros(n_words, d)
        seen   = set()
        for subw_idx, w_id in enumerate(word_ids):
            if w_id is None:         # [CLS] or [SEP]
                continue
            if w_id not in seen and w_id < n_words:
                result[w_id] = hidden[subw_idx]
                seen.add(w_id)
        return result

    # ── Public interface ──────────────────────────────────────────────────────
    def encode_one(self, lang: str, words: List[str],
                   cached_xlmr: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Encode a single sentence.  Returns [1, n_words, hidden].
        If cached_xlmr is provided (CPU tensor [1, n_words, d]), skips XLM-R
        and runs only the language adapter — much faster in training loops.
        """
        assert lang in self.LANG_NAMES, f"Unknown language: {lang}"
        if cached_xlmr is not None:
            adapter_device = next(self.adapters[lang].parameters()).device
            return self.adapters[lang](cached_xlmr.to(adapter_device))
        return self._encode_words(words, lang)

    def encode_pair(
        self,
        hindi_words:    List[str],
        bhojpuri_words: List[str],
        cached_hi:  Optional[torch.Tensor] = None,
        cached_bho: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode both sentences.
        Returns (H_hi, H_bho), each [1, n_words, hidden].
        """
        H_hi  = self.encode_one("hindi",    hindi_words,    cached_hi)
        H_bho = self.encode_one("bhojpuri", bhojpuri_words, cached_bho)
        return H_hi, H_bho

    # ── Pre-computation cache (call once before training) ─────────────────────
    def precompute_xlmr(self, sentences_words: List[List[str]],
                        desc: str = "") -> List[torch.Tensor]:
        """
        Run frozen XLM-R on every sentence ONCE and cache word-level
        representations on CPU.  Re-use these across all training epochs —
        since XLM-R is frozen its output never changes, so there is no point
        running it repeatedly.

        Returns a list of CPU tensors, each [1, n_words, 768].
        """
        print(f"  Pre-computing XLM-R embeddings for {len(sentences_words):,}"
              f" {desc} sentences …", flush=True)
        cache = []
        self.xlmr.eval()
        with torch.no_grad():
            for i, words in enumerate(sentences_words):
                if not words:
                    cache.append(None)
                    continue
                encoding = self.tokenizer(
                    words, is_split_into_words=True,
                    return_tensors="pt", padding=False,
                    truncation=True, max_length=512,
                )
                out = self.xlmr(input_ids=encoding["input_ids"],
                                attention_mask=encoding["attention_mask"])
                hidden = out.last_hidden_state   # [1, n_sub, 768] on CPU
                word_ids   = encoding.word_ids(batch_index=0)
                word_h     = self._first_subword_pool(hidden[0], word_ids, len(words))
                cache.append(word_h.unsqueeze(0))   # [1, n_words, 768] CPU
                if (i + 1) % 500 == 0:
                    print(f"    {i+1:,}/{len(sentences_words):,}", flush=True)
        print(f"  Done — cache size: {len(cache)}")
        return cache

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

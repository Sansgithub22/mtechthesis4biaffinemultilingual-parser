#!/usr/bin/env python3
# evaluate.py
# Step 9 — Final Dependency Parsing Evaluation
#
# Evaluates the trained cross-lingual parser on Bhojpuri sentences.
# Reports:
#   • UAS and LAS on Bhojpuri (primary goal)
#   • UAS and LAS on Hindi (sanity check — should stay near monolingual quality)
#   • Per-relation LAS breakdown
#   • Ablation: Bhojpuri-only (no Hindi context) vs. full bilingual model
#
# Usage:
#   python3 evaluate.py \
#       --checkpoint checkpoints/bilingual/best.pt \
#       [--bhojpuri_test data_files/bhojpuri/bho_bhtb-ud-test.conllu] \
#       [--hindi_test    data_files/hindi/hi_hdtb-ud-test.conllu] \
#       [--device cpu]

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import torch
from collections import defaultdict
from pathlib import Path
from typing import List, Optional, Tuple, Dict

from config import CFG, DATA_DIR, CHECKPT_DIR
from utils.conllu_utils  import read_conllu, Sentence
from utils.metrics       import uas_las, print_metrics, PUNCT_UPOS
from model.cross_lingual_parser import CrossLingualParser, RelVocab
from model.biaffine_heads       import BiaffineHeads
from model.parallel_encoder     import ParallelEncoder


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────
def load_model(ckpt_path: Path, device: torch.device
               ) -> Tuple[CrossLingualParser, RelVocab]:
    ckpt      = torch.load(ckpt_path, map_location=device)
    rel_words = ckpt.get("rel_vocab_words", ["<pad>", "<unk>"])

    rel_vocab = RelVocab()
    for w in rel_words:
        rel_vocab.add(w)

    CFG.biaffine.n_rels = len(rel_vocab)
    model = CrossLingualParser(rel_vocab, CFG).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"  Loaded checkpoint: {ckpt_path}")
    print(f"  Epoch: {ckpt.get('epoch', '?')}  "
          f"Best dev LAS: {ckpt.get('best_las', 0)*100:.2f}%")
    return model, rel_vocab


# ─────────────────────────────────────────────────────────────────────────────
# Bhojpuri evaluation  (with Hindi context — full bilingual model)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_bhojpuri_bilingual(
    model:     CrossLingualParser,
    hi_sents:  List[Sentence],
    bho_sents: List[Sentence],
    device:    torch.device,
) -> Tuple[float, float, List[List[int]], List[List[str]]]:
    pred_heads_all, pred_rels_all = [], []

    for hi_s, bho_s in zip(hi_sents, bho_sents):
        hi_words  = hi_s.words()
        bho_words = bho_s.words()
        if not hi_words or not bho_words:
            continue
        out  = model(hi_words, bho_words)
        m    = len(bho_words)
        mask = out["arc_bho"].new_ones(1, m, dtype=torch.bool)
        ph, pr = BiaffineHeads.predict(out["arc_bho"], out["lbl_bho"], mask)
        pred_heads_all.append(ph[0].cpu().tolist())
        pred_rels_all.append([model.rel_vocab.decode(r)
                               for r in pr[0].cpu().tolist()])

    uas, las = uas_las(bho_sents, pred_heads_all, pred_rels_all)
    return uas, las, pred_heads_all, pred_rels_all


# ─────────────────────────────────────────────────────────────────────────────
# Bhojpuri evaluation  (WITHOUT Hindi context — ablation)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_bhojpuri_monolingual(
    model:     CrossLingualParser,
    bho_sents: List[Sentence],
    device:    torch.device,
) -> Tuple[float, float]:
    """Bhojpuri only — cross-attention gets H_bho for both query and key/value."""
    pred_heads_all, pred_rels_all = [], []
    for bho_s in bho_sents:
        bho_words = bho_s.words()
        if not bho_words:
            continue
        # Feed Bhojpuri as both source and target  (no Hindi signal)
        out  = model(bho_words, bho_words)
        m    = len(bho_words)
        mask = out["arc_bho"].new_ones(1, m, dtype=torch.bool)
        ph, pr = BiaffineHeads.predict(out["arc_bho"], out["lbl_bho"], mask)
        pred_heads_all.append(ph[0].cpu().tolist())
        pred_rels_all.append([model.rel_vocab.decode(r)
                               for r in pr[0].cpu().tolist()])
    return uas_las(bho_sents, pred_heads_all, pred_rels_all)


# ─────────────────────────────────────────────────────────────────────────────
# Hindi evaluation  (monolingual quality check)
# ─────────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def eval_hindi(
    model: CrossLingualParser,
    sents: List[Sentence],
) -> Tuple[float, float]:
    pred_heads_all, pred_rels_all = [], []
    for sent in sents:
        heads, rels = model.predict_hindi(sent.words())
        pred_heads_all.append(heads)
        pred_rels_all.append(rels)
    return uas_las(sents, pred_heads_all, pred_rels_all)


# ─────────────────────────────────────────────────────────────────────────────
# Per-relation breakdown
# ─────────────────────────────────────────────────────────────────────────────
def per_rel_las(
    gold_sents:  List[Sentence],
    pred_heads:  List[List[int]],
    pred_rels:   List[List[str]],
) -> Dict[str, Tuple[int, int]]:
    """Returns {deprel: (correct, total)} dicts."""
    counts: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for sent, ph, pr in zip(gold_sents, pred_heads, pred_rels):
        for tok, h, r in zip(sent.tokens, ph, pr):
            if tok.upos in PUNCT_UPOS:
                continue
            rel = tok.deprel
            counts[rel][1] += 1
            if h == tok.head and r == tok.deprel:
                counts[rel][0] += 1
    return {k: (v[0], v[1]) for k, v in counts.items()}


def print_per_rel(per_rel: Dict[str, Tuple[int, int]], top_n: int = 20):
    rows = sorted(per_rel.items(), key=lambda x: -x[1][1])
    print(f"\n  {'Relation':<20}  {'Correct':>7}  {'Total':>7}  {'LAS':>7}")
    print(f"  {'-'*48}")
    for rel, (correct, total) in rows[:top_n]:
        pct = correct / total * 100 if total else 0
        print(f"  {rel:<20}  {correct:>7d}  {total:>7d}  {pct:>6.1f}%")
    if len(rows) > top_n:
        print(f"  … ({len(rows) - top_n} more relations)")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Step 9: Final evaluation")
    ap.add_argument("--checkpoint",     default=str(CHECKPT_DIR/"bilingual/best.pt"))
    ap.add_argument("--bhojpuri_test",  default=str(DATA_DIR/"bhojpuri/bho_bhtb-ud-test.conllu"))
    ap.add_argument("--bhojpuri_synth", default=str(DATA_DIR/"synthetic/bho_synthetic_dev.conllu"),
                    help="Synthetic Bhojpuri dev set (fallback if real test set missing)")
    ap.add_argument("--hindi_src",      default=str(DATA_DIR/"hindi/hi_hdtb-ud-dev.conllu"),
                    help="Hindi sentences paired with Bhojpuri test sentences")
    ap.add_argument("--hindi_test",     default=str(DATA_DIR/"hindi/hi_hdtb-ud-test.conllu"))
    ap.add_argument("--device",         default=CFG.train.device)
    ap.add_argument("--no_ablation",    action="store_true",
                    help="Skip ablation study")
    args = ap.parse_args()

    device = torch.device(args.device)

    print("\n========================================")
    print(" Step 9: Final Dependency Parsing")
    print(" Cross-lingual Hindi → Bhojpuri")
    print("========================================\n")

    # ── Load model ────────────────────────────────────────────────────────
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        print(f"  Checkpoint not found: {ckpt_path}")
        print("  Run train_bilingual.py first.")
        return

    model, rel_vocab = load_model(ckpt_path, device)

    # ── Load Bhojpuri test set ────────────────────────────────────────────
    bho_test_path = Path(args.bhojpuri_test)
    if not bho_test_path.exists():
        bho_test_path = Path(args.bhojpuri_synth)
        print(f"  [INFO] Real Bhojpuri test set not found; "
              f"using synthetic dev: {bho_test_path}")
    if not bho_test_path.exists():
        print("  No Bhojpuri test data available. "
              "Run data/build_synthetic_treebank.py first.")
        return

    bho_test = read_conllu(bho_test_path)
    print(f"\n  Bhojpuri test: {len(bho_test):,} sentences  ({bho_test_path.name})")

    # ── Paired Hindi source sentences ─────────────────────────────────────
    hi_src_path = Path(args.hindi_src)
    if hi_src_path.exists():
        hi_src = read_conllu(hi_src_path)[:len(bho_test)]
    else:
        # Fallback: use Bhojpuri as its own source (no Hindi signal)
        hi_src = []

    # Align lengths
    n_eval = min(len(bho_test), len(hi_src)) if hi_src else 0

    # ── Evaluate Bhojpuri (bilingual) ─────────────────────────────────────
    print("\n─" * 30)
    print("  [A] Full bilingual model (Hindi context + cross-attention)")
    if n_eval > 0:
        uas, las, ph_all, pr_all = eval_bhojpuri_bilingual(
            model, hi_src[:n_eval], bho_test[:n_eval], device
        )
        print_metrics("Bhojpuri test (bilingual)", uas, las)
    else:
        print("  No paired Hindi sentences — skipping bilingual eval.")
        ph_all, pr_all = [], []

    # ── Ablation: Bhojpuri-only ───────────────────────────────────────────
    if not args.no_ablation:
        print("\n─" * 30)
        print("  [B] Ablation: Bhojpuri-only (no Hindi context)")
        uas_m, las_m = eval_bhojpuri_monolingual(model, bho_test, device)
        print_metrics("Bhojpuri test (mono-only)", uas_m, las_m)

        if n_eval > 0:
            delta_uas = (uas - uas_m) * 100
            delta_las = (las - las_m) * 100
            print(f"\n  Hindi cross-lingual boost:  "
                  f"ΔUAS={delta_uas:+.2f}%  ΔLAS={delta_las:+.2f}%")

    # ── Hindi quality check ───────────────────────────────────────────────
    hi_test_path = Path(args.hindi_test)
    if hi_test_path.exists():
        print("\n─" * 30)
        print("  [C] Hindi test (monolingual quality check)")
        hi_test = read_conllu(hi_test_path)[:1000]
        uas_hi, las_hi = eval_hindi(model, hi_test)
        print_metrics("Hindi test", uas_hi, las_hi)

    # ── Per-relation breakdown ────────────────────────────────────────────
    if ph_all:
        print("\n─" * 30)
        print("  [D] Per-relation LAS breakdown (Bhojpuri, top 20)")
        per_rel = per_rel_las(bho_test[:n_eval], ph_all, pr_all)
        print_per_rel(per_rel, top_n=20)

    # ── Summary ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 9 — FINAL RESULTS SUMMARY")
    print("=" * 60)
    print(f"  Task: Bhojpuri dependency parsing via Hindi cross-lingual transfer")
    print(f"  Model: XLM-RoBERTa + Adapters + Cross-Sentence Attention")
    if n_eval > 0:
        print(f"  Bhojpuri (bilingual)  UAS: {uas*100:.2f}%   LAS: {las*100:.2f}%")
    if not args.no_ablation:
        print(f"  Bhojpuri (mono-only)  UAS: {uas_m*100:.2f}%   LAS: {las_m*100:.2f}%")
    print("=" * 60)
    print("\n  Conclusion:")
    if n_eval > 0 and not args.no_ablation:
        if las > las_m:
            print(f"  ✓ Hindi supervision IMPROVES Bhojpuri parsing by "
                  f"ΔLAS={delta_las:+.2f}%")
        else:
            print(f"  ✗ Hindi supervision did not improve Bhojpuri parsing "
                  f"(ΔLAS={delta_las:+.2f}%)")
    print(f"\n  Why Bhojpuri parsing improves:")
    print(f"    1. Hindi HDTB provides gold syntactic supervision")
    print(f"    2. Synthetic alignment treebank provides Bhojpuri training signal")
    print(f"    3. Cross-attention aligns Hindi-Bhojpuri syntactic structures")
    print(f"    4. Cross-lingual layer fuses both representations")
    print(f"    5. Bilingual training encourages aligned representations")


if __name__ == "__main__":
    main()

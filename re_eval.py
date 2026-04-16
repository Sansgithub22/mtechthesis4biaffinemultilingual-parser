#!/usr/bin/env python3
"""
re_eval.py — Re-evaluate saved checkpoints with the FIXED parallel_encoder.

The original runs reported ~8% BHTB UAS because _encode_words applied the
adapter at the subword level, while training used the cache (word-level).
This script loads the stored checkpoint weights (trained correctly with
word-level input) and evaluates with the now-correct _encode_words path.

Usage (on HPC after `git pull`):
    python3 re_eval.py --system g --device cuda
    python3 re_eval.py --system h --device cuda
    python3 re_eval.py --system i --device cuda
"""

from __future__ import annotations

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import argparse
import datetime
import torch
from pathlib import Path

from config import CHECKPT_DIR, DATA_DIR, XLM_R_LOCAL
from utils.conllu_utils import read_conllu
from utils.metrics import uas_las
from model.parallel_encoder import ParallelEncoder
from model.biaffine_heads import BiaffineHeads

ROOT_DIR  = Path(__file__).parent
PROF_BHO  = ROOT_DIR / "bhojpuri_matched_transferred.conllu"
BHTB_TEST = DATA_DIR / "bhojpuri" / "bho_bhtb-ud-test.conllu"

# Results from the buggy runs (for comparison)
PREV_RESULTS = {
    "g": dict(int_uas=55.49, int_las=49.79, bhtb_uas=7.92,  bhtb_las=0.28),
    "h": dict(int_uas=55.08, int_las=49.36, bhtb_uas=8.63,  bhtb_las=0.45),
    "i": dict(int_uas=55.08, int_las=49.36, bhtb_uas=8.63,  bhtb_las=0.45),
}


def evaluate(encoder, parser, vocab, sents, device):
    encoder.eval(); parser.eval()
    pred_heads_all, pred_rels_all = [], []
    with torch.no_grad():
        for sent in sents:
            if not sent.tokens:
                pred_heads_all.append([]); pred_rels_all.append([]); continue
            H = encoder.encode_one("bhojpuri", sent.words()).to(device)
            arc_s, lbl_s = parser(H)
            mask = torch.ones(1, len(sent.tokens), dtype=torch.bool, device=device)
            ph, pr = BiaffineHeads.predict(arc_s, lbl_s, mask)
            pred_heads_all.append(ph[0].cpu().tolist())
            pred_rels_all.append([vocab.decode(r) for r in pr[0].cpu().tolist()])
    return uas_las(sents, pred_heads_all, pred_rels_all)


def main():
    ap = argparse.ArgumentParser(description="Re-evaluate saved checkpoints with fixed encoder")
    ap.add_argument("--system", choices=["g", "h", "i"], required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    sys_name = args.system.lower()
    ckpt_path = CHECKPT_DIR / f"system_{sys_name}/system_{sys_name}.pt"

    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        return

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available — falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"  Re-evaluating System {sys_name.upper()} with fixed encoder")
    print(f"{'='*60}")
    print(f"  Checkpoint : {ckpt_path}")
    print(f"  Device     : {device}")

    # ── Load checkpoint ────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint …")
    ckpt       = torch.load(str(ckpt_path), map_location=device)
    vocab      = ckpt["vocab"]
    saved_args = ckpt.get("args", {})
    n_rels     = len(vocab)
    print(f"  Epoch      : {ckpt['epoch']}")
    print(f"  Best DevLAS: {ckpt.get('best_las', 0)*100:.2f}%")
    print(f"  n_rels     : {n_rels}")

    # ── Build model ────────────────────────────────────────────────────────────
    print(f"\nBuilding model ({XLM_R_LOCAL}) …")
    encoder = ParallelEncoder(
        model_name=XLM_R_LOCAL, adapter_dim=64,
        adapter_dropout=0.1, freeze_xlmr=True,
    )
    encoder.adapters.to(device)
    parser_bho = BiaffineHeads(768, 500, 100, n_rels, 0.33).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    parser_bho.load_state_dict(ckpt["parser_bho"])
    print("  Model weights loaded.")

    # ── BHTB evaluation ────────────────────────────────────────────────────────
    print(f"\nLoading BHTB test set …")
    bhtb_sents = read_conllu(BHTB_TEST)
    print(f"  BHTB: {len(bhtb_sents):,} sentences")
    print(f"Evaluating on BHTB …")
    bhtb_uas, bhtb_las = evaluate(encoder, parser_bho, vocab, bhtb_sents, device)

    # ── Internal test set ──────────────────────────────────────────────────────
    dev_ratio  = saved_args.get("dev_ratio",  0.1)
    test_ratio = saved_args.get("test_ratio", 0.1)
    bho_sents  = read_conllu(PROF_BHO)
    n_total    = len(bho_sents)
    n_test     = max(1, int(n_total * test_ratio))
    n_dev      = max(1, int(n_total * dev_ratio))
    n_train    = n_total - n_dev - n_test
    test_idx   = list(range(n_train + n_dev, n_total))
    int_test   = [bho_sents[i] for i in test_idx]
    print(f"\nEvaluating on internal test ({len(test_idx):,} sentences) …")
    int_uas, int_las = evaluate(encoder, parser_bho, vocab, int_test, device)

    # ── Print results ──────────────────────────────────────────────────────────
    prev = PREV_RESULTS[sys_name]
    print(f"\n{'='*60}")
    print(f"  System {sys_name.upper()} — Results (fixed encoder vs buggy)")
    print(f"{'='*60}")
    print(f"  {'Test Set':<35} {'UAS':>7} {'LAS':>7}")
    print(f"  {'─'*51}")
    print(f"  {'Internal (10% prof data)':<35} {int_uas*100:>6.2f}% {int_las*100:>6.2f}%"
          f"   [was {prev['int_uas']:.2f}% / {prev['int_las']:.2f}%]")
    print(f"  {'BHTB (external gold)':<35} {bhtb_uas*100:>6.2f}% {bhtb_las*100:>6.2f}%"
          f"   [was {prev['bhtb_uas']:.2f}% / {prev['bhtb_las']:.2f}%]")
    print(f"  {'─'*51}")
    print(f"  Baseline System A (zero-shot): UAS 53.48% / LAS 34.84%")
    print(f"{'='*60}")

    # ── Save ───────────────────────────────────────────────────────────────────
    results_dir = ROOT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rf_path = results_dir / f"system_{sys_name}_fixed_{ts}.txt"
    with open(rf_path, "w") as rf:
        rf.write(f"System {sys_name.upper()} — Re-evaluation with FIXED parallel_encoder\n")
        rf.write(f"{'='*60}\n")
        rf.write(f"Date          : {datetime.datetime.now()}\n")
        rf.write(f"Checkpoint    : {ckpt_path}\n")
        rf.write(f"Best epoch    : {ckpt['epoch']}\n")
        rf.write(f"Best Dev LAS  : {ckpt.get('best_las', 0)*100:.2f}%\n\n")
        rf.write(f"{'Test Set':<35} {'UAS':>8} {'LAS':>8}\n")
        rf.write(f"{'─'*53}\n")
        rf.write(f"{'Internal (10% prof data)':<35} {int_uas*100:>7.2f}% {int_las*100:>7.2f}%  ({len(test_idx)} sents)\n")
        rf.write(f"{'BHTB (external gold)':<35} {bhtb_uas*100:>7.2f}% {bhtb_las*100:>7.2f}%\n")
        rf.write(f"{'─'*53}\n")
        rf.write(f"Baseline (System A zero-shot): UAS 53.48% / LAS 34.84%\n\n")
        rf.write(f"Previous (buggy) results:\n")
        rf.write(f"  Internal  : UAS {prev['int_uas']:.2f}% LAS {prev['int_las']:.2f}%\n")
        rf.write(f"  BHTB      : UAS {prev['bhtb_uas']:.2f}% LAS {prev['bhtb_las']:.2f}%\n")
    print(f"\n  Results saved → {rf_path}")


if __name__ == "__main__":
    main()

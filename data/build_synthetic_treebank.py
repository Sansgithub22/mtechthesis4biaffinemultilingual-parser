#!/usr/bin/env python3
# data/build_synthetic_treebank.py
# Step 2 orchestrator: runs the full synthetic treebank pipeline:
#
#   Hindi CoNLL-U
#       │
#       ├── translate_hindi.py  →  translations TSV  (Hindi \t Bhojpuri)
#       │
#       ├── word_alignment.py   →  alignments TXT    (Pharaoh format)
#       │
#       └── project_annotations.py  →  Bhojpuri CoNLL-U  (projected labels)
#
# Usage:
#   python3 data/build_synthetic_treebank.py [--max_sents N] [--method dict]

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
from pathlib import Path

from config import CFG, DATA_DIR
from data.translate_hindi    import translate_conllu
from data.word_alignment      import align_translations
from data.project_annotations import project_treebank


def build_split(
    split:     str,      # "train" | "dev"
    method:    str,
    max_sents: int,
):
    hindi_file  = DATA_DIR / "hindi" / f"hi_hdtb-ud-{split}.conllu"
    trans_file  = DATA_DIR / "synthetic" / f"translations_{split}.txt"
    align_file  = DATA_DIR / "synthetic" / f"alignments_{split}.txt"
    output_file = DATA_DIR / "synthetic" / f"bho_synthetic_{split}.conllu"

    if not hindi_file.exists():
        print(f"  MISSING: {hindi_file}\n"
              f"  Run:  python3 data/download_ud_data.py  first.")
        return

    print(f"\n{'─'*60}")
    print(f"  Building synthetic Bhojpuri treebank — {split} split")
    print(f"{'─'*60}")

    # ── 2a. Translate ─────────────────────────────────────────────────────
    print(f"\n[2a] Translating Hindi → Bhojpuri  (method={method})")
    translate_conllu(hindi_file, trans_file, method, max_sents)

    # ── 2b. Align ────────────────────────────────────────────────────────
    print(f"\n[2b] Word alignment (SimAlign)")
    align_translations(trans_file, align_file, method="inter")

    # ── 2c. Project ──────────────────────────────────────────────────────
    print(f"\n[2c] Projecting annotations  (Steps 2c + 6)")
    project_treebank(
        hindi_conllu     = hindi_file,
        translations_tsv = trans_file,
        alignments_txt   = align_file,
        output_conllu    = output_file,
        max_sents        = max_sents,
    )

    # Quick stats
    lines = output_file.read_text(encoding="utf-8").splitlines()
    n_sents = sum(1 for l in lines if l == "")
    n_toks  = sum(1 for l in lines if l and not l.startswith("#"))
    print(f"\n  [{split}] {n_sents:,} sentences, {n_toks:,} tokens → {output_file.name}")


def main():
    ap = argparse.ArgumentParser(
        description="Build synthetic Hindi→Bhojpuri aligned treebank (Steps 2a–2c + 6)"
    )
    ap.add_argument("--method",    choices=["dict", "indic", "google"], default="dict",
                    help="Translation method (default: dict)")
    ap.add_argument("--max_sents", type=int, default=0,
                    help="Limit sentences per split (0 = all)")
    ap.add_argument("--splits",    nargs="+", default=["train", "dev"],
                    choices=["train", "dev"])
    args = ap.parse_args()

    CFG.make_dirs()

    print("\n========================================")
    print(" Step 2: Synthetic Alignment Treebank")
    print("========================================")

    for split in args.splits:
        build_split(split, args.method, args.max_sents)

    print("\n\nAll splits done.")
    print("Next: python3 train_monolingual.py   (Step 7 — monolingual pre-training)")


if __name__ == "__main__":
    main()

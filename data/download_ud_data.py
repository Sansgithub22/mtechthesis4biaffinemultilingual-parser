#!/usr/bin/env python3
# data/download_ud_data.py
# Step 1: Download Universal Dependencies treebanks needed for this project.
#
#   Hindi  : UD_Hindi-HDTB  (16,647 train / 1,659 dev / 1,684 test sentences)
#   Bhojpuri: UD_Bhojpuri-BHTB (evaluation set; ~100 test sentences)
#
# Usage:
#   python3 data/download_ud_data.py

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import requests
import gzip
import shutil
from pathlib import Path
from config import CFG, DATA_DIR

CFG.make_dirs()

# ─────────────────────────────────────────────────────────────────────────────
# Raw GitHub URLs for UD v2.13 (latest stable)
# ─────────────────────────────────────────────────────────────────────────────
UD_BASE = "https://raw.githubusercontent.com/UniversalDependencies"
UD_REF  = "r2.13"      # tag for UD v2.13

HINDI_FILES = {
    "hi_hdtb-ud-train.conllu": f"{UD_BASE}/UD_Hindi-HDTB/{UD_REF}/hi_hdtb-ud-train.conllu",
    "hi_hdtb-ud-dev.conllu":   f"{UD_BASE}/UD_Hindi-HDTB/{UD_REF}/hi_hdtb-ud-dev.conllu",
    "hi_hdtb-ud-test.conllu":  f"{UD_BASE}/UD_Hindi-HDTB/{UD_REF}/hi_hdtb-ud-test.conllu",
}

BHOJPURI_FILES = {
    "bho_bhtb-ud-test.conllu": f"{UD_BASE}/UD_Bhojpuri-BHTB/{UD_REF}/bho_bhtb-ud-test.conllu",
}


def download_file(url: str, dest: Path, desc: str):
    if dest.exists():
        print(f"  [skip] {desc} already exists.")
        return
    print(f"  Downloading {desc} …", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        dest.write_bytes(r.content)
        lines = dest.read_text(encoding="utf-8").count("\n")
        print(f"done  ({lines:,} lines, {dest.stat().st_size//1024} KB)")
    except Exception as e:
        print(f"\n  ERROR: {e}")
        print(f"  Please download manually from:\n  {url}")
        print(f"  and save to: {dest}")


def main():
    print("\n=== Step 1: Downloading UD treebanks ===\n")

    print("Hindi HDTB:")
    for fname, url in HINDI_FILES.items():
        download_file(url, DATA_DIR / "hindi" / fname, fname)

    print("\nBhojpuri BHTB:")
    for fname, url in BHOJPURI_FILES.items():
        download_file(url, DATA_DIR / "bhojpuri" / fname, fname)

    # Verify counts
    print("\n--- Summary ---")
    for path in (DATA_DIR / "hindi").glob("*.conllu"):
        n = sum(1 for l in open(path, encoding="utf-8") if l.strip() == "")
        print(f"  {path.name}: {n:,} sentences")
    for path in (DATA_DIR / "bhojpuri").glob("*.conllu"):
        n = sum(1 for l in open(path, encoding="utf-8") if l.strip() == "")
        print(f"  {path.name}: {n:,} sentences")

    print("\nDone. Next step: python3 data/build_synthetic_treebank.py")


if __name__ == "__main__":
    main()

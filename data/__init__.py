"""Data pipeline for the Hindi → Bhojpuri cross-lingual parser.

Step 1 — download_ud_data.py   : Download Hindi HDTB + Bhojpuri BHTB from GitHub
Step 2 — translate_hindi.py    : Translate Hindi sentences → Bhojpuri
        word_alignment.py      : Align Hindi ↔ Bhojpuri words (SimAlign / XLM-R)
        project_annotations.py : Project UD dependency arcs across the alignment
        build_synthetic_treebank.py : Orchestrate the whole Step 2 pipeline
"""

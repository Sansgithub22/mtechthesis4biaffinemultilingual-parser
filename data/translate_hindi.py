#!/usr/bin/env python3
# data/translate_hindi.py
# Step 2a: Translate Hindi sentences to Bhojpuri.
#
# Three back-ends are supported (select with --method):
#
#   dict   — Fast rule-based word substitution using a curated Hindi→Bhojpuri
#             lexicon.  Good for demos; covers the most frequent function-word
#             differences between the two languages.
#
#   indic  — IndicTrans2 neural MT (best quality).  Requires the
#             ai4bharat/indictrans2-hi-indic-1B model from HuggingFace.
#             Install: pip install indic-transliteration sacremoses
#
#   google — Google Cloud Translation API (requires GOOGLE_API_KEY env var).
#
# Usage:
#   python3 data/translate_hindi.py --method dict  \
#       --input data_files/hindi/hi_hdtb-ud-train.conllu \
#       --output data_files/synthetic/translations_train.txt \
#       --max_sents 5000

from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import argparse
from pathlib import Path
from typing import List

from utils.conllu_utils import read_conllu


# ─────────────────────────────────────────────────────────────────────────────
# Hindi → Bhojpuri word-level substitution lexicon
# Bhojpuri is closely related to Hindi (both Indo-Aryan); primary differences
# are in verb inflection, pronouns, and some postpositions.
# ─────────────────────────────────────────────────────────────────────────────
HINDI_TO_BHOJPURI: dict = {
    # Pronouns
    "मैं": "हम",       "मुझे": "हमके",      "मुझको": "हमके",
    "मेरा": "हमार",    "मेरी": "हमार",       "मेरे": "हमार",
    "हम": "हमनी",      "हमें": "हमनी के",
    "तुम": "तू",       "तुम्हें": "तोहके",   "तुम्हारा": "तोहार",
    "आप": "रउआ",       "आपको": "रउआके",      "आपका": "रउआकर",
    "वह": "ऊ",         "उसे": "ओके",          "उसको": "ओके",
    "उसका": "ओकर",     "उसकी": "ओकर",         "उसके": "ओकर",
    "वे": "ऊलोग",      "उन्हें": "उनका",
    "यह": "ई",         "इसे": "एके",          "इसको": "एके",
    "इसका": "एकर",
    "यहाँ": "इहाँ",    "यहां": "इहाँ",
    "वहाँ": "उहाँ",    "वहां": "उहाँ",
    "कौन": "के",       "क्या": "का",          "कब": "कब",
    "कहाँ": "कहाँ",    "कैसे": "कइसे",        "कितना": "केतना",
    # Postpositions / case markers
    "को": "के",        "का": "के",            "की": "के",
    "ने": "",          "से": "से",            "में": "में",
    "पर": "पर",        "के लिए": "खातिर",     "तक": "ले",
    "के साथ": "संगे",  "के बाद": "के बाद",    "के पास": "लगे",
    # Copula / auxiliaries
    "है": "बा",        "हैं": "बाड़न",         "हूँ": "बानी",
    "था": "रहे",       "थी": "रहे",           "थे": "रहलन",
    "होगा": "होई",     "होगी": "होई",
    "है।": "बा।",      "हैं।": "बाड़न।",
    # Common verbs (infinitive)
    "जाना": "जाये",    "जाता": "जाला",        "जाती": "जाले",
    "जाते": "जालन",    "गया": "गइल",          "गई": "गइल",
    "आना": "आये",      "आता": "आवेला",        "आती": "आवेले",
    "आते": "आवेलन",    "आया": "अइल",
    "खाना": "खाये",    "खाता": "खाला",        "खाती": "खाले",
    "खाते": "खालन",    "खाया": "खाइल",
    "पढ़ना": "पढ़े",   "पढ़ता": "पढ़ेला",     "पढ़ती": "पढ़ेले",
    "पढ़ते": "पढ़ेलन", "पढ़ा": "पढ़ल",
    "करना": "करे",     "करता": "करेला",       "करती": "करेले",
    "करते": "करेलन",   "किया": "कइल",         "की": "कइल",
    "देखना": "देखे",   "देखता": "देखेला",
    "बोलना": "बोले",   "बोलता": "बोलेला",
    "रहना": "रहे",     "रहता": "रहेला",       "रहती": "रहेले",
    # Common nouns / conjunctions
    "और": "आउर",       "लेकिन": "बाकिर",      "क्योंकि": "काहे कि",
    "जब": "जब",        "तब": "तब",            "अगर": "अगर",
    "घर": "घर",        "पानी": "पनिया",       "खाना": "खाना",
    "किताब": "किताब",  "स्कूल": "स्कूल",     "आज": "आज",
    "कल": "काल्ह",    "अब": "अब",
}


def translate_dict(sentence: str) -> str:
    """
    Word-level substitution using the Hindi→Bhojpuri lexicon.
    Preserves word order (same in both languages).
    Unknown words are kept as-is (Hindi and Bhojpuri share most content words).
    """
    words = sentence.split()
    translated = []
    i = 0
    while i < len(words):
        # Try 2-word phrases first
        if i + 1 < len(words):
            bigram = words[i] + " " + words[i + 1]
            if bigram in HINDI_TO_BHOJPURI:
                replacement = HINDI_TO_BHOJPURI[bigram]
                if replacement:
                    translated.append(replacement)
                i += 2
                continue
        w = words[i]
        replacement = HINDI_TO_BHOJPURI.get(w, w)
        if replacement:           # empty string means drop the word
            translated.append(replacement)
        i += 1
    return " ".join(translated)


def translate_indic(sentences: List[str]) -> List[str]:
    """
    Uses IndicTrans2 (ai4bharat/indictrans2-hi-indic-1B) for neural MT.
    Source: Hindi (hin_Deva)  →  Target: Bhojpuri (bho_Deva)
    """
    try:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        MODEL_ID = "ai4bharat/indictrans2-hi-indic-1B"
        print(f"  Loading {MODEL_ID} …")
        tok = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
        mdl = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID, trust_remote_code=True)
        results = []
        for s in sentences:
            inputs = tok(s, return_tensors="pt", padding=True, truncation=True,
                         max_length=256)
            out = mdl.generate(**inputs, num_beams=4, max_length=256)
            results.append(tok.decode(out[0], skip_special_tokens=True))
        return results
    except ImportError:
        raise ImportError(
            "IndicTrans2 not available. "
            "Install: pip install transformers sentencepiece sacremoses\n"
            "Or use --method dict for the rule-based fallback."
        )


def translate_google(sentences: List[str], api_key: str) -> List[str]:
    """
    Google Cloud Translation API v2.
    Needs GOOGLE_API_KEY environment variable or --api_key argument.
    Note: Google Translate Hindi→Bhojpuri quality is limited.
    """
    import requests as req
    url = "https://translation.googleapis.com/language/translate/v2"
    results = []
    for s in sentences:
        r = req.post(url, params={"key": api_key}, json={
            "q": s, "source": "hi", "target": "bho", "format": "text"
        })
        r.raise_for_status()
        results.append(r.json()["data"]["translations"][0]["translatedText"])
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def translate_conllu(
    input_path:  str | Path,
    output_path: str | Path,
    method:      str = "dict",
    max_sents:   int = 0,
    api_key:     str = "",
) -> None:
    """
    Read a Hindi CoNLL-U file, translate every sentence, write a plain-text
    file with one Bhojpuri translation per line (matching the sentence order).
    """
    sentences = read_conllu(input_path)
    if max_sents:
        sentences = sentences[:max_sents]

    src_texts = [" ".join(t.form for t in s.tokens) for s in sentences]

    print(f"  Translating {len(src_texts):,} sentences (method={method}) …")

    if method == "dict":
        tgt_texts = [translate_dict(s) for s in src_texts]
    elif method == "indic":
        tgt_texts = translate_indic(src_texts)
    elif method == "google":
        if not api_key:
            api_key = os.environ.get("GOOGLE_API_KEY", "")
        if not api_key:
            raise ValueError("Set GOOGLE_API_KEY env var or pass --api_key")
        tgt_texts = translate_google(src_texts, api_key)
    else:
        raise ValueError(f"Unknown method: {method}")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        for src, tgt in zip(src_texts, tgt_texts):
            fh.write(f"{src}\t{tgt}\n")

    print(f"  Written {len(tgt_texts):,} sentence pairs → {output_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Translate Hindi CoNLL-U → Bhojpuri plain text")
    ap.add_argument("--input",     default="data_files/hindi/hi_hdtb-ud-train.conllu")
    ap.add_argument("--output",    default="data_files/synthetic/translations_train.txt")
    ap.add_argument("--method",    choices=["dict", "indic", "google"], default="dict")
    ap.add_argument("--max_sents", type=int, default=0, help="0 = all")
    ap.add_argument("--api_key",   default="")
    args = ap.parse_args()

    translate_conllu(args.input, args.output, args.method, args.max_sents, args.api_key)

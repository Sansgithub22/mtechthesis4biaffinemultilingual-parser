#!/usr/bin/env python3
# data/build_selective_treebank.py
# Innovation 3 — Relation-selective projection.
#
# Strategy per token in the projected treebank:
#   HIGH_CONF_RELS (System A LAS ≥ 25%) → keep the projected (head, deprel)
#   LOW_CONF_RELS  (System A LAS  < 25%) → replace with System A's prediction
#
# HIGH_CONF_RELS (derived from baseline per-relation LAS on real BHTB):
#   case(83%), root(50%), punct(46%), mark(42%), conj(41%), advmod(38%),
#   amod(37%), nummod(33%), obl(30%), nmod(29%)
#
# LOW_CONF_RELS (unreliable projection, System A wins here):
#   aux(23%), compound(24%), nsubj(21%), det(20%), obj(15%),
#   acl(1%), advcl(4%), xcomp(0%), ccomp(0%), cc(3%)
#
# Outputs:
#   data_files/synthetic/bho_selective_train.conllu
#   data_files/synthetic/bho_selective_dev.conllu
#
# Usage:
#   python3 data/build_selective_treebank.py [--split train|dev|both]

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import argparse
import shutil
import torch
from pathlib import Path

from config import DATA_DIR, CHECKPT_DIR
from utils.conllu_utils import read_conllu, write_conllu


# ── Relation confidence sets (from System A per-relation LAS on BHTB) ────────
HIGH_CONF_RELS = frozenset({
    "case",     # 83.2% — postpositions, very consistent across Hindi/Bhojpuri
    "root",     # 49.6% — sentence root
    "punct",    # 45.6% — punctuation
    "mark",     # 42.3% — subordinating conjunctions
    "conj",     # 40.9% — coordination
    "advmod",   # 37.9% — adverbial modifier
    "amod",     # 36.6% — adjectival modifier
    "nummod",   # 33.3% — numeric modifier
    "obl",      # 29.8% — oblique nominal
    "nmod",     # 28.8% — nominal modifier
})

LOW_CONF_RELS = frozenset({
    "aux",      # 22.5% — auxiliary
    "compound", # 23.7% — compound
    "nsubj",    # 20.5% — nominal subject
    "det",      # 20.1% — determiner
    "obj",      # 14.8% — object
    "acl",      #  1.0% — clausal modifier
    "advcl",    #  3.7% — adverbial clause
    "xcomp",    #  0.0% — open clausal complement
    "ccomp",    #  0.0% — clausal complement
    "cc",       #  3.3% — coordinating conjunction
})


def _ensure_xlmr_cache_symlink(save_dir: str, lang: str):
    hub_cache = Path.home() / ".cache/huggingface/hub/models--xlm-roberta-base"
    if not hub_cache.exists():
        return
    trankit_save = Path(save_dir) / "xlm-roberta-base" / lang
    for root in [trankit_save, trankit_save / "xlm-roberta-base"]:
        root.mkdir(parents=True, exist_ok=True)
        target = root / "models--xlm-roberta-base"
        if target.exists() and not target.is_symlink():
            shutil.rmtree(target)
        if not target.exists():
            target.symlink_to(hub_cache)


def run_sysA_inference(sentences, split_name: str) -> list:
    """
    Run the Hindi (System A) Trankit model on pre-tokenised Bhojpuri sentences.
    Returns a list of Sentence objects with System A's (head, deprel) predictions.
    """
    from trankit import TPipeline

    hindi_save = str(CHECKPT_DIR / "trankit_hindi")
    hindi_mdl  = CHECKPT_DIR / "trankit_hindi/xlm-roberta-base/hindi/hindi.tagger.mdl"
    hi_train   = DATA_DIR / "hindi" / "hi_hdtb-ud-train.conllu"

    if not hindi_mdl.exists():
        raise FileNotFoundError(f"Hindi checkpoint not found: {hindi_mdl}\n"
                                "Run: python3 train_trankit_hindi.py")

    # Write sentences to a temp file for TPipeline
    tmp_input = CHECKPT_DIR / "trankit_hindi" / f"selective_{split_name}_input.conllu"
    write_conllu(sentences, tmp_input)

    _ensure_xlmr_cache_symlink(hindi_save, lang="hindi")

    print(f"  Initialising System A (Hindi TPipeline) for {split_name} inference …")
    pipeline = TPipeline(training_config={
        "category":           "hindi",
        "task":               "posdep",
        "save_dir":           hindi_save,
        "train_conllu_fpath": str(hi_train),
        "dev_conllu_fpath":   str(tmp_input),
        "max_epoch":          0,
        "batch_size":         32,
        "gpu":                False,
        "embedding":          "xlm-roberta-base",
    })

    # Load Hindi checkpoint
    state    = torch.load(str(hindi_mdl), map_location="cpu")
    adapters = state["adapters"]
    epoch    = state.get("epoch", "?")
    print(f"  Loaded Hindi checkpoint (epoch {epoch})")

    emb_sd = pipeline._embedding_layers.state_dict()
    tag_sd = pipeline._tagger.state_dict()
    for k, v in adapters.items():
        if k in emb_sd:
            emb_sd[k] = v
        if k in tag_sd:
            tag_sd[k] = v
    pipeline._embedding_layers.load_state_dict(emb_sd)
    pipeline._tagger.load_state_dict(tag_sd)
    pipeline._embedding_layers.eval()
    pipeline._tagger.eval()

    print(f"  Running inference on {len(sentences)} sentences …")
    _, pred_path = pipeline._eval_posdep(
        data_set  = pipeline.dev_set,
        batch_num = pipeline.dev_batch_num,
        name      = f"sysA_{split_name}",
        epoch     = epoch,
    )
    return read_conllu(Path(pred_path))


def _would_create_cycle(tok_id: int, new_head: int, heads: dict) -> bool:
    """
    Check whether setting heads[tok_id] = new_head would introduce a cycle.
    Walk the head chain from new_head; if we reach tok_id before reaching 0,
    a cycle would be created.
    """
    cur = new_head
    for _ in range(len(heads) + 2):
        if cur == 0:
            return False   # reached ROOT safely
        if cur == tok_id:
            return True    # cycle detected
        cur = heads.get(cur, 0)
    return True            # infinite loop / dangling pointer — treat as cycle


def build_selective(proj_sents, sysA_sents, out_path: Path):
    """
    Merge projected treebank with System A predictions token-by-token:
      - Token's projected deprel in HIGH_CONF_RELS → keep projected (head, deprel)
      - Otherwise                                  → use System A (head, deprel)

    Safety guards (applied before committing each replacement):
      1. Multiple-root guard: if System A predicts head=0 for a non-root token,
         keep the projected annotation.
      2. Cycle guard: if System A's predicted head would introduce a dependency
         cycle, keep the projected annotation instead.
    Both guards preserve the original projected tree structure, ensuring every
    output sentence remains a valid projective tree with no cycles.
    """
    n_kept = n_replaced = n_total = n_guarded = n_cycle_guarded = 0
    merged = []

    for proj_s, sysA_s in zip(proj_sents, sysA_sents):
        n_tok = min(len(proj_s.tokens), len(sysA_s.tokens))
        # Live head map — updated as we commit each replacement
        heads = {t.id: t.head for t in proj_s.tokens}
        # Identify the projected root token(s)
        proj_root_ids = {pt.id for pt in proj_s.tokens if pt.head == 0}

        for i in range(n_tok):
            pt  = proj_s.tokens[i]
            at  = sysA_s.tokens[i]
            # Strip UD subtype (e.g. "nsubj:pass" → "nsubj") for matching
            base_rel = pt.deprel.lower().split(":")[0]
            n_total += 1

            if base_rel in HIGH_CONF_RELS:
                n_kept += 1          # keep projected

            else:
                # Guard 1: don't let System A introduce a second root
                if at.head == 0 and pt.id not in proj_root_ids:
                    n_guarded += 1
                    n_kept += 1

                # Guard 2: don't apply if the new head would create a cycle
                elif _would_create_cycle(pt.id, at.head, heads):
                    n_cycle_guarded += 1
                    n_kept += 1

                else:
                    pt.head   = at.head   # commit replacement
                    pt.deprel = at.deprel
                    heads[pt.id] = at.head   # update live head map
                    n_replaced += 1

        merged.append(proj_s)

    write_conllu(merged, out_path)
    pct = n_replaced / n_total * 100 if n_total else 0
    print(f"  {len(merged)} sentences | {n_kept:,} tokens kept (projection) "
          f"| {n_replaced:,} tokens replaced by System A ({pct:.1f}%)")
    print(f"  [{n_guarded} multiple-root guards, {n_cycle_guarded} cycle guards applied]")
    print(f"  Written → {out_path}")
    return merged


def main():
    ap = argparse.ArgumentParser(
        description="Build relation-selective projected Bhojpuri treebank (System D data)"
    )
    ap.add_argument("--split", choices=["train", "dev", "both"], default="both")
    args = ap.parse_args()

    syn_dir = DATA_DIR / "synthetic"

    print("\n========================================")
    print(" Innovation 3 — Relation-selective Projection")
    print("========================================")
    print(f"  HIGH_CONF (keep projected):  {sorted(HIGH_CONF_RELS)}")
    print(f"  LOW_CONF  (use System A):    {sorted(LOW_CONF_RELS)}")

    if args.split in ("train", "both"):
        print("\n[Train split]")
        proj_train = read_conllu(syn_dir / "bho_synthetic_train.conllu")
        sysA_train = run_sysA_inference(proj_train, "train")
        build_selective(proj_train, sysA_train,
                        syn_dir / "bho_selective_train.conllu")

    if args.split in ("dev", "both"):
        print("\n[Dev split]")
        proj_dev = read_conllu(syn_dir / "bho_synthetic_dev.conllu")
        sysA_dev = run_sysA_inference(proj_dev, "dev")
        build_selective(proj_dev, sysA_dev,
                        syn_dir / "bho_selective_dev.conllu")

    print("\nDone. Next: python3 train_trankit_bhojpuri_warmstart.py")


if __name__ == "__main__":
    main()

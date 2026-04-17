# patch_trankit_env.py
# Idempotent monkey-patch for Trankit 1.1.x against modern transformers/torch.
#
# Modern transformers (>=4.40) removed `AdamW`, so Trankit's `tpipeline.py`
# fails to import. Professor's Bhojpuri data also contains cycles and
# multiple roots, which Trankit's UD scorer rejects as hard errors.
#
# This helper patches both on disk, using trankit.__file__ to find the
# installed package location (works for user-installs at ~/.local/lib as well
# as conda-env site-packages — site.getsitepackages() does NOT).
#
# Safe to call multiple times: each patch is a no-op if already applied.
#
# Usage (at the very top of any trainer script):
#   from patch_trankit_env import patch_trankit_env
#   patch_trankit_env()
#   from trankit import TPipeline

from __future__ import annotations
from pathlib import Path


def _patch_file(path: Path, replacements: list[tuple[str, str]]) -> int:
    """Apply each (old, new) replacement if `old` still present. Return #applied."""
    if not path.exists():
        return 0
    txt = path.read_text()
    n_applied = 0
    for old, new in replacements:
        if old in txt:
            txt = txt.replace(old, new)
            n_applied += 1
    if n_applied:
        path.write_text(txt)
    return n_applied


def patch_trankit_env(verbose: bool = True) -> None:
    """
    Patch installed Trankit 1.1.x for compatibility with transformers>=4.40
    and for training on noisy cross-lingual projected treebanks.

    Fixes:
      1. `from transformers import AdamW` → `from torch.optim import AdamW`
         (Trankit 1.1.x tpipeline.py)
      2. `raise UDError("There is a cycle in a sentence")` → no-op pass
      3. `raise UDError("There are multiple roots in a sentence")` → no-op pass
         (Trankit's conll18_ud_eval.py — these errors fire during eval on
          noisy transferred Bhojpuri and halt training.)
    """
    try:
        import trankit
    except ImportError:
        if verbose:
            print("[patch_trankit_env] trankit not installed — skipping")
        return

    pkg_root = Path(trankit.__file__).parent

    # ── 1) AdamW import shim in tpipeline.py ──────────────────────────────────
    tpipe = pkg_root / "tpipeline.py"
    n1 = _patch_file(tpipe, [(
        "from transformers import AdamW, get_linear_schedule_with_warmup",
        "from torch.optim import AdamW\nfrom transformers import get_linear_schedule_with_warmup",
    )])
    if verbose and n1:
        print(f"[patch_trankit_env] AdamW shim applied     : {tpipe}")

    # ── 2) UD scorer: allow cycles & multiple roots ───────────────────────────
    scorer = pkg_root / "utils" / "scorers" / "conll18_ud_eval.py"
    n2 = _patch_file(scorer, [
        ('raise UDError("There is a cycle in a sentence")',
         'pass  # patched: allow cycles in transferred data'),
        ('raise UDError("There are multiple roots in a sentence")',
         'pass  # patched: allow multiple roots in transferred data'),
    ])
    if verbose and n2:
        print(f"[patch_trankit_env] UD scorer patched      : {scorer}")

    if verbose and (n1 or n2):
        print(f"[patch_trankit_env] {n1 + n2} patch(es) applied.")
    elif verbose:
        print(f"[patch_trankit_env] Trankit already patched (no-op).")


if __name__ == "__main__":
    patch_trankit_env()

#!/usr/bin/env bash
# run_full_pipeline.sh
# Runs the complete Hindi→Bhojpuri cross-lingual parsing pipeline.
# Expected total wall-clock time on Apple M-series MPS: ~3–5 hours.
#
# Usage:
#   bash run_full_pipeline.sh 2>&1 | tee pipeline.log

set -e
cd "$(dirname "$0")"

echo "=================================================="
echo " Cross-Lingual Hindi → Bhojpuri Dependency Parser"
echo " Full Pipeline Run"
echo " $(date)"
echo "=================================================="

# ── Step 7: Monolingual pre-training ─────────────────────────────────────────
echo ""
echo "[$(date +%H:%M:%S)] Step 7a — Monolingual training: HINDI"
python3 train_monolingual.py \
    --lang    hindi \
    --epochs  30    \
    --device  mps   \
    2>&1 | grep -v "NotOpenSSL\|warnings.warn\|urllib3"

echo ""
echo "[$(date +%H:%M:%S)] Step 7b — Monolingual training: BHOJPURI"
python3 train_monolingual.py \
    --lang    bhojpuri \
    --epochs  30       \
    --device  mps      \
    2>&1 | grep -v "NotOpenSSL\|warnings.warn\|urllib3"

# ── Steps 7-8: Bilingual fine-tuning ─────────────────────────────────────────
echo ""
echo "[$(date +%H:%M:%S)] Steps 7-8 — Bilingual fine-tuning"
python3 train_bilingual.py \
    --epochs       20    \
    --lambda_bho   0.5   \
    --lambda_align 0.1   \
    --device       mps   \
    2>&1 | grep -v "NotOpenSSL\|warnings.warn\|urllib3"

# ── Step 9: Final evaluation ──────────────────────────────────────────────────
echo ""
echo "[$(date +%H:%M:%S)] Step 9 — Final evaluation"
python3 evaluate.py \
    --device mps   \
    2>&1 | grep -v "NotOpenSSL\|warnings.warn\|urllib3"

echo ""
echo "=================================================="
echo " Pipeline complete: $(date)"
echo "=================================================="

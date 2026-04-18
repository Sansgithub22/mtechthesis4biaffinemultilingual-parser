#!/bin/bash
#SBATCH --job-name=thesis_sysM
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=10:00:00
#SBATCH --output=/home/ksanskruti.s.cse21.iitbhu/mtechthesis4biaffinemultilingual-parser/logs/sysM_%j.out
#SBATCH --error=/home/ksanskruti.s.cse21.iitbhu/mtechthesis4biaffinemultilingual-parser/logs/sysM_%j.err

# System M — UD-Bridge on MuRIL backbone  (Comp 2: beat K on BHTB)
#
# Three phases in one job:
#   Phase 1: train_hindi_muril.py    — Hindi parser on MuRIL (HDTB)
#   Phase 2: generate_silver_muril.py — UD-style silver Bhojpuri
#   Phase 3: train_system_m.py       — UD-Bridge on MuRIL, BHTB eval
#
# Prereq:  MuRIL must be in the HF hub cache. On login node with internet:
#   python3 -c "from transformers import AutoModel, AutoTokenizer; \
#               AutoModel.from_pretrained('google/muril-base-cased'); \
#               AutoTokenizer.from_pretrained('google/muril-base-cased')"
# If that's not feasible, the first pre-flight step below will try to fetch it.

echo "============================================"
echo "  System M — UD-Bridge on MuRIL"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  Start  : $(date)"
echo "============================================"

source /home/apps/miniconda3/etc/profile.d/conda.sh
conda activate /home/ksanskruti.s.cse21.iitbhu/thesis_env

cd ~/mtechthesis4biaffinemultilingual-parser
mkdir -p logs results data_files/synthetic cache

TS=$(date +%Y%m%d_%H%M%S)
RESULT_FILE=results/system_M_${TS}.txt

# ── Pre-flight: ensure MuRIL is cached ───────────────────────────────────────
echo ""
echo "─── Pre-flight: check MuRIL cache ────────────────────────────────────"
python3 - <<'PY'
import os
# Try offline first; fall back to online download if needed.
for k in ("TRANSFORMERS_OFFLINE", "HF_HUB_OFFLINE"):
    os.environ.pop(k, None)
try:
    from transformers import AutoModel, AutoTokenizer
    print("  Downloading MuRIL (if not cached) …")
    AutoTokenizer.from_pretrained("google/muril-base-cased")
    AutoModel.from_pretrained("google/muril-base-cased")
    print("  MuRIL ready.")
except Exception as e:
    print(f"  [ERROR] MuRIL fetch failed: {e}")
    print("  Please download manually on a machine with internet and copy")
    print("  the ~/.cache/huggingface/hub/models--google--muril-base-cased")
    print("  directory to the HPC ~/.cache/huggingface/hub/")
    raise SystemExit(1)
PY

# ── Phase 1: Hindi parser on MuRIL ───────────────────────────────────────────
echo ""
echo "─── Phase 1/3: train_hindi_muril.py ──────────────────────────────────"
python3 train_hindi_muril.py \
    --epochs 30 \
    --device cuda \
    --lr 5e-4 \
    --patience 5 2>&1 | tee -a ${RESULT_FILE}

if [ ! -f checkpoints/muril_hindi/best.pt ]; then
    echo "[ERROR] Phase 1 did not produce a checkpoint — aborting."
    exit 1
fi

# ── Phase 2: Silver UD-Bhojpuri via MuRIL-Hindi ──────────────────────────────
echo ""
echo "─── Phase 2/3: generate_silver_muril.py ──────────────────────────────"
python3 generate_silver_muril.py \
    --device cuda \
    --min_len 3 --max_len 100 \
    --require_single_root 2>&1 | tee -a ${RESULT_FILE}

if [ ! -s data_files/synthetic/bho_silver_muril_ud.conllu ]; then
    echo "[ERROR] Phase 2 did not produce silver treebank — aborting."
    exit 1
fi

# ── Phase 3: System M training + BHTB eval ───────────────────────────────────
echo ""
echo "─── Phase 3/3: train_system_m.py ─────────────────────────────────────"
python3 train_system_m.py \
    --epochs 30 \
    --device cuda \
    --lr 5e-4 \
    --patience 5 \
    --lambda_hi 0.3 2>&1 | tee -a ${RESULT_FILE}

echo ""
echo "============================================"
echo "  Finished at: $(date)"
echo "  Results   : ${RESULT_FILE}"
echo "============================================"

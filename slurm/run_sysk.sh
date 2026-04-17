#!/bin/bash
#SBATCH --job-name=thesis_sysK
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --output=/home/ksanskruti.s.cse21.iitbhu/mtechthesis4biaffinemultilingual-parser/logs/sysK_%j.out
#SBATCH --error=/home/ksanskruti.s.cse21.iitbhu/mtechthesis4biaffinemultilingual-parser/logs/sysK_%j.err

# System K — UD-Bridge via Silver Self-Training (Competition 2: beat A on BHTB)
#
# Pipeline (single SLURM job):
#   1. generate_silver_ud_labels.py — System A labels prof's Bhojpuri in UD style
#   2. train_system_k.py           — Trankit fine-tune on HDTB + silver UD-Bho
#   3. evaluate_trankit.py --include_k — BHTB test score vs System A

echo "============================================"
echo "  System K — UD-Bridge via Silver Self-Training"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Node    : $SLURM_NODELIST"
echo "  Start   : $(date)"
echo "============================================"

source /home/apps/miniconda3/etc/profile.d/conda.sh
conda activate /home/ksanskruti.s.cse21.iitbhu/thesis_env

cd ~/mtechthesis4biaffinemultilingual-parser
mkdir -p logs results data_files/sysk data_files/synthetic

# ── Offline HuggingFace (required on IIT BHU Param Shivay) ──────────────────
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

TS=$(date +%Y%m%d_%H%M%S)
RESULT_FILE=results/system_K_${TS}.txt

echo ""
echo "─── Step 1/3: Generate UD-style silver labels ────────────────────────"
python3 generate_silver_ud_labels.py \
    --input bhojpuri_matched_transferred.conllu \
    --output data_files/synthetic/bho_silver_ud.conllu \
    --min_len 3 --max_len 100 \
    --require_single_root \
    --batch_size 32 \
    --gpu 2>&1 | tee -a ${RESULT_FILE}

if [ ! -s data_files/synthetic/bho_silver_ud.conllu ]; then
    echo "[ERROR] Silver treebank generation failed — aborting."
    exit 1
fi

echo ""
echo "─── Step 2/3: Train System K ─────────────────────────────────────────"
python3 train_system_k.py \
    --epochs 60 \
    --batch_size 16 \
    --warm_start_sysa \
    --gpu 2>&1 | tee -a ${RESULT_FILE}

echo ""
echo "─── Step 3/3: Evaluate on BHTB ───────────────────────────────────────"
python3 evaluate_trankit.py \
    --gpu \
    --skip_d --skip_f \
    --include_k 2>&1 | tee -a ${RESULT_FILE}

echo ""
echo "============================================"
echo "  Finished at: $(date)"
echo "  Results    : ${RESULT_FILE}"
echo "============================================"

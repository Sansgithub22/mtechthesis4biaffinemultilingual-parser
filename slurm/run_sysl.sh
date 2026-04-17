#!/bin/bash
#SBATCH --job-name=thesis_sysL
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=/home/ksanskruti.s.cse21.iitbhu/mtechthesis4biaffinemultilingual-parser/logs/sysL_%j.out
#SBATCH --error=/home/ksanskruti.s.cse21.iitbhu/mtechthesis4biaffinemultilingual-parser/logs/sysL_%j.err

# System L — Iterative Self-Training + Agreement Filter (Comp 2 fallback)
#
# Prerequisites:
#   - System A trained: checkpoints/trankit_hindi/xlm-roberta-base/hindi/hindi.tagger.mdl
#   - System K trained: checkpoints/trankit_bho_sysk/xlm-roberta-base/bhojpuri_sysk/bhojpuri_sysk.tagger.mdl
#   - silver v1 exists : data_files/synthetic/bho_silver_ud.conllu (from run_sysk.sh step 1)
#
# Pipeline (single SLURM job):
#   1. generate_silver_ud_labels.py using System K as teacher  → silver_ud_v2.conllu
#   2. compare_silver_labels.py (agreement between A and K)    → silver_ud_filtered.conllu
#   3. train_system_l.py (HDTB + filtered silver, warm-start K)
#   4. evaluate_trankit.py --include_k --include_l             → BHTB scores

echo "============================================"
echo "  System L — Iter Self-Training + Agreement Filter"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Node    : $SLURM_NODELIST"
echo "  Start   : $(date)"
echo "============================================"

source /home/apps/miniconda3/etc/profile.d/conda.sh
conda activate /home/ksanskruti.s.cse21.iitbhu/thesis_env

cd ~/mtechthesis4biaffinemultilingual-parser
mkdir -p logs results data_files/sysl data_files/synthetic

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

TS=$(date +%Y%m%d_%H%M%S)
RESULT_FILE=results/system_L_${TS}.txt

SYSK_CKPT=checkpoints/trankit_bho_sysk/xlm-roberta-base/bhojpuri_sysk/bhojpuri_sysk.tagger.mdl
SILVER_V1=data_files/synthetic/bho_silver_ud.conllu
SILVER_V2=data_files/synthetic/bho_silver_ud_v2.conllu
SILVER_FIL=data_files/synthetic/bho_silver_ud_filtered.conllu

if [ ! -f "${SYSK_CKPT}" ]; then
    echo "[ERROR] System K checkpoint not found: ${SYSK_CKPT}"
    echo "        Run slurm/run_sysk.sh first."
    exit 1
fi
if [ ! -f "${SILVER_V1}" ]; then
    echo "[ERROR] Silver v1 not found: ${SILVER_V1}"
    echo "        It should have been produced by run_sysk.sh step 1."
    exit 1
fi

echo ""
echo "─── Step 1/4: Re-parse Bhojpuri with System K → silver v2 ───────────"
python3 generate_silver_ud_labels.py \
    --input bhojpuri_matched_transferred.conllu \
    --teacher_dir checkpoints/trankit_bho_sysk \
    --teacher_category bhojpuri_sysk \
    --teacher_train_conllu data_files/sysk/sysk_train.conllu \
    --output ${SILVER_V2} \
    --min_len 3 --max_len 100 \
    --require_single_root \
    --batch_size 32 \
    --gpu 2>&1 | tee -a ${RESULT_FILE}

if [ ! -s ${SILVER_V2} ]; then
    echo "[ERROR] Silver v2 generation failed — aborting."
    exit 1
fi

echo ""
echo "─── Step 2/4: Agreement filter (System A ∩ System K) ────────────────"
python3 compare_silver_labels.py \
    --silver_v1 ${SILVER_V1} \
    --silver_v2 ${SILVER_V2} \
    --output ${SILVER_FIL} \
    --min_agreement 0.80 \
    --use_labels_from v2 2>&1 | tee -a ${RESULT_FILE}

if [ ! -s ${SILVER_FIL} ]; then
    echo "[ERROR] Agreement filtering produced empty corpus — try lowering --min_agreement."
    exit 1
fi

echo ""
echo "─── Step 3/4: Train System L ────────────────────────────────────────"
python3 train_system_l.py \
    --epochs 60 \
    --batch_size 16 \
    --silver_filtered ${SILVER_FIL} \
    --gpu 2>&1 | tee -a ${RESULT_FILE}

echo ""
echo "─── Step 4/4: Evaluate on BHTB (A vs K vs L) ────────────────────────"
python3 evaluate_trankit.py \
    --gpu \
    --skip_d --skip_f \
    --include_k --include_l 2>&1 | tee -a ${RESULT_FILE}

echo ""
echo "============================================"
echo "  Finished at: $(date)"
echo "  Results    : ${RESULT_FILE}"
echo "============================================"

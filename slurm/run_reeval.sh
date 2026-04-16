#!/bin/bash
#SBATCH -J thesis_reeval
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH -t 01:00:00
#SBATCH -o logs/reeval_%j.out
#SBATCH -e logs/reeval_%j.err

echo "===== Re-evaluation with fixed encoder ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node  : $SLURMD_NODENAME"
echo "Start : $(date)"

source /home/apps/miniconda3/etc/profile.d/conda.sh
conda activate /home/ksanskruti.s.cse21.iitbhu/thesis_env

cd ~/mtechthesis4biaffinemultilingual-parser
mkdir -p logs

echo ""
echo "--- System G ---"
python3 re_eval.py --system g --device cpu

echo ""
echo "--- System H ---"
python3 re_eval.py --system h --device cpu

echo ""
echo "--- System I ---"
python3 re_eval.py --system i --device cpu

echo ""
echo "--- System J ---"
python3 re_eval.py --system j --device cpu

echo ""
echo "===== Re-evaluation done: $(date) ====="

#!/bin/bash
#SBATCH -J thesis_sysh
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -o logs/sysh_%j.out
#SBATCH -e logs/sysh_%j.err

echo "===== System H: Syntax-Aware Cross-lingual Transfer (SACT) ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node  : $SLURMD_NODENAME"
echo "Start : $(date)"

module load miniconda_23.5.2_python_3.11.4
source /home/apps/miniconda3/etc/profile.d/conda.sh
conda activate /home/ksanskruti.s.cse21.iitbhu/thesis_env

cd ~/mtechthesis4biaffinemultilingual-parser
mkdir -p logs

python3 train_system_h.py \
    --epochs 40 \
    --device cuda \
    --lambda_hi 0.3 \
    --lambda_cosine 0.4 \
    --lambda_arc 0.1 \
    --lambda_cts 0.6 \
    --warmup_epochs 3

echo "===== System H done: $(date) ====="

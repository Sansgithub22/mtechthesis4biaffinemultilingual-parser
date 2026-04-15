#!/bin/bash
#SBATCH -J thesis_sysg
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -o logs/sysg_%j.out
#SBATCH -e logs/sysg_%j.err

echo "===== System G: Exact Alignment Joint Training ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node  : $SLURMD_NODENAME"
echo "Start : $(date)"

source /home/apps/miniconda3/etc/profile.d/conda.sh
conda activate /home/ksanskruti.s.cse21.iitbhu/thesis_env

cd ~/mtechthesis4biaffinemultilingual-parser
mkdir -p logs

python3 train_system_g.py --epochs 40 --device cuda --lambda_hi 0.3 --lambda_align 0.5

echo "===== System G done: $(date) ====="

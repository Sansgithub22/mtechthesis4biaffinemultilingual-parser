#!/bin/bash
#SBATCH -J thesis_sysf
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=16G
#SBATCH -t 24:00:00
#SBATCH -o logs/sysf_%j.out
#SBATCH -e logs/sysf_%j.err

echo "===== System F: High-Quality Fine-tuning ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node  : $SLURMD_NODENAME"
echo "Start : $(date)"

module load miniconda_23.5.2_python_3.11.4
source activate ~/thesis_env

cd ~/mtechthesis4biaffinemultilingual-parser
mkdir -p logs

python3 train_system_f.py --epochs 60 --batch_size 32 --gpu

echo "===== System F done: $(date) ====="

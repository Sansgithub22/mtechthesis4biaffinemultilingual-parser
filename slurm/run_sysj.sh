#!/bin/bash
#SBATCH --job-name=thesis_sysJ
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --output=/home/ksanskruti.s.cse21.iitbhu/mtechthesis4biaffinemultilingual-parser/logs/sysJ_%j.out
#SBATCH --error=/home/ksanskruti.s.cse21.iitbhu/mtechthesis4biaffinemultilingual-parser/logs/sysJ_%j.err

echo "============================================"
echo "  System J — RS-SACT"
echo "  Job ID  : $SLURM_JOB_ID"
echo "  Node    : $SLURM_NODELIST"
echo "  Start   : $(date)"
echo "============================================"

source /home/apps/miniconda3/etc/profile.d/conda.sh
conda activate /home/ksanskruti.s.cse21.iitbhu/thesis_env

cd ~/mtechthesis4biaffinemultilingual-parser
mkdir -p logs

python3 train_system_j.py \
    --epochs 40 \
    --device cuda \
    --lambda_hi 0.5 \
    --lambda_cosine 0.4 \
    --lambda_arc 2.0 \
    --lambda_cts 0.2 \
    --lambda_contrast 0.5 \
    --tau 0.07 \
    --warmup_epochs 2 \
    --patience 10 \
    --lr 5e-5

echo "Finished at: $(date)"

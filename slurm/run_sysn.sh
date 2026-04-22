#!/bin/bash
#SBATCH -J thesis_sysn
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -o logs/sysn_%j.out
#SBATCH -e logs/sysn_%j.err

echo "===== System N: MuRIL-SACT + MFEF + Curriculum + SWA ====="
echo "Job ID: $SLURM_JOB_ID"
echo "Node  : $SLURMD_NODENAME"
echo "Start : $(date)"

module load miniconda_23.5.2_python_3.11.4
source /home/apps/miniconda3/etc/profile.d/conda.sh
conda activate /home/ksanskruti.s.cse21.iitbhu/thesis_env

cd ~/mtechthesis4biaffinemultilingual-parser
mkdir -p logs

python3 train_system_n.py \
    --backbone muril \
    --epochs 40 \
    --device cuda \
    --lr 5e-5 \
    --patience 8 \
    --curriculum_epochs 5 \
    --swa_start 15 \
    --morph_dim 128 \
    --max_feats 8 \
    --lambda_hi 0.5 \
    --lambda_cosine 0.4 \
    --lambda_arc 2.0 \
    --lambda_cts 0.2 \
    --warmup_epochs 0

echo "===== System N done: $(date) ====="

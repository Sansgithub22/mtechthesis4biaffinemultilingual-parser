#!/bin/bash
#SBATCH --job-name=xlmr_cache
#SBATCH --partition=standard
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=/home/ksanskruti.s.cse21.iitbhu/mtechthesis4biaffinemultilingual-parser/logs/cache_%j.out
#SBATCH --error=/home/ksanskruti.s.cse21.iitbhu/mtechthesis4biaffinemultilingual-parser/logs/cache_%j.err

echo "============================================"
echo "  XLM-R Cache Pre-computation"
echo "  Job ID : $SLURM_JOB_ID"
echo "  Node   : $SLURM_NODELIST"
echo "  CPUs   : $SLURM_CPUS_PER_TASK"
echo "  Start  : $(date)"
echo "============================================"

source /home/apps/miniconda3/etc/profile.d/conda.sh
conda activate thesis

cd /home/ksanskruti.s.cse21.iitbhu/mtechthesis4biaffinemultilingual-parser

mkdir -p logs cache

python3 precompute_cache.py

echo ""
echo "Finished at: $(date)"
echo "Cache file:"
ls -lh cache/xlmr_cache.pt 2>/dev/null || echo "ERROR: cache file not found"

#!/bin/bash
# submit_all.sh
# Submits all four jobs with proper dependencies:
#
#   Job 1 (Hindi)  — runs first, no dependency
#   Job 2 (Sys F)  — starts after Hindi finishes successfully
#   Job 3 (Sys G)  — starts after Hindi finishes successfully
#   Job 4 (Sys H)  — starts after Hindi finishes successfully
#
# Jobs 2, 3, 4 run SIMULTANEOUSLY on separate GPU nodes.
#
# Usage:
#   cd ~/mtechthesis4biaffinemultilingual-parser
#   bash slurm/submit_all.sh

set -e

cd ~/mtechthesis4biaffinemultilingual-parser
mkdir -p logs

echo "============================================"
echo " Submitting thesis training jobs to SLURM"
echo "============================================"

# Job 1: Train Hindi model (System A) — F, G, H all depend on this
HINDI_JOB=$(sbatch --parsable slurm/run_hindi.sh)
echo "  Submitted Hindi training   → Job ID: $HINDI_JOB"

# Job 2: Train System F — starts after Hindi is done
SYSF_JOB=$(sbatch --parsable --dependency=afterok:$HINDI_JOB slurm/run_sysf.sh)
echo "  Submitted System F         → Job ID: $SYSF_JOB  (starts after Job $HINDI_JOB)"

# Job 3: Train System G — starts after Hindi is done (same time as F)
SYSG_JOB=$(sbatch --parsable --dependency=afterok:$HINDI_JOB slurm/run_sysg.sh)
echo "  Submitted System G         → Job ID: $SYSG_JOB  (starts after Job $HINDI_JOB)"

# Job 4: Train System H — starts after Hindi is done (same time as F and G)
SYSH_JOB=$(sbatch --parsable --dependency=afterok:$HINDI_JOB slurm/run_sysh.sh)
echo "  Submitted System H         → Job ID: $SYSH_JOB  (starts after Job $HINDI_JOB)"

echo ""
echo "============================================"
echo " All jobs submitted!"
echo "============================================"
echo ""
echo " Timeline:"
echo "   Hindi training  (Job $HINDI_JOB)  → runs first"
echo "   System F        (Job $SYSF_JOB)   → runs after Hindi, simultaneously with G and H"
echo "   System G        (Job $SYSG_JOB)   → runs after Hindi, simultaneously with F and H"
echo "   System H        (Job $SYSH_JOB)   → runs after Hindi, simultaneously with F and G"
echo ""
echo " Monitor:"
echo "   squeue -u \$USER"
echo ""
echo " Watch logs:"
echo "   tail -f logs/hindi_${HINDI_JOB}.out"
echo "   tail -f logs/sysf_${SYSF_JOB}.out"
echo "   tail -f logs/sysg_${SYSG_JOB}.out"
echo "   tail -f logs/sysh_${SYSH_JOB}.out"
echo ""
echo " Cancel all:"
echo "   scancel $HINDI_JOB $SYSF_JOB $SYSG_JOB $SYSH_JOB"

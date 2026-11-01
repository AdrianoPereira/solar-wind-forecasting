#!/bin/bash
# Hybrid-model variants with one boosting library (same splits and features as docs/hybrid.md).
# Login nodes kill any process after 30 min of CPU time (ulimit -t 1800), so run it here.
#
# Submit from the project root; MODEL is hgb, lightgbm, xgboost, catboost or ngboost; MAX_ITER sets the most trees:
#   mkdir -p data/slurm_logs && MODEL=xgboost sbatch scripts/slurm/hybrid_experiment.sbatch
# The output directory must be new or empty.
#SBATCH --job-name=benchlab-hybrid
#SBATCH --account=cptec
#SBATCH --partition=lncc-cpu_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=data/slurm_logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
MODEL="${MODEL:-xgboost}"

echo "job ${SLURM_JOB_ID} on $(hostname), ${SLURM_CPUS_PER_TASK} CPUs, model ${MODEL}, start $(date -u +%FT%TZ)"
.venv/bin/python -m benchlab_forecasting.hybrid experiment --model "${MODEL}" \
    --variants ${VARIANTS:-mse mse_no_ch timse} ${MAX_ITER:+--max-iter "${MAX_ITER}"} \
    --output-dir "${OUTPUT_DIR:-data/experiments/hybrid_${MODEL}_2011_2026}"
echo "end $(date -u +%FT%TZ)"

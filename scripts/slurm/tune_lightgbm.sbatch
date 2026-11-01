#!/bin/bash
# Random search of LightGBM hyperparameters for the hybrid model (chosen by validation RMSE).
# Login nodes kill any process after 30 min of CPU time (ulimit -t 1800), so run it here.
#
# Submit from the project root:
#   mkdir -p data/slurm_logs && sbatch scripts/slurm/tune_lightgbm.sbatch
# Resubmitting with the same output directory resumes: finished trials are skipped.
#SBATCH --job-name=benchlab-tune
#SBATCH --account=cptec
#SBATCH --partition=lncc-cpu_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=data/slurm_logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

echo "job ${SLURM_JOB_ID} on $(hostname), ${SLURM_CPUS_PER_TASK} CPUs, start $(date -u +%FT%TZ)"
.venv/bin/python -m benchlab_forecasting.hybrid tune \
    --trials "${TRIALS:-40}" --threads "${SLURM_CPUS_PER_TASK}" \
    --output-dir "${OUTPUT_DIR:-data/experiments/hybrid_lightgbm_tuning}"
echo "end $(date -u +%FT%TZ)"

#!/bin/bash
# TIMSE-oriented forecast robust to the TIMSE aggregation (benchlab_forecasting.timse_blend).
# Submit from the project root:
#   mkdir -p data/slurm_logs && sbatch scripts/slurm/timse_blend.sbatch
#SBATCH --job-name=benchlab-timse
#SBATCH --account=cptec
#SBATCH --partition=lncc-cpu_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --output=data/slurm_logs/%x-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1
echo "job ${SLURM_JOB_ID} on $(hostname), start $(date -u +%FT%TZ)"
PYTHONPATH=src .venv/bin/python -m benchlab_forecasting.timse_blend --threads "${SLURM_CPUS_PER_TASK}" \
    --output-dir "${OUTPUT_DIR:-data/experiments/timse_blend_2011_2026}"
echo "end $(date -u +%FT%TZ)"

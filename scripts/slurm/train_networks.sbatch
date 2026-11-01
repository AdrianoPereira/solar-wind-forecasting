#!/bin/bash
# Retrain the neural networks on the hybrid's splits (train 2011-02 – 2021-12, validation 2022-01 – 2023-11),
# one architecture per array task, with the hyperparameters of their earlier experiments.
# The 2024–2026 test is scored afterwards by scripts/benchmark_all_models.py.
#
# Submit from the project root:
#   mkdir -p data/slurm_logs && sbatch scripts/slurm/train_networks.sbatch
#SBATCH --job-name=benchlab-nets
#SBATCH --account=cptec
#SBATCH --partition=lncc-cpu_amd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --array=0-5
#SBATCH --output=data/slurm_logs/%x-%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export PYTHONUNBUFFERED=1

NAMES=(lstm gru attention_gru attention_lstm transformer patch_transformer)
NAME="${NAMES[${SLURM_ARRAY_TASK_ID}]}"
OUTPUT="data/experiments/${NAME}_2011_2021"
echo "job ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID} ${NAME} on $(hostname), start $(date -u +%FT%TZ)"
# Validation runs from 2022-01-01 to 2023-11-09, before the ACE gap of 10 Nov 2023 that leaves an
# issuance without recent data (the networks refuse to skip validation issuances).
.venv/bin/python -m "benchlab_forecasting.${NAME}_cli" experiment \
    --data data/ace_hourly_2011_2026/ace_swepam_hourly.parquet \
    --mag-data data/ace_mag_hourly_2011_2026/ace_mag_hourly.parquet \
    --train-start 2011-02-05 --validation-start 2022-01-01 --validation-hours 16260 \
    --test-start 2024-01-01 --test-hours 168 --threads "${SLURM_CPUS_PER_TASK}" \
    --output-dir "${OUTPUT}"
echo "end $(date -u +%FT%TZ)"

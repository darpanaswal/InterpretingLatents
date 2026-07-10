#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################
EXPERIMENT="variance_decomposition"
MODEL_FAMILY="llama"
N_GPUS=1
WALLTIME="02:00:00"
########################################

LOG_DIR="runs/variance_decomposition"
LOG_FILE="${LOG_DIR}/${MODEL_FAMILY}.txt"

SCRIPT_PATH="$(readlink -f "$0")"

# If not inside a SLURM job -> submit self
if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "${LOG_DIR}"
    sbatch \
        --job-name="${EXPERIMENT}" \
        --output="${LOG_FILE}" \
        --error="${LOG_FILE}" \
        --partition=gpu_p13 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=$((N_GPUS * 4)) \
        --gres=gpu:${N_GPUS} \
        --time="${WALLTIME}" \
        "${SCRIPT_PATH}"
    exit 0
fi


# Inside SLURM job -> run experiment
module purge
module load anaconda-py3/2024.06
source $WORK/env_cache_guard.sh
conda activate lrm

> "${LOG_FILE}"
echo "Running variance decomposition for all"
echo "Log file        : ${LOG_FILE}"

python -u -m experiments.geometry.variance_decomposition \
    --all \
    --model_family "${MODEL_FAMILY}" \
    >> "${LOG_FILE}" 2>&1

# TO RUN, COPY: bash args/variance_decomposition.sh

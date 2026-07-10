#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################
EXPERIMENT="markovianity_test"
TASK="gsm"
MODEL_FAMILY="llama"
MODEL="codi"
PROJECT_TO_SUBSPACE="both"
N_GPUS=1
WALLTIME="10:00:00"
########################################

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}/${TASK}_${MODEL_FAMILY}_${MODEL}.txt"

SCRIPT_PATH="$(readlink -f "$0")"

# If not inside a SLURM job -> submit self
if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "${LOG_DIR}"
    sbatch \
        --job-name="${EXPERIMENT}_${TASK}_${MODEL_FAMILY}_${MODEL}" \
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
echo "Task            : ${TASK}"
echo "Model Family    : ${MODEL_FAMILY}"
echo "Model           : ${MODEL}"
echo "GPUs            : ${N_GPUS}"
echo "Project Subspace: ${PROJECT_TO_SUBSPACE}"
echo "Log file        : ${LOG_FILE}"

python -u -m experiments.geometry.markovianity_test \
    --task "${TASK}" \
    --model "${MODEL}" \
    --model_family "${MODEL_FAMILY}" \
    --num_gpus "${N_GPUS}" \
    --project_to_subspace "${PROJECT_TO_SUBSPACE}" \
    >> "${LOG_FILE}" 2>&1

# TO RUN, COPY: bash args/markovianity_test.sh

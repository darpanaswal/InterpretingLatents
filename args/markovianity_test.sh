#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################
EXPERIMENT="markovianity_test"
TASK="gsm"
MODEL_FAMILY="llama"
MODEL="codi"
PROJECT_TO_SUBSPACE="all"
N_GPUS=1
CPUS_PER_TASK=4
WALLTIME="10:00:00"
########################################

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}/${TASK}_${MODEL_FAMILY}_${MODEL}.txt"

# If not inside a SLURM job -> submit self
if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "${LOG_DIR}"
    SNAPSHOT="$(mktemp "${LOG_DIR}/.snapshot.XXXXXX.sh")"
    cp "$(readlink -f "$0")" "${SNAPSHOT}"
    chmod +x "${SNAPSHOT}"
    sbatch \
        --job-name="${EXPERIMENT}_${TASK}_${MODEL_FAMILY}_${MODEL}" \
        --output="${LOG_FILE}" \
        --error="${LOG_FILE}" \
        --partition=gpu_p13 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=${CPUS_PER_TASK} \
        --gres=gpu:${N_GPUS} \
        --time="${WALLTIME}" \
        "${SNAPSHOT}"
    exit 0
fi


# Inside SLURM job -> run experiment
rm -f "$0"
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
    --project_to_subspace "${PROJECT_TO_SUBSPACE}" \
    >> "${LOG_FILE}" 2>&1

# TO RUN, COPY: bash args/markovianity_test.sh

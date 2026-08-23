#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################
EXPERIMENT="random_corruption"
TASK="gsm"
MODEL_FAMILY="llama"
MODEL="codi"
CODI_BATCH_SIZE=8
NUM_SEEDS=3
N_GPUS=4
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
        --export=ALL,SNAPSHOT_FILE="${SNAPSHOT}" \
        --job-name="${EXPERIMENT}_${TASK}_${MODEL_FAMILY}_${MODEL}" \
        --output="${LOG_FILE}" \
        --error="${LOG_FILE}" \
        --partition=gpu_p2,gpu_p2s,gpu_p2l \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=$((N_GPUS * 4)) \
        --gres=gpu:${N_GPUS} \
        --time="${WALLTIME}" \
        "${SNAPSHOT}"
    exit 0
fi


# Inside SLURM job -> run experiment
if [ -n "${SNAPSHOT_FILE:-}" ]; then
    rm -f "${SNAPSHOT_FILE}"
fi
module purge
module load anaconda-py3/2024.06
source $WORK/env_cache_guard.sh
conda activate lrm

> "${LOG_FILE}"
echo "Task            : ${TASK}"
echo "Model           : ${MODEL}"
echo "Model Family    : ${MODEL_FAMILY}"
echo "GPUs            : ${N_GPUS}"
echo "CODI Batch Size : ${CODI_BATCH_SIZE}"
echo "Num Seeds       : ${NUM_SEEDS}"
echo "Log file        : ${LOG_FILE}"

python -u -m experiments.random_corruption \
    --task "${TASK}" \
    --model_family "${MODEL_FAMILY}" \
    --model "${MODEL}" \
    --n_gpus "${N_GPUS}" \
    --codi_batch_size "${CODI_BATCH_SIZE}" \
    --n_seeds "${NUM_SEEDS}" \
    >> "${LOG_FILE}" 2>&1

# TO RUN, COPY: bash slurm/random_corruption.sh

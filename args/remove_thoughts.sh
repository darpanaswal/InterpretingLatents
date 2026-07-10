#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################
EXPERIMENT="remove_thoughts"
TASK="prosqa"
MODEL_FAMILY="gpt2"
MODEL="pause"
N_GPUS=2
WALLTIME="02:00:00"
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
echo "Model Family    : ${MODEL_FAMILY}"
echo "Task            : ${TASK}"
echo "Models          : ${MODEL}"
echo "GPUs            : ${N_GPUS}"
echo "Log file        : ${LOG_FILE}"

torchrun --nproc_per_node="${N_GPUS}" -m experiments.ablation.remove_thoughts \
    --model_family "${MODEL_FAMILY}" \
    --task "${TASK}" \
    --models "${MODEL}" \
    >> "${LOG_FILE}" 2>&1

# TO RUN, COPY: bash args/remove_thoughts.sh

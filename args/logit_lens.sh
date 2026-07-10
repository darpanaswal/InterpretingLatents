#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################
EXPERIMENT="logit_lens"
TASK="gsm"
MODEL_FAMILY="llama"
MODEL="coconut_u"
K=6
N_GPUS=1
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
echo "Model           : ${MODEL}"
echo "K               : ${K}"
echo "Log file        : ${LOG_FILE}"

python -u -m experiments.dead_salmon.logit_lens \
    --model_family "${MODEL_FAMILY}" \
    --task "${TASK}" \
    --model "${MODEL}" \
    --k "${K}" \
    >> "${LOG_FILE}" 2>&1

# TO RUN, COPY: bash args/logit_lens.sh

#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################
EXPERIMENT="gradient_subspace_predtoken"
TASK="gsm"
MODEL_FAMILY="gpt2"
MODEL="codi"
MAX_NEW=12
N_GPUS=1
WALLTIME="12:00:00"
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
        --cpus-per-task=$((N_GPUS * 4)) \
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
echo "Max new tokens  : ${MAX_NEW}"
echo "GPUs            : ${N_GPUS}"
echo "Log file        : ${LOG_FILE}"

# Full test set (no --max_instances).
python -u -m experiments.ablation.gradient_subspace_predtoken \
    --task "${TASK}" \
    --model_family "${MODEL_FAMILY}" \
    --model "${MODEL}" \
    --max_new "${MAX_NEW}" \
    >> "${LOG_FILE}" 2>&1

# TO RUN, COPY: bash args/gradient_subspace_predtoken.sh
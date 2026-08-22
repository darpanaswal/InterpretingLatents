#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE (env-overridable, e.g.
#   TASK=gsm MODEL_FAMILY=llama MODEL=codi bash args/variance_decomposition.sh
# to run one (task, model) config; omit TASK/MODEL to keep the original
# --all behavior (every task/model for the family in one job).)
########################################
EXPERIMENT="variance_decomposition"
MODEL_FAMILY="${MODEL_FAMILY:-llama}"
TASK="${TASK:-}"
MODEL="${MODEL:-}"
N_GPUS="${N_GPUS:-1}"
WALLTIME="${WALLTIME:-02:00:00}"
########################################

LOG_DIR="runs/${EXPERIMENT}"
if [ -n "${TASK}" ] && [ -n "${MODEL}" ]; then
    LOG_FILE="${LOG_DIR}/${TASK}_${MODEL_FAMILY}_${MODEL}.txt"
    JOB_TAG="${MODEL_FAMILY}_${TASK}_${MODEL}"
else
    LOG_FILE="${LOG_DIR}/${MODEL_FAMILY}.txt"
    JOB_TAG="${MODEL_FAMILY}_all"
fi

# If not inside a SLURM job -> submit self
if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "${LOG_DIR}"
    SNAPSHOT="$(mktemp "${LOG_DIR}/.snapshot.XXXXXX.sh")"
    cp "$(readlink -f "$0")" "${SNAPSHOT}"
    chmod +x "${SNAPSHOT}"
    JOBID=$(sbatch --parsable \
        --export=ALL,SNAPSHOT_FILE="${SNAPSHOT}" \
        --job-name="${EXPERIMENT}_${JOB_TAG}" \
        --output="${LOG_FILE}" \
        --error="${LOG_FILE}" \
        --partition=gpu_p2,gpu_p2s,gpu_p2l \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=$((N_GPUS * 4)) \
        --gres=gpu:${N_GPUS} \
        --time="${WALLTIME}" \
        ${SBATCH_EXTRA_ARGS:-} \
        "${SNAPSHOT}"
    )
    echo "${JOBID}"
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
echo "Model Family    : ${MODEL_FAMILY}"
echo "Log file        : ${LOG_FILE}"

if [ -n "${TASK}" ] && [ -n "${MODEL}" ]; then
    echo "Task            : ${TASK}"
    echo "Model           : ${MODEL}"
    python -u -m experiments.geometry.variance_decomposition \
        --task "${TASK}" \
        --model "${MODEL}" \
        --model_family "${MODEL_FAMILY}" \
        >> "${LOG_FILE}" 2>&1
else
    echo "Running variance decomposition for all (task, model) pairs"
    python -u -m experiments.geometry.variance_decomposition \
        --all \
        --model_family "${MODEL_FAMILY}" \
        >> "${LOG_FILE}" 2>&1
fi

# TO RUN, COPY: bash args/variance_decomposition.sh
# TO OVERRIDE, COPY: TASK=gsm MODEL_FAMILY=llama MODEL=codi bash args/variance_decomposition.sh

#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE (env-overridable, e.g.
#   TASK=gsm MODEL_FAMILY=llama MODEL=codi bash args/mean_ablation.sh)
########################################
EXPERIMENT="mean_ablation"
TASK="${TASK:-gsm}"
MODEL_FAMILY="${MODEL_FAMILY:-llama}"
MODEL="${MODEL:-codi}"
N_THOUGHTS="${N_THOUGHTS:-6}"
INTERVENTION="${INTERVENTION:-all}"
N_GPUS="${N_GPUS:-4}"
WALLTIME="${WALLTIME:-20:00:00}"
########################################

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}/${TASK}_${MODEL_FAMILY}_${MODEL}.txt"

# If not inside a SLURM job -> submit self
if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "${LOG_DIR}"
    SNAPSHOT="$(mktemp "${LOG_DIR}/.snapshot.XXXXXX.sh")"
    cp "$(readlink -f "$0")" "${SNAPSHOT}"
    chmod +x "${SNAPSHOT}"
    JOBID=$(sbatch --parsable \
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
echo "Task            : ${TASK}"
echo "Model Family    : ${MODEL_FAMILY}"
echo "Model           : ${MODEL}"
echo "GPUs            : ${N_GPUS}"
echo "Thoughts        : ${N_THOUGHTS}"
echo "Intervention    : ${INTERVENTION}"
echo "Log file        : ${LOG_FILE}"

PYTHONUNBUFFERED=1 python -u -m experiments.geometry.mean_ablation \
    --n_gpus "${N_GPUS}" \
    --n_thoughts "${N_THOUGHTS}" \
    --intervention "${INTERVENTION}" \
    --task "${TASK}" \
    --model "${MODEL}" \
    --model_family "${MODEL_FAMILY}" \
    >> "${LOG_FILE}" 2>&1

# TO RUN, COPY: bash args/mean_ablation.sh
# TO OVERRIDE, COPY: TASK=gsm MODEL_FAMILY=llama MODEL=codi bash args/mean_ablation.sh

#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE (env-overridable, e.g.
#   TASK=gsm MODEL_FAMILY=llama MODEL=codi bash slurm/logit_lens.sh)
########################################
EXPERIMENT="logit_lens"
TASK="${TASK:-gsm}"
MODEL_FAMILY="${MODEL_FAMILY:-llama}"
MODEL="${MODEL:-coconut_u}"
K="${K:-6}"
N_GPUS="${N_GPUS:-1}"
WALLTIME="${WALLTIME:-02:00:00}"
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

# TO RUN, COPY: bash slurm/logit_lens.sh
# TO OVERRIDE, COPY: TASK=gsm MODEL_FAMILY=llama MODEL=codi bash slurm/logit_lens.sh

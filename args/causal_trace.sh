#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################
EXPERIMENT="causal_trace"
TASK="gsm"
MODEL_FAMILY="llama"
MODEL="codi"
GRANULARITY="single"
BATCH_SIZE=320
MAX_INSTANCES=2000
DEBUG=false
VERIFY_BATCHED=false
N_GPUS=1
WALLTIME="20:00:00"
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
echo "Task              : ${TASK}"
echo "Model Family      : ${MODEL_FAMILY}"
echo "Model             : ${MODEL}"
echo "GPUs              : ${N_GPUS}"
echo "Granularity       : ${GRANULARITY}"
echo "Batch Size        : ${BATCH_SIZE}"
echo "Max Instances     : ${MAX_INSTANCES}"
echo "Debug             : ${DEBUG}"
echo "Verify Batched    : ${VERIFY_BATCHED}"
echo "Log file          : ${LOG_FILE}"

CMD=(
    python -u -m experiments.causal_tracing.causal_trace
    --task "${TASK}"
    --model_family "${MODEL_FAMILY}"
    --model "${MODEL}"
    --n_gpus "${N_GPUS}"
    --granularity "${GRANULARITY}"
    --batch_size "${BATCH_SIZE}"
    --max_instances "${MAX_INSTANCES}"
)

if [ "${DEBUG}" = "true" ]; then
    CMD+=(--debug)
fi

if [ "${VERIFY_BATCHED}" = "true" ]; then
    CMD+=(--verify_batched)
fi

"${CMD[@]}" > "${LOG_FILE}" 2>&1

# TO RUN, COPY: bash args/causal_trace.sh

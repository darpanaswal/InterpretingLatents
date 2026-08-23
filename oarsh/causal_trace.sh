#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################
EXPERIMENT="causal_trace"
TASK="gsm"
MODEL_FAMILY="llama"
MODEL="coconut_u"
GRANULARITY="single"
BATCH_SIZE=160
MAX_INSTANCES=2000
DEBUG=false
VERIFY_BATCHED=false
N_GPUS=2
WALLTIME="20:00:00"
########################################

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}/${TASK}_${MODEL_FAMILY}_${MODEL}.txt"

        # -p "network_address='lig-gpu1.imag.fr' OR network_address='lig-gpu2.imag.fr' OR network_address='lig-gpu3.imag.fr' OR network_address='lig-gpu4.imag.fr' OR network_address='lig-gpu5.imag.fr' OR network_address='lig-gpu6.imag.fr'" \
# If not inside OAR job → submit self
if [ -z "${OAR_JOB_ID:-}" ]; then
    mkdir -p "${LOG_DIR}"
    SNAPSHOT="$(mktemp "${LOG_DIR}/.snapshot.XXXXXX.sh")"
    cp "$0" "${SNAPSHOT}"
    chmod +x "${SNAPSHOT}"
    export SNAPSHOT_FILE="${SNAPSHOT}"
    oarsub \
        -n "${EXPERIMENT}_${TASK}_${MODEL_FAMILY}_${MODEL}" \
        -p "network_address='lig-gpu8.imag.fr'" \
        -l /host=1/gpu=${N_GPUS},walltime=${WALLTIME} \
        -O "${LOG_FILE}" \
        -E "${LOG_FILE}" \
        "${SNAPSHOT}"
    exit 0
fi

# Inside OAR job → run experiment
if [ -n "${SNAPSHOT_FILE:-}" ]; then
    rm -f "${SNAPSHOT_FILE}"
fi
# source reason/bin/activate
source primitive/bin/activate

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

# TO RUN, COPY: bash oarsh/causal_trace.sh
#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################
EXPERIMENT="superposition"
MODEL_FAMILY="llama"
MODE="coconut_u"
N_GPUS=1
WALLTIME="02:00:00"
########################################

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}/${MODEL_FAMILY}_${MODE}.txt"

# If not inside OAR job → submit self
if [ -z "${OAR_JOB_ID:-}" ]; then
    mkdir -p "${LOG_DIR}"
    SNAPSHOT="$(mktemp "${LOG_DIR}/.snapshot.XXXXXX.sh")"
    cp "$0" "${SNAPSHOT}"
    chmod +x "${SNAPSHOT}"
    export SNAPSHOT_FILE="${SNAPSHOT}"
    oarsub \
        -n "${EXPERIMENT}_${MODEL_FAMILY}_${MODE}" \
        -p "network_address='lig-gpu1.imag.fr' OR network_address='lig-gpu2.imag.fr' OR network_address='lig-gpu3.imag.fr' OR network_address='lig-gpu4.imag.fr' OR network_address='lig-gpu5.imag.fr' OR network_address='lig-gpu6.imag.fr'" \
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
source primitive/bin/activate

> "${LOG_FILE}"
echo "Model Family    : ${MODEL_FAMILY}"
echo "Mode            : ${MODE}"
echo "Log file        : ${LOG_FILE}"

python -u -m experiments.dead_salmon.superposition \
    --model_family "${MODEL_FAMILY}" \
    --mode "${MODE}" \
    >> "${LOG_FILE}" 2>&1

# TO RUN, COPY: bash oarsh/superposition.sh
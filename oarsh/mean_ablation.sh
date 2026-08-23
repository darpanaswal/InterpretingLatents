#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################
EXPERIMENT="mean_ablation"
TASK="gsm"
MODEL_FAMILY="llama"
MODEL="coconut_u"
N_THOUGHTS=6
INTERVENTION="all"
N_GPUS=4
WALLTIME="20:00:00"
########################################

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}/${TASK}_${MODEL_FAMILY}_${MODEL}.txt"

# If not inside OAR job → submit self
if [ -z "${OAR_JOB_ID:-}" ]; then
    mkdir -p "${LOG_DIR}"
    SNAPSHOT="$(mktemp "${LOG_DIR}/.snapshot.XXXXXX.sh")"
    cp "$0" "${SNAPSHOT}"
    chmod +x "${SNAPSHOT}"
    export SNAPSHOT_FILE="${SNAPSHOT}"
    oarsub \
        -n "${EXPERIMENT}_${TASK}_${MODEL_FAMILY}_${MODEL}" \
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

# TO RUN, COPY: bash oarsh/mean_ablation.sh
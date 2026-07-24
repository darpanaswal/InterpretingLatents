#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################
EXPERIMENT="variance_decomposition"
MODEL_FAMILY="llama"
N_GPUS=1
WALLTIME="02:00:00"
########################################

LOG_DIR="runs/variance_decomposition"
LOG_FILE="${LOG_DIR}/${MODEL_FAMILY}.txt"

# If not inside OAR job → submit self
if [ -z "${OAR_JOB_ID:-}" ]; then
    mkdir -p "${LOG_DIR}"
    SNAPSHOT="$(mktemp "${LOG_DIR}/.snapshot.XXXXXX.sh")"
    cp "$0" "${SNAPSHOT}"
    chmod +x "${SNAPSHOT}"
    oarsub \
        -n "${EXPERIMENT}" \
        -p "network_address='lig-gpu1.imag.fr' OR network_address='lig-gpu2.imag.fr' OR network_address='lig-gpu3.imag.fr' OR network_address='lig-gpu4.imag.fr' OR network_address='lig-gpu5.imag.fr' OR network_address='lig-gpu6.imag.fr'" \
        -l /host=1/gpu=${N_GPUS},walltime=${WALLTIME} \
        -O "${LOG_FILE}" \
        -E "${LOG_FILE}" \
        "${SNAPSHOT}"
    exit 0
fi

# Inside OAR job → run experiment
rm -f "$0"
source primitive/bin/activate

> "${LOG_FILE}"
echo "Running variance decomposition for all"
echo "Log file        : ${LOG_FILE}"

python -u -m experiments.geometry.variance_decomposition \
    --all \
    --model_family "${MODEL_FAMILY}" \
    >> "${LOG_FILE}" 2>&1

# TO RUN, COPY: bash args_old/variance_decomposition.sh
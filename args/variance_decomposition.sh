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
    oarsub \
        -n "${EXPERIMENT}" \
        -p "network_address='lig-gpu7.imag.fr'" \
        -l /host=1/gpu=${N_GPUS},walltime=${WALLTIME} \
        -O "${LOG_FILE}" \
        -E "${LOG_FILE}" \
        "$0"
    exit 0
fi

# Inside OAR job → run experiment
source primitive/bin/activate

> "${LOG_FILE}"
echo "Running variance decomposition for all"
echo "Log file        : ${LOG_FILE}"

python -u -m experiments.geometry.variance_decomposition \
    --all \
    --model_family "${MODEL_FAMILY}" \
    >> "${LOG_FILE}" 2>&1

# TO RUN, COPY: bash args/variance_decomposition.sh
#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################
EXPERIMENT="extract_thoughts"
TASK="prosqa"
MODEL_FAMILY="llama"
MODEL="coconut_u"
N_THOUGHTS=6
N_GPUS=1
WALLTIME="02:00:00"
########################################

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}/${TASK}_${MODEL_FAMILY}_${MODEL}.txt"

# If not inside OAR job → submit self
if [ -z "${OAR_JOB_ID:-}" ]; then
    mkdir -p "${LOG_DIR}"
    oarsub \
        -n "${EXPERIMENT}_${TASK}_${MODEL_FAMILY}_${MODEL}" \
        -p "network_address='lig-gpu1.imag.fr' OR network_address='lig-gpu2.imag.fr' OR network_address='lig-gpu3.imag.fr' OR network_address='lig-gpu4.imag.fr' OR network_address='lig-gpu5.imag.fr' OR network_address='lig-gpu6.imag.fr'" \
        -l /host=1/gpu=${N_GPUS},walltime=${WALLTIME} \
        -O "${LOG_FILE}" \
        -E "${LOG_FILE}" \
        "$0"
    exit 0
fi

# Inside OAR job → run experiment
source primitive/bin/activate

> "${LOG_FILE}"
echo "Task         : ${TASK}"
echo "Model Family : ${MODEL_FAMILY}"
echo "Model        : ${MODEL}"
echo "N Thoughts   : ${N_THOUGHTS}"
echo "GPUs         : ${N_GPUS}"
echo "Log file     : ${LOG_FILE}"

python -u -m experiments.extract_thoughts \
    --task "${TASK}" \
    --model_family "${MODEL_FAMILY}" \
    --model "${MODEL}" \
    --n_thoughts "${N_THOUGHTS}" \
    >> "${LOG_FILE}" 2>&1

# TO RUN, COPY: bash args/extract_thoughts.sh
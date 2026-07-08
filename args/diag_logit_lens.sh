#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################
EXPERIMENT="diag_logit_lens"
N_GPUS=1
WALLTIME="20:00:00"
########################################

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}.txt"

        # 
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
# source reason/bin/activate
source primitive/bin/activate

> "${LOG_FILE}"
echo "GPUs              : ${N_GPUS}"
echo "Log file          : ${LOG_FILE}"

CMD=(
    python diag_logit_lens.py --family llama --model cot --task gsm
)

"${CMD[@]}" > "${LOG_FILE}" 2>&1

# TO RUN, COPY: bash args/causal_trace.sh
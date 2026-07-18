#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################
EXPERIMENT="markovianity_test"
TASK="gsm"
MODEL_FAMILY="llama"
MODEL="codi"
PROJECT_TO_SUBSPACE="both"
N_GPUS=1
WALLTIME="10:00:00"
########################################

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}/${TASK}_${MODEL_FAMILY}_${MODEL}.txt"

# If not inside OAR job → submit self
if [ -z "${OAR_JOB_ID:-}" ]; then
    mkdir -p "${LOG_DIR}"
    oarsub \
        -n "${EXPERIMENT}_${TASK}_${MODEL_FAMILY}_${MODEL}" \
        -p "network_address='lig-gpu7.imag.fr' OR network_address='lig-gpu8.imag.fr' OR network_address='lig-gpu9.imag.fr' OR network_address='lig-gpu10.imag.fr'" \
        -l /host=1/gpu=${N_GPUS},walltime=${WALLTIME} \
        -O "${LOG_FILE}" \
        -E "${LOG_FILE}" \
        "$0"
    exit 0
fi

# Inside OAR job → run experiment
source primitive/bin/activate

> "${LOG_FILE}"
echo "Task            : ${TASK}"
echo "Model Family    : ${MODEL_FAMILY}"
echo "Model           : ${MODEL}"
echo "GPUs            : ${N_GPUS}"
echo "Project Subspace: ${PROJECT_TO_SUBSPACE}"
echo "Log file        : ${LOG_FILE}"

python -u -m experiments.geometry.markovianity_test \
    --task "${TASK}" \
    --model "${MODEL}" \
    --model_family "${MODEL_FAMILY}" \
    --num_gpus "${N_GPUS}" \
    --project_to_subspace "${PROJECT_TO_SUBSPACE}" \
    >> "${LOG_FILE}" 2>&1

# TO RUN, COPY: bash args/markovianity_test.sh
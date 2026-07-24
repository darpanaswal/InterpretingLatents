#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################
EXPERIMENT="gradient_subspace_interventions_predtoken"
TASK="gsm"
MODEL_FAMILY="gpt2"
MODEL="coconut_u"
N_GPUS=4
WALLTIME="6:00:00"
########################################

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}/${TASK}_${MODEL_FAMILY}_${MODEL}.txt"

BASES_PATH="outputs/gradient_geometry_predtoken/${MODEL_FAMILY}/${TASK}/${MODEL}/bases.npz"
OUTPUT_DIR="outputs/grad_subspace_predtoken/${MODEL_FAMILY}/${TASK}/${MODEL}"

# If not inside OAR job → submit self
if [ -z "${OAR_JOB_ID:-}" ]; then
    mkdir -p "${LOG_DIR}"
    SNAPSHOT="$(mktemp "${LOG_DIR}/.snapshot.XXXXXX.sh")"
    cp "$0" "${SNAPSHOT}"
    chmod +x "${SNAPSHOT}"
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
rm -f "$0"
source primitive/bin/activate

> "${LOG_FILE}"
echo "Task            : ${TASK}"
echo "Model Family    : ${MODEL_FAMILY}"
echo "Model           : ${MODEL}"
echo "GPUs            : ${N_GPUS}"
echo "Bases path      : ${BASES_PATH}"
echo "Output dir      : ${OUTPUT_DIR}"
echo "Log file        : ${LOG_FILE}"

if [ ! -f "${BASES_PATH}" ]; then
    echo "ERROR: ${BASES_PATH} not found. Run args_old/gradient_subspace_predtoken.sh first." >&2
    exit 1
fi

python -u -m experiments.ablation.gradient_subspace_interventions \
    --task "${TASK}" \
    --model_family "${MODEL_FAMILY}" \
    --model "${MODEL}" \
    --n_gpus "${N_GPUS}" \
    --bases_path "${BASES_PATH}" \
    --output_dir "${OUTPUT_DIR}" \
    >> "${LOG_FILE}" 2>&1

# TO RUN, COPY: bash args_old/gradient_subspace_interventions_predtoken.sh

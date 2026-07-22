#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################
EXPERIMENT="extract_thoughts"
TASK="prosqa"
MODEL_FAMILY="llama"
MODEL="coconut_u"
SPLIT="test"  # test | train | both
N_THOUGHTS=6
N_GPUS=1
WALLTIME="02:00:00"
########################################

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}/${TASK}_${MODEL_FAMILY}_${MODEL}_${SPLIT}.txt"

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
echo "Task         : ${TASK}"
echo "Model Family : ${MODEL_FAMILY}"
echo "Model        : ${MODEL}"
echo "Split        : ${SPLIT}"
echo "N Thoughts   : ${N_THOUGHTS}"
echo "GPUs         : ${N_GPUS}"
echo "Log file     : ${LOG_FILE}"

python -u -m experiments.extract_thoughts \
    --task "${TASK}" \
    --model_family "${MODEL_FAMILY}" \
    --model "${MODEL}" \
    --split "${SPLIT}" \
    --n_thoughts "${N_THOUGHTS}" \
    --n_gpus "${N_GPUS}" \
    >> "${LOG_FILE}" 2>&1

# TO RUN, COPY: bash args/extract_thoughts.sh

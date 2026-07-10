#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE
########################################
EXPERIMENT="gradient_subspace_interventions_predtoken"
TASK="gsm"
MODEL_FAMILY="gpt2"
MODEL="codi"
N_GPUS=8
WALLTIME="12:00:00"
########################################

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}/${TASK}_${MODEL_FAMILY}_${MODEL}.txt"

SCRIPT_PATH="$(readlink -f "$0")"

# Predicted-token bases (from the extraction step) and a namespaced output dir
# so nothing collides with the gold-subspace intervention run.
BASES_PATH="outputs/gradient_geometry_predtoken/${MODEL_FAMILY}/${TASK}/${MODEL}/bases.npz"
OUT_DIR="outputs/grad_subspace_predtoken/${MODEL_FAMILY}/${TASK}/${MODEL}"

# If not inside a SLURM job -> submit self
if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "${LOG_DIR}"
    sbatch \
        --job-name="${EXPERIMENT}_${TASK}_${MODEL_FAMILY}_${MODEL}" \
        --output="${LOG_FILE}" \
        --error="${LOG_FILE}" \
        --partition=gpu_p2 \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=$((N_GPUS * 4)) \
        --gres=gpu:${N_GPUS} \
        --time="${WALLTIME}" \
        "${SCRIPT_PATH}"
    exit 0
fi

# Inside SLURM job -> run experiment
module purge
module load anaconda-py3/2024.06
source $WORK/env_cache_guard.sh
conda activate lrm

> "${LOG_FILE}"
echo "Task            : ${TASK}"
echo "Model Family    : ${MODEL_FAMILY}"
echo "Model           : ${MODEL}"
echo "GPUs            : ${N_GPUS}"
echo "Bases           : ${BASES_PATH}"
echo "Output dir      : ${OUT_DIR}"
echo "Log file        : ${LOG_FILE}"

python -u -m experiments.ablation.gradient_subspace_interventions \
    --task "${TASK}" \
    --model_family "${MODEL_FAMILY}" \
    --model "${MODEL}" \
    --n_gpus "${N_GPUS}" \
    --bases_path "${BASES_PATH}" \
    --output_dir "${OUT_DIR}" \
    >> "${LOG_FILE}" 2>&1

# TO RUN, COPY: bash args/gradient_subspace_predtoken_interventions.sh
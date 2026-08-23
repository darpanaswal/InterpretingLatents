#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE (env-overridable, e.g.
#   TASK=gsm MODEL_FAMILY=llama MODEL=codi bash slurm/gradient_subspace_predtoken_interventions.sh)
########################################
EXPERIMENT="gradient_subspace_interventions_predtoken"
TASK="${TASK:-gsm}"
MODEL_FAMILY="${MODEL_FAMILY:-gpt2}"
MODEL="${MODEL:-codi}"
N_GPUS="${N_GPUS:-4}"
WALLTIME="${WALLTIME:-12:00:00}"
########################################

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}/${TASK}_${MODEL_FAMILY}_${MODEL}.txt"

# Predicted-token bases (from the extraction step) and a namespaced output dir
# so nothing collides with the gold-subspace intervention run.
BASES_PATH="outputs/gradient_geometry_predtoken/${MODEL_FAMILY}/${TASK}/${MODEL}/bases.npz"
OUT_DIR="outputs/grad_subspace_predtoken/${MODEL_FAMILY}/${TASK}/${MODEL}"

# If not inside a SLURM job -> submit self
if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "${LOG_DIR}"
    SNAPSHOT="$(mktemp "${LOG_DIR}/.snapshot.XXXXXX.sh")"
    cp "$(readlink -f "$0")" "${SNAPSHOT}"
    chmod +x "${SNAPSHOT}"
    JOBID=$(sbatch --parsable \
        --export=ALL,SNAPSHOT_FILE="${SNAPSHOT}" \
        --job-name="${EXPERIMENT}_${TASK}_${MODEL_FAMILY}_${MODEL}" \
        --output="${LOG_FILE}" \
        --error="${LOG_FILE}" \
        --partition=gpu_p2,gpu_p2s,gpu_p2l \
        --nodes=1 \
        --ntasks=1 \
        --cpus-per-task=$((N_GPUS * 4)) \
        --gres=gpu:${N_GPUS} \
        --time="${WALLTIME}" \
        ${SBATCH_EXTRA_ARGS:-} \
        "${SNAPSHOT}"
    )
    echo "${JOBID}"
    exit 0
fi

# Inside SLURM job -> run experiment
if [ -n "${SNAPSHOT_FILE:-}" ]; then
    rm -f "${SNAPSHOT_FILE}"
fi
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

# RUN GPT2

# TASK=prosqa MODEL_FAMILY=gpt2 MODEL=pause bash slurm/gradient_subspace_predtoken_interventions.sh
# TASK=prosqa MODEL_FAMILY=gpt2 MODEL=coconut bash slurm/gradient_subspace_predtoken_interventions.sh
# TASK=prosqa MODEL_FAMILY=gpt2 MODEL=coconut_u bash slurm/gradient_subspace_predtoken_interventions.sh
# TASK=prosqa MODEL_FAMILY=gpt2 MODEL=codi bash slurm/gradient_subspace_predtoken_interventions.sh
# TASK=gsm    MODEL_FAMILY=gpt2 MODEL=pause bash slurm/gradient_subspace_predtoken_interventions.sh
# TASK=gsm    MODEL_FAMILY=gpt2 MODEL=coconut bash slurm/gradient_subspace_predtoken_interventions.sh
# TASK=gsm    MODEL_FAMILY=gpt2 MODEL=coconut_u bash slurm/gradient_subspace_predtoken_interventions.sh
# TASK=gsm    MODEL_FAMILY=gpt2 MODEL=codi bash slurm/gradient_subspace_predtoken_interventions.sh

# RUN LLAMA

# TASK=prosqa MODEL_FAMILY=llama MODEL=pause bash slurm/gradient_subspace_predtoken_interventions.sh
# TASK=prosqa MODEL_FAMILY=llama MODEL=coconut bash slurm/gradient_subspace_predtoken_interventions.sh
# TASK=prosqa MODEL_FAMILY=llama MODEL=coconut_u bash slurm/gradient_subspace_predtoken_interventions.sh
# TASK=prosqa MODEL_FAMILY=llama MODEL=codi bash slurm/gradient_subspace_predtoken_interventions.sh
# TASK=gsm    MODEL_FAMILY=llama MODEL=pause bash slurm/gradient_subspace_predtoken_interventions.sh
# TASK=gsm    MODEL_FAMILY=llama MODEL=coconut bash slurm/gradient_subspace_predtoken_interventions.sh
# TASK=gsm    MODEL_FAMILY=llama MODEL=coconut_u bash slurm/gradient_subspace_predtoken_interventions.sh
# TASK=gsm    MODEL_FAMILY=llama MODEL=codi bash slurm/gradient_subspace_predtoken_interventions.sh
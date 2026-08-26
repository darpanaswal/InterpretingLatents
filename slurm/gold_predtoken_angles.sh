#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE (env-overridable, e.g.
#   TASK=gsm MODEL_FAMILY=llama MODEL=coconut bash slurm/gold_predtoken_angles.sh)
########################################
EXPERIMENT="gold_predtoken_angles"
TASK="${TASK:-prosqa}"
MODEL_FAMILY="${MODEL_FAMILY:-gpt2}"
MODEL="${MODEL:-pause}"
N_GPUS="${N_GPUS:-1}"
WALLTIME="${WALLTIME:-00:20:00}"
########################################

# Note: pure-NumPy job (SVD of B_gold^T B_pred, k x k matrices) — never
# touches the GPU. The gres request only keeps it uniform with the rest of
# slurm/; set N_GPUS=0 and swap the partition for a CPU/prepost queue if you
# prefer not to hold a GPU allocation for it.

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}/${TASK}_${MODEL_FAMILY}_${MODEL}.txt"

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
echo "Task         : ${TASK}"
echo "Model Family : ${MODEL_FAMILY}"
echo "Model        : ${MODEL}"
echo "GPUs         : ${N_GPUS}"
echo "Log file     : ${LOG_FILE}"

python -u -B -m experiments.geometry.gold_predtoken_angles \
    --task "${TASK}" \
    --model_family "${MODEL_FAMILY}" \
    --model "${MODEL}" \
    >> "${LOG_FILE}" 2>&1

# RUN GPT2

# TASK=prosqa MODEL_FAMILY=gpt2 MODEL=pause     bash slurm/gold_predtoken_angles.sh
# TASK=prosqa MODEL_FAMILY=gpt2 MODEL=coconut   bash slurm/gold_predtoken_angles.sh
# TASK=prosqa MODEL_FAMILY=gpt2 MODEL=coconut_u bash slurm/gold_predtoken_angles.sh
# TASK=prosqa MODEL_FAMILY=gpt2 MODEL=codi      bash slurm/gold_predtoken_angles.sh
# TASK=gsm    MODEL_FAMILY=gpt2 MODEL=pause     bash slurm/gold_predtoken_angles.sh
# TASK=gsm    MODEL_FAMILY=gpt2 MODEL=coconut   bash slurm/gold_predtoken_angles.sh
# TASK=gsm    MODEL_FAMILY=gpt2 MODEL=coconut_u bash slurm/gold_predtoken_angles.sh
# TASK=gsm    MODEL_FAMILY=gpt2 MODEL=codi      bash slurm/gold_predtoken_angles.sh

# RUN LLAMA

# TASK=prosqa MODEL_FAMILY=llama MODEL=pause     bash slurm/gold_predtoken_angles.sh
# TASK=prosqa MODEL_FAMILY=llama MODEL=coconut   bash slurm/gold_predtoken_angles.sh
# TASK=prosqa MODEL_FAMILY=llama MODEL=coconut_u bash slurm/gold_predtoken_angles.sh
# TASK=prosqa MODEL_FAMILY=llama MODEL=codi      bash slurm/gold_predtoken_angles.sh
# TASK=gsm    MODEL_FAMILY=llama MODEL=pause     bash slurm/gold_predtoken_angles.sh
# TASK=gsm    MODEL_FAMILY=llama MODEL=coconut   bash slurm/gold_predtoken_angles.sh
# TASK=gsm    MODEL_FAMILY=llama MODEL=coconut_u bash slurm/gold_predtoken_angles.sh
# TASK=gsm    MODEL_FAMILY=llama MODEL=codi      bash slurm/gold_predtoken_angles.sh
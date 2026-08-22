#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE (env-overridable, e.g.
#   TASK=gsm MODEL_FAMILY=llama MODEL=codi bash args/remove_thoughts.sh)
########################################
EXPERIMENT="remove_thoughts"
TASK="${TASK:-gsm}"
MODEL_FAMILY="${MODEL_FAMILY:-llama}"
MODEL="${MODEL:-pause}"
N_GPUS="${N_GPUS:-1}"
WALLTIME="${WALLTIME:-02:00:00}"
########################################

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
echo "Model Family    : ${MODEL_FAMILY}"
echo "Task            : ${TASK}"
echo "Models          : ${MODEL}"
echo "GPUs            : ${N_GPUS}"
echo "Log file        : ${LOG_FILE}"

# torchrun defaults to master port 29500 with no --master_port given, so
# any two jobs that land on the same compute node at once collide with
# EADDRINUSE (this happened running several remove_thoughts.sh configs in
# parallel: single-GPU jobs get packed multiple-per-node). Derive a port
# from the SLURM job ID -- unique per job, no coordination needed.
MASTER_PORT=$((20000 + SLURM_JOB_ID % 10000))

torchrun --nproc_per_node="${N_GPUS}" --master_port="${MASTER_PORT}" \
    -m experiments.ablation.remove_thoughts \
    --model_family "${MODEL_FAMILY}" \
    --task "${TASK}" \
    --models "${MODEL}" \
    >> "${LOG_FILE}" 2>&1

# TO RUN, COPY: bash args/remove_thoughts.sh
# TO OVERRIDE, COPY: TASK=gsm MODEL_FAMILY=llama MODEL=codi bash args/remove_thoughts.sh

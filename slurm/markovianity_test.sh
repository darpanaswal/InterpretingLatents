#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE (env-overridable, e.g.
#   TASK=all MODEL_FAMILY=llama MODEL=all PROJECT_TO_SUBSPACE=pred_and_gold \
#   bash slurm/markovianity_test.sh)
########################################
EXPERIMENT="markovianity_test"
TASK="${TASK:-prosqa}"
MODEL_FAMILY="${MODEL_FAMILY:-llama}"
MODEL="${MODEL:-codi}"
PROJECT_TO_SUBSPACE="${PROJECT_TO_SUBSPACE:-all}"
# Sharding unit is (order, "core"|mlp_seed) -- an order's core analysis
# plus one unit per --mlp_seeds value (default 3 seeds) -- so parallelism
# isn't capped by the number of orders: even a single valid order gives
# 1 + 3 = 4 units to spread across GPUs. Default 4 matches that.
N_GPUS="${N_GPUS:-4}"
# Host RAM on this cluster scales with --cpus-per-task, not --gres=gpu, so
# this is NOT tied to N_GPUS (that coupling is what caused a host-RAM OOM
# once already: N_GPUS 4->1 silently dropped this from 16->4). The
# streaming train-side pipeline no longer needs much RAM regardless of
# dataset size, but keep a modest fixed floor as insurance for chunk
# buffers + overhead, and a little more per GPU worker process.
CPUS_PER_TASK="${CPUS_PER_TASK:-$((N_GPUS*4))}"
WALLTIME="${WALLTIME:-20:00:00}"
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
        --cpus-per-task=${CPUS_PER_TASK} \
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
echo "Project Subspace: ${PROJECT_TO_SUBSPACE}"
echo "Log file        : ${LOG_FILE}"

python -u -B -m experiments.geometry.markovianity_test \
    --task "${TASK}" \
    --model "${MODEL}" \
    --model_family "${MODEL_FAMILY}" \
    --project_to_subspace "${PROJECT_TO_SUBSPACE}" \
    --n_gpus "${N_GPUS}" \
    >> "${LOG_FILE}" 2>&1

# TO RUN, COPY: bash slurm/markovianity_test.sh
# TO OVERRIDE, COPY: TASK=all MODEL_FAMILY=llama MODEL=codi PROJECT_TO_SUBSPACE=pred_and_gold bash slurm/markovianity_test.sh

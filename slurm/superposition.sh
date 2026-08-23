#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE (env-overridable, e.g.
#   MODEL_FAMILY=llama MODE=codi bash slurm/superposition.sh)
# prosqa only, deliberately -- the python script defaults --task to
# prosqa and this launcher never exposed it as a variable to override.
########################################
EXPERIMENT="superposition"
MODEL_FAMILY="${MODEL_FAMILY:-llama}"
MODE="${MODE:-coconut_u}"
N_GPUS="${N_GPUS:-1}"
WALLTIME="${WALLTIME:-02:00:00}"
########################################

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}/${MODEL_FAMILY}_${MODE}.txt"

# If not inside a SLURM job -> submit self
if [ -z "${SLURM_JOB_ID:-}" ]; then
    mkdir -p "${LOG_DIR}"
    SNAPSHOT="$(mktemp "${LOG_DIR}/.snapshot.XXXXXX.sh")"
    cp "$(readlink -f "$0")" "${SNAPSHOT}"
    chmod +x "${SNAPSHOT}"
    JOBID=$(sbatch --parsable \
        --export=ALL,SNAPSHOT_FILE="${SNAPSHOT}" \
        --job-name="${EXPERIMENT}_${MODEL_FAMILY}_${MODE}" \
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
echo "Mode            : ${MODE}"
echo "Log file        : ${LOG_FILE}"

python -u -m experiments.dead_salmon.superposition \
    --model_family "${MODEL_FAMILY}" \
    --mode "${MODE}" \
    >> "${LOG_FILE}" 2>&1

# RUN GPT2

# MODEL_FAMILY=gpt2 MODE=pause bash slurm/superposition.sh
# MODEL_FAMILY=gpt2 MODE=coconut bash slurm/superposition.sh
# MODEL_FAMILY=gpt2 MODE=coconut_u bash slurm/superposition.sh
# MODEL_FAMILY=gpt2 MODE=codi bash slurm/superposition.sh

# RUN LLAMA

# MODEL_FAMILY=llama MODE=pause bash slurm/superposition.sh
# MODEL_FAMILY=llama MODE=coconut bash slurm/superposition.sh
# MODEL_FAMILY=llama MODE=coconut_u bash slurm/superposition.sh
# MODEL_FAMILY=llama MODE=codi bash slurm/superposition.sh

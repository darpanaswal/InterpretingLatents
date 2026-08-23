#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE (env-overridable, e.g.
#   TASK=gsm MODEL_FAMILY=llama MODEL=codi bash slurm/causal_trace.sh)
########################################
EXPERIMENT="causal_trace"
TASK="${TASK:-gsm}"
MODEL_FAMILY="${MODEL_FAMILY:-llama}"
MODEL="${MODEL:-codi}"
GRANULARITY="${GRANULARITY:-single}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_INSTANCES="${MAX_INSTANCES:-2000}"
DEBUG="${DEBUG:-false}"
VERIFY_BATCHED="${VERIFY_BATCHED:-false}"
N_GPUS="${N_GPUS:-6}"
WALLTIME="${WALLTIME:-10:00:00}"
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
        --cpus-per-task=$((N_GPUS)) \
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
module load anaconda-py3/2024.06 git-lfs
source $WORK/env_cache_guard.sh
conda activate lrm

> "${LOG_FILE}"
echo "Task              : ${TASK}"
echo "Model Family      : ${MODEL_FAMILY}"
echo "Model             : ${MODEL}"
echo "GPUs              : ${N_GPUS}"
echo "Granularity       : ${GRANULARITY}"
echo "Batch Size        : ${BATCH_SIZE}"
echo "Max Instances     : ${MAX_INSTANCES}"
echo "Debug             : ${DEBUG}"
echo "Verify Batched    : ${VERIFY_BATCHED}"
echo "Log file          : ${LOG_FILE}"

CMD=(
    python -u -m experiments.causal_tracing.causal_trace
    --task "${TASK}"
    --model_family "${MODEL_FAMILY}"
    --model "${MODEL}"
    --n_gpus "${N_GPUS}"
    --granularity "${GRANULARITY}"
    --batch_size "${BATCH_SIZE}"
    --max_instances "${MAX_INSTANCES}"
)

if [ "${DEBUG}" = "true" ]; then
    CMD+=(--debug)
fi

if [ "${VERIFY_BATCHED}" = "true" ]; then
    CMD+=(--verify_batched)
fi

"${CMD[@]}" > "${LOG_FILE}" 2>&1

# RUN GPT2

# TASK=prosqa MODEL_FAMILY=gpt2 MODEL=pause bash slurm/causal_trace.sh
# TASK=prosqa MODEL_FAMILY=gpt2 MODEL=coconut bash slurm/causal_trace.sh
# TASK=prosqa MODEL_FAMILY=gpt2 MODEL=coconut_u bash slurm/causal_trace.sh
# TASK=prosqa MODEL_FAMILY=gpt2 MODEL=codi bash slurm/causal_trace.sh
# TASK=gsm    MODEL_FAMILY=gpt2 MODEL=pause bash slurm/causal_trace.sh
# TASK=gsm    MODEL_FAMILY=gpt2 MODEL=coconut bash slurm/causal_trace.sh
# TASK=gsm    MODEL_FAMILY=gpt2 MODEL=coconut_u bash slurm/causal_trace.sh
# TASK=gsm    MODEL_FAMILY=gpt2 MODEL=codi bash slurm/causal_trace.sh

# RUN LLAMA

# TASK=prosqa MODEL_FAMILY=llama MODEL=pause bash slurm/causal_trace.sh
# TASK=prosqa MODEL_FAMILY=llama MODEL=coconut bash slurm/causal_trace.sh
# TASK=prosqa MODEL_FAMILY=llama MODEL=coconut_u bash slurm/causal_trace.sh
# TASK=prosqa MODEL_FAMILY=llama MODEL=codi bash slurm/causal_trace.sh
# TASK=gsm    MODEL_FAMILY=llama MODEL=pause bash slurm/causal_trace.sh
# TASK=gsm    MODEL_FAMILY=llama MODEL=coconut bash slurm/causal_trace.sh
# TASK=gsm    MODEL_FAMILY=llama MODEL=coconut_u bash slurm/causal_trace.sh
# TASK=gsm    MODEL_FAMILY=llama MODEL=codi bash slurm/causal_trace.sh

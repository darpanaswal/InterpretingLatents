#!/bin/bash
set -euo pipefail

########################################
# CONFIGURE HERE (env-overridable, e.g.
#   TASK=gsm MODEL_FAMILY=llama MODEL=codi SPLIT=both bash slurm/extract_thoughts.sh)
########################################
EXPERIMENT="extract_thoughts"
TASK="${TASK:-prosqa}"
MODEL_FAMILY="${MODEL_FAMILY:-llama}"
MODEL="${MODEL:-codi}"
SPLIT="${SPLIT:-both}"  # test | train | both
N_THOUGHTS="${N_THOUGHTS:-6}"
N_GPUS="${N_GPUS:-4}"
WALLTIME="${WALLTIME:-12:00:00}"
########################################

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}/${TASK}_${MODEL_FAMILY}_${MODEL}_${SPLIT}.txt"

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
echo "Split        : ${SPLIT}"
echo "N Thoughts   : ${N_THOUGHTS}"
echo "GPUs         : ${N_GPUS}"
echo "Log file     : ${LOG_FILE}"

python -u -B -m experiments.extract_thoughts \
    --task "${TASK}" \
    --model_family "${MODEL_FAMILY}" \
    --model "${MODEL}" \
    --split "${SPLIT}" \
    --n_thoughts "${N_THOUGHTS}" \
    --n_gpus "${N_GPUS}" \
    >> "${LOG_FILE}" 2>&1

# RUN GPT2

# TASK=prosqa MODEL_FAMILY=gpt2 MODEL=pause SPLIT=both bash slurm/extract_thoughts.sh
# TASK=prosqa MODEL_FAMILY=gpt2 MODEL=coconut SPLIT=both bash slurm/extract_thoughts.sh
# TASK=prosqa MODEL_FAMILY=gpt2 MODEL=coconut_u SPLIT=both bash slurm/extract_thoughts.sh
# TASK=prosqa MODEL_FAMILY=gpt2 MODEL=codi SPLIT=both bash slurm/extract_thoughts.sh
# TASK=gsm    MODEL_FAMILY=gpt2 MODEL=pause SPLIT=both bash slurm/extract_thoughts.sh
# TASK=gsm    MODEL_FAMILY=gpt2 MODEL=coconut SPLIT=both bash slurm/extract_thoughts.sh
# TASK=gsm    MODEL_FAMILY=gpt2 MODEL=coconut_u SPLIT=both bash slurm/extract_thoughts.sh
# TASK=gsm    MODEL_FAMILY=gpt2 MODEL=codi SPLIT=both bash slurm/extract_thoughts.sh

# RUN LLAMA

# TASK=prosqa MODEL_FAMILY=llama MODEL=pause SPLIT=both bash slurm/extract_thoughts.sh
# TASK=prosqa MODEL_FAMILY=llama MODEL=coconut SPLIT=both bash slurm/extract_thoughts.sh
# TASK=prosqa MODEL_FAMILY=llama MODEL=coconut_u SPLIT=both bash slurm/extract_thoughts.sh
# TASK=prosqa MODEL_FAMILY=llama MODEL=codi SPLIT=both bash slurm/extract_thoughts.sh
# TASK=gsm    MODEL_FAMILY=llama MODEL=pause SPLIT=both bash slurm/extract_thoughts.sh
# TASK=gsm    MODEL_FAMILY=llama MODEL=coconut SPLIT=both bash slurm/extract_thoughts.sh
# TASK=gsm    MODEL_FAMILY=llama MODEL=coconut_u SPLIT=both bash slurm/extract_thoughts.sh
# TASK=gsm    MODEL_FAMILY=llama MODEL=codi SPLIT=both bash slurm/extract_thoughts.sh
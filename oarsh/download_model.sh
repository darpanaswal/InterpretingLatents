#!/bin/bash
set -euo pipefail

EXPERIMENT="download_model"
WALLTIME="00:05:00"
VENV="primitive/bin/activate"

LOG_DIR="runs/${EXPERIMENT}"
LOG_FILE="${LOG_DIR}/download_model.log"

# Submit if not already inside an OAR job
if [ -z "${OAR_JOB_ID:-}" ]; then
    mkdir -p "${LOG_DIR}"
    SNAPSHOT="$(mktemp "${LOG_DIR}/.snapshot.XXXXXX.sh")"
    cp "$0" "${SNAPSHOT}"
    chmod +x "${SNAPSHOT}"
    export SNAPSHOT_FILE="${SNAPSHOT}"
    oarsub \
        -n "download_model" \
        -l /host=1/core=1,walltime=${WALLTIME} \
        -O "${LOG_FILE}" \
        -E "${LOG_FILE}" \
        "${SNAPSHOT}"

    exit 0
fi

if [ -n "${SNAPSHOT_FILE:-}" ]; then
    rm -f "${SNAPSHOT_FILE}"
fi
source "${VENV}"

mkdir -p "${LOG_DIR}"

# Start fresh log
> "${LOG_FILE}"

echo "Log file : ${LOG_FILE}"
echo "Job ID   : ${OAR_JOB_ID}"
echo "----------------------------------------"

python -u -m src.download_model

echo "----------------------------------------"
echo "Download completed."
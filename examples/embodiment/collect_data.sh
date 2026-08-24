#! /bin/bash

set -o pipefail

export EMBODIED_PATH="$( cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd )"
export REPO_PATH=$(dirname $(dirname "$EMBODIED_PATH"))
export SRC_FILE="${EMBODIED_PATH}/collect_real_data.py"

export PYTHONPATH=${REPO_PATH}:$PYTHONPATH

export HYDRA_FULL_ERROR=1


if [ -z "${1:-}" ]; then
    CONFIG_NAME="realworld_collect_data"
else
    CONFIG_NAME=$1
    shift
fi

PYTHON_BIN="${RLINF_PYTHON:-${REPO_PATH}/.venv/bin/python}"
if [[ "${PYTHON_BIN}" == */* ]]; then
    if [[ ! -x "${PYTHON_BIN}" ]]; then
        echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
        exit 127
    fi
else
    PYTHON_BIN="$(command -v "${PYTHON_BIN}")" || {
        echo "Python interpreter not found: ${RLINF_PYTHON}" >&2
        exit 127
    }
fi

echo "Using Python at ${PYTHON_BIN}"
LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')" #/$(date +'%Y%m%d-%H:%M:%S')"
MEGA_LOG_FILE="${LOG_DIR}/run_embodiment.log"
mkdir -p "${LOG_DIR}"
CMD=(
    "${PYTHON_BIN}" "${SRC_FILE}"
    --config-path "${EMBODIED_PATH}/config/"
    --config-name "${CONFIG_NAME}"
    "$@"
    "runner.logger.log_path=${LOG_DIR}"
)
printf '%q ' "${CMD[@]}" > "${MEGA_LOG_FILE}"
printf '\n' >> "${MEGA_LOG_FILE}"
"${CMD[@]}" 2>&1 | (
    # The Python driver handles the first Ctrl-C as a graceful stop request.
    # Keep the logging side of the foreground pipeline alive until finalization.
    trap '' INT
    tee -a "${MEGA_LOG_FILE}"
)

#!/usr/bin/env bash
set -euo pipefail

REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENPI_PATH="${OPENPI_PATH:-${HOME}/openpi}"
CHECKPOINT_DIR="${RLINF_PI05_CHECKPOINT:-${HOME}/checkpoints/openpi/pi05_franka_fold_steam_cfgrl_a8004_29999}"
SERVER_HOST="${OPENPI_SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${OPENPI_SERVER_PORT:-8000}"
SERVER_START_TIMEOUT="${OPENPI_SERVER_START_TIMEOUT:-600}"
RUN_DIR="${RLINF_PI05_EVAL_DIR:-${REPO_PATH}/logs/franka_fold_pi05_eval/$(date +'%Y%m%d-%H%M%S')-recap-cfgrl-condition-only}"
SERVER_LOG="${RUN_DIR}/policy_server.log"
ROLLOUT_DIR="${RUN_DIR}/rollouts"
SERVER_SCRIPT="${REPO_PATH}/examples/embodiment/franka_fold_pi05_cfgrl_server.py"
OPENPI_PYTHON="${OPENPI_PYTHON:-${OPENPI_PATH}/.venv/bin/python}"
SERVER_PID=""

usage() {
  cat <<'EOF'
Usage: ./run_franka_fold_pi05_cfgrl_eval.sh [client options]

Starts the positive-condition CFGRL policy server, waits for its warmup and
health check, then runs the dual-Franka evaluation client. Client options such
as --num-episodes, --enable-pico, and hardware overrides are forwarded.

Environment overrides:
  RLINF_PI05_CHECKPOINT          CFGRL checkpoint directory
  RLINF_PI05_EVAL_DIR            Server log and rollout output directory
  OPENPI_PATH                    OpenPI checkout (default: ~/openpi)
  OPENPI_PYTHON                  OpenPI Python executable
  OPENPI_SERVER_HOST             Server/client host (default: 127.0.0.1)
  OPENPI_SERVER_PORT             Server/client port (default: 8000)
  OPENPI_SERVER_START_TIMEOUT    Warmup timeout in seconds (default: 600)

Example:
  ./run_franka_fold_pi05_cfgrl_eval.sh --num-episodes 1
EOF
}

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "Stopping policy server (pid=${SERVER_PID})"
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

for path in "${OPENPI_PYTHON}" "${SERVER_SCRIPT}" "${REPO_PATH}/run_franka_fold_pi05_client.sh"; do
  if [[ ! -e "${path}" ]]; then
    echo "Required path does not exist: ${path}" >&2
    exit 1
  fi
done
if [[ ! -d "${CHECKPOINT_DIR}" ]]; then
  echo "Checkpoint directory does not exist: ${CHECKPOINT_DIR}" >&2
  exit 1
fi
if ! [[ "${SERVER_PORT}" =~ ^[0-9]+$ ]] || (( SERVER_PORT < 1 || SERVER_PORT > 65535 )); then
  echo "OPENPI_SERVER_PORT must be between 1 and 65535" >&2
  exit 1
fi
if ! [[ "${SERVER_START_TIMEOUT}" =~ ^[0-9]+$ ]] || (( SERVER_START_TIMEOUT < 1 )); then
  echo "OPENPI_SERVER_START_TIMEOUT must be a positive integer" >&2
  exit 1
fi
if curl --silent --fail --max-time 1 "http://${SERVER_HOST}:${SERVER_PORT}/healthz" >/dev/null 2>&1; then
  echo "A policy server is already listening at ${SERVER_HOST}:${SERVER_PORT}" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}"
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

echo "Checkpoint: ${CHECKPOINT_DIR}"
echo "Evaluation output: ${RUN_DIR}"
echo "Starting positive-condition policy server"
(
  cd "${OPENPI_PATH}"
  export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
  export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
  export PYTHONPATH="${OPENPI_PATH}/src:${REPO_PATH}:${PYTHONPATH:-}"
  exec "${OPENPI_PYTHON}" "${SERVER_SCRIPT}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" \
    --host "${SERVER_HOST}" \
    --port "${SERVER_PORT}"
) > >(tee "${SERVER_LOG}") 2>&1 &
SERVER_PID=$!

deadline=$((SECONDS + SERVER_START_TIMEOUT))
next_report=$((SECONDS + 30))
while ! curl --silent --fail --max-time 1 "http://${SERVER_HOST}:${SERVER_PORT}/healthz" >/dev/null 2>&1; do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    wait "${SERVER_PID}" || true
    echo "Policy server exited before becoming ready. See ${SERVER_LOG}" >&2
    exit 1
  fi
  if (( SECONDS >= deadline )); then
    echo "Policy server did not become ready within ${SERVER_START_TIMEOUT}s" >&2
    exit 1
  fi
  if (( SECONDS >= next_report )); then
    echo "Still waiting for model load and warmup..."
    next_report=$((SECONDS + 30))
  fi
  sleep 1
done

echo "Policy server is ready; starting the robot client"
cd "${REPO_PATH}"
./run_franka_fold_pi05_client.sh \
  --host "${SERVER_HOST}" \
  --port "${SERVER_PORT}" \
  --rollout-dir "${ROLLOUT_DIR}" \
  "$@"

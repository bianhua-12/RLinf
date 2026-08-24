#!/usr/bin/env bash

set -eo pipefail

EPISODES="${1:-10}"
if [[ ! "${EPISODES}" =~ ^[1-9][0-9]*$ ]]; then
    echo "Usage: $0 [positive_episode_count] [hydra_override ...]" >&2
    exit 2
fi
if (( $# > 0 )); then
    shift
fi

REPO_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_PATH}"

ROS_SETUP="${RLINF_ROS_SETUP:-/opt/ros/humble/setup.bash}"
FRANKA_ROS2_SETUP="${RLINF_FRANKA_ROS2_SETUP:-${HOME}/franka_ros2_ws/install/setup.bash}"
RLINF_ROS2_SETUP="${RLINF_ROS2_SETUP:-${REPO_PATH}/ros2_ws/install/setup.bash}"
for setup_file in "${ROS_SETUP}" "${FRANKA_ROS2_SETUP}" "${RLINF_ROS2_SETUP}"; do
    if [[ ! -f "${setup_file}" ]]; then
        echo "Required ROS 2 environment is missing: ${setup_file}" >&2
        exit 1
    fi
    # shellcheck disable=SC1090
    source "${setup_file}"
done
set -u

PYTHON_BIN="${RLINF_PYTHON:-${REPO_PATH}/.venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python interpreter is not executable: ${PYTHON_BIN}" >&2
    exit 127
fi
if ! "${PYTHON_BIN}" -c "import rclpy"; then
    echo "ROS 2 Python package rclpy is unavailable in ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ "${RLINF_COLLECT_PREFLIGHT_ONLY:-0}" == "1" ]]; then
    exit 0
fi

ray stop --force >/dev/null 2>&1 || true
pkill -TERM -f 'single_fr3\.launch\.py|__ns:=/?(left|right)([[:space:]]|$)' 2>/dev/null || true
sleep 2
pkill -KILL -f 'single_fr3\.launch\.py|__ns:=/?(left|right)([[:space:]]|$)' 2>/dev/null || true
export RLINF_TASK_DESCRIPTION="stack the boxes"
bash examples/embodiment/collect_data.sh \
    realworld_collect_data_ros2_gello_dual_franka_pnp.yaml \
    "runner.num_data_episodes=${EPISODES}" \
    "$@"

#!/usr/bin/env bash

set -euo pipefail

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

ray stop --force >/dev/null 2>&1 || true
pkill -TERM -f 'single_fr3\.launch\.py|__ns:=/?(left|right)([[:space:]]|$)' 2>/dev/null || true
sleep 2
pkill -KILL -f 'single_fr3\.launch\.py|__ns:=/?(left|right)([[:space:]]|$)' 2>/dev/null || true
export RLINF_TASK_DESCRIPTION="fold the clothes"
bash examples/embodiment/collect_data.sh \
    realworld_collect_data_ros2_gello_dual_franka_pnp.yaml \
    "runner.num_data_episodes=${EPISODES}" \
    "$@"

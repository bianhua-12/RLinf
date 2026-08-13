#!/usr/bin/env bash
set -e

client="$(cd "$(dirname "$0")" && pwd)/examples/embodiment/franka_fold_pi05_client.py"
source /opt/ros/humble/setup.bash
source /data/RLinf-franka/ros2_ws/install/setup.bash
cd /data/RLinf-franka
export PYTHONPATH="/data/RLinf-franka${PYTHONPATH:+:$PYTHONPATH}"
ray stop --force >/dev/null 2>&1 || true; pkill -TERM -f 'single_fr3\.launch\.py|__ns:=/?(left|right)([[:space:]]|$)' 2>/dev/null || true; sleep 2; pkill -KILL -f 'single_fr3\.launch\.py|__ns:=/?(left|right)([[:space:]]|$)' 2>/dev/null || true

/data/RLinf/.venv/bin/python \
  "$client" \
  --enable-policy \
  "$@"

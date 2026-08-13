#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash
source /data/RLinf-franka/ros2_ws/install/setup.bash
export PYTHONPATH=/data/RLinf-franka:$PYTHONPATH
cd /data/RLinf-franka
ray stop --force >/dev/null 2>&1 || true; pkill -TERM -f 'single_fr3\.launch\.py|__ns:=/?(left|right)([[:space:]]|$)' 2>/dev/null || true; sleep 2; pkill -KILL -f 'single_fr3\.launch\.py|__ns:=/?(left|right)([[:space:]]|$)' 2>/dev/null || true

/data/RLinf/.venv/bin/python \
  examples/embodiment/franka_fold_gr00t_client.py \
  --enable-policy \
  "$@"

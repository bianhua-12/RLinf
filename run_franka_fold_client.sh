#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash
source /data/RLinf/ros2_ws/install/setup.bash
export PYTHONPATH=/data/RLinf-franka-fold-gr00t-rtc:$PYTHONPATH

/data/RLinf/.venv/bin/python \
  examples/embodiment/franka_fold_gr00t_client.py \
  --enable-policy

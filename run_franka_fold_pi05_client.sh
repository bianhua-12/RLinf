#!/usr/bin/env bash
set -e

client="$(cd "$(dirname "$0")" && pwd)/examples/embodiment/franka_fold_pi05_client.py"
source /opt/ros/humble/setup.bash
source /data/RLinf/ros2_ws/install/setup.bash
cd /data/RLinf
export PYTHONPATH=/data/RLinf

/data/RLinf/.venv/bin/python \
  "$client" \
  --enable-policy

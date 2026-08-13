ray stop --force >/dev/null 2>&1 || true; pkill -TERM -f 'single_fr3\.launch\.py|__ns:=/?(left|right)([[:space:]]|$)' 2>/dev/null || true; sleep 2; pkill -KILL -f 'single_fr3\.launch\.py|__ns:=/?(left|right)([[:space:]]|$)' 2>/dev/null || true

cd /data/RLinf-franka
source /opt/ros/humble/setup.bash
source ros2_ws/install/setup.bash

RLINF_REPLAY_DATASET=/data/RLinf-franka/logs/franka_fold_pi05_rollouts/rank_0/id_0 \
RLINF_REPLAY_EPISODE=0 \
/data/RLinf/.venv/bin/python -m examples.embodiment.replay_lerobot_episode

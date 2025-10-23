# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing as mp
from types import SimpleNamespace

import numpy as np

from rlinf.envs.robotwin.RoboTwin_env import RoboTwin

if __name__ == "__main__":
    mp.set_start_method("spawn")  # solve CUDA compatibility problem
    task_name = "place_shoe"
    n_envs = 2
    steps = 30
    horizon = 10
    action_dim = 14
    times = 10

    # 构造与 env worker 一致的最小 cfg（不要把 n_envs 直接传入构造函数）
    cfg = SimpleNamespace(
        init_params={"num_envs": n_envs},
        horizon=horizon,
        image_size=(224, 224),
        twin2_task_config="demo_clean",
        twin2_ckpt_setting="demo_clean",
        twin2_instruction_type="unseen",
    )
    robotwin = RoboTwin(cfg, seed_offset=0, total_num_processes=1)
    actions = np.zeros((n_envs, horizon, action_dim))
    for t in range(times):
        # 获取初始观测（不再调用 init_process）
        prev_obs_venv, info_venv = robotwin.reset()
        for step in range(steps):
            actions += np.random.randn(n_envs, horizon, action_dim) * 0.05
            actions = np.clip(actions, 0, 1)
            obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
                robotwin.step(actions)
            )

            if step % 10 == 0:
                robotwin.reset()
            if terminated_venv[0] == 1:
                print("main", f"terminated_venv: {terminated_venv}")
            if truncated_venv[0] == 1:
                print("main", f"truncated_venv: {truncated_venv}")
            print("main", f"info_venv: {info_venv}")
        robotwin.clear()

# Copyright 2026 The RLinf Authors.
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

import copy
import os
import pathlib
import time
from functools import partial
from typing import OrderedDict

import gymnasium as gym
import numpy as np
import psutil
import torch
from filelock import FileLock
from omegaconf import OmegaConf

from rlinf.envs.realworld.venv import NoAutoResetSyncVectorEnv
from rlinf.envs.utils import to_tensor
from rlinf.scheduler import WorkerInfo


class RealWorldEnv(gym.Env):
    def __init__(
        self,
        cfg,
        num_envs,
        seed_offset,
        total_num_processes,
        worker_info,
        return_numpy: bool = False,
    ):
        assert num_envs == 1, (
            f"Currently, only 1 realworld env can be started per worker, but {num_envs=} is received."
        )

        self.cfg = cfg
        self.override_cfg = OmegaConf.to_container(
            cfg.get("override_cfg", OmegaConf.create({})), resolve=True
        )

        self.video_cfg = cfg.video_cfg

        self.seed = cfg.seed + seed_offset
        self.num_envs = num_envs
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.auto_reset = cfg.auto_reset
        self.ignore_terminations = cfg.ignore_terminations
        self.num_group = num_envs // cfg.group_size
        self.group_size = cfg.group_size
        self.main_image_key = cfg.main_image_key
        extra_view_image_keys = cfg.get("extra_view_image_keys")
        self.extra_view_image_keys = (
            tuple(extra_view_image_keys) if extra_view_image_keys is not None else None
        )
        self.manual_episode_control_only = bool(
            self.override_cfg.get("manual_episode_control_only", False)
        )
        self.return_numpy = return_numpy
        self._closed = False

        self._init_env()
        self._is_start = True
        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self._init_reset_state_ids()

    def close(self) -> None:
        """Close the vector environment and all owned real-world hardware."""
        if getattr(self, "_closed", False):
            return
        self._closed = True
        if hasattr(self, "env"):
            self.env.close()

    def _create_env(self, env_idx: int):
        worker_info: WorkerInfo = self.worker_info
        hardware_info = None
        if worker_info is not None and env_idx < len(worker_info.hardware_infos):
            hardware_info = worker_info.hardware_infos[env_idx]
        override_cfg = copy.deepcopy(self.override_cfg)
        env = gym.make(
            id=self.cfg.init_params.id,
            override_cfg=override_cfg,
            worker_info=worker_info,
            hardware_info=hardware_info,
            env_idx=env_idx,
            env_cfg=self.cfg,
        )
        return env

    @staticmethod
    def realworld_setup():
        """Setup RealWorld environment upon env class import.

        This is for any node-level setup required by RealWorld environments. For example, ROS
        requires a single roscore instance per node, so we ensure that any existing roscore
        processes are terminated before starting a new one.

        This function is called once when the RealWorldEnv class is first imported.
        """
        # Concurrency control is needed for multiple processes on the same node
        node_lock_file = "/tmp/.realworld.lock"
        # Check if the path is valid
        if not os.path.exists(os.path.dirname(node_lock_file)):
            node_lock_file = os.path.join(pathlib.Path.home(), ".realworld.lock")
        node_lock = FileLock(node_lock_file)

        with node_lock:
            ros_proc_names = ["roscore", "rosmaster", "rosout"]
            for proc in psutil.process_iter():
                if proc.name() in ros_proc_names:
                    proc.kill()
                    time.sleep(0.5)

    def _init_env(self):
        env_fns = [
            partial(self._create_env, env_idx=env_idx)
            for env_idx in range(self.num_envs)
        ]
        self.env = NoAutoResetSyncVectorEnv(env_fns, copy=not self.return_numpy)
        self.task_descriptions = list(
            self.env.call("get_wrapper_attr", "task_description")
        )

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def total_num_group_envs(self):
        return np.iinfo(np.uint8).max // 2  # TODO

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    def _init_metrics(self):
        self.prev_step_reward = np.zeros(self.num_envs)

        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)
        self.intervened_once = np.zeros(self.num_envs, dtype=bool)
        self.intervened_steps = np.zeros(self.num_envs, dtype=int)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self._elapsed_steps[mask] = 0
            self.intervened_once[mask] = False
            self.intervened_steps[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0
            self.intervened_once[:] = False
            self.intervened_steps[:] = 0

    def _record_metrics(
        self,
        step_reward,
        terminations,
        success_current_step,
        intervene_current_step,
        infos,
    ):
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | success_current_step
        self.intervened_once = self.intervened_once | intervene_current_step
        self.intervened_steps += intervene_current_step.astype(int)

        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        episode_info["intervened_once"] = self.intervened_once.copy()
        episode_info["intervened_steps"] = self.intervened_steps.copy()
        episode_info["success_no_intervened"] = self.success_once.copy() & (
            ~self.intervened_once
        )
        infos["episode"] = (
            episode_info if self.return_numpy else to_tensor(episode_info)
        )
        return infos

    def _format_output(self, value):
        """Keep collection outputs in NumPy while preserving the default API."""
        return value if self.return_numpy else to_tensor(value)

    def reset(self, *, reset_state_ids=None, seed=None, options=None, env_idx=None):
        # TODO: handle partial reset
        raw_obs, infos = self.env.reset(seed=seed, options=options)
        self._stamp_observation_info(infos)

        extracted_obs = self._wrap_obs(raw_obs)
        if env_idx is not None:
            self._reset_metrics(env_idx)
        else:
            self._reset_metrics()
        return extracted_obs, infos

    def _stamp_observation_info(self, infos):
        """Attach the monotonic time at which a camera observation was received."""
        infos["observation_timestamp_ns"] = np.full(
            self.num_envs, time.monotonic_ns(), dtype=np.int64
        )

    def _wrap_obs(self, raw_obs):
        """
        raw_obs: Dict of list
        """
        obs = {}

        state = raw_obs["state"]
        full_states = np.concatenate([state[k] for k in sorted(state)], axis=-1)
        obs["states"] = full_states

        frames = raw_obs["frames"]
        if self.main_image_key not in frames:
            raise KeyError(
                f"main_image_key {self.main_image_key!r} not in {list(frames)}"
            )
        main_images = frames[self.main_image_key]
        # The copy=False vector buffer is reused on the next step. Collection
        # keeps observations for the full episode, so take ownership once here.
        obs["main_images"] = main_images.copy() if self.return_numpy else main_images
        extra_view_image_keys = getattr(self, "extra_view_image_keys", None)
        if extra_view_image_keys is None:
            raw_images = OrderedDict(sorted(frames.items()))
            raw_images.pop(self.main_image_key)
        else:
            configured_keys = extra_view_image_keys
            available_keys = set(frames) - {self.main_image_key}
            if len(configured_keys) != len(set(configured_keys)):
                raise ValueError("extra_view_image_keys must not contain duplicates")
            if set(configured_keys) != available_keys:
                missing = sorted(available_keys - set(configured_keys))
                unexpected = sorted(set(configured_keys) - available_keys)
                raise KeyError(
                    "extra_view_image_keys must list every non-main camera exactly "
                    f"once; missing={missing}, unexpected={unexpected}"
                )
            raw_images = OrderedDict((key, frames[key]) for key in configured_keys)

        if raw_images:
            obs["extra_view_images"] = np.stack(list(raw_images.values()), axis=1)

        if not self.return_numpy:
            obs = to_tensor(obs)
        obs["task_descriptions"] = self.task_descriptions
        return obs

    def step(self, actions=None, auto_reset=True):
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, truncations, infos = self.env.step(actions)
        self._stamp_observation_info(infos)
        # max_episode_steps: null → external wrapper owns episode end.
        if self.cfg.max_episode_steps is None:
            timeout_truncations = np.zeros_like(truncations, dtype=bool)
        else:
            timeout_truncations = self.elapsed_steps >= self.cfg.max_episode_steps
        if not self.manual_episode_control_only:
            truncations = timeout_truncations

        obs = self._wrap_obs(raw_obs)
        step_reward = self._calc_step_reward(_reward)
        success_current_step = np.isclose(step_reward, 1.0)
        intervene_flag = np.zeros(self.num_envs, dtype=bool)
        if "intervene_action" in infos:
            for env_id in range(self.num_envs):
                if infos["intervene_action"][env_id] is not None:
                    intervene_flag[env_id] = True

        infos = self._record_metrics(
            step_reward,
            terminations,
            success_current_step,
            intervene_flag,
            infos,
        )
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = self._format_output(
                terminations.copy()
            )
            terminations[:] = False

        intervene_action = np.zeros_like(actions)
        if "intervene_action" in infos:
            for env_id in range(self.num_envs):
                env_intervene_action = infos["intervene_action"][env_id]
                if env_intervene_action is not None:
                    intervene_action[env_id] = env_intervene_action.copy()
        infos["intervene_action"] = self._format_output(intervene_action)
        infos["intervene_flag"] = self._format_output(intervene_flag)
        if "rlt_switch_flags" in infos:
            infos["rlt_switch_flags"] = self._format_output(
                np.asarray(infos["rlt_switch_flags"], dtype=bool).copy()
            )

        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return (
            obs,
            self._format_output(step_reward),
            self._format_output(terminations),
            self._format_output(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []

        raw_chunk_intervene_actions = []
        raw_chunk_intervene_flag = []
        raw_chunk_rlt_switch_flags = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)
            if "intervene_action" in infos:
                raw_chunk_intervene_actions.append(infos["intervene_action"])
                raw_chunk_intervene_flag.append(infos["intervene_flag"])
            if "rlt_switch_flags" in infos:
                raw_chunk_rlt_switch_flags.append(infos["rlt_switch_flags"])

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        if self.return_numpy:
            stack = partial(np.stack, axis=1)
            chunk_rewards = stack(chunk_rewards)
            raw_chunk_terminations = stack(raw_chunk_terminations)
            raw_chunk_truncations = stack(raw_chunk_truncations)
            past_terminations = raw_chunk_terminations.any(axis=1)
            past_truncations = raw_chunk_truncations.any(axis=1)
            past_dones = np.logical_or(past_terminations, past_truncations)
        else:
            stack = partial(torch.stack, dim=1)
            chunk_rewards = stack(chunk_rewards)
            raw_chunk_terminations = stack(raw_chunk_terminations)
            raw_chunk_truncations = stack(raw_chunk_truncations)
            past_terminations = raw_chunk_terminations.any(dim=1)
            past_truncations = raw_chunk_truncations.any(dim=1)
            past_dones = torch.logical_or(past_terminations, past_truncations)

        infos_last = infos_list[-1] if infos_list else {}
        if raw_chunk_intervene_actions:
            infos_last["intervene_action"] = stack(raw_chunk_intervene_actions).reshape(
                self.num_envs, -1
            )
            infos_last["intervene_flag"] = stack(raw_chunk_intervene_flag)
            infos_list[-1] = infos_last
        if raw_chunk_rlt_switch_flags:
            infos_last["rlt_switch_flags"] = stack(raw_chunk_rlt_switch_flags)
            infos_list[-1] = infos_last

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones if self.return_numpy else past_dones.cpu().numpy(),
                obs_list[-1],
                infos_list[-1],
            )

        if self.auto_reset or self.ignore_terminations:
            zeros_like = np.zeros_like if self.return_numpy else torch.zeros_like
            chunk_terminations = zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations

            chunk_truncations = zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
            if self.return_numpy:
                chunk_terminations = raw_chunk_terminations.copy()
                chunk_truncations = raw_chunk_truncations.copy()
            else:
                chunk_terminations = raw_chunk_terminations.clone()
                chunk_truncations = raw_chunk_truncations.clone()
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        obs, infos = self.reset(
            env_idx=env_idx,
            reset_state_ids=(
                self.reset_state_ids[env_idx]
                if self.use_fixed_reset_state_ids
                else None
            ),
        )
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(self, reward: np.ndarray):
        return reward.astype(np.float32)

    def _get_random_reset_state_ids(self, num_reset_states):
        reset_state_ids = self._generator.integers(
            low=0, high=self.total_num_group_envs, size=(num_reset_states,)
        )
        return reset_state_ids

    def _init_reset_state_ids(self):
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.update_reset_state_ids()

    def update_reset_state_ids(self):
        reset_state_ids = torch.randint(
            low=0,
            high=self.total_num_group_envs,
            size=(self.num_group,),
            generator=self._generator,
        )
        self.reset_state_ids = reset_state_ids.repeat_interleave(
            repeats=self.group_size
        )

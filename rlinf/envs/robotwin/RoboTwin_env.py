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

import importlib
import os
from multiprocessing import Array, Event, Semaphore, Value

import cv2
import gymnasium as gym
import numpy as np
import torch
import torch.multiprocessing as mp
import yaml
from omegaconf.omegaconf import OmegaConf
from rlinf.envs.robotwin.RoboTwin_wrapper import RobotwinEnvWrapper
from rlinf.envs.libero.utils import (
    get_benchmark_overridden,
    get_libero_image,
    get_libero_wrist_image,
    list_of_dict_to_dict_of_list,
    put_info_on_image,
    quat2axisangle,
    save_rollout_video,
    tile_images,
    to_tensor,
)
from typing import Optional
"""
Input values:
actions[step,14]
Complete return values:
images[step,3,3,480,640]
states[step,action_dim]
"""


def worker(
    process_id: int,
    task_name: str,
    args: dict,
    seed_id: Value,
    actions: Array,
    results: Array,
    action_input_sem: Semaphore,
    result_output_sem: Semaphore,
    reset_event: Event,
):
    global GPU_ID
    NUM_PROCESSES = args["n_envs"]
    HORIZON = args["horizon"]
    ACTION_DIM = args["action_dim"]
    RESULT_SIZE = args["result_size"]

    def filling_up(return_poses: np.ndarray):
        if len(return_poses) < HORIZON:
            new_poses = np.empty((HORIZON, *return_poses.shape[1:]))
            new_poses[: len(return_poses)] = return_poses
            new_poses[len(return_poses) :] = return_poses[-1]
            return new_poses
        return return_poses

    env_wrapper = RobotwinEnvWrapper(
        task_name,
        trial_id=0,
        trial_seed=0,
        config=args,
        version="2.0",
    )
    env_wrapper.initialize()

    # Initial observation push
    action_input_sem.acquire()
    prev_obs = env_wrapper.get_obs()
    imgs, state = update_obs(prev_obs)
    image_return = imgs + imgs  # duplicate 3 cams to 6
    state_return = state
    return_poses = np.zeros((HORIZON, 6), dtype=np.float64)
    return_poses = filling_up(return_poses[0:1, :])

    image_return = np.array(image_return)
    state_return = np.array(state_return)

    result = np.concatenate(
        [
            image_return.flatten(),
            state_return.flatten(),
            np.array([0]),
            np.array([0]),
            np.array([0]),
            return_poses.flatten(),
        ]
    )
    results[RESULT_SIZE * process_id : (process_id + 1) * RESULT_SIZE] = result
    result_output_sem.release()

    while True:
        action_input_sem.acquire()
        numpy_actions = np.frombuffer(actions.get_obj()).reshape(
            NUM_PROCESSES, HORIZON, ACTION_DIM
        )
        input_actions = numpy_actions[process_id]

        if reset_event.is_set():
            valid_seed = False
            while not valid_seed:
                try:
                    with seed_id.get_lock():
                        now_seed = seed_id.value
                        seed_id.value += 1
                        if seed_id.value >= 30:
                            seed_id.value %= 30
                    # Recreate wrapper with new seed
                    try:
                        env_wrapper.close()
                    except Exception:
                        pass
                    env_wrapper = RobotwinEnvWrapper(
                        task_name,
                        trial_id=now_seed,
                        trial_seed=now_seed,
                        config=args,
                        version="2.0",
                    )
                    env_wrapper.initialize()
                    valid_seed = True
                except Exception as e:
                    print(
                        f"Error in process {process_id} during reset with seed {now_seed}: {e}"
                    )

            prev_obs = env_wrapper.get_obs()
            imgs, state = update_obs(prev_obs)
            image_return = imgs + imgs
            state_return = state
            return_poses = np.zeros((HORIZON, 6), dtype=np.float64)
            return_poses = filling_up(return_poses[0:1, :])

            image_return = np.array(image_return)
            state_return = np.array(state_return)
            result = np.concatenate(
                [
                    image_return.flatten(),
                    state_return.flatten(),
                    np.array([0]),
                    np.array([0]),
                    np.array([1]),
                    return_poses.flatten(),
                ]
            )
            results[RESULT_SIZE * process_id : RESULT_SIZE * (process_id + 1)] = result
            result_output_sem.release()
            continue

        # Rollout HORIZON steps
        last_reward = 0.0
        last_terminated = False
        last_truncated = False
        obs_list = []
        cur_obs = prev_obs
        for t in range(HORIZON):
            action = input_actions[t].reshape(1, ACTION_DIM)
            cur_obs, r, terminated, truncated, info = env_wrapper.step(action)
            last_reward = float(r)
            last_terminated = bool(terminated)
            last_truncated = bool(truncated)
            obs_list.append(cur_obs)
            if last_terminated:
                break

        # Build image/state return using last two obs
        if len(obs_list) >= 1:
            last_obs = obs_list[-1]
        else:
            last_obs = prev_obs

        if len(obs_list) >= 2:
            penult_obs = obs_list[-2]
        else:
            penult_obs = prev_obs

        imgs_a, _ = update_obs(penult_obs)
        imgs_b, state_b = update_obs(last_obs)
        image_return = imgs_a + imgs_b
        state_return = state_b

        return_poses = np.zeros((HORIZON, 6), dtype=np.float64)
        return_poses = filling_up(return_poses[0:1, :])

        # Auto reset if terminated: immediately start new episode and send its first obs next time
        if last_terminated:
            try:
                with seed_id.get_lock():
                    now_seed = seed_id.value
                    seed_id.value += 1
                    if seed_id.value >= 30:
                        seed_id.value %= 30
            except Exception:
                now_seed = 0
            try:
                env_wrapper.close()
            except Exception:
                pass
            env_wrapper = RobotwinEnvWrapper(
                task_name,
                trial_id=now_seed,
                trial_seed=now_seed,
                config=args,
                version="2.0",
            )
            env_wrapper.initialize()
            prev_obs = env_wrapper.get_obs()
        else:
            prev_obs = last_obs

        image_return = np.array(image_return)
        state_return = np.array(state_return)

        result = np.concatenate(
            [
                image_return.flatten(),
                state_return.flatten(),
                np.array([last_reward]),
                np.array([1 if last_terminated else 0]),
                np.array([1 if last_truncated else 0]),
                return_poses.flatten(),
            ]
        )
        results[RESULT_SIZE * process_id : RESULT_SIZE * (process_id + 1)] = result
        result_output_sem.release()


def update_obs(observation):
    imgs = []
    imgs.append(observation["observation"]["head_camera"]["rgb"][:, :, ::-1])
    imgs.append(observation["observation"]["right_camera"]["rgb"][:, :, ::-1])
    imgs.append(observation["observation"]["left_camera"]["rgb"][:, :, ::-1])
    state = observation["joint_action"]["vector"]
    return imgs, state


class RoboTwin(gym.Env):
    def __init__(self, cfg, seed_offset, total_num_processes, record_metrics=True):
        # Get parameters from configuration
        self.cfg = cfg
        self.seed_offset = seed_offset
        self.total_num_processes = total_num_processes
        self.record_metrics = record_metrics
        self._is_start = True
        self.info_logging_keys = ["is_src_obj_grasped", "consecutive_grasp", "success"]
        try:
            self.env_args = OmegaConf.to_container(cfg.init_params, resolve=True)
        except Exception:
            # 兼容直接传入 dict 的用法
            self.env_args = dict(cfg.init_params)
        if self.record_metrics:
            self._init_metrics()

        self.task_name = "place_shoe"
        self.n_envs = self.num_envs
        self.horizon = getattr(cfg, "horizon", 1)
        self.action_dim = 14
        self.image_size = self.cfg.image_size

        self.NUM_IMAGES = 6
        self.IMAGE_SHAPE = (240, 320, 3)
        self.STATE_SHAPE = (1, 14)
        self.TARGET_SHAPE = (self.horizon, 6)

        self.IMAGE_SIZE = np.prod(self.IMAGE_SHAPE)
        self.STATE_SIZE = np.prod(self.STATE_SHAPE)
        self.TARGET_SIZE = np.prod(self.TARGET_SHAPE)

        args = {}
        args["n_envs"] = self.n_envs
        args["horizon"] = self.horizon
        args["action_dim"] = self.action_dim
        args["result_size"] = int(
            self.NUM_IMAGES * self.IMAGE_SIZE + self.STATE_SIZE + 3 + self.TARGET_SIZE
        )
        args["input_size"] = int(self.horizon * self.action_dim)
        # RobotWin 2.0 configuration for wrapper
        args["twin2_task_config"] = getattr(self.cfg, "twin2_task_config", "demo_clean")
        args["twin2_ckpt_setting"] = getattr(self.cfg, "twin2_ckpt_setting", "demo_clean")
        args["twin2_instruction_type"] = getattr(self.cfg, "twin2_instruction_type", "unseen")

        self.args = args
        self.auto_reset = False
        self.use_rel_reward = False

        self.process = []

        self.seed = 0
        self.input_sem = None
        self.output_sem = None
        self.reset_event = None
        self.share_seed = None
        self.share_actions = None
        self.share_results = None

        self.init_process()

    @property
    def num_envs(self):
        return self.env_args["num_envs"]

    @property
    def device(self):
        return "cpu"  # RoboTwin uses CPU

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_reset_state_ids(self):
        self._generator = torch.Generator()
        self._generator.manual_seed(self.seed)
        self.all_reset_state_ids = torch.randperm(
            self.total_num_group_envs, generator=self._generator
        ).to(self.device)
        self.update_reset_state_ids()


    def _extract_obs_image(self, raw_obs):
        obs_image = raw_obs["sensor_data"]["3rd_view_camera"]["rgb"].to(torch.uint8)
        obs_image = obs_image.permute(0, 3, 1, 2)  # [B, C, H, W]
        extracted_obs = {"images": obs_image, "task_descriptions": self.instruction}
        return extracted_obs

    def _calc_step_reward(self, info):
        reward = torch.zeros(self.num_envs, dtype=torch.float32).to(
            self.env.device
        )  # [B, ]
        reward += info["is_src_obj_grasped"] * 0.1
        reward += info["consecutive_grasp"] * 0.1
        reward += (info["success"] & info["is_src_obj_grasped"]) * 1.0
        # diff
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def _init_metrics(self):
        self.success_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.fail_once = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.bool
        )
        self.returns = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.float32
        )

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = torch.zeros(self.num_envs, dtype=bool, device=self.device)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            if self.record_metrics:
                self.success_once[mask] = False
                self.fail_once[mask] = False
                self.returns[mask] = 0
        else:
            self.prev_step_reward[:] = 0
            if self.record_metrics:
                self.success_once[:] = False
                self.fail_once[:] = False
                self.returns[:] = 0.0

    def _record_metrics(self, step_reward, infos):
        episode_info = {}
        self.returns += step_reward
        if "success" in infos:
            self.success_once = self.success_once | infos["success"]
            episode_info["success_once"] = self.success_once.clone()
        if "fail" in infos:
            self.fail_once = self.fail_once | infos["fail"]
            episode_info["fail_once"] = self.fail_once.clone()
        episode_info["return"] = self.returns.clone()
        episode_info["episode_len"] = self.elapsed_steps.clone()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = episode_info
        return infos

    # After successful initialization, need to return initial scene
    def init_process(self):
        self.context = mp.get_context("spawn")
        self.input_sem = self.context.Semaphore(0)
        self.output_sem = self.context.Semaphore(0)
        self.reset_event = self.context.Event()

        self.share_seed = self.context.Value("i", self.seed)
        self.share_actions = self.context.Array(
            "d", self.args["n_envs"] * self.args["input_size"]
        )
        self.share_results = self.context.Array(
            "d", self.args["n_envs"] * self.args["result_size"]
        )
        for i in range(self.n_envs):
            os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            os.environ["TORCH_CUDA_ARCH_LIST"] = "8.0"
            p = self.context.Process(
                target=worker,
                args=(
                    i,
                    self.task_name,
                    self.args,
                    self.share_seed,
                    self.share_actions,
                    self.share_results,
                    self.input_sem,
                    self.output_sem,
                    self.reset_event,
                ),
                daemon=True,
            )
            self.process.append(p)
            p.start()

        for _ in range(self.n_envs):
            self.input_sem.release()

        for _ in range(self.n_envs):
            self.output_sem.acquire()

        results = np.frombuffer(self.share_results.get_obj()).reshape(
            self.n_envs, self.args["result_size"]
        )
        results = self.de_initialize_result(results)
        obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
            self.transform(results)
        )
        return obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv

    def step(self, actions=None):
        if actions is None:
            self.share_actions[:] = np.zeros(
                (self.n_envs, self.horizon, self.action_dim)
            ).flatten()
        else:
            self.share_actions[:] = actions.flatten()

        # Release semaphore, indicating actions have been updated
        for i in range(self.n_envs):
            self.input_sem.release()

        # Wait for all envs to complete output
        for i in range(self.n_envs):
            self.output_sem.acquire()

        results = np.frombuffer(self.share_results.get_obj()).reshape(
            self.n_envs, self.args["result_size"]
        )
        results = self.de_initialize_result(results)
        obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv = (
            self.transform(results)
        )
        if actions is None:
            reward_venv = None
        return obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv

    def reset(self):
        self.reset_event.set()
        # Release semaphore, indicating actions have been updated
        for i in range(self.n_envs):
            self.input_sem.release()

        # Wait for all envs to complete output
        for i in range(self.n_envs):
            self.output_sem.acquire()

        self.reset_event.clear()
        results = np.frombuffer(self.share_results.get_obj()).reshape(
        self.n_envs, self.args["result_size"]
    )
        results = self.de_initialize_result(results)
        obs_venv, _, _, _, info_venv = self.transform(results)
        return obs_venv, info_venv

    def transform(self, results):
        def jpeg_mapping(img):
            if img is None:
                return None
            img = cv2.imencode(".jpg", img)[1].tobytes()
            img = cv2.imdecode(np.frombuffer(img, np.uint8), cv2.IMREAD_COLOR)
            return img

        obs_venv = {"images": [], "state": [], "task_descriptions": []}
        reward_venv = torch.zeros(
            self.num_envs, dtype=torch.float32, device=self.device
        )  # [B, ]
        terminated_venv = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        truncated_venv = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        info_venv = {"return_poses": []}
        for i in range(self.n_envs):
            result = results[i]
            imgs = result["imgs"]
            imgs = [cv2.resize(img, self.image_size) for img in imgs]
            imgs = [jpeg_mapping(img) for img in imgs]
            imgs = np.array(imgs)

            # TODO output is 6 3 224 224, we just use first images
            obs_venv["images"].append(torch.from_numpy(imgs).to(self.device))
            obs_venv["state"].append(torch.from_numpy(result["state"]).to(self.device))
            obs_venv["task_descriptions"].append("")
            reward_venv[i] = torch.from_numpy(result["reward"]).to(self.device)
            terminated_venv[i] = torch.from_numpy(result["terminated"]).to(self.device)
            truncated_venv[i] = torch.from_numpy(result["truncated"]).to(self.device)
            info_venv["return_poses"].append(
                torch.from_numpy(result["return_poses"]).to(self.device)
            )
        obs_venv["images"] = torch.stack(obs_venv["images"]).permute(0, 1, 4, 2, 3)
        obs_venv["state"] = torch.stack(obs_venv["state"])
        return obs_venv, reward_venv, terminated_venv, truncated_venv, info_venv

    def chunk_step(self, chunk_actions):
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions
            )

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(
                past_dones, extracted_obs, infos
            )

        chunk_terminations = torch.zeros_like(raw_chunk_terminations)
        chunk_terminations[:, -1] = past_terminations

        chunk_truncations = torch.zeros_like(raw_chunk_truncations)
        chunk_truncations[:, -1] = past_truncations
        return (
            extracted_obs,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos,
        )

    def de_initialize_result(self, results):
        conds = []
        for i in range(self.n_envs):
            cond = {}
            start = 0
            result = results[i]
            cond["imgs"] = result[
                start : start + self.NUM_IMAGES * self.IMAGE_SIZE
            ].reshape(self.NUM_IMAGES, *self.IMAGE_SHAPE)
            start += self.NUM_IMAGES * self.IMAGE_SIZE
            cond["state"] = result[start : start + self.STATE_SIZE].reshape(
                self.STATE_SHAPE
            )
            start += self.STATE_SIZE
            cond["reward"] = result[start : start + 1]
            start += 1
            cond["terminated"] = result[start : start + 1]
            start += 1
            cond["truncated"] = result[start : start + 1]
            start += 1
            cond["return_poses"] = result[start : start + self.TARGET_SIZE].reshape(
                self.TARGET_SHAPE
            )
            start += self.TARGET_SIZE
            conds.append(cond)
        return conds

    def clear(self):
        # End all current processes
        for i in range(self.n_envs):
            if self.process[i].is_alive():
                self.process[i].terminate()
                # Wait for process to completely end
                self.process[i].join()
                # Release occupied memory
                self.process[i].close()
        self.seed = self.share_seed.value
        if self.seed > 3000:
            self.seed = 0
        self.process = []

    def flush_video(self, video_sub_dir: Optional[str] = None):
        output_dir = os.path.join(self.video_cfg.video_base_dir, f"seed_{self.seed}")
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")
        save_rollout_video(
            self.render_images,
            output_dir=output_dir,
            video_name=f"{self.video_cnt}",
        )
        self.video_cnt += 1
        self.render_images = []
    
    def add_new_frames(self, raw_obs, plot_infos):
        images = []
        for env_id, raw_single_obs in enumerate(raw_obs):
            info_item = {
                k: v if np.size(v) == 1 else v[env_id] for k, v in plot_infos.items()
            }
            img = raw_single_obs["agentview_image"][::-1, ::-1]
            img = put_info_on_image(img, info_item)
            images.append(img)
        full_image = tile_images(images, nrows=int(np.sqrt(self.num_envs)))
        self.render_images.append(full_image)

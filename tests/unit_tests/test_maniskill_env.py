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

import os
import sys
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch
from mani_skill.utils import common, sapien_utils
from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import rlinf.envs.maniskill  # noqa: F401
from rlinf.envs.maniskill.maniskill_env import ManiskillEnv
from rlinf.envs.maniskill.tasks.pick_cube_dualview import build_orbit_view_poses
from rlinf.models.embodiment.reward.vlm_reward_utils.input_builder import (
    SimpleDualViewTernaryInputBuilder,
)


class _FakeScene:
    def __init__(self, render_camera_rgb: torch.Tensor | None = None):
        self.render_camera_rgb = render_camera_rgb
        self.render_updates: list[dict[str, bool]] = []

    def update_render(self, **kwargs):
        self.render_updates.append(kwargs)

    def get_human_render_camera_images(self, camera_name: str):
        return {camera_name: self.render_camera_rgb}


class _FakeUnwrapped:
    def __init__(self, scene: _FakeScene):
        self.obs_mode = "rgb"
        self.device = torch.device("cpu")
        self.scene = scene

    def get_language_instruction(self):
        return ["pick cube"]


def _make_env(scene: _FakeScene) -> ManiskillEnv:
    env = object.__new__(ManiskillEnv)
    env.cfg = OmegaConf.create({"wrap_obs_mode": "simple_prompt"})
    env.use_full_state = True
    env.env = SimpleNamespace(
        unwrapped=_FakeUnwrapped(scene),
        spec=SimpleNamespace(id="PickCube-v1"),
    )
    env._warned_missing_extra_view_images = False
    env._get_full_state_obs = lambda: torch.ones((1, 4), dtype=torch.float32)
    return env


def _make_rgb_frame(value: int) -> torch.Tensor:
    frame = torch.full((1, 2, 2, 3), fill_value=value, dtype=torch.uint8)
    return frame


def _make_pickcube_dualview_env():
    pytest.importorskip("mani_skill.envs")
    return gym.make(
        "PickCubeDualView-v1",
        obs_mode="rgb",
        render_mode="rgb_array",
        num_envs=1,
        sim_backend="physx_cpu",
        sensor_configs={"width": 64, "height": 64, "shader_pack": "minimal"},
        human_render_camera_configs={
            "width": 64,
            "height": 64,
            "shader_pack": "minimal",
        },
    )


def _get_allowed_secondary_pose_raws(base_env) -> list[np.ndarray]:
    center_xy = common.to_numpy(base_env.agent.robot.pose.p)[0, :2].astype(np.float32)
    reference_pose = sapien_utils.look_at(
        eye=base_env.human_cam_eye_pos,
        target=base_env.human_cam_target_pos,
    ).sp
    poses = build_orbit_view_poses(
        center_xy=center_xy,
        reference_eye=np.asarray(base_env.human_cam_eye_pos, dtype=np.float32),
        target=np.asarray(base_env.human_cam_target_pos, dtype=np.float32),
        azimuth_views=base_env.orbit_azimuth_views,
        reference_pose=reference_pose,
    )
    return [np.concatenate([pose.p, pose.q]).astype(np.float32) for pose in poses[1:]]


def _get_orbit_camera_pose_raw(base_env) -> np.ndarray:
    return common.to_numpy(base_env.orbit_camera_mount.pose.raw_pose)[0].astype(
        np.float32
    )


def test_wrap_obs_uses_render_camera_when_no_extra_sensor_view():
    env = _make_env(scene=_FakeScene(render_camera_rgb=_make_rgb_frame(9)))
    raw_obs = {
        "sensor_data": {
            "base_camera": {"rgb": _make_rgb_frame(1)},
        },
        "sensor_param": {},
    }

    obs = env._wrap_obs(raw_obs)

    assert tuple(obs["extra_view_images"].shape) == (1, 2, 2, 3)
    assert int(obs["extra_view_images"][0, 0, 0, 0].item()) == 9
    assert env.env.unwrapped.scene.render_updates == [
        {"update_sensors": False, "update_human_render_cameras": True}
    ]


def test_wrap_obs_selects_first_sorted_extra_sensor_view():
    env = _make_env(scene=_FakeScene(render_camera_rgb=_make_rgb_frame(99)))
    raw_obs = {
        "sensor_data": {
            "cam_b": {"rgb": _make_rgb_frame(7)},
            "base_camera": {"rgb": _make_rgb_frame(1)},
            "cam_a": {"rgb": _make_rgb_frame(3)},
        },
        "sensor_param": {},
    }

    obs = env._wrap_obs(raw_obs)

    assert tuple(obs["extra_view_images"].shape) == (1, 2, 2, 3)
    assert int(obs["extra_view_images"][0, 0, 0, 0].item()) == 3
    assert env.env.unwrapped.scene.render_updates == []


def test_pickcube_dualview_env_exposes_orbit_camera_and_allowed_pose():
    env = _make_pickcube_dualview_env()
    try:
        obs, _ = env.reset(seed=0)

        assert set(obs["sensor_data"]) >= {"base_camera", "orbit_camera"}
        assert (
            obs["sensor_data"]["orbit_camera"]["rgb"].shape
            == obs["sensor_data"]["base_camera"]["rgb"].shape
        )

        base_extrinsic = np.asarray(obs["sensor_param"]["base_camera"]["extrinsic_cv"])
        orbit_extrinsic = np.asarray(
            obs["sensor_param"]["orbit_camera"]["extrinsic_cv"]
        )
        assert not np.allclose(base_extrinsic, orbit_extrinsic)

        base_env = env.unwrapped
        orbit_pose_raw = _get_orbit_camera_pose_raw(base_env)
        allowed_secondary_poses = _get_allowed_secondary_pose_raws(base_env)
        assert any(
            np.allclose(orbit_pose_raw, allowed_pose, atol=1e-5)
            for allowed_pose in allowed_secondary_poses
        )
    finally:
        env.close()


def test_pickcube_dualview_orbit_camera_changes_across_resets_and_stays_fixed_in_episode():
    env = _make_pickcube_dualview_env()
    try:
        unique_pose_keys: set[tuple[float, ...]] = set()
        fixed_episode_extrinsic = None

        for seed in range(8):
            obs, _ = env.reset(seed=seed)
            base_env = env.unwrapped
            pose_key = tuple(np.round(_get_orbit_camera_pose_raw(base_env), decimals=6))
            unique_pose_keys.add(pose_key)

            if seed == 0:
                fixed_episode_extrinsic = np.asarray(
                    obs["sensor_param"]["orbit_camera"]["extrinsic_cv"]
                )
                for _ in range(2):
                    obs, _, _, _, _ = env.step(env.action_space.sample())
                    step_extrinsic = np.asarray(
                        obs["sensor_param"]["orbit_camera"]["extrinsic_cv"]
                    )
                    np.testing.assert_allclose(
                        step_extrinsic,
                        fixed_episode_extrinsic,
                        atol=1e-6,
                    )

        assert len(unique_pose_keys) > 1
    finally:
        env.close()


def test_pickcube_dualview_real_env_produces_sensor_extra_view_images():
    pytest.importorskip("mani_skill.envs")

    cfg = OmegaConf.create(
        {
            "seed": 0,
            "auto_reset": True,
            "use_rel_reward": False,
            "ignore_terminations": True,
            "use_full_state": True,
            "group_size": 1,
            "use_fixed_reset_state_ids": False,
            "wrap_obs_mode": "simple_prompt",
            "reward_mode": "raw",
            "video_cfg": {"save_video": False, "info_on_video": False},
            "init_params": {
                "id": "PickCubeDualView-v1",
                "obs_mode": "rgb",
                "control_mode": "pd_joint_delta_pos",
                "sim_backend": "physx_cpu",
                "render_mode": "rgb_array",
                "sensor_configs": {
                    "shader_pack": "minimal",
                    "width": 64,
                    "height": 64,
                },
                "human_render_camera_configs": {
                    "shader_pack": "minimal",
                    "width": 64,
                    "height": 64,
                },
            },
        }
    )
    env = ManiskillEnv(
        cfg=cfg,
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info={},
        record_metrics=True,
    )
    try:
        obs, _ = env.reset()
        assert obs["extra_view_images"] is not None
        assert tuple(obs["main_images"].shape) == tuple(obs["extra_view_images"].shape)
        assert obs["task_descriptions"] == [
            "Pick up the red cube and place it on the green spot on the table."
        ]

        next_obs, *_ = env.step(env.env.action_space.sample())
        assert next_obs["extra_view_images"] is not None
        assert tuple(next_obs["main_images"].shape) == tuple(
            next_obs["extra_view_images"].shape
        )

        builder = SimpleDualViewTernaryInputBuilder(
            _processor=None,
            history_buffer_names=["history_window"],
        )
        history_input = {
            "history_window": {
                "main_images": [
                    [obs["main_images"][0].cpu(), next_obs["main_images"][0].cpu()]
                ],
                "extra_view_images": [
                    [
                        obs["extra_view_images"][0].cpu(),
                        next_obs["extra_view_images"][0].cpu(),
                    ]
                ],
            }
        }
        assert builder.get_valid_input_ids(next_obs, history_input) == [0]
    finally:
        env.env.close()

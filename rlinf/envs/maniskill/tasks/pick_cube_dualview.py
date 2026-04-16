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

from __future__ import annotations

from typing import Optional

import numpy as np
import sapien
import torch
from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import common, sapien_utils
from mani_skill.utils.registration import register_env
from mani_skill.utils.structs.pose import Pose

ORBIT_AZIMUTH_VIEWS = 16


def compute_orbit_eyes(
    center_xy: np.ndarray,
    reference_eye: np.ndarray,
    azimuth_views: int = ORBIT_AZIMUTH_VIEWS,
) -> np.ndarray:
    center_xy = np.asarray(center_xy, dtype=np.float32)
    reference_eye = np.asarray(reference_eye, dtype=np.float32)

    if center_xy.shape != (2,):
        raise ValueError(f"center_xy must have shape (2,), got {center_xy.shape}")
    if reference_eye.shape != (3,):
        raise ValueError(
            f"reference_eye must have shape (3,), got {reference_eye.shape}"
        )
    if azimuth_views <= 0:
        raise ValueError("azimuth_views must be positive")

    radius = np.linalg.norm(reference_eye[:2] - center_xy)
    if np.isclose(radius, 0):
        raise ValueError("reference_eye lies on the robot center in the xy plane")

    base_azimuth = np.arctan2(
        reference_eye[1] - center_xy[1],
        reference_eye[0] - center_xy[0],
    )
    azimuths = base_azimuth + np.arange(azimuth_views, dtype=np.float32) * (
        2 * np.pi / azimuth_views
    )

    eyes = np.zeros((azimuth_views, 3), dtype=np.float32)
    eyes[:, 0] = center_xy[0] + radius * np.cos(azimuths)
    eyes[:, 1] = center_xy[1] + radius * np.sin(azimuths)
    eyes[:, 2] = reference_eye[2]
    return eyes


def build_orbit_view_poses(
    center_xy: np.ndarray,
    reference_eye: np.ndarray,
    target: np.ndarray,
    azimuth_views: int = ORBIT_AZIMUTH_VIEWS,
    reference_pose: Optional[sapien.Pose] = None,
) -> list[sapien.Pose]:
    target = np.asarray(target, dtype=np.float32)
    if target.shape != (3,):
        raise ValueError(f"target must have shape (3,), got {target.shape}")

    eyes = compute_orbit_eyes(center_xy, reference_eye, azimuth_views)
    poses: list[sapien.Pose] = []
    for view_idx, eye in enumerate(eyes):
        if view_idx == 0 and reference_pose is not None:
            poses.append(reference_pose)
        else:
            poses.append(sapien_utils.look_at(eye=eye, target=target).sp)
    return poses


@register_env("PickCubeDualView-v1", max_episode_steps=50)
class PickCubeDualViewEnv(PickCubeEnv):
    def __init__(self, *args, **kwargs):
        self.orbit_camera_mount = None
        self.orbit_camera_name = "orbit_camera"
        self.orbit_azimuth_views = ORBIT_AZIMUTH_VIEWS
        super().__init__(*args, **kwargs)

    @property
    def _default_sensor_configs(self):
        sensor_configs = list(super()._default_sensor_configs)
        base_camera_config = sensor_configs[0]
        sensor_configs.append(
            CameraConfig(
                uid=self.orbit_camera_name,
                pose=sapien.Pose(),
                width=base_camera_config.width,
                height=base_camera_config.height,
                fov=base_camera_config.fov,
                near=base_camera_config.near,
                far=base_camera_config.far,
                intrinsic=base_camera_config.intrinsic,
                mount=self.orbit_camera_mount,
                shader_pack=base_camera_config.shader_pack,
            )
        )
        return sensor_configs

    def _load_scene(self, options: dict):
        super()._load_scene(options)
        builder = self.scene.create_actor_builder()
        builder.initial_pose = sapien.Pose()
        self.orbit_camera_mount = builder.build_kinematic("orbit_camera_mount")

    def _get_orbit_candidate_poses(self) -> Pose:
        center_xy = common.to_numpy(self.agent.robot.pose.p)[0, :2].astype(np.float32)
        reference_pose = sapien_utils.look_at(
            eye=self.human_cam_eye_pos,
            target=self.human_cam_target_pos,
        ).sp
        poses = build_orbit_view_poses(
            center_xy=center_xy,
            reference_eye=np.asarray(self.human_cam_eye_pos, dtype=np.float32),
            target=np.asarray(self.human_cam_target_pos, dtype=np.float32),
            azimuth_views=self.orbit_azimuth_views,
            reference_pose=reference_pose,
        )
        candidate_raw_poses = np.stack(
            [np.concatenate([pose.p, pose.q]).astype(np.float32) for pose in poses[1:]]
        )
        return Pose.create(candidate_raw_poses, device=self.device)

    def _sample_orbit_camera_pose(self, num_envs: int) -> Pose:
        candidate_poses = self._get_orbit_candidate_poses()
        sampled_ids = torch.randint(
            low=0,
            high=len(candidate_poses),
            size=(num_envs,),
            device=self.device,
        )
        return Pose.create(candidate_poses.raw_pose[sampled_ids], device=self.device)

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        super()._initialize_episode(env_idx, options)
        self.orbit_camera_mount.set_pose(self._sample_orbit_camera_pose(len(env_idx)))


PickCubeDualViewEnv.__doc__ = (
    (PickCubeEnv.__doc__ or "")
    + "\n\nThis variant exposes a reset-sampled sensor camera `orbit_camera` in "
    + "addition to `base_camera` for dual-view RGB observation pipelines."
)

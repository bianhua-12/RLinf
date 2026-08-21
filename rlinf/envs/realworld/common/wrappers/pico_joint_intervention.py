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

"""PICO takeover for the dual-Franka 16-D joint action environment."""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.common.pico.pico_expert import PicoExpert
from rlinf.envs.realworld.common.wrappers.pico_intervention import (
    _split_dual_pico_config,
)


class _Fr3RobotiqKinematics:
    """FR3 kinematics with the Robotiq TCP configured in Franka Desk."""

    def __init__(self) -> None:
        import pinocchio as pin
        import xacro
        from ament_index_python.packages import get_package_share_directory

        urdf_path = (
            Path(get_package_share_directory("franka_description"))
            / "robots/fr3/fr3.urdf.xacro"
        )
        robot = ET.fromstring(
            xacro.process_file(
                str(urdf_path),
                mappings={
                    "hand": "false",
                    "ros2_control": "false",
                    "no_prefix": "true",
                },
            ).toxml()
        )
        ET.SubElement(robot, "link", {"name": "robotiq_tcp"})
        joint = ET.SubElement(
            robot, "joint", {"name": "robotiq_tcp_joint", "type": "fixed"}
        )
        ET.SubElement(joint, "parent", {"link": "link8"})
        ET.SubElement(joint, "child", {"link": "robotiq_tcp"})
        ET.SubElement(joint, "origin", {"xyz": "0 0 0.174", "rpy": "0 0 0"})

        self.pin = pin
        self.model = pin.buildModelFromXML(ET.tostring(robot, encoding="unicode"))
        self.data = self.model.createData()
        self.tcp_frame = self.model.getFrameId("robotiq_tcp")

    def compute(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.pin.forwardKinematics(self.model, self.data, joints)
        self.pin.updateFramePlacements(self.model, self.data)
        placement = self.data.oMf[self.tcp_frame]
        pose = np.concatenate(
            [placement.translation, R.from_matrix(placement.rotation).as_quat()]
        )
        jacobian = self.pin.computeFrameJacobian(
            self.model,
            self.data,
            joints,
            self.tcp_frame,
            self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return pose, jacobian


class DualFrankaJointPicoIntervention(gym.ActionWrapper):
    """Latch PICO takeover while keeping the 16-D joint controller active."""

    def __init__(
        self,
        env: gym.Env,
        *,
        experts: dict[str, Any] | None = None,
        ready_timeout_s: float = 10.0,
        ik_damping: float = 0.05,
        **pico_config: Any,
    ) -> None:
        super().__init__(env)
        if self.action_space.shape != (16,):
            raise ValueError(
                "DualFrankaJointPicoIntervention requires a 16-D dual-arm "
                f"joint action space, got {self.action_space.shape}"
            )
        if ready_timeout_s <= 0.0:
            raise ValueError("ready_timeout_s must be positive")
        if ik_damping <= 0.0:
            raise ValueError("ik_damping must be positive")
        if experts is None:
            left_cfg, right_cfg = _split_dual_pico_config(
                {**pico_config, "hand": "dual"}
            )
            left_expert = PicoExpert(**left_cfg)
            try:
                right_expert = PicoExpert(**right_cfg)
            except Exception:
                left_expert.stop()
                raise
            self.experts = {"left": left_expert, "right": right_expert}
        else:
            if set(experts) != {"left", "right"}:
                raise ValueError("experts must contain exactly 'left' and 'right'")
            self.experts = dict(experts)
        self.ready_timeout_s = float(ready_timeout_s)
        self.ik_damping = float(ik_damping)
        self.takeover_latched = False
        self._active = {"left": False, "right": False}
        self._ready = {"left": False, "right": False}
        self._gripper_action = np.zeros(2, dtype=np.float32)
        self._kinematics = None

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.takeover_latched = False
        self._active = {"left": False, "right": False}
        self._ready = {"left": False, "right": False}
        self._gripper_action.fill(0.0)
        self.prepare()
        return observation, info

    def prepare(self) -> None:
        """Wait for calibrated, released controllers without commanding the robot."""
        deadline = time.monotonic() + self.ready_timeout_s
        status: dict[str, tuple[bool, bool, bool]] = {}
        while True:
            tcp_poses = self._tcp_poses()
            for arm_index, side in enumerate(("left", "right")):
                expert = self.experts[side]
                _, _, side_info = expert.get_action(
                    tcp_poses[arm_index],
                    None,
                    gripper_enabled=True,
                    direct=True,
                )
                active = bool(side_info.get("pico_active", False))
                data_ready = bool(side_info.get("pico_ready", False))
                calibration_enabled = bool(
                    getattr(expert, "calibration_enabled", False)
                )
                calibrated = bool(
                    side_info.get("pico_calibrated", not calibration_enabled)
                )
                valid = not bool(side_info.get("pico_invalid_pose", False))
                ready = data_ready and calibrated and valid
                self._active[side] = active
                self._ready[side] = ready
                status[side] = (ready, calibrated, active)
            if all(self._ready.values()) and not any(self._active.values()):
                return
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "PICO startup timed out waiting for fresh calibrated data "
                    f"with both grip triggers released: {status}"
                )
            time.sleep(0.02)

    def _tcp_poses(self) -> np.ndarray:
        joints = np.asarray(
            self.get_wrapper_attr("get_joint_positions")(), dtype=np.float64
        )
        if joints.shape != (2, 7) or not np.isfinite(joints).all():
            raise ValueError(
                "Dual-Franka joint positions must be finite shape (2,7), "
                f"got {joints.shape}"
            )
        return self._model_kinematics(joints)[0]

    def _model_kinematics(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._kinematics is None:
            self._kinematics = _Fr3RobotiqKinematics()
        states = [self._kinematics.compute(q) for q in joints]
        return np.stack([state[0] for state in states]), np.stack(
            [state[1] for state in states]
        )

    def _joint_action(self, current_joints: np.ndarray) -> np.ndarray:
        action = np.zeros(16, dtype=np.float32)
        action[:7] = current_joints[0]
        action[7] = self._gripper_action[0]
        action[8:15] = current_joints[1]
        action[15] = self._gripper_action[1]
        return action

    def get_hold_action(self) -> np.ndarray:
        """Return the current 16-D joint hold target for RTC conditioning."""
        current = np.asarray(
            self.get_wrapper_attr("get_joint_positions")(), dtype=np.float32
        )
        return self._joint_action(current)

    def release_to_policy(self) -> None:
        """Release a latched hold after the controllers are released."""
        if any(self._active.values()):
            raise RuntimeError(
                "Cannot release PICO takeover while a controller is active"
            )
        if not all(self._ready.values()):
            raise ValueError(
                "Cannot release PICO takeover while controller data is stale"
            )
        self.takeover_latched = False

    def action(self, action: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        current_joints = np.asarray(
            self.get_wrapper_attr("get_joint_positions")(), dtype=np.float32
        )
        if current_joints.shape != (2, 7) or not np.isfinite(current_joints).all():
            raise ValueError(
                "Dual-Franka joint positions must be finite shape (2,7), "
                f"got {current_joints.shape}"
            )
        tcp_poses, jacobians = self._model_kinematics(current_joints)

        hold_action = self._joint_action(current_joints)
        pico_info: dict[str, Any] = {}
        active_any = False
        for arm_index, side in enumerate(("left", "right")):
            expert = self.experts[side]
            expert_action, replaced, side_info = expert.get_action(
                tcp_poses[arm_index],
                None,
                gripper_enabled=True,
                direct=True,
            )
            if not side_info.get("pico_ready", False):
                raise RuntimeError(f"PICO {side} controller data timed out")
            side_active = bool(side_info.get("pico_active", False))
            self._active[side] = side_active
            calibration_enabled = bool(getattr(expert, "calibration_enabled", False))
            calibrated = bool(side_info.get("pico_calibrated", not calibration_enabled))
            self._ready[side] = (
                bool(side_info.get("pico_ready", False))
                and calibrated
                and not bool(side_info.get("pico_invalid_pose", False))
            )
            active_any = active_any or side_active
            for key, value in side_info.items():
                pico_info[f"{side}_{key}"] = value

            if replaced:
                expert_action = np.asarray(expert_action, dtype=np.float32)
                jacobian = jacobians[arm_index]
                regularized = jacobian @ jacobian.T + self.ik_damping**2 * np.eye(6)
                joint_delta = jacobian.T @ np.linalg.solve(
                    regularized, expert_action[:6]
                )
                target_joints = current_joints[arm_index] + joint_delta
                action_offset = arm_index * 8
                hold_action[action_offset : action_offset + 7] = target_joints
                if expert_action.size >= 7:
                    hold_action[action_offset + 7] = expert_action[6]

        if active_any:
            self.takeover_latched = True

        pico_info["pico_active"] = active_any
        pico_info["pico_takeover"] = self.takeover_latched
        pico_info["pico_ready"] = all(self._ready.values())
        if self.takeover_latched:
            self._gripper_action[:] = hold_action[[7, 15]]
            return (
                np.clip(hold_action, self.action_space.low, self.action_space.high),
                pico_info,
            )
        policy_action = np.asarray(action, dtype=np.float32).copy()
        if policy_action.shape != (16,) or not np.isfinite(policy_action).all():
            raise ValueError("Policy action must be finite shape (16,)")
        self._gripper_action[:] = policy_action[[7, 15]]
        return policy_action, pico_info

    def step(self, action):
        new_action, pico_info = self.action(action)
        observation, reward, terminated, truncated, info = self.env.step(new_action)
        if pico_info["pico_takeover"]:
            info["intervene_action"] = new_action
            info["intervene_flag"] = np.ones(1, dtype=bool)
        info.update(pico_info)
        return observation, reward, terminated, truncated, info

    def close(self):
        for expert in self.experts.values():
            expert.stop()
        return super().close()

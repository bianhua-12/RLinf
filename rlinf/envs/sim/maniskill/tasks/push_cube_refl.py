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

import torch
from mani_skill.envs.tasks.tabletop.push_cube import PushCubeEnv
from mani_skill.utils.registration import register_env

# Layout of :meth:`PushCubeReFLEnv.get_physical_state`, the world-model state.
PHYSICAL_STATE_LAYOUT = (
    ("qpos", 9),
    ("qvel", 9),
    ("tcp_pose", 7),
    ("cube_pose", 7),
    ("goal_position", 3),
    ("cube_linear_velocity", 3),
    ("cube_angular_velocity", 3),
)


@register_env("PushCubeReFL-v1", max_episode_steps=50)
class PushCubeReFLEnv(PushCubeEnv):
    """Native PushCube with a language instruction and the physical state a
    world-model ReFL scorer starts from. Dynamics, reward and success unchanged."""

    INSTRUCTION = "push the cube to the red and white target"

    def get_language_instruction(self):
        return [self.INSTRUCTION] * self.num_envs

    def get_physical_state(self) -> torch.Tensor:
        """[num_envs, 41]; poses are position + WXYZ quaternion in the env frame."""
        agent, obj = self.agent, self.obj
        parts = [
            agent.robot.get_qpos(),
            agent.robot.get_qvel(),
            agent.tcp.pose.raw_pose,
            obj.pose.raw_pose,
            self.goal_region.pose.p,
            obj.linear_velocity,
            obj.angular_velocity,
        ]
        parts = [p.reshape(self.num_envs, -1) for p in parts]
        widths = [p.shape[-1] for p in parts]
        if widths != [width for _, width in PHYSICAL_STATE_LAYOUT]:
            raise RuntimeError(f"Physical state layout mismatch: {widths}")
        return torch.cat(parts, -1).float()

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

"""Physical-state world model, its ReFL scorer and the env-side carry."""

import torch

from rlinf.models.embodiment.physical_wm.geometry import RobotFK
from rlinf.models.embodiment.physical_wm.model import (
    SCHEMAS,
    Carry,
    PhysicalWM,
    feature_floor,
    load_wm,
    pushcube_reward,
)


def _toy_asset():
    """Seven revolute joints about z with 0.1 m links; the TCP is link7."""
    links = "".join(f'<link name="link{i}"/>' for i in range(8))
    joints = "".join(
        f'<joint name="joint{i}" type="revolute"><parent link="link{i - 1}"/>'
        f'<child link="link{i}"/><origin xyz="0.1 0 0" rpy="0 0 0"/>'
        '<axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="1" velocity="1"/>'
        "</joint>"
        for i in range(1, 8)
    )
    return {
        "urdf": f'<robot name="toy">{links}{joints}</robot>',
        "ee_link": "link7",
        "arm_joint_names": [f"joint{i}" for i in range(1, 8)],
        "base_pose": torch.eye(4).tolist(),
    }


def test_robot_fk_composes_urdf_joints():
    fk = RobotFK(_toy_asset())
    qpos = torch.zeros(2, 9)
    qpos[1, 0] = torch.pi / 2
    pose = fk(qpos)
    torch.testing.assert_close(pose[0], torch.tensor([0.7, 0, 0, 1, 0, 0, 0]))
    half = 2**-0.5
    torch.testing.assert_close(
        pose[1], torch.tensor([0.1, 0.6, 0, half, 0, 0, half]), atol=1e-6, rtol=0
    )
    assert not {"origins", "axes", "tail"} & set(fk.state_dict())


def _wm():
    floor = feature_floor(SCHEMAS["pushcube"])
    wm = PhysicalWM("pushcube", _toy_asset(), torch.zeros_like(floor), floor, width=32)
    # A trained WM has a nonzero residual head; the zero init has no action path.
    torch.manual_seed(0)
    torch.nn.init.normal_(wm.continuous.weight, std=0.05)
    return wm


def _state(batch):
    """PushCube states at rest: random arm joints, the cube on the table."""
    state = torch.zeros(batch, 41)
    generator = torch.Generator().manual_seed(1)
    state[:, :9] = 0.1 * torch.randn(batch, 9, generator=generator)
    state[:, 21] = 1  # TCP quaternion
    state[:, 25:29] = torch.tensor([0.0, 0.0, 0.02, 1.0])  # cube pose
    state[:, 32:35] = torch.tensor([0.3, 0.0, 0.0])  # goal
    return state


def test_pushcube_reward_matches_native_terms():
    state = _state(2)
    # Env 0: TCP at the push point and the cube on the goal: success.
    state[0, 25:28] = torch.tensor([0.3, 0.0, 0.02])
    state[0, 18:21] = torch.tensor([0.3 - 0.025, 0.0, 0.02])
    # Env 1: TCP 0.2 m above the push point, the cube far from the goal.
    state[1, 18:21] = torch.tensor([-0.025, 0.0, 0.22])
    reward, success = pushcube_reward(state)
    assert success.tolist() == [1.0, 0.0]
    assert reward[0] == 1
    torch.testing.assert_close(reward[1], (1 - torch.tanh(torch.tensor(1.0))) / 4)


def test_world_model_step_keeps_kinematics_and_survives_checkpoint():
    wm = _wm()
    state = _state(3)
    carry = Carry(
        state,
        state.clone(),
        torch.zeros(3, 7),
        torch.ones(3, dtype=torch.bool),
        torch.zeros(3, 1),
    )
    nxt = wm(carry, torch.zeros(3, 7))
    torch.testing.assert_close(nxt.state[:, 18:25], wm.fk(nxt.state[:, :9]))
    torch.testing.assert_close(nxt.state[:, 28:32].norm(dim=-1), torch.ones(3))
    torch.testing.assert_close(nxt.previous, carry.state)
    assert not nxt.reset.any()
    torch.testing.assert_close(nxt.context, torch.full((3, 1), 1 / 50))
    torch.testing.assert_close(
        load_wm(wm.checkpoint())(carry, torch.zeros(3, 7)).state, nxt.state
    )

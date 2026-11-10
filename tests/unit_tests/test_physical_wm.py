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

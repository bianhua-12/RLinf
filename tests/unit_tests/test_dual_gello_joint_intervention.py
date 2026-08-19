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

from types import SimpleNamespace

import numpy as np

from rlinf.envs.realworld.common.wrappers.dual_gello_joint_intervention import (
    DualGelloJointIntervention,
)


def test_gello_forwards_intermediate_gripper_positions_while_arm_is_stationary():
    wrapper = object.__new__(DualGelloJointIntervention)
    wrapper.gripper_enabled = True
    wrapper.use_delta = False
    wrapper._direct_stream = False
    wrapper._aligned = False
    wrapper.last_intervene = 0.0
    wrapper.left_expert = SimpleNamespace(
        ready=True,
        get_action=lambda: (np.arange(7, dtype=np.float32), np.array([0.25])),
    )
    wrapper.right_expert = SimpleNamespace(
        ready=True,
        get_action=lambda: (np.arange(7, dtype=np.float32), np.array([0.75])),
    )
    wrapper._get_current_joint_positions = lambda: np.zeros((2, 7))

    action, replaced = wrapper.action(np.zeros(16, dtype=np.float32))

    assert replaced
    assert action[7] == 0.5
    assert action[15] == -0.5

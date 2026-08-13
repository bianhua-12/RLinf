# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
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
from rlinf.envs.realworld.franka.tasks.ros2_dual_franka_joint_env import (
    Ros2DualFrankaJointEnv,
)


def test_ros2_joint_env_uses_placeholder_reward_for_manual_collection():
    env = object.__new__(Ros2DualFrankaJointEnv)

    assert env._calc_step_reward([True, True]) == 0.0


class _Expert:
    ready = True

    def __init__(self, joints, gripper):
        self._action = (np.asarray(joints), np.asarray([gripper]))

    def get_action(self):
        return self._action


def test_direct_stream_records_stationary_gello_action():
    wrapper = object.__new__(DualGelloJointIntervention)
    wrapper.left_expert = _Expert(np.arange(7), 0.25)
    wrapper.right_expert = _Expert(np.arange(7) + 10, 0.75)
    wrapper.gripper_enabled = True
    wrapper.use_delta = False
    wrapper.action_scale = 0.1
    wrapper.last_intervene = 0.0
    wrapper._direct_stream = True
    wrapper._aligned = True
    wrapper._get_current_joint_positions = lambda: np.stack(
        [np.arange(7), np.arange(7) + 10]
    )

    action, replaced = wrapper.action(np.zeros(16))

    np.testing.assert_allclose(
        action,
        np.concatenate([np.arange(7), [0.5], np.arange(7) + 10, [-0.5]]),
    )
    assert replaced


class _Result:
    def __init__(self, value, event=None, events=None):
        self.value = value
        self.event = event
        self.events = events

    def wait(self):
        if self.event is not None:
            self.events.append(self.event)
        return [self.value]


class _Controller:
    def __init__(self, side, events):
        self.side = side
        self.events = events
        self.state = object()

    def open_gripper(self):
        return None

    def reset_joint(self, target):
        self.events.append(f"{self.side}_reset")
        return _Result(None, f"{self.side}_wait", self.events)

    def get_state(self):
        return _Result(self.state)


def test_ros2_rest_reset_publishes_both_targets_before_waiting():
    events = []
    env = object.__new__(Ros2DualFrankaJointEnv)
    env._left_ctrl = _Controller("left", events)
    env._right_ctrl = _Controller("right", events)
    env.config = SimpleNamespace(joint_reset_qpos=[[0.0] * 7, [0.0] * 7])

    env._go_to_rest()

    assert events == ["left_reset", "right_reset", "left_wait", "right_wait"]
    assert env._left_state is env._left_ctrl.state
    assert env._right_state is env._right_ctrl.state

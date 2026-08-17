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

import threading
from types import SimpleNamespace

from rlinf.envs.realworld.franka.ros2_controller import Ros2DualFrankaBackend
from rlinf.envs.realworld.franka.tasks.ros2_dual_franka_joint_env import (
    Ros2DualFrankaJointEnv,
)


def test_ros2_joint_env_uses_placeholder_reward_for_manual_collection():
    env = object.__new__(Ros2DualFrankaJointEnv)

    assert env._calc_step_reward([True, True]) == 0.0


def test_backend_subscribes_to_throttled_joint_states():
    assert Ros2DualFrankaBackend._state_topic("left") == "/left/joint_states"
    assert Ros2DualFrankaBackend._state_topic("right") == "/right/joint_states"


def test_go_to_rest_waits_for_both_arms(monkeypatch):
    events = []

    class Result:
        def __init__(self, event, value=None):
            self.event = event
            self.value = value

        def wait(self):
            events.append(self.event)
            return [self.value]

    class Controller:
        def __init__(self, side):
            self.side = side

        def open_gripper(self):
            pass

        def reset_joint(self, target):
            events.append(f"{self.side}_reset")
            return Result(f"{self.side}_reset_wait")

        def get_state(self):
            events.append(f"{self.side}_get_state")
            return Result(f"{self.side}_state_wait", self.side)

    env = object.__new__(Ros2DualFrankaJointEnv)
    env.config = SimpleNamespace(joint_reset_qpos=[[0.0] * 7, [0.0] * 7])
    env._left_ctrl = Controller("left")
    env._right_ctrl = Controller("right")
    monkeypatch.setattr("time.sleep", lambda _: None)

    env._go_to_rest()

    assert events == [
        "left_reset",
        "right_reset",
        "left_reset_wait",
        "right_reset_wait",
        "left_get_state",
        "left_state_wait",
        "right_get_state",
        "right_state_wait",
    ]


def test_controller_health_timeout_only_after_seen_active(monkeypatch):
    class PendingFuture:
        def done(self):
            return False

    backend = object.__new__(Ros2DualFrankaBackend)
    backend.config = SimpleNamespace(
        controller_health_period=0.1, controller_health_timeout=0.5
    )
    backend._health_lock = threading.Lock()
    backend._controller_health_futures = {"left": PendingFuture()}
    backend._controller_health_requested = {"left": 1.0}
    backend._controller_health_clients = {
        "left": SimpleNamespace(service_is_ready=lambda: True)
    }
    backend._controller_seen_active = {"left": False}
    backend._controller_health_errors = {"left": None}
    monkeypatch.setattr(
        "rlinf.envs.realworld.franka.ros2_controller.time.monotonic", lambda: 2.0
    )

    backend._poll_controller_health("left")
    assert backend._controller_health_errors["left"] is None

    backend._controller_seen_active["left"] = True
    backend._poll_controller_health("left")
    assert isinstance(backend._controller_health_errors["left"], TimeoutError)

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

import numpy as np
import pytest

from rlinf.envs.realworld.franka.ros2_controller import (
    FR3_JOINT_LIMITS_LOWER,
    FR3_JOINT_LIMITS_UPPER,
    JOINT_LIMIT_ROUNDING_TOLERANCE_RAD,
    Ros2ControllerConfig,
    Ros2DualFrankaBackend,
)
from rlinf.envs.realworld.franka.tasks.ros2_dual_franka_joint_env import (
    Ros2DualFrankaJointEnv,
)


def test_ros2_joint_env_uses_placeholder_reward_for_manual_collection():
    env = object.__new__(Ros2DualFrankaJointEnv)

    assert env._calc_step_reward([True, True]) == 0.0


def test_backend_subscribes_to_throttled_joint_states():
    assert Ros2DualFrankaBackend._state_topic("left") == "/left/joint_states"
    assert Ros2DualFrankaBackend._state_topic("right") == "/right/joint_states"


def test_robotiq_polling_defaults_to_30_hz():
    config = Ros2ControllerConfig(
        left_robot_ip="left",
        right_robot_ip="right",
        left_gripper_type="robotiq",
        right_gripper_type="robotiq",
        left_gripper_connection="left-port",
        right_gripper_connection="right-port",
    )

    assert config.gripper_poll_interval == 1.0 / 30.0
    assert config.left_gripper_close_force == 130.0
    assert config.right_gripper_close_force == 130.0


@pytest.mark.parametrize("joint_index", range(7))
@pytest.mark.parametrize("limits", [FR3_JOINT_LIMITS_LOWER, FR3_JOINT_LIMITS_UPPER])
def test_backend_accepts_float32_round_trip_at_all_joint_limits(joint_index, limits):
    backend = object.__new__(Ros2DualFrankaBackend)
    backend.config = SimpleNamespace(
        joint_names=[f"fr3_joint{index}" for index in range(1, 8)]
    )
    target = (FR3_JOINT_LIMITS_LOWER + FR3_JOINT_LIMITS_UPPER) / 2.0
    target[joint_index] = np.float64(np.float32(limits[joint_index]))

    sanitized = backend._sanitize_joint_target("right", target)

    assert sanitized.dtype == np.float64
    assert np.all(sanitized > FR3_JOINT_LIMITS_LOWER)
    assert np.all(sanitized < FR3_JOINT_LIMITS_UPPER)


def test_backend_rejects_real_joint_limit_violation_with_joint_details():
    backend = object.__new__(Ros2DualFrankaBackend)
    backend.config = SimpleNamespace(
        joint_names=[f"fr3_joint{index}" for index in range(1, 8)]
    )
    target = (FR3_JOINT_LIMITS_LOWER + FR3_JOINT_LIMITS_UPPER) / 2.0
    target[0] = FR3_JOINT_LIMITS_UPPER[0] + 2 * JOINT_LIMIT_ROUNDING_TOLERANCE_RAD

    with pytest.raises(
        ValueError, match=r"right.*fr3_joint1.*\(J1\).*above FR3 software limit"
    ):
        backend._sanitize_joint_target("right", target)


@pytest.mark.parametrize(
    ("action", "expected_position"),
    [(1.0, 0), (0.0, 128), (-1.0, 255), (2.0, 0), (-2.0, 255)],
)
def test_ros2_joint_env_maps_continuous_gripper_action(action, expected_position):
    class Controller:
        def __init__(self):
            self.positions = []

        def move_gripper(self, position):
            self.positions.append(position)

    env = object.__new__(Ros2DualFrankaJointEnv)
    controller = Controller()

    assert env._gripper_action(controller, None, action)
    assert controller.positions == [expected_position]


def test_ros2_backend_queues_absolute_gripper_position():
    backend = object.__new__(Ros2DualFrankaBackend)
    backend._lock = threading.Lock()
    backend._gripper_targets = {"left": None, "right": None}

    backend.move_gripper("left", 127.6)
    backend.open_gripper("right")

    assert backend._gripper_targets == {"left": 128, "right": 0}


def test_ros2_backend_applies_intermediate_gripper_position(monkeypatch):
    class Gripper:
        def __init__(self):
            self.calls = []

        def open(self, speed):
            self.calls.append(("open", speed))

        def close(self, speed, force=130.0):
            self.calls.append(("close", speed, force))

        def move(self, position, speed):
            self.calls.append(("move", position, speed))

        @property
        def position(self):
            backend._gripper_running = False
            return 0.0425

        @property
        def is_open(self):
            return True

    gripper = Gripper()
    backend = object.__new__(Ros2DualFrankaBackend)
    backend.config = SimpleNamespace(gripper_poll_interval=0.0)
    backend._lock = threading.Lock()
    backend._grippers = {"left": gripper}
    backend._gripper_targets = {"left": 128}
    backend._gripper_positions = {"left": None}
    backend._gripper_open = {"left": True}
    backend._gripper_errors = {"left": None}
    backend._gripper_running = True
    monkeypatch.setattr(
        "rlinf.envs.realworld.franka.ros2_controller.time.sleep", lambda _: None
    )

    backend._gripper_loop("left")

    assert gripper.calls == [("move", 128, 1.0)]
    assert backend._gripper_positions["left"] == 0.0425
    assert not backend._gripper_open["left"]


@pytest.mark.parametrize(("side", "force"), [("left", 255.0), ("right", 255.0)])
def test_ros2_backend_applies_per_arm_force_at_fully_closed_target(
    monkeypatch, side, force
):
    class Gripper:
        def __init__(self):
            self.calls = []

        def close(self, speed, force):
            self.calls.append(("close", speed, force))

        @property
        def position(self):
            backend._gripper_running = False
            return 0.0

        @property
        def is_open(self):
            return False

    gripper = Gripper()
    backend = object.__new__(Ros2DualFrankaBackend)
    backend.config = SimpleNamespace(
        gripper_poll_interval=0.0,
        left_gripper_close_force=255.0,
        right_gripper_close_force=255.0,
    )
    backend._lock = threading.Lock()
    backend._grippers = {side: gripper}
    backend._gripper_targets = {side: 255}
    backend._gripper_positions = {side: None}
    backend._gripper_open = {side: True}
    backend._gripper_errors = {side: None}
    backend._gripper_running = True
    monkeypatch.setattr(
        "rlinf.envs.realworld.franka.ros2_controller.time.sleep", lambda _: None
    )

    backend._gripper_loop(side)

    assert gripper.calls == [("close", 1.0, force)]


@pytest.mark.parametrize("position", [-1, 256, float("nan"), float("inf")])
def test_ros2_backend_rejects_invalid_gripper_position(position):
    backend = object.__new__(Ros2DualFrankaBackend)
    backend._lock = threading.Lock()
    backend._gripper_targets = {"left": None, "right": None}

    with pytest.raises(ValueError, match=r"within \[0, 255\]"):
        backend.move_gripper("left", position)


def test_observation_deadline_subtracts_previous_post_read_work(monkeypatch):
    env = object.__new__(Ros2DualFrankaJointEnv)
    env.config = SimpleNamespace(step_frequency=30.0)
    env._step_deadline = 0.0
    sleeps = []
    times = iter([0.002, 1.0 / 30.0 + 0.002])
    monkeypatch.setattr(
        "rlinf.envs.realworld.franka.dual_franka_env.time.perf_counter",
        lambda: next(times),
    )
    monkeypatch.setattr(
        "rlinf.envs.realworld.franka.dual_franka_env.time.sleep", sleeps.append
    )

    env._wait_for_observation_deadline()
    env._wait_for_observation_deadline()

    assert sleeps == pytest.approx([1.0 / 30.0 - 0.002] * 2)


def test_observation_deadline_reanchors_after_overrun(monkeypatch):
    env = object.__new__(Ros2DualFrankaJointEnv)
    env.config = SimpleNamespace(step_frequency=30.0)
    env._step_deadline = 0.0
    sleeps = []
    times = iter([0.1, 0.102])
    monkeypatch.setattr(
        "rlinf.envs.realworld.franka.dual_franka_env.time.perf_counter",
        lambda: next(times),
    )
    monkeypatch.setattr(
        "rlinf.envs.realworld.franka.dual_franka_env.time.sleep", sleeps.append
    )

    env._wait_for_observation_deadline()
    env._wait_for_observation_deadline()

    assert sleeps == pytest.approx([1.0 / 30.0 - 0.002])


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

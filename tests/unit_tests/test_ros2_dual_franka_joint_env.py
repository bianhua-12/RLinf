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
import time
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np
import pytest

from rlinf.envs.realworld.common.camera.base_camera import (
    CameraHealth,
    CameraSnapshot,
)
from rlinf.envs.realworld.franka.dual_franka_env import (
    _DaemonCameraRecoveryExecutor,
)
from rlinf.envs.realworld.franka.ros2_controller import (
    Ros2ControllerConfig,
    Ros2DualFrankaBackend,
)
from rlinf.envs.realworld.franka.tasks.ros2_dual_franka_joint_env import (
    Ros2DualFrankaJointEnv,
)


class _SnapshotCamera:
    def __init__(self, name="base_0_rgb", health=CameraHealth.HEALTHY):
        self._camera_info = SimpleNamespace(
            name=name, serial_number=f"serial-{name}", camera_type="fake"
        )
        self.health = health
        self.frame = np.zeros((4, 4, 3), dtype=np.uint8)
        self.timestamp = 1.0
        self.sequence = 1
        self.last_failure = None
        self.closed = False

    @property
    def name(self):
        return self._camera_info.name

    def snapshot(self):
        return CameraSnapshot(
            frame=self.frame,
            frame_timestamp_monotonic=self.timestamp,
            frame_sequence=self.sequence,
            health=self.health,
            consecutive_read_failures=3 if self.health == CameraHealth.FAILED else 0,
            last_success_monotonic=self.timestamp,
            last_failure=self.last_failure,
            last_failure_monotonic=self.timestamp if self.last_failure else None,
        )

    def close(self):
        self.closed = True


class _RecordingExecutor:
    def __init__(self):
        self.submissions = []

    def submit(self, function, *args):
        future = Future()
        self.submissions.append((function, args, future))
        return future


def _camera_env(cameras):
    env = object.__new__(Ros2DualFrankaJointEnv)
    env._cameras = cameras
    env._last_camera_frame = {}
    env._last_camera_sequence = {}
    env._camera_stale_since = {}
    env._camera_consecutive_stale_steps = {}
    env._camera_diagnostics = {}
    env._camera_recovery_futures = {}
    env._camera_recovery_attempts = {}
    env._camera_recovery_next_retry = {}
    env._camera_needs_recovery = {}
    env._camera_recovery_executor = _RecordingExecutor()
    env.config = SimpleNamespace(
        camera_first_frame_timeout_seconds=0.2,
        camera_max_stale_seconds=1.0,
        camera_recovery_initial_backoff_seconds=0.5,
        camera_recovery_max_backoff_seconds=4.0,
        camera_recovery_shutdown_timeout_seconds=0.5,
    )
    env.observation_space = {
        "frames": {camera.name: SimpleNamespace(shape=(4, 4, 3)) for camera in cameras}
    }
    env.camera_player = SimpleNamespace(put_frame=lambda frames: None)
    env._logger = SimpleNamespace(
        error=lambda *args: None,
        info=lambda *args: None,
        exception=lambda *args: None,
    )
    return env


def test_failed_camera_schedules_one_recovery_from_capture_health():
    camera = _SnapshotCamera(health=CameraHealth.FAILED)
    camera.last_failure = "three real read failures"
    env = _camera_env([camera])

    frames = env._get_camera_frames()
    assert frames["base_0_rgb"].shape == (4, 4, 3)
    assert len(env._camera_recovery_executor.submissions) == 1
    assert "base_0_rgb" in env._camera_recovery_futures

    env._get_camera_frames()
    assert len(env._camera_recovery_executor.submissions) == 1


def test_three_camera_snapshot_path_does_not_wait_per_camera():
    cameras = [_SnapshotCamera(f"camera_{index}") for index in range(3)]
    env = _camera_env(cameras)
    started = time.perf_counter()

    frames = env._get_camera_frames()

    assert time.perf_counter() - started < 0.05
    assert set(frames) == {camera.name for camera in cameras}
    assert all(frame.shape == (4, 4, 3) for frame in frames.values())
    assert all(frame.dtype == np.uint8 for frame in frames.values())


def test_cached_frame_is_marked_stale_and_fails_after_named_limit(monkeypatch):
    camera = _SnapshotCamera()
    env = _camera_env([camera])
    times = iter([1.0, 1.1, 2.2])
    monkeypatch.setattr(
        "rlinf.envs.realworld.franka.dual_franka_env.time.monotonic",
        lambda: next(times),
    )

    env._get_camera_frames()
    env._get_camera_frames()
    diagnostic = env.camera_diagnostics["base_0_rgb"]
    assert diagnostic["is_stale"]
    assert diagnostic["consecutive_stale_steps"] == 1

    with pytest.raises(RuntimeError, match="camera_max_stale_seconds"):
        env._get_camera_frames()


def test_recovery_failure_uses_exponential_backoff(monkeypatch):
    camera = _SnapshotCamera(health=CameraHealth.FAILED)
    env = _camera_env([camera])
    now = [10.0]
    monkeypatch.setattr(
        "rlinf.envs.realworld.franka.dual_franka_env.time.monotonic",
        lambda: now[0],
    )
    env._camera_needs_recovery[camera.name] = True
    env._schedule_camera_recovery(camera)
    future = env._camera_recovery_futures[camera.name]
    future.set_exception(RuntimeError("enumeration failed"))

    env._poll_camera_recovery(0, camera)
    assert env._camera_recovery_next_retry[camera.name] == pytest.approx(10.5)
    env._schedule_camera_recovery(camera)
    assert len(env._camera_recovery_executor.submissions) == 1
    now[0] = 10.5
    env._schedule_camera_recovery(camera)
    assert len(env._camera_recovery_executor.submissions) == 2


def test_replacement_waits_for_first_frame_before_returning(monkeypatch):
    old = _SnapshotCamera(health=CameraHealth.FAILED)
    replacement = _SnapshotCamera()
    events = []
    old.close = lambda: events.append("old_close")
    replacement.open = lambda: events.append("replacement_open")
    replacement.wait_for_first_frame = lambda timeout: events.append(
        ("first_frame", timeout)
    )
    replacement.close = lambda: events.append("replacement_close")
    monkeypatch.setattr(
        "rlinf.envs.realworld.franka.dual_franka_env.create_camera",
        lambda info: replacement,
    )

    result = Ros2DualFrankaJointEnv._replace_camera(old, 0.25)

    assert result is replacement
    assert events == ["old_close", "replacement_open", ("first_frame", 0.25)]


def test_camera_recovery_executor_shutdown_is_bounded_and_reaps_normal_job():
    executor = _DaemonCameraRecoveryExecutor()
    future = executor.submit(lambda: "done")
    assert future.result(timeout=1.0) == "done"
    assert executor.shutdown(timeout=1.0)
    assert not executor.thread_alive


def test_close_reaps_inflight_recovery_and_recovered_camera():
    old = _SnapshotCamera()
    replacement = _SnapshotCamera(name="replacement")
    env = _camera_env([old])
    executor = _DaemonCameraRecoveryExecutor()
    env._camera_recovery_executor = executor
    entered = threading.Event()

    def recover():
        entered.set()
        time.sleep(0.02)
        return replacement

    env._camera_recovery_futures = {old.name: executor.submit(recover)}
    assert entered.wait(timeout=1.0)

    env._close_cameras()

    assert not executor.thread_alive
    assert old.closed
    assert replacement.closed
    assert env._cameras == []


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

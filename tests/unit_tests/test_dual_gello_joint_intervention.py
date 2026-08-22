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

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

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


class _CompletedResult:
    def wait(self):
        return [None]


def _stream_wrapper():
    wrapper = object.__new__(DualGelloJointIntervention)
    wrapper._stream_gate = threading.Event()
    wrapper._stream_gate.set()
    wrapper._stream_command_lock = threading.Lock()
    wrapper._stream_period = 0.0
    wrapper._stream_error = None
    wrapper._stream_last_success_time = None
    wrapper._aligned = True
    return wrapper


def test_direct_stream_keeps_joint_targets_in_float64():
    wrapper = _stream_wrapper()
    targets = []

    class Controller:
        def __init__(self, stop=False):
            self.stop = stop

        def move_joints(self, target):
            targets.append(target)
            if self.stop:
                wrapper._stream_running = False
            return _CompletedResult()

    wrapper.left_expert = SimpleNamespace(
        ready=True,
        get_action=lambda: (np.arange(7, dtype=np.float32), np.array([0.0])),
    )
    wrapper.right_expert = wrapper.left_expert
    wrapper._resolve_controllers = lambda: (Controller(), Controller(stop=True))
    wrapper._stream_running = True

    wrapper._stream_loop()

    assert wrapper._stream_error is None
    assert all(target.dtype == np.float64 for target in targets)
    assert wrapper._stream_last_success_time is not None


def test_direct_stream_logs_and_records_controller_failure():
    wrapper = _stream_wrapper()
    log_calls = []

    class FailingController:
        def move_joints(self, target):
            raise ValueError("right joint target is outside FR3 joint limits")

    wrapper.left_expert = SimpleNamespace(
        ready=True,
        get_action=lambda: (np.arange(7, dtype=np.float32), np.array([0.0])),
    )
    wrapper.right_expert = wrapper.left_expert
    wrapper._resolve_controllers = lambda: (
        SimpleNamespace(move_joints=lambda target: _CompletedResult()),
        FailingController(),
    )
    wrapper._logger = SimpleNamespace(
        exception=lambda message, *args: log_calls.append((message, args))
    )
    wrapper._stream_running = True

    wrapper._stream_loop()

    assert isinstance(wrapper._stream_error, ValueError)
    assert not wrapper._stream_running
    assert not wrapper._aligned
    assert not wrapper._stream_gate.is_set()
    assert len(log_calls) == 1
    assert "left_target" in log_calls[0][0]


def test_foreground_raises_recorded_stream_failure():
    wrapper = _stream_wrapper()
    cause = ValueError("right J1 target exceeded its software limit")
    wrapper._stream_error = cause

    with pytest.raises(RuntimeError, match="dual GELLO stream failed") as exc_info:
        wrapper._raise_if_stream_unhealthy()

    assert exc_info.value.__cause__ is cause


def test_direct_stream_rejects_stale_command_heartbeat(monkeypatch):
    wrapper = _stream_wrapper()
    wrapper._direct_stream = True
    wrapper._stream_running = True
    wrapper._stream_thread = SimpleNamespace(is_alive=lambda: True)
    wrapper._stream_last_success_time = 1.0
    wrapper._stream_watchdog_timeout = 0.25
    monkeypatch.setattr(
        "rlinf.envs.realworld.common.wrappers.dual_gello_joint_intervention.time.monotonic",
        lambda: 1.5,
    )

    with pytest.raises(RuntimeError, match=r"heartbeat is stale \(0.500s > 0.250s\)"):
        wrapper._raise_if_stream_unhealthy()


def _reset_wrapper(*, aligned: bool):
    wrapper = _stream_wrapper()
    wrapper._direct_stream = True
    wrapper._aligned = aligned
    wrapper._stream_running = aligned
    wrapper._stream_thread = SimpleNamespace(is_alive=lambda: aligned)
    wrapper._stream_last_success_time = time.monotonic() if aligned else None
    wrapper._stream_watchdog_timeout = 1.0
    wrapper._wait_for_experts = lambda: None
    return wrapper


def test_healthy_direct_stream_stays_open_across_episode_reset():
    wrapper = _reset_wrapper(aligned=True)
    calls = []

    class Env:
        def reset(self, **kwargs):
            calls.append(("reset", kwargs))
            assert wrapper._stream_gate.is_set()
            return "observation", {"episode": 2}

    wrapper.env = Env()
    wrapper._align_to_gello = lambda: calls.append(("align", {}))
    wrapper._start_stream_thread = lambda: calls.append(("start", {}))

    result = wrapper.reset()

    assert result == ("observation", {"episode": 2})
    assert calls == [
        ("reset", {"options": {"skip_reset_to_home": True}}),
    ]
    assert wrapper._aligned
    assert wrapper._stream_gate.is_set()


def test_initial_direct_stream_reset_aligns_before_opening_gate():
    wrapper = _reset_wrapper(aligned=False)
    calls = []

    class Env:
        def reset(self, **kwargs):
            calls.append(("reset", kwargs))
            assert not wrapper._stream_gate.is_set()
            return "pre_alignment_observation", {"episode": 1}

        def get_wrapper_attr(self, name):
            assert name == "_get_observation"
            calls.append(("observation", {}))
            return lambda: "aligned_observation"

    def align():
        calls.append(("align", {}))
        assert not wrapper._stream_gate.is_set()
        wrapper._aligned = True
        return True

    def start():
        calls.append(("start", {}))
        assert wrapper._stream_gate.is_set()

    wrapper.env = Env()
    wrapper._align_to_gello = align
    wrapper._start_stream_thread = start

    result = wrapper.reset()

    assert result == ("aligned_observation", {"episode": 1})
    assert [name for name, _ in calls] == ["reset", "align", "observation", "start"]
    assert wrapper._aligned
    assert wrapper._stream_gate.is_set()


def test_explicit_home_reset_pauses_stream_before_realigning():
    wrapper = _reset_wrapper(aligned=True)
    calls = []

    class Env:
        def reset(self, **kwargs):
            calls.append(("reset", kwargs))
            assert not wrapper._stream_gate.is_set()
            return "home_observation", {"episode": 2}

        def get_wrapper_attr(self, name):
            assert name == "_get_observation"
            return lambda: "realigned_observation"

    def align():
        calls.append(("align", {}))
        assert not wrapper._stream_gate.is_set()
        wrapper._aligned = True
        return True

    wrapper.env = Env()
    wrapper._align_to_gello = align
    wrapper._start_stream_thread = lambda: calls.append(("start", {}))

    result = wrapper.reset(options={"skip_reset_to_home": False})

    assert result == ("realigned_observation", {"episode": 2})
    assert calls[0] == ("reset", {"options": {"skip_reset_to_home": False}})
    assert [name for name, _ in calls] == ["reset", "align", "start"]
    assert wrapper._stream_gate.is_set()


def test_explicit_home_reset_waits_for_inflight_stream_tick():
    wrapper = _reset_wrapper(aligned=True)
    reset_entered = threading.Event()
    result = []

    class Env:
        def reset(self, **kwargs):
            reset_entered.set()
            return "home_observation", {}

        def get_wrapper_attr(self, name):
            assert name == "_get_observation"
            return lambda: "realigned_observation"

    wrapper.env = Env()
    wrapper._align_to_gello = lambda: True
    wrapper._start_stream_thread = lambda: None

    wrapper._stream_command_lock.acquire()
    reset_thread = threading.Thread(
        target=lambda: result.append(
            wrapper.reset(options={"skip_reset_to_home": False})
        )
    )
    reset_thread.start()
    deadline = time.monotonic() + 1.0
    while wrapper._stream_gate.is_set() and time.monotonic() < deadline:
        time.sleep(0.001)

    assert not wrapper._stream_gate.is_set()
    assert not reset_entered.is_set()

    wrapper._stream_command_lock.release()
    assert reset_entered.wait(timeout=1.0)
    reset_thread.join(timeout=1.0)

    assert not reset_thread.is_alive()
    assert result == [("realigned_observation", {})]


def test_direct_stream_reset_failure_closes_gate():
    wrapper = _reset_wrapper(aligned=True)
    wrapper.env = SimpleNamespace(
        reset=lambda **kwargs: (_ for _ in ()).throw(
            RuntimeError("camera reset failed")
        )
    )

    with pytest.raises(RuntimeError, match="camera reset failed"):
        wrapper.reset()

    assert not wrapper._aligned
    assert not wrapper._stream_gate.is_set()

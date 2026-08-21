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

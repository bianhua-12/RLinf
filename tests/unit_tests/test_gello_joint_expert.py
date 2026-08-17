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

import numpy as np
import pytest

from rlinf.envs.realworld.common.gello.gello_joint_expert import GelloJointExpert


def test_gello_reader_polls_at_50_hz(monkeypatch):
    expert = GelloJointExpert.__new__(GelloJointExpert)
    expert._stop = False
    expert._prev_joints = np.zeros(7)
    expert._unwrap_reference = np.zeros(7)
    expert._joint_limits_lower = np.full(7, -10.0)
    expert._joint_limits_upper = np.full(7, 10.0)
    expert.state_lock = threading.Lock()
    expert.latest_data = {}
    sleep_calls = []

    def read_once():
        expert._stop = True
        return np.zeros(7), 0.0

    expert._action_source = read_once
    monkeypatch.setattr(
        "rlinf.envs.realworld.common.gello.gello_joint_expert.time.sleep",
        sleep_calls.append,
    )

    expert._read_gello()

    assert sleep_calls == [pytest.approx(1.0 / 50.0)]

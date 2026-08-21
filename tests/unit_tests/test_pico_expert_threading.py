# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import threading
from types import SimpleNamespace

import numpy as np
from scipy.spatial.transform import Rotation as R

from rlinf.envs.realworld.common.pico import pico_expert


def test_zmq_socket_lifecycle_stays_on_receiver_thread(monkeypatch):
    calls = []
    received = threading.Event()

    class Again(Exception):
        pass

    class Socket:
        def _record(self, name):
            calls.append((name, threading.get_ident()))

        def set_hwm(self, value):
            del value
            self._record("set_hwm")

        def setsockopt(self, option, value):
            del option, value
            self._record("setsockopt")

        def connect(self, address):
            del address
            self._record("connect")

        def recv(self):
            self._record("recv")
            received.set()
            raise Again

        def close(self, linger):
            del linger
            self._record("close")

    class Context:
        def __init__(self):
            calls.append(("context", threading.get_ident()))

        def socket(self, socket_type):
            del socket_type
            calls.append(("socket", threading.get_ident()))
            return Socket()

        def term(self):
            calls.append(("term", threading.get_ident()))

    fake_zmq = SimpleNamespace(
        Context=Context,
        SUB=1,
        RCVTIMEO=2,
        SUBSCRIBE=3,
        error=SimpleNamespace(Again=Again),
    )
    monkeypatch.setattr(pico_expert, "zmq", fake_zmq)

    expert = pico_expert.PicoExpert(
        hand="left",
        timeout_ms=1,
        calibration={"enabled": False},
    )
    assert received.wait(timeout=1.0)
    receiver_thread_id = expert._thread.ident
    expert.stop()

    thread_ids = {thread_id for _, thread_id in calls}
    assert thread_ids == {receiver_thread_id}
    assert threading.get_ident() not in thread_ids
    assert {"context", "socket", "recv", "close", "term"} <= {name for name, _ in calls}


def test_repeated_source_timestamp_stays_ready_while_messages_arrive(monkeypatch):
    now = [1.0]
    monkeypatch.setattr(pico_expert.time, "monotonic", lambda: now[0])
    expert = pico_expert.PicoExpert.__new__(pico_expert.PicoExpert)
    expert._lock = threading.Lock()
    expert._latest_data = None
    expert._last_update_time = 0.0
    expert.max_stale_s = 0.2

    expert._set_latest_data({"timestamp_ns": 1})
    assert expert.ready

    now[0] = 1.3
    expert._set_latest_data({"timestamp_ns": 1})
    assert expert.ready

    now[0] = 1.51
    assert not expert.ready


def test_one_euro_pose_filter_smooths_and_adapts_to_motion():
    fixed = pico_expert._OneEuroPoseFilter(1.0, 0.0, 1.0)
    adaptive = pico_expert._OneEuroPoseFilter(1.0, 0.1, 1.0)
    position = np.array([1.0, 0.0, 0.0])
    rotation = R.from_euler("z", 90.0, degrees=True)
    for pose_filter in (fixed, adaptive):
        pose_filter.filter(np.zeros(3), R.identity(), 0.0)

    fixed_position, fixed_rotation = fixed.filter(position, rotation, 1.0 / 30.0)
    adaptive_position, adaptive_rotation = adaptive.filter(
        position, rotation, 1.0 / 30.0
    )

    assert 0.0 < fixed_position[0] < adaptive_position[0] < position[0]
    assert 0.0 < fixed_rotation.magnitude() < adaptive_rotation.magnitude() < np.pi / 2


def test_direct_action_returns_full_tcp_error():
    expert = pico_expert.PicoExpert.__new__(pico_expert.PicoExpert)
    expert.hand = "left"
    expert.control_trigger = "grip"
    expert.control_threshold = 0.5
    expert.calibration_enabled = False
    expert.require_calibration = False
    expert._calibrated = True
    expert._active = True
    expert._trajectory_filter = None
    expert._last_action = np.zeros(7, dtype=np.float32)
    expert._snapshot = lambda: {}
    expert._maybe_update_calibration = lambda data: None
    expert._controller_pose = lambda data, hand: np.array([0, 0, 0, 0, 0, 0, 1])
    expert._control_value = lambda data, hand, trigger: 1.0
    expert._transform_raw_pose_to_world = lambda pose: (np.zeros(3), R.identity())
    target_pos = np.array([0.5, -0.4, 0.3])
    target_rot = R.from_rotvec([0.8, 0.0, 0.0])
    expert._target_tcp_pose = lambda position, rotation: (target_pos, target_rot)

    action, replaced, _ = expert.get_action(
        np.array([0, 0, 0, 0, 0, 0, 1]),
        None,
        gripper_enabled=False,
        direct=True,
    )

    assert replaced
    np.testing.assert_allclose(action, [0.5, -0.4, 0.3, 0.8, 0.0, 0.0])

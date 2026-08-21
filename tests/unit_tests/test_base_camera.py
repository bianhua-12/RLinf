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

import queue
import threading
import time

import numpy as np
import pytest

from rlinf.envs.realworld.common.camera.base_camera import BaseCamera, CameraHealth


class _BlockingCaptureCamera(BaseCamera):
    def __init__(self):
        # Deliberately omit fps: blocking SDK reads, not a software sleep, pace capture.
        super().__init__(camera_info=type("CameraInfo", (), {"name": "test"})())

    def _read_frame(self):
        self._frame_capturing_start = False
        return True, np.zeros((2, 2, 3), dtype=np.uint8)

    def _close_device(self):
        self.closed = True


def test_capture_loop_does_not_apply_an_additional_software_rate_limit():
    camera = _BlockingCaptureCamera()
    camera.closed = False
    camera._frame_capturing_start = True

    camera._capture_frames()

    assert camera._frame_queue.qsize() == 1


class _TransientFailureCamera(BaseCamera):
    _READ_RETRY_DELAY_S = 0.0

    def __init__(self, outcomes):
        super().__init__(camera_info=type("CameraInfo", (), {"name": "flaky"})())
        self.outcomes = list(outcomes)
        self.calls = 0
        self.closed = False

    def _read_frame(self):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        if not outcome:
            return False, None
        self._frame_capturing_start = False
        return True, np.ones((2, 2, 3), dtype=np.uint8)

    def _close_device(self):
        self.closed = True


@pytest.mark.parametrize(
    "outcomes", [[False, True], [RuntimeError("transient"), True], [False, False, True]]
)
def test_capture_loop_recovers_and_clears_real_read_failures(outcomes):
    camera = _TransientFailureCamera(outcomes=outcomes)
    camera._frame_capturing_start = True

    camera._capture_frames()

    assert camera.calls == len(outcomes)
    snapshot = camera.snapshot()
    assert snapshot.health == CameraHealth.HEALTHY
    assert snapshot.consecutive_read_failures == 0
    assert snapshot.last_success_monotonic is not None
    np.testing.assert_array_equal(
        camera._frame_queue.get_nowait(), np.ones((2, 2, 3), dtype=np.uint8)
    )


def test_capture_loop_stops_after_bounded_failures():
    camera = _TransientFailureCamera(outcomes=[False] * 10)
    camera._frame_capturing_start = True

    camera._capture_frames()

    assert camera.calls == camera._MAX_CONSECUTIVE_READ_FAILURES
    assert camera._frame_queue.empty()
    snapshot = camera.snapshot()
    assert snapshot.health == CameraHealth.FAILED
    assert snapshot.consecutive_read_failures == 3
    assert snapshot.last_failure == "_read_frame returned no frame"


def test_queue_miss_does_not_change_capture_failure_count():
    camera = _TransientFailureCamera(outcomes=[True])
    camera._frame_capturing_start = True
    camera._capture_frames()
    camera._frame_queue.get_nowait()

    with pytest.raises(queue.Empty):
        camera.get_frame(timeout=0.0)

    assert camera.snapshot().consecutive_read_failures == 0


class _CloseOrderingCamera(BaseCamera):
    _CAPTURE_THREAD_JOIN_TIMEOUT_S = 0.2

    def __init__(self):
        info = type("CameraInfo", (), {"name": "close-order"})()
        super().__init__(camera_info=info)
        self.read_started = threading.Event()
        self.release_read = threading.Event()
        self.read_finished = threading.Event()
        self.device_closed = threading.Event()

    def _read_frame(self):
        self.read_started.set()
        self.release_read.wait(timeout=1.0)
        self.read_finished.set()
        return False, None

    def _close_device(self):
        assert self.read_finished.is_set()
        self.device_closed.set()


def test_close_joins_capture_before_destroying_device():
    camera = _CloseOrderingCamera()
    camera.open()
    assert camera.read_started.wait(timeout=1.0)
    closer = threading.Thread(target=camera.close)
    closer.start()
    time.sleep(0.02)
    assert not camera.device_closed.is_set()

    camera.release_read.set()
    closer.join(timeout=1.0)

    assert not closer.is_alive()
    assert camera.device_closed.is_set()
    assert camera.health == CameraHealth.CLOSED
    camera.close()


def test_close_timeout_retains_device_handle_until_read_exits():
    camera = _CloseOrderingCamera()
    camera._CAPTURE_THREAD_JOIN_TIMEOUT_S = 0.02
    camera.open()
    assert camera.read_started.wait(timeout=1.0)

    with pytest.raises(TimeoutError, match="device handle retained"):
        camera.close()
    assert not camera.device_closed.is_set()

    camera.release_read.set()
    camera._frame_capturing_thread.join(timeout=1.0)
    camera.close()
    assert camera.device_closed.is_set()

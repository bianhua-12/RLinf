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

import numpy as np

from rlinf.envs.realworld.common.camera.base_camera import BaseCamera


class _BlockingCaptureCamera(BaseCamera):
    def __init__(self):
        # Deliberately omit fps: blocking SDK reads, not a software sleep, pace capture.
        super().__init__(camera_info=type("CameraInfo", (), {"name": "test"})())

    def _read_frame(self):
        self._frame_capturing_start = False
        return True, np.zeros((2, 2, 3), dtype=np.uint8)

    def _close_device(self):
        pass


def test_capture_loop_does_not_apply_an_additional_software_rate_limit():
    camera = _BlockingCaptureCamera()
    camera._frame_capturing_start = True

    camera._capture_frames()

    assert camera._frame_queue.qsize() == 1

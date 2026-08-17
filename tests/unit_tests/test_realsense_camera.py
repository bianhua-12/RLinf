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

from rlinf.envs.realworld.common.camera.realsense_camera import RealSenseCamera


class _ColorFrame:
    def __init__(self, image: np.ndarray):
        self._image = image

    def is_video_frame(self) -> bool:
        return True

    def get_data(self) -> np.ndarray:
        return self._image


class _FrameSet:
    def __init__(self, color_frame: _ColorFrame):
        self._color_frame = color_frame

    def get_color_frame(self) -> _ColorFrame:
        return self._color_frame


class _Pipeline:
    def __init__(self, frames: _FrameSet):
        self._frames = frames

    def wait_for_frames(self) -> _FrameSet:
        return self._frames


class _UnexpectedAlign:
    def process(self, frames):
        del frames
        raise AssertionError("RGB-only capture must not align frames")


def test_read_frame_skips_alignment_when_depth_is_disabled():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    camera = RealSenseCamera.__new__(RealSenseCamera)
    camera._enable_depth = False
    camera._pipeline = _Pipeline(_FrameSet(_ColorFrame(image)))
    camera._align = _UnexpectedAlign()

    success, frame = camera._read_frame()

    assert success
    assert frame is image

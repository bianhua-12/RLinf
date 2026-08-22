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

import sys
from types import SimpleNamespace

import numpy as np

from rlinf.envs.realworld.common.camera.base_camera import CameraInfo
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


class _Config:
    def __init__(self):
        self.serial_number = None
        self.streams = []

    def enable_device(self, serial_number):
        self.serial_number = serial_number

    def enable_stream(self, *args):
        self.streams.append(args)


class _StartingPipeline:
    def __init__(self):
        self.config = None

    def start(self, config):
        self.config = config
        return "profile"


def test_read_frame_skips_alignment_when_depth_is_disabled():
    image = np.zeros((4, 5, 3), dtype=np.uint8)
    camera = RealSenseCamera.__new__(RealSenseCamera)
    camera._enable_depth = False
    camera._pipeline = _Pipeline(_FrameSet(_ColorFrame(image)))
    camera._align = _UnexpectedAlign()

    success, frame = camera._read_frame()

    assert success
    assert frame is image


def test_init_binds_target_without_enumerating_all_devices(monkeypatch):
    pipeline = _StartingPipeline()

    def unexpected_context():
        raise AssertionError("initialization must not enumerate all devices")

    fake_rs = SimpleNamespace(
        pipeline=lambda: pipeline,
        config=_Config,
        context=unexpected_context,
        stream=SimpleNamespace(color="color", depth="depth"),
        format=SimpleNamespace(bgr8="bgr8", z16="z16"),
        align=lambda stream: ("align", stream),
    )
    monkeypatch.setitem(sys.modules, "pyrealsense2", fake_rs)
    camera_info = CameraInfo(
        name="left_wrist_0_rgb",
        serial_number="261922076829",
        resolution=(640, 480),
        fps=30,
    )

    camera = RealSenseCamera(camera_info)

    assert camera._serial_number == "261922076829"
    assert pipeline.config.serial_number == "261922076829"
    assert pipeline.config.streams == [("color", 640, 480, "bgr8", 30)]
    assert camera.profile == "profile"
    assert camera._align is None

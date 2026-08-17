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

import math
from types import SimpleNamespace
from unittest.mock import MagicMock, call

from rlinf.envs.realworld.common.camera import CameraInfo, create_camera
from rlinf.envs.realworld.common.camera.base_camera import BaseCamera
from rlinf.envs.realworld.common.camera.hikrobot_camera import HikrobotCamera


def test_create_camera_dispatches_hikrobot_without_loading_sdk(monkeypatch):
    monkeypatch.setattr(
        HikrobotCamera,
        "__init__",
        lambda self, camera_info: BaseCamera.__init__(self, camera_info),
    )

    camera = create_camera(
        CameraInfo(
            name="base_0_rgb",
            serial_number="DA6135161",
            camera_type="hikrobot",
        )
    )

    assert isinstance(camera, HikrobotCamera)


def test_hikrobot_undistort_maps_are_square_with_100_degree_fov():
    import cv2

    map1, map2 = HikrobotCamera._create_undistort_maps(cv2)
    output_matrix = HikrobotCamera._output_camera_matrix()
    focal_length = output_matrix[0, 0]
    half_size = HikrobotCamera._OUTPUT_SIZE / 2
    field_of_view = math.degrees(2 * math.atan(half_size / focal_length))
    scale = HikrobotCamera._OUTPUT_SIZE / HikrobotCamera._REFERENCE_OUTPUT_SIZE

    assert map1.shape[:2] == (224, 224)
    assert map2.shape == (224, 224)
    assert math.isclose(
        focal_length,
        HikrobotCamera._K_100_SQUARE_REFERENCE[0, 0] * scale,
    )
    assert math.isclose(output_matrix[0, 2], half_size)
    assert math.isclose(output_matrix[1, 2], half_size)
    assert math.isclose(field_of_view, 100.0)


def test_hikrobot_enables_continuous_auto_features():
    calls = []
    camera = HikrobotCamera.__new__(HikrobotCamera)
    camera._sdk = SimpleNamespace(MV_OK=0)
    camera._camera = SimpleNamespace(
        MV_CC_SetEnumValueByString=lambda key, value: calls.append((key, value)) or 0,
    )

    camera._enable_continuous_auto_features()

    assert calls == [
        ("ExposureAuto", "Continuous"),
        ("GainAuto", "Continuous"),
        ("BalanceWhiteAuto", "Continuous"),
    ]


def test_configure_frame_rate_enables_limit_and_reads_result():
    camera = HikrobotCamera.__new__(HikrobotCamera)
    camera._sdk = SimpleNamespace(
        MV_OK=0,
        MVCC_FLOATVALUE=lambda: SimpleNamespace(fCurValue=29.97),
    )
    camera._camera = MagicMock()
    camera._camera.MV_CC_SetBoolValue.return_value = 0
    camera._camera.MV_CC_SetFloatValue.return_value = 0
    camera._camera.MV_CC_GetFloatValue.return_value = 0

    camera._configure_frame_rate(30)

    assert camera._camera.method_calls == [
        call.MV_CC_SetBoolValue("AcquisitionFrameRateEnable", True),
        call.MV_CC_SetFloatValue("AcquisitionFrameRate", 30),
        call.MV_CC_GetFloatValue(
            "ResultingFrameRate", camera._camera.MV_CC_GetFloatValue.call_args.args[1]
        ),
    ]
    assert camera._resulting_frame_rate == 29.97

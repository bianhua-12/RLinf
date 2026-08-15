# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

import math
from types import SimpleNamespace

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
    focal_length = HikrobotCamera._K_100_SQUARE[0, 0]
    half_size = HikrobotCamera._OUTPUT_SIZE / 2
    field_of_view = math.degrees(2 * math.atan(half_size / focal_length))

    assert map1.shape[:2] == (1080, 1080)
    assert map2.shape == (1080, 1080)
    assert math.isclose(field_of_view, 100.0)


def test_hikrobot_enables_continuous_auto_features():
    calls = []
    camera = HikrobotCamera.__new__(HikrobotCamera)
    camera._sdk = SimpleNamespace(MV_OK=0)
    camera._camera = SimpleNamespace(
        MV_CC_SetEnumValueByString=lambda key, value: calls.append(
            (key, value)
        )
        or 0,
    )

    camera._enable_continuous_auto_features()

    assert calls == [
        ("ExposureAuto", "Continuous"),
        ("GainAuto", "Continuous"),
        ("BalanceWhiteAuto", "Continuous"),
    ]

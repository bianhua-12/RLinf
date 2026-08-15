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

"""Hikrobot MVS capture with square 100-degree fisheye rectification."""

from __future__ import annotations

import importlib
import os
import sys
from ctypes import POINTER, c_ubyte, cast
from types import ModuleType
from typing import Optional

import numpy as np

from rlinf.utils.logging import get_logger

from .base_camera import BaseCamera, CameraInfo

_MVS_BINDING_PATH = "/opt/MVS/Samples/64/Python/MvImport"
_MVS_RUNTIME_PATH = "/opt/MVS/lib"
_logger = get_logger()


def _load_mvs_sdk() -> ModuleType:
    os.environ.setdefault("MVCAM_COMMON_RUNENV", _MVS_RUNTIME_PATH)
    if _MVS_BINDING_PATH not in sys.path:
        sys.path.insert(0, _MVS_BINDING_PATH)
    try:
        return importlib.import_module("MvCameraControl_class")
    except (ModuleNotFoundError, OSError) as exc:
        raise ModuleNotFoundError(
            "Hikrobot MVS SDK is required. Install MVS under /opt/MVS."
        ) from exc


class HikrobotCamera(BaseCamera):
    """Capture and rectify the calibrated Hikrobot base fisheye camera."""

    _SERIAL_NUMBER = "DA6135161"
    _NATIVE_W = 1440
    _NATIVE_H = 1080
    _OUTPUT_SIZE = 1080
    _K = np.array(
        [
            [329.3108299651, 0.0, 755.1510914205],
            [0.0, 329.0356913045, 546.5887611416],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    _D = np.array(
        [-0.0112917342, -0.0066525146, -0.0019194655, 0.0005036831],
        dtype=np.float64,
    )
    _K_100_SQUARE = np.array(
        [
            [453.1138008357, 0.0, 540.0],
            [0.0, 453.1138008357, 540.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    def __init__(self, camera_info: CameraInfo):
        import cv2

        super().__init__(camera_info)
        if camera_info.enable_depth:
            raise ValueError("HikrobotCamera does not support depth capture.")
        if camera_info.serial_number != self._SERIAL_NUMBER:
            raise ValueError(
                "The bundled fisheye calibration is only valid for Hikrobot camera "
                f"serial {self._SERIAL_NUMBER}, got {camera_info.serial_number}."
            )

        self._cv2 = cv2
        self._map1, self._map2 = self._create_undistort_maps(cv2)
        self._sdk = _load_mvs_sdk()
        self._camera = None
        self._sdk_initialized = False
        self._handle_created = False
        self._device_open = False
        self._grabbing = False
        self._buffer = (c_ubyte * (self._NATIVE_W * self._NATIVE_H * 3))()
        self._frame_info = self._sdk.MV_FRAME_OUT_INFO_EX()

        try:
            self._check(self._sdk.MvCamera.MV_CC_Initialize(), "initialize SDK")
            self._sdk_initialized = True
            device = self._find_device(camera_info.serial_number)

            self._camera = self._sdk.MvCamera()
            self._check(self._camera.MV_CC_CreateHandle(device), "create handle")
            self._handle_created = True
            self._check(
                self._camera.MV_CC_OpenDevice(self._sdk.MV_ACCESS_Exclusive, 0),
                "open device",
            )
            self._device_open = True
            self._check(
                self._camera.MV_CC_SetEnumValue(
                    "TriggerMode", self._sdk.MV_TRIGGER_MODE_OFF
                ),
                "disable trigger mode",
            )
            self._enable_continuous_auto_features()
            self._check(self._camera.MV_CC_StartGrabbing(), "start grabbing")
            self._grabbing = True
        except Exception:
            self._close_device()
            raise

    @classmethod
    def _create_undistort_maps(cls, cv2):
        return cv2.fisheye.initUndistortRectifyMap(
            cls._K,
            cls._D,
            np.eye(3),
            cls._K_100_SQUARE,
            (cls._OUTPUT_SIZE, cls._OUTPUT_SIZE),
            cv2.CV_16SC2,
        )

    def _find_device(self, serial_number: str):
        devices = self._sdk.MV_CC_DEVICE_INFO_LIST()
        self._check(
            self._sdk.MvCamera.MV_CC_EnumDevices(self._sdk.MV_USB_DEVICE, devices),
            "enumerate USB cameras",
        )
        available: list[str] = []
        for index in range(devices.nDeviceNum):
            device = cast(
                devices.pDeviceInfo[index], POINTER(self._sdk.MV_CC_DEVICE_INFO)
            ).contents
            serial = (
                bytes(device.SpecialInfo.stUsb3VInfo.chSerialNumber)
                .split(b"\0", 1)[0]
                .decode()
            )
            available.append(serial)
            if serial == serial_number:
                return device
        raise ValueError(
            f"Hikrobot camera serial {serial_number} is not connected. "
            f"Available MVS USB cameras: {available}."
        )

    def _check(self, result: int, operation: str) -> None:
        if result != self._sdk.MV_OK:
            raise RuntimeError(f"MVS failed to {operation}: {result:#x}")

    def _enable_continuous_auto_features(self) -> None:
        for feature in ("ExposureAuto", "GainAuto", "BalanceWhiteAuto"):
            result = self._camera.MV_CC_SetEnumValueByString(feature, "Continuous")
            if result != self._sdk.MV_OK:
                _logger.warning(
                    "Failed to set Hikrobot %s=Continuous: %#x", feature, result
                )

    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        result = self._camera.MV_CC_GetImageForBGR(
            self._buffer, len(self._buffer), self._frame_info, 1000
        )
        if result != self._sdk.MV_OK:
            return False, None
        if (self._frame_info.nWidth, self._frame_info.nHeight) != (
            self._NATIVE_W,
            self._NATIVE_H,
        ):
            raise RuntimeError(
                "Hikrobot fisheye calibration requires 1440x1080 frames, got "
                f"{self._frame_info.nWidth}x{self._frame_info.nHeight}."
            )

        raw = np.ctypeslib.as_array(self._buffer).reshape(
            self._NATIVE_H, self._NATIVE_W, 3
        )
        frame = self._cv2.remap(
            raw, self._map1, self._map2, interpolation=self._cv2.INTER_LINEAR
        )
        return True, frame

    def _close_device(self) -> None:
        if self._camera is not None:
            if self._grabbing:
                self._camera.MV_CC_StopGrabbing()
                self._grabbing = False
            if self._device_open:
                self._camera.MV_CC_CloseDevice()
                self._device_open = False
            if self._handle_created:
                self._camera.MV_CC_DestroyHandle()
                self._handle_created = False
        if self._sdk_initialized:
            self._sdk.MvCamera.MV_CC_Finalize()
            self._sdk_initialized = False

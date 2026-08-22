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
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from rlinf.utils.logging import get_logger

_logger = get_logger()


@dataclass
class CameraInfo:
    """Descriptor for a single camera device."""

    name: str
    serial_number: str
    camera_type: str = "realsense"
    resolution: tuple[int, int] = (640, 480)
    fps: int = 15
    enable_depth: bool = False
    crop_region: Optional[tuple[float, float, float, float]] = None


class CameraHealth(str, Enum):
    """Lifecycle and acquisition health reported by the capture thread."""

    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPING = "stopping"
    CLOSED = "closed"


@dataclass(frozen=True)
class CameraSnapshot:
    """Thread-safe non-blocking snapshot of the latest camera state."""

    frame: Optional[np.ndarray]
    frame_timestamp_monotonic: Optional[float]
    frame_sequence: int
    health: CameraHealth
    consecutive_read_failures: int
    last_success_monotonic: Optional[float]
    last_failure: Optional[str]
    last_failure_monotonic: Optional[float]


class BaseCamera(ABC):
    """Abstract base class for threaded camera capture.

    Subclasses must implement ``_read_frame`` (hardware-specific frame
    acquisition) and ``_close_device`` (hardware-specific cleanup).
    The threading, queue management, and public API (``open``, ``close``,
    ``get_frame``) are handled here.
    """

    _MAX_CONSECUTIVE_READ_FAILURES = 3
    _READ_RETRY_DELAY_S = 0.01
    _CAPTURE_THREAD_JOIN_TIMEOUT_S = 2.0

    def __init__(self, camera_info: CameraInfo):
        self._camera_info = camera_info
        self._frame_queue: queue.Queue = queue.Queue()
        self._frame_capturing_thread = threading.Thread(
            target=self._capture_frames,
            name=f"camera_capture_{camera_info.name}",
            daemon=True,
        )
        self._frame_capturing_start = False
        self._state_lock = threading.Lock()
        self._first_frame_ready = threading.Condition(self._state_lock)
        self._health = CameraHealth.CLOSED
        self._consecutive_read_failures = 0
        self._last_success_monotonic: float | None = None
        self._last_failure: str | None = None
        self._last_failure_monotonic: float | None = None
        self._latest_frame: np.ndarray | None = None
        self._latest_frame_timestamp_monotonic: float | None = None
        self._frame_sequence = 0
        self._device_closed = False

    @property
    def name(self) -> str:
        return self._camera_info.name

    def open(self):
        """Start the background frame-capturing thread."""
        with self._state_lock:
            if self._health not in (CameraHealth.CLOSED, CameraHealth.STARTING):
                return
            self._health = CameraHealth.STARTING
            self._device_closed = False
        self._frame_capturing_start = True
        self._frame_capturing_thread.start()

    def close(self):
        """Stop capture before releasing a handle that a read may still use."""
        with self._state_lock:
            if self._health == CameraHealth.CLOSED and self._device_closed:
                return
            self._health = CameraHealth.STOPPING
        self._frame_capturing_start = False
        if self._frame_capturing_thread.is_alive():
            self._frame_capturing_thread.join(
                timeout=self._CAPTURE_THREAD_JOIN_TIMEOUT_S
            )
        if self._frame_capturing_thread.is_alive():
            with self._state_lock:
                self._health = CameraHealth.FAILED
            raise TimeoutError(
                f"camera {self.name} capture thread did not stop within "
                f"{self._CAPTURE_THREAD_JOIN_TIMEOUT_S:.3f}s; device handle retained"
            )
        if not self._device_closed:
            self._close_device()
            self._device_closed = True
        with self._state_lock:
            self._health = CameraHealth.CLOSED

    def get_frame(self, timeout: float = 5) -> np.ndarray:
        """Return the most recent frame (blocks up to *timeout* seconds).

        Args:
            timeout: Maximum seconds to wait for a frame.
        """
        assert self.health not in (CameraHealth.CLOSED, CameraHealth.STOPPING), (
            "Frame capturing is not started. Call open() first."
        )
        return self._frame_queue.get(timeout=timeout)

    @property
    def health(self) -> CameraHealth:
        """Current capture-thread-owned camera health."""
        with self._state_lock:
            return self._health

    def snapshot(self) -> CameraSnapshot:
        """Return the latest frame and diagnostics without waiting for I/O."""
        with self._state_lock:
            return CameraSnapshot(
                frame=self._latest_frame,
                frame_timestamp_monotonic=self._latest_frame_timestamp_monotonic,
                frame_sequence=self._frame_sequence,
                health=self._health,
                consecutive_read_failures=self._consecutive_read_failures,
                last_success_monotonic=self._last_success_monotonic,
                last_failure=self._last_failure,
                last_failure_monotonic=self._last_failure_monotonic,
            )

    def wait_for_first_frame(self, timeout: float) -> CameraSnapshot:
        """Wait up to one shared-deadline remainder for the first valid frame."""
        deadline = time.monotonic() + timeout
        with self._first_frame_ready:
            while self._latest_frame is None:
                if self._health == CameraHealth.FAILED:
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._first_frame_ready.wait(timeout=remaining)
        snapshot = self.snapshot()
        if snapshot.frame is None:
            raise TimeoutError(
                f"camera {self.name} produced no valid frame within {timeout:.3f}s; "
                f"health={snapshot.health.value}, last_failure={snapshot.last_failure}"
            )
        return snapshot

    @property
    def capture_thread_alive(self) -> bool:
        """Whether the background acquisition loop is still running."""
        return self._frame_capturing_thread.is_alive()

    # ── internal ──────────────────────────────────────────────────────

    def _capture_frames(self):
        while self._frame_capturing_start:
            failure = None
            try:
                has_frame, frame = self._read_frame()
            except Exception as e:
                failure = f"{type(e).__name__}: {e}"
                _logger.warning(
                    "[%s] _read_frame raised %s.",
                    self._camera_info.name,
                    failure,
                )
                has_frame, frame = False, None
            if not has_frame or frame is None:
                if failure is None:
                    failure = "_read_frame returned no frame"
                now = time.monotonic()
                with self._state_lock:
                    self._consecutive_read_failures += 1
                    self._last_failure = failure
                    self._last_failure_monotonic = now
                    if (
                        self._consecutive_read_failures
                        >= self._MAX_CONSECUTIVE_READ_FAILURES
                    ):
                        self._health = CameraHealth.FAILED
                    else:
                        self._health = CameraHealth.DEGRADED
                    failure_count = self._consecutive_read_failures
                    self._first_frame_ready.notify_all()
                if failure_count >= self._MAX_CONSECUTIVE_READ_FAILURES:
                    _logger.error(
                        "[%s] camera read failed %d consecutive times; "
                        "exiting capture thread.",
                        self._camera_info.name,
                        failure_count,
                    )
                    break
                time.sleep(self._READ_RETRY_DELAY_S)
                continue
            now = time.monotonic()
            with self._state_lock:
                self._consecutive_read_failures = 0
                self._last_success_monotonic = now
                self._latest_frame = frame
                self._latest_frame_timestamp_monotonic = now
                self._frame_sequence += 1
                self._health = CameraHealth.HEALTHY
                self._first_frame_ready.notify_all()
            if not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
            self._frame_queue.put(frame)
        self._frame_capturing_start = False

    @abstractmethod
    def _read_frame(self) -> tuple[bool, Optional[np.ndarray]]:
        """Read a single frame from the camera hardware.

        Returns:
            ``(success, frame)`` where *frame* is a BGR ``uint8`` numpy array,
            or ``(False, None)`` on failure.
        """
        raise NotImplementedError

    @abstractmethod
    def _close_device(self) -> None:
        """Release hardware-specific resources (pipeline, SDK handle, …)."""
        raise NotImplementedError

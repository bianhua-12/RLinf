# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Crash-recoverable, direct-to-MP4 LeRobot v3 episode writer.

The stock LeRobot 0.3.4 writer first materializes every video frame as PNG and
encodes those PNGs when an episode (or dataset) is finalized.  That makes a
controller restart proportional to all previously collected pixels.  This
module instead streams synchronized camera groups to AV1 files while the robot
is running, then publishes one immutable parquet/MP4 group per episode.

Publishing is transaction-like: closed media and parquet live below
``.streaming/partial`` until a ``manifest.ready.json`` marker is atomically
written.  Recovery can replay that manifest idempotently; episode metadata is
the final commit marker and is rebuilt from manifests on every recovery.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import multiprocessing as mp
import os
import queue
import shutil
import time
import traceback
from fractions import Fraction
from pathlib import Path
from typing import Any

import numpy as np

from rlinf.utils.logging import get_logger

PROTOCOL_ID = "rlinf_lerobot_stream_mp4_v1"
SUPPORTED_LEROBOT_VERSION = "0.3.4"
DEFAULT_QUEUE_SIZE = 60
DEFAULT_MAX_PENDING_COMMITS = 2
DEFAULT_COMMIT_WATCHDOG_TIMEOUT_S = 30.0
ENCODING_CONTRACT = {
    "container": "mp4",
    "encoder": "libsvtav1",
    "codec": "av1",
    "crf": 30,
    "pixel_format": "yuv420p",
    "gop": 2,
    "preset": 8,
}


def _require_supported_lerobot_version() -> None:
    version = importlib.metadata.version("lerobot")
    if version != SUPPORTED_LEROBOT_VERSION:
        raise RuntimeError(
            f"{PROTOCOL_ID} requires lerobot=={SUPPORTED_LEROBOT_VERSION}; "
            f"found {version}. The writer uses LeRobot v3 private schema APIs."
        )


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _numpy_stats(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _numpy_stats(item) for key, item in value.items()}
    if isinstance(value, list):
        return np.asarray(value)
    return value


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(_jsonable(payload), handle, indent=2, sort_keys=True)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sampled_image_stats(per_frame: list[dict[str, np.ndarray]]) -> dict[str, Any]:
    from lerobot.datasets.compute_stats import sample_indices

    indices = sample_indices(len(per_frame))
    selected = [per_frame[index] for index in indices]
    pixel_count = sum(int(item["pixel_count"]) for item in selected)
    total = np.sum([item["sum"] for item in selected], axis=0, dtype=np.float64)
    square_total = np.sum(
        [item["square_sum"] for item in selected], axis=0, dtype=np.float64
    )
    mean = total / pixel_count
    variance = np.maximum(square_total / pixel_count - np.square(mean), 0.0)
    minimum = np.min([item["min"] for item in selected], axis=0)
    maximum = np.max([item["max"] for item in selected], axis=0)

    # Match LeRobot's normalized Cx1x1 video/image statistics.
    def reshape(value):
        return np.asarray(value, dtype=np.float64).reshape(-1, 1, 1) / 255.0

    return {
        "min": reshape(minimum),
        "max": reshape(maximum),
        "mean": reshape(mean),
        "std": reshape(np.sqrt(variance)),
        "count": np.asarray([len(indices)]),
    }


def _encoder_worker(command_queue, result_connection) -> None:
    """Own PyAV containers in a spawned process and preserve command order."""
    import av

    containers: dict[str, Any] = {}
    streams: dict[str, Any] = {}
    stats: dict[str, list[dict[str, np.ndarray]]] = {}
    frame_count = 0
    active = False

    def close_containers(flush: bool) -> None:
        nonlocal containers, streams
        for key, container in containers.items():
            if flush:
                for packet in streams[key].encode():
                    container.mux(packet)
            container.close()
        containers = {}
        streams = {}

    try:
        while True:
            command = command_queue.get()
            kind = command[0]
            if kind == "start":
                if active:
                    raise RuntimeError(
                        "encoder received start while an episode is active"
                    )
                _, output_dir, camera_shapes, fps = command
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                frame_count = 0
                stats = {key: [] for key in camera_shapes}
                for key, shape in camera_shapes.items():
                    height, width, channels = shape
                    if channels != 3:
                        raise ValueError(f"{key} must be HxWx3, got {shape}")
                    path = output_dir / f"{key}.mp4"
                    container = av.open(
                        str(path), mode="w", options={"movflags": "+faststart"}
                    )
                    stream = container.add_stream(
                        ENCODING_CONTRACT["encoder"],
                        rate=fps,
                        options={
                            "crf": str(ENCODING_CONTRACT["crf"]),
                            "g": str(ENCODING_CONTRACT["gop"]),
                            "preset": str(ENCODING_CONTRACT["preset"]),
                        },
                    )
                    stream.width = width
                    stream.height = height
                    stream.pix_fmt = ENCODING_CONTRACT["pixel_format"]
                    stream.time_base = Fraction(1, fps)
                    containers[key] = container
                    streams[key] = stream
                active = True
                result_connection.send(("started", None))
            elif kind == "frame":
                if not active:
                    raise RuntimeError("encoder received frame without active episode")
                _, index, images = command
                if index != frame_count:
                    raise RuntimeError(
                        f"non-contiguous encoder frame: expected {frame_count}, got {index}"
                    )
                for key, image in images.items():
                    array = np.asarray(image, dtype=np.uint8)
                    frame = av.VideoFrame.from_ndarray(array, format="rgb24")
                    for packet in streams[key].encode(frame):
                        containers[key].mux(packet)
                    flat = array.reshape(-1, 3).astype(np.float64)
                    stats[key].append(
                        {
                            "min": flat.min(axis=0),
                            "max": flat.max(axis=0),
                            "sum": flat.sum(axis=0),
                            "square_sum": np.square(flat).sum(axis=0),
                            "pixel_count": np.asarray(flat.shape[0], dtype=np.int64),
                        }
                    )
                frame_count += 1
            elif kind == "finish":
                _, expected_frames = command
                if not active or frame_count != expected_frames:
                    raise RuntimeError(
                        f"encoder finish expected {expected_frames} frames, got {frame_count}"
                    )
                close_containers(flush=True)
                image_stats = {
                    key: _sampled_image_stats(values) for key, values in stats.items()
                }
                active = False
                result_connection.send(("finished", {"stats": image_stats}))
            elif kind == "abort":
                if active:
                    close_containers(flush=False)
                active = False
                frame_count = 0
                stats = {}
                result_connection.send(("aborted", None))
            elif kind == "close":
                if active:
                    close_containers(flush=False)
                result_connection.send(("closed", None))
                return
            else:
                raise ValueError(f"unknown encoder command {kind!r}")
    except BaseException as exc:
        try:
            close_containers(flush=False)
        finally:
            result_connection.send(("error", f"{type(exc).__name__}: {exc}"))


def _commit_worker(command_queue, result_queue) -> None:
    """Publish staged episodes in order from a separately killable process."""
    while True:
        command = command_queue.get()
        kind = command[0]
        if kind == "close":
            result_queue.put({"kind": "closed"})
            return
        if kind != "commit":
            raise ValueError(f"unknown stream commit command {kind!r}")

        _, root, partial_dir, manifest = command
        episode_index = int(manifest["episode_index"])
        started_at = time.monotonic()
        try:
            StreamingLeRobotDatasetWriter._publish_ready(
                Path(root),
                Path(partial_dir),
                manifest,
                verify_sources=False,
            )
            StreamingLeRobotDatasetWriter._append_manifest_metadata(
                Path(root), manifest
            )
        except BaseException as exc:
            result_queue.put(
                {
                    "kind": "commit_result",
                    "episode_index": episode_index,
                    "started_at": started_at,
                    "completed_at": time.monotonic(),
                    "success": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            )
            return
        result_queue.put(
            {
                "kind": "commit_result",
                "episode_index": episode_index,
                "started_at": started_at,
                "completed_at": time.monotonic(),
                "success": True,
                "error": None,
                "traceback": None,
            }
        )


class StreamingLeRobotDatasetWriter:
    """Direct MP4 writer pinned to LeRobot 0.3.4 / dataset format v3.0."""

    _COMMIT_WORKER_TARGET = staticmethod(_commit_worker)

    def __init__(
        self,
        queue_size: int = DEFAULT_QUEUE_SIZE,
        max_pending_commits: int = DEFAULT_MAX_PENDING_COMMITS,
        commit_watchdog_timeout: float = DEFAULT_COMMIT_WATCHDOG_TIMEOUT_S,
    ):
        if queue_size <= 0:
            raise ValueError("stream_video_queue_size must be positive")
        if max_pending_commits <= 0:
            raise ValueError("stream_max_pending_commits must be positive")
        if commit_watchdog_timeout <= 0:
            raise ValueError("stream_commit_watchdog_timeout must be positive")
        _require_supported_lerobot_version()
        self.queue_size = queue_size
        self.max_pending_commits = int(max_pending_commits)
        self.commit_watchdog_timeout = float(commit_watchdog_timeout)
        self.dataset = None
        self.root: Path | None = None
        self.fps = 0
        self._frames: list[dict[str, Any]] = []
        self._camera_shapes: dict[str, tuple[int, int, int]] = {}
        self._partial_dir: Path | None = None
        self._active = False
        self._closed = False
        self._next_episode = 0
        self._next_frame_index = 0
        self._tasks: list[str] = []
        self._pending_commits: dict[int, dict[str, float]] = {}
        self._commit_error: RuntimeError | None = None
        self._last_successful_commit_monotonic = time.monotonic()
        self._commit_worker_closed = False
        self._max_pending_commit_count = 0
        self._max_encoder_queue_depth = 0
        self._rejected_frame_count = 0
        self._last_commit_duration = 0.0
        self._context = mp.get_context("spawn")
        self._queue = self._context.Queue(maxsize=queue_size)
        receive, send = self._context.Pipe(duplex=False)
        self._results = receive
        self._process = self._context.Process(
            target=_encoder_worker,
            args=(self._queue, send),
            name="lerobot_stream_mp4_encoder",
        )
        self._process.start()
        send.close()
        self._commit_queue = self._context.Queue(maxsize=self.max_pending_commits)
        self._commit_results = self._context.Queue()
        self._commit_process = self._context.Process(
            target=self._COMMIT_WORKER_TARGET,
            args=(self._commit_queue, self._commit_results),
            name="lerobot_stream_commit",
        )
        self._commit_process.start()
        self.logger = get_logger()

    def create(
        self,
        *,
        repo_id: str,
        robot_type: str,
        fps: int,
        image_shape: tuple[int, int, int] | None,
        state_dim: int,
        action_dim: int,
        has_image: bool,
        wrist_image_keys: dict[str, tuple[int, ...]] | None,
        extra_view_image_keys: dict[str, tuple[int, ...]] | None,
        has_intervene_flag: bool,
        has_segment_id: bool,
        has_observation_timestamp: bool,
    ) -> None:
        from rlinf.data.storage.lerobot.writer import LeRobotDatasetWriter

        if self.dataset is not None:
            raise RuntimeError("streaming dataset already created")
        legacy = LeRobotDatasetWriter()
        legacy.create(
            repo_id=repo_id,
            robot_type=robot_type,
            fps=fps,
            image_shape=image_shape,
            state_dim=state_dim,
            action_dim=action_dim,
            has_image=has_image,
            wrist_image_keys=wrist_image_keys,
            extra_view_image_keys=extra_view_image_keys,
            has_intervene_flag=has_intervene_flag,
            has_segment_id=has_segment_id,
            has_observation_timestamp=has_observation_timestamp,
            use_videos=True,
            image_writer_threads=0,
            image_writer_processes=0,
        )
        self.dataset = legacy.dataset
        self.root = Path(repo_id)
        self.fps = int(fps)
        self.dataset.meta.info["rlinf_streaming_video"] = {
            "protocol_id": PROTOCOL_ID,
            "lerobot_version": SUPPORTED_LEROBOT_VERSION,
            "queue_size_frame_groups": self.queue_size,
            "max_pending_commits": self.max_pending_commits,
            "commit_watchdog_timeout_seconds": self.commit_watchdog_timeout,
            "encoding": dict(ENCODING_CONTRACT),
            "episode_file_layout": "one_file_per_episode",
        }
        _atomic_json(self.root / "meta" / "info.json", self.dataset.meta.info)
        (self.root / ".streaming" / "partial").mkdir(parents=True, exist_ok=True)
        (self.root / ".streaming" / "committed").mkdir(parents=True, exist_ok=True)
        self._rebuild_from_manifests(self.root)
        with (self.root / "meta" / "info.json").open(encoding="utf-8") as handle:
            info = json.load(handle)
        self._next_episode = int(info["total_episodes"])
        self._next_frame_index = int(info["total_frames"])
        for path in sorted(
            (self.root / ".streaming" / "committed").glob("episode_*.json")
        ):
            with path.open(encoding="utf-8") as handle:
                task = json.load(handle)["task"]
            if task not in self._tasks:
                self._tasks.append(task)

    @property
    def active(self) -> bool:
        return self._active

    @property
    def pending_commit_count(self) -> int:
        """Number of staged episodes awaiting durable metadata commit."""
        self._drain_commit_results()
        return len(self._pending_commits)

    @property
    def oldest_pending_commit_age(self) -> float:
        """Age in seconds of the oldest staged but uncommitted episode."""
        if not self._pending_commits:
            return 0.0
        oldest = min(item["queued_at"] for item in self._pending_commits.values())
        return max(0.0, time.monotonic() - oldest)

    @property
    def last_successful_commit_heartbeat(self) -> float:
        """Monotonic timestamp of the last completed commit."""
        return self._last_successful_commit_monotonic

    @property
    def max_pending_commit_count(self) -> int:
        """High-water mark of ready episodes awaiting commit."""
        return self._max_pending_commit_count

    @property
    def max_encoder_queue_depth(self) -> int:
        """Observed high-water mark of queued synchronized frame groups."""
        return self._max_encoder_queue_depth

    @property
    def rejected_frame_count(self) -> int:
        """Number of frame groups rejected because the encoder queue was full."""
        return self._rejected_frame_count

    @property
    def last_commit_duration(self) -> float:
        """Worker-side duration in seconds of the most recent commit."""
        return self._last_commit_duration

    def append_frame(self, frame: dict[str, Any]) -> None:
        if self.dataset is None or self.root is None:
            raise RuntimeError("streaming dataset not created")
        self._raise_worker_error()
        self._check_commit_health(reject_if_full=not self._active)
        video_keys = set(self.dataset.meta.video_keys)
        images = {key: frame[key] for key in video_keys if key in frame}
        if set(images) != video_keys:
            raise ValueError(
                f"camera group mismatch: expected {sorted(video_keys)}, got {sorted(images)}"
            )
        shapes = {key: tuple(np.asarray(value).shape) for key, value in images.items()}
        if not self._active:
            self._camera_shapes = shapes
            episode_index = self._next_episode_index()
            self._partial_dir = (
                self.root / ".streaming" / "partial" / f"episode_{episode_index:06d}"
            )
            if self._partial_dir.exists():
                shutil.rmtree(self._partial_dir)
            self._queue.put_nowait(
                ("start", str(self._partial_dir / "videos"), shapes, self.fps)
            )
            self._expect("started")
            self._active = True
        elif shapes != self._camera_shapes:
            raise ValueError(
                f"camera shape changed mid-episode: {self._camera_shapes} -> {shapes}"
            )

        numeric = {key: value for key, value in frame.items() if key not in video_keys}
        try:
            self._queue.put_nowait(("frame", len(self._frames), images))
        except queue.Full as exc:
            self._rejected_frame_count += 1
            raise RuntimeError(
                f"stream MP4 queue exceeded {self.queue_size} frame groups; "
                "failing closed to prevent silent frame loss"
            ) from exc
        try:
            self._max_encoder_queue_depth = max(
                self._max_encoder_queue_depth, self._queue.qsize()
            )
        except (AttributeError, NotImplementedError):
            pass
        self._frames.append(numeric)

    def finish_episode(self, *, task: str, is_success: bool) -> int:
        if not self._active or self._partial_dir is None or not self._frames:
            raise RuntimeError("cannot finish an empty streaming episode")
        started_at = time.monotonic()
        self._check_commit_health()
        frame_count = len(self._frames)
        self._queue.put(("finish", frame_count), timeout=10)
        result = self._expect("finished", timeout=max(30.0, frame_count / self.fps))
        # The media is now closed. If staging/publish fails after this point,
        # leave any partial/ready transaction on disk for restart recovery;
        # abort_episode must not discard a fully encoded episode.
        self._active = False
        for frame in self._frames:
            frame["is_success"] = np.asarray([is_success], dtype=bool)
            frame["done"] = np.asarray([False], dtype=bool)
        self._frames[-1]["done"] = np.asarray([True], dtype=bool)
        partial_dir = self._partial_dir
        if task not in self._tasks:
            self._tasks.append(task)
        manifest = self._stage_ready_manifest(
            task,
            result["stats"],
            episode_index=self._next_episode,
            dataset_from_index=self._next_frame_index,
            task_index=self._tasks.index(task),
        )
        self._frames = []
        self._camera_shapes = {}
        self._partial_dir = None
        self._next_episode += 1
        self._next_frame_index += frame_count
        staged_seconds = time.monotonic() - started_at
        episode_index = int(manifest["episode_index"])
        self._pending_commits[episode_index] = {
            "queued_at": time.monotonic(),
            "staged_at": started_at,
        }
        self._max_pending_commit_count = max(
            self._max_pending_commit_count, len(self._pending_commits)
        )
        try:
            self._commit_queue.put_nowait(
                ("commit", str(self.root), str(partial_dir), manifest)
            )
        except queue.Full as exc:
            self._commit_error = RuntimeError(
                "stream commit queue is full; ready manifest preserved for "
                f"episode {episode_index}"
            )
            raise self._commit_error from exc
        self.logger.info(
            "Staged streaming LeRobot episode %d (%d frames) in %.3fs; "
            "metadata commit queued",
            manifest["episode_index"],
            frame_count,
            staged_seconds,
        )
        return frame_count

    def _drain_commit_results(self) -> None:
        while True:
            try:
                result = self._commit_results.get_nowait()
            except queue.Empty:
                return
            if result["kind"] == "closed":
                self._commit_worker_closed = True
                continue
            if result["kind"] != "commit_result":
                self._commit_error = RuntimeError(
                    f"unexpected stream commit result: {result!r}"
                )
                continue
            episode_index = int(result["episode_index"])
            if not result["success"]:
                self._commit_error = RuntimeError(
                    f"stream commit failed for episode {episode_index}: "
                    f"{result['error']}\n{result['traceback']}"
                )
                continue
            self._pending_commits.pop(episode_index, None)
            self._last_successful_commit_monotonic = float(result["completed_at"])
            self._last_commit_duration = float(result["completed_at"]) - float(
                result["started_at"]
            )
            if self.root is not None and self.dataset is not None:
                with (self.root / "meta" / "info.json").open(
                    encoding="utf-8"
                ) as handle:
                    self.dataset.meta.info = json.load(handle)
            self.logger.info(
                "Committed streaming LeRobot episode %d asynchronously in %.3fs",
                episode_index,
                self._last_commit_duration,
            )

    def _terminate_commit_process(self) -> None:
        if self._commit_process.is_alive():
            self._commit_process.terminate()
            self._commit_process.join(timeout=2.0)

    def _check_commit_health(self, *, reject_if_full: bool = False) -> None:
        self._drain_commit_results()
        if self._commit_error is not None:
            raise self._commit_error
        if not self._commit_process.is_alive() and not self._commit_worker_closed:
            pending = sorted(self._pending_commits)
            self._commit_error = RuntimeError(
                "stream commit worker exited unexpectedly with code "
                f"{self._commit_process.exitcode}; pending episodes={pending}"
            )
            raise self._commit_error
        age = self.oldest_pending_commit_age
        if self._pending_commits and age > self.commit_watchdog_timeout:
            pending = sorted(self._pending_commits)
            self._terminate_commit_process()
            self._commit_error = RuntimeError(
                "stream commit watchdog expired after "
                f"{age:.3f}s; pending episodes={pending}"
            )
            raise self._commit_error
        if reject_if_full and len(self._pending_commits) >= self.max_pending_commits:
            raise RuntimeError(
                "stream pending commit limit reached "
                f"({len(self._pending_commits)}/{self.max_pending_commits}); "
                f"oldest_age={age:.3f}s"
            )

    def _wait_commits_bounded(self) -> None:
        deadline = time.monotonic() + self.commit_watchdog_timeout
        while self._pending_commits:
            self._check_commit_health()
            if time.monotonic() >= deadline:
                pending = sorted(self._pending_commits)
                self._terminate_commit_process()
                raise TimeoutError(
                    f"timed out finalizing stream commits; pending episodes={pending}"
                )
            time.sleep(0.01)
        self._commit_queue.put(("close",), timeout=1.0)
        self._commit_process.join(timeout=2.0)
        self._drain_commit_results()
        if self._commit_process.is_alive():
            self._terminate_commit_process()
            raise TimeoutError("stream commit worker did not close within 2 seconds")

    def abort_episode(self) -> None:
        if not self._active:
            return
        self._process.join(timeout=0.2)
        if self._process.is_alive():
            self._queue.put(("abort",), timeout=5)
            self._expect("aborted")
        if self._partial_dir is not None and self._partial_dir.exists():
            shutil.rmtree(self._partial_dir)
        self._frames = []
        self._camera_shapes = {}
        self._partial_dir = None
        self._active = False

    def finalize(self, prior_error: BaseException | None = None) -> None:
        del prior_error
        if self._closed:
            return
        first_error: BaseException | None = None
        try:
            self.abort_episode()
        except BaseException as exc:
            first_error = exc
        try:
            if self._process.is_alive():
                self._queue.put(("close",), timeout=5)
                self._expect("closed")
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        try:
            self._wait_commits_bounded()
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        finally:
            self._process.join(timeout=10)
            if self._process.is_alive():
                self._process.terminate()
                self._process.join(timeout=5)
            self._queue.close()
            self._terminate_commit_process()
            self._commit_queue.cancel_join_thread()
            self._commit_results.cancel_join_thread()
            self._commit_queue.close()
            self._commit_results.close()
            self._closed = True
        if first_error is not None:
            raise first_error

    def _next_episode_index(self) -> int:
        return self._next_episode

    def _expect(self, expected: str, timeout: float = 15.0) -> Any:
        if not self._results.poll(timeout):
            self._raise_worker_error()
            raise TimeoutError(f"timed out waiting for streaming encoder {expected}")
        try:
            kind, payload = self._results.recv()
        except EOFError as exc:
            raise RuntimeError(
                f"streaming encoder result pipe closed; exit code {self._process.exitcode}"
            ) from exc
        if kind == "error":
            raise RuntimeError(f"streaming encoder failed: {payload}")
        if kind != expected:
            raise RuntimeError(f"expected encoder {expected}, got {kind}")
        return payload

    def _raise_worker_error(self) -> None:
        if self._results.poll():
            try:
                kind, payload = self._results.recv()
            except EOFError as exc:
                raise RuntimeError(
                    "streaming encoder exited unexpectedly with code "
                    f"{self._process.exitcode}"
                ) from exc
            if kind == "error":
                raise RuntimeError(f"streaming encoder failed: {payload}")
            raise RuntimeError(f"unexpected encoder result {kind}: {payload}")
        if not self._process.is_alive():
            raise RuntimeError(
                f"streaming encoder exited unexpectedly with code {self._process.exitcode}"
            )

    def _stage_ready_manifest(
        self,
        task: str,
        image_stats: dict[str, dict[str, Any]],
        *,
        episode_index: int,
        dataset_from_index: int,
        task_index: int,
    ) -> dict[str, Any]:
        import datasets
        from lerobot.datasets.compute_stats import compute_episode_stats

        frame_count = len(self._frames)
        tasks = [task] * frame_count
        episode_buffer: dict[str, Any] = {}
        for key, feature in self.dataset.features.items():
            if feature["dtype"] in ("image", "video"):
                continue
            if key == "index":
                episode_buffer[key] = np.arange(
                    dataset_from_index,
                    dataset_from_index + frame_count,
                )
            elif key == "frame_index":
                episode_buffer[key] = np.arange(frame_count)
            elif key == "episode_index":
                episode_buffer[key] = np.full(frame_count, episode_index)
            elif key == "task_index":
                episode_buffer[key] = np.full(frame_count, task_index)
            elif key == "timestamp":
                episode_buffer[key] = (
                    np.arange(frame_count, dtype=np.float64) / self.fps
                )
            else:
                episode_buffer[key] = np.stack([frame[key] for frame in self._frames])

        numeric_stats = compute_episode_stats(episode_buffer, self.dataset.features)
        episode_stats = dict(numeric_stats)
        episode_stats.update(image_stats)
        hf_dict = {key: episode_buffer[key] for key in self.dataset.hf_features}
        hf_dataset = datasets.Dataset.from_dict(
            hf_dict, features=self.dataset.hf_features, split="train"
        )
        parquet_path = self._partial_dir / "data.parquet"
        hf_dataset.to_parquet(str(parquet_path))

        chunk_index = episode_index // self.dataset.meta.chunks_size
        file_index = episode_index % self.dataset.meta.chunks_size
        files = {
            "data": {
                "source": "data.parquet",
                "destination": self.dataset.meta.data_path.format(
                    chunk_index=chunk_index, file_index=file_index
                ),
                "sha256": _sha256(parquet_path),
            }
        }
        video_metadata = {}
        for key in self.dataset.meta.video_keys:
            source = self._partial_dir / "videos" / f"{key}.mp4"
            self._validate_video(source, frame_count, self._camera_shapes[key])
            destination = self.dataset.meta.video_path.format(
                video_key=key, chunk_index=chunk_index, file_index=file_index
            )
            files[f"video:{key}"] = {
                "source": f"videos/{key}.mp4",
                "destination": destination,
                "sha256": _sha256(source),
            }
            video_metadata.update(
                {
                    f"videos/{key}/chunk_index": chunk_index,
                    f"videos/{key}/file_index": file_index,
                    f"videos/{key}/from_timestamp": 0.0,
                    f"videos/{key}/to_timestamp": frame_count / self.fps,
                }
            )
        manifest = {
            "protocol_id": PROTOCOL_ID,
            "episode_index": episode_index,
            "frame_count": frame_count,
            "task": task,
            "tasks": sorted(set(tasks)),
            "stats": episode_stats,
            "chunk_index": chunk_index,
            "file_index": file_index,
            "files": files,
            "video_metadata": video_metadata,
            "encoding": dict(ENCODING_CONTRACT),
        }
        _atomic_json(self._partial_dir / "manifest.ready.json", manifest)
        return manifest

    def _validate_video(
        self, path: Path, frame_count: int, shape: tuple[int, int, int]
    ) -> None:
        import av

        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            actual_shape = (stream.codec_context.height, stream.codec_context.width, 3)
            actual_fps = float(stream.average_rate)
            if stream.frames != frame_count:
                raise RuntimeError(
                    f"{path} contains {stream.frames} frames, expected {frame_count}"
                )
            if actual_shape != shape or abs(actual_fps - self.fps) > 1e-6:
                raise RuntimeError(
                    f"{path} format mismatch: shape={actual_shape}, fps={actual_fps}"
                )
            if stream.codec_context.codec.id != av.codec.Codec("av1", "r").id:
                raise RuntimeError(
                    f"{path} codec={stream.codec_context.name}, expected AV1 bitstream"
                )
            if stream.codec_context.format.name != ENCODING_CONTRACT["pixel_format"]:
                raise RuntimeError(
                    f"{path} pixel format={stream.codec_context.format.name}, expected yuv420p"
                )

    @classmethod
    def recover_rank(cls, save_dir: str, rank: int) -> None:
        _require_supported_lerobot_version()
        rank_dir = Path(save_dir) / f"rank_{rank}"
        if not rank_dir.is_dir():
            return
        for shard in sorted(rank_dir.glob("id_*")):
            info_path = shard / "meta" / "info.json"
            try:
                with info_path.open(encoding="utf-8") as handle:
                    info = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            protocol = info.get("rlinf_streaming_video", {}).get("protocol_id")
            if protocol == PROTOCOL_ID:
                cls._rebuild_from_manifests(shard)

    @classmethod
    def _publish_ready(
        cls,
        root: Path,
        partial_dir: Path,
        manifest: dict[str, Any],
        *,
        verify_sources: bool = True,
    ) -> None:
        for item in manifest["files"].values():
            source = partial_dir / item["source"]
            destination = root / item["destination"]
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                if _sha256(destination) != item["sha256"]:
                    raise RuntimeError(f"stream transaction conflict at {destination}")
                if source.exists():
                    source.unlink()
            else:
                if not source.exists() or (
                    verify_sources and _sha256(source) != item["sha256"]
                ):
                    raise RuntimeError(
                        f"stream transaction source missing/corrupt: {source}"
                    )
                os.replace(source, destination)
        committed = (
            root
            / ".streaming"
            / "committed"
            / f"episode_{manifest['episode_index']:06d}.json"
        )
        if committed.exists():
            with committed.open(encoding="utf-8") as handle:
                prior_manifest = json.load(handle)
            if prior_manifest != _jsonable(manifest):
                raise RuntimeError(
                    f"stream transaction manifest conflict at {committed}"
                )
        else:
            _atomic_json(committed, manifest)
        if partial_dir.exists():
            shutil.rmtree(partial_dir)

    @staticmethod
    def _write_episode_metadata(
        root: Path, manifest: dict[str, Any], dataset_from_index: int
    ) -> None:
        import pandas as pd
        from lerobot.datasets.utils import flatten_dict

        stats = _numpy_stats(manifest["stats"])
        episode_dict = {
            "episode_index": manifest["episode_index"],
            "tasks": manifest["tasks"],
            "length": manifest["frame_count"],
            "data/chunk_index": manifest["chunk_index"],
            "data/file_index": manifest["file_index"],
            "dataset_from_index": dataset_from_index,
            "dataset_to_index": dataset_from_index + manifest["frame_count"],
            **manifest["video_metadata"],
            **flatten_dict({"stats": stats}),
            "meta/episodes/chunk_index": manifest["chunk_index"],
            "meta/episodes/file_index": manifest["file_index"],
        }
        meta_path = (
            root
            / "meta"
            / "episodes"
            / f"chunk-{manifest['chunk_index']:03d}"
            / f"file-{manifest['file_index']:03d}.parquet"
        )
        meta_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = meta_path.with_name(f".{meta_path.name}.tmp-{os.getpid()}")
        pd.DataFrame([_jsonable(episode_dict)]).to_parquet(temporary, index=False)
        os.replace(temporary, meta_path)

    @staticmethod
    def _write_tasks(root: Path, tasks: list[str]) -> None:
        import pandas as pd

        tasks_path = root / "meta" / "tasks.parquet"
        tasks_tmp = tasks_path.with_name(f".{tasks_path.name}.tmp-{os.getpid()}")
        pd.DataFrame(
            {"task_index": range(len(tasks))}, index=pd.Index(tasks)
        ).to_parquet(tasks_tmp)
        os.replace(tasks_tmp, tasks_path)

    @classmethod
    def _append_manifest_metadata(cls, root: Path, manifest: dict[str, Any]) -> None:
        """Incrementally commit one episode without scanning historical media."""
        import pandas as pd
        from lerobot.datasets.compute_stats import aggregate_stats
        from lerobot.datasets.utils import load_stats, serialize_dict
        from lerobot.datasets.video_utils import get_video_info

        with (root / "meta" / "info.json").open(encoding="utf-8") as handle:
            info = json.load(handle)
        episode_index = int(manifest["episode_index"])
        if int(info["total_episodes"]) != episode_index:
            raise RuntimeError(
                "incremental stream metadata is not contiguous: "
                f"info has {info['total_episodes']} episodes, manifest is {episode_index}"
            )
        dataset_from_index = int(info["total_frames"])
        tasks_path = root / "meta" / "tasks.parquet"
        tasks = list(pd.read_parquet(tasks_path).index) if tasks_path.exists() else []
        if manifest["task"] not in tasks:
            tasks.append(manifest["task"])
        cls._write_tasks(root, tasks)

        current_stats = load_stats(root)
        episode_stats = _numpy_stats(manifest["stats"])
        aggregate = (
            aggregate_stats([current_stats, episode_stats])
            if current_stats is not None
            else episode_stats
        )
        _atomic_json(root / "meta" / "stats.json", serialize_dict(aggregate))

        info["total_episodes"] = episode_index + 1
        info["total_frames"] = dataset_from_index + manifest["frame_count"]
        info["total_tasks"] = len(tasks)
        info["splits"] = {"train": f"0:{episode_index + 1}"}
        if episode_index == 0:
            for name, item in manifest["files"].items():
                if name.startswith("video:"):
                    key = name.split(":", 1)[1]
                    info["features"][key]["info"] = get_video_info(
                        root / item["destination"]
                    )
        _atomic_json(root / "meta" / "info.json", info)
        # Episode parquet is the final metadata commit marker.
        cls._write_episode_metadata(root, manifest, dataset_from_index)

    @classmethod
    def _rebuild_from_manifests(cls, root: Path) -> None:
        from lerobot.datasets.compute_stats import aggregate_stats
        from lerobot.datasets.utils import serialize_dict
        from lerobot.datasets.video_utils import get_video_info

        streaming = root / ".streaming"
        committed_dir = streaming / "committed"
        committed_dir.mkdir(parents=True, exist_ok=True)
        partial_dir = streaming / "partial"
        partial_dir.mkdir(parents=True, exist_ok=True)
        for candidate in sorted(partial_dir.glob("episode_*")):
            ready = candidate / "manifest.ready.json"
            if ready.exists():
                with ready.open(encoding="utf-8") as handle:
                    manifest = json.load(handle)
                cls._publish_ready(root, candidate, manifest)
            else:
                shutil.rmtree(candidate)

        manifests = []
        for path in sorted(committed_dir.glob("episode_*.json")):
            with path.open(encoding="utf-8") as handle:
                manifests.append(json.load(handle))
        indices = [manifest["episode_index"] for manifest in manifests]
        if indices != list(range(len(manifests))):
            raise RuntimeError(f"non-contiguous stream manifests in {root}: {indices}")
        if not manifests:
            return

        total_frames = 0
        tasks: list[str] = []
        episode_stats = []
        for manifest in manifests:
            for item in manifest["files"].values():
                path = root / item["destination"]
                if not path.exists() or _sha256(path) != item["sha256"]:
                    raise RuntimeError(
                        f"committed stream artifact missing/corrupt: {path}"
                    )
            task = manifest["task"]
            if task not in tasks:
                tasks.append(task)
            episode_stats.append(_numpy_stats(manifest["stats"]))
            cls._write_episode_metadata(root, manifest, total_frames)
            total_frames += manifest["frame_count"]

        cls._write_tasks(root, tasks)

        aggregate = aggregate_stats(episode_stats)
        _atomic_json(root / "meta" / "stats.json", serialize_dict(aggregate))
        with (root / "meta" / "info.json").open(encoding="utf-8") as handle:
            info = json.load(handle)
        info["total_episodes"] = len(manifests)
        info["total_frames"] = total_frames
        info["total_tasks"] = len(tasks)
        info["splits"] = {"train": f"0:{len(manifests)}"}
        first = manifests[0]
        for name, item in first["files"].items():
            if name.startswith("video:"):
                key = name.split(":", 1)[1]
                info["features"][key]["info"] = get_video_info(
                    root / item["destination"]
                )
        _atomic_json(root / "meta" / "info.json", info)

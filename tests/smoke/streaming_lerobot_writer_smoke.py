#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""CPU-only real-time and recovery smoke for the stream MP4 writer."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

from rlinf.data.storage.lerobot.streaming_writer import (
    StreamingLeRobotDatasetWriter,
)


def frame(index: int, height: int, width: int) -> dict:
    rows = np.arange(height, dtype=np.uint16)[:, None]
    columns = np.arange(width, dtype=np.uint16)[None, :]
    base = ((rows + columns + index) % 256).astype(np.uint8)
    image = np.stack([base, np.roll(base, index % width, 1), 255 - base], axis=-1)
    return {
        "image": image,
        "wrist_image": np.roll(image, index % height, axis=0),
        "extra_view_image": np.flip(image, axis=1),
        "state": np.full(16, index, dtype=np.float32),
        "actions": np.full(16, index + 1, dtype=np.float32),
        "done": np.asarray([False]),
        "is_success": np.asarray([False]),
        "intervene_flag": np.asarray([False]),
        "segment_id": np.asarray([index // 1000], dtype=np.uint8),
        "observation_timestamp_ns": np.asarray([time.monotonic_ns()], dtype=np.int64),
    }


def add_recovery_copies(root: Path, count: int) -> None:
    committed = root / ".streaming" / "committed"
    with (committed / "episode_000000.json").open() as handle:
        template = json.load(handle)
    chunks_size = 1000
    for episode_index in range(1, count):
        manifest = json.loads(json.dumps(template))
        manifest["episode_index"] = episode_index
        chunk_index = episode_index // chunks_size
        file_index = episode_index % chunks_size
        manifest["chunk_index"] = chunk_index
        manifest["file_index"] = file_index
        for name, item in manifest["files"].items():
            source = root / template["files"][name]["destination"]
            suffix = ".mp4" if name.startswith("video:") else ".parquet"
            if name.startswith("video:"):
                key = name.split(":", 1)[1]
                destination = (
                    Path("videos")
                    / key
                    / f"chunk-{chunk_index:03d}"
                    / f"file-{file_index:03d}{suffix}"
                )
                manifest["video_metadata"][f"videos/{key}/chunk_index"] = chunk_index
                manifest["video_metadata"][f"videos/{key}/file_index"] = file_index
            else:
                destination = (
                    Path("data")
                    / f"chunk-{chunk_index:03d}"
                    / f"file-{file_index:03d}{suffix}"
                )
            target = root / destination
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(source, target)
            item["destination"] = str(destination)
        path = committed / f"episode_{episode_index:06d}.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration-s", type=float, default=130.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--recovery-copies", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    temporary = None
    if args.output is None:
        temporary = tempfile.TemporaryDirectory(prefix="rlinf-stream-smoke-")
        root = Path(temporary.name) / "dataset"
    else:
        root = args.output
        if root.exists():
            raise FileExistsError(root)

    writer = StreamingLeRobotDatasetWriter(queue_size=60)
    writer.create(
        repo_id=str(root),
        robot_type="dual_FR3",
        fps=args.fps,
        image_shape=(args.height, args.width, 3),
        state_dim=16,
        action_dim=16,
        has_image=True,
        wrist_image_keys={"wrist_image": (args.height, args.width, 3)},
        extra_view_image_keys={"extra_view_image": (args.height, args.width, 3)},
        has_intervene_flag=True,
        has_segment_id=True,
        has_observation_timestamp=True,
    )
    frame_count = round(args.duration_s * args.fps)
    start = time.monotonic()
    try:
        for index in range(frame_count):
            deadline = start + index / args.fps
            delay = deadline - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            writer.append_frame(frame(index, args.height, args.width))
        flush_start = time.monotonic()
        writer.finish_episode(task="stack boxes", is_success=True)
        flush_seconds = time.monotonic() - flush_start

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        dataset = LeRobotDataset(repo_id="local/stream-smoke", root=root)
        decoded_shape = tuple(dataset[frame_count - 1]["image"].shape)
        if list(root.rglob("*.png")):
            raise RuntimeError("stream smoke produced PNG files")
        media_mtimes = {path: path.stat().st_mtime_ns for path in root.rglob("*.mp4")}
        add_recovery_copies(root, args.recovery_copies)
        recovery_start = time.monotonic()
        StreamingLeRobotDatasetWriter._rebuild_from_manifests(root)
        recovery_seconds = time.monotonic() - recovery_start
        if any(
            path.stat().st_mtime_ns != mtime for path, mtime in media_mtimes.items()
        ):
            raise RuntimeError("recovery modified existing MP4 media")
        with (root / "meta" / "info.json").open(encoding="utf-8") as handle:
            writer.dataset.meta.info = json.load(handle)
        for index in range(args.fps):
            writer.append_frame(frame(index, args.height, args.width))
        append_start = time.monotonic()
        writer.finish_episode(task="stack boxes", is_success=True)
        append_after_history_seconds = time.monotonic() - append_start
    finally:
        writer.finalize()
    result = {
        "frames": frame_count,
        "cameras": 3,
        "shape": [args.height, args.width, 3],
        "fps": args.fps,
        "wall_seconds": time.monotonic() - start,
        "flush_seconds": flush_seconds,
        "native_decoded_shape": decoded_shape,
        "png_count": 0,
        "recovery_episode_manifests": args.recovery_copies,
        "recovery_seconds": recovery_seconds,
        "recovery_transcodes": 0,
        "append_after_history_seconds": append_after_history_seconds,
        "output": str(root),
    }
    print(json.dumps(result, indent=2))
    if flush_seconds >= 2.0:
        raise RuntimeError(f"episode flush exceeded 2s: {flush_seconds:.3f}s")
    if recovery_seconds >= 15.0:
        raise RuntimeError(
            f"100-episode recovery exceeded 15s: {recovery_seconds:.3f}s"
        )
    if append_after_history_seconds >= 2.0:
        raise RuntimeError(
            "episode flush after 100 historical episodes exceeded 2s: "
            f"{append_after_history_seconds:.3f}s"
        )
    if temporary is not None:
        shutil.rmtree(root)


if __name__ == "__main__":
    main()

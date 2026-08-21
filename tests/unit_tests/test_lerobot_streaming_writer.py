# Copyright 2026 The RLinf Authors.

import json
import queue
from unittest.mock import MagicMock

import numpy as np
import pytest

from rlinf.data.storage.lerobot.streaming_writer import (
    ENCODING_CONTRACT,
    PROTOCOL_ID,
    StreamingLeRobotDatasetWriter,
    _sampled_image_stats,
)


def _frame(index: int) -> dict:
    return {
        "image": np.full((16, 16, 3), index, dtype=np.uint8),
        "wrist_image": np.full((16, 16, 3), index + 1, dtype=np.uint8),
        "extra_view_image": np.full((16, 16, 3), index + 2, dtype=np.uint8),
        "state": np.asarray([index, index + 0.5], dtype=np.float32),
        "actions": np.asarray([index + 1, index + 2], dtype=np.float32),
        "done": np.asarray([False]),
        "is_success": np.asarray([False]),
        "intervene_flag": np.asarray([index % 2 == 0]),
        "segment_id": np.asarray([index // 2], dtype=np.uint8),
        "observation_timestamp_ns": np.asarray([100 + index], dtype=np.int64),
    }


def test_sampled_image_stats_match_direct_population_statistics():
    images = [
        np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3) + index
        for index in range(120)
    ]
    summaries = []
    for image in images:
        flat = image.reshape(-1, 3).astype(np.float64)
        summaries.append(
            {
                "min": flat.min(axis=0),
                "max": flat.max(axis=0),
                "sum": flat.sum(axis=0),
                "square_sum": np.square(flat).sum(axis=0),
                "pixel_count": flat.shape[0],
            }
        )

    from lerobot.datasets.compute_stats import sample_indices

    selected = np.stack([images[index] for index in sample_indices(len(images))])
    direct = selected.transpose(0, 3, 1, 2).astype(np.float64) / 255.0
    actual = _sampled_image_stats(summaries)

    np.testing.assert_allclose(actual["min"].reshape(3), direct.min((0, 2, 3)))
    np.testing.assert_allclose(actual["max"].reshape(3), direct.max((0, 2, 3)))
    np.testing.assert_allclose(actual["mean"].reshape(3), direct.mean((0, 2, 3)))
    np.testing.assert_allclose(actual["std"].reshape(3), direct.std((0, 2, 3)))


def test_streaming_writer_creates_native_lerobot_v3_without_png(tmp_path):
    root = tmp_path / "dataset"
    writer = StreamingLeRobotDatasetWriter(queue_size=8)
    writer.create(
        repo_id=str(root),
        robot_type="dual_FR3",
        fps=30,
        image_shape=(16, 16, 3),
        state_dim=2,
        action_dim=2,
        has_image=True,
        wrist_image_keys={"wrist_image": (16, 16, 3)},
        extra_view_image_keys={"extra_view_image": (16, 16, 3)},
        has_intervene_flag=True,
        has_segment_id=True,
        has_observation_timestamp=True,
    )
    try:
        for index in range(6):
            writer.append_frame(_frame(index))
        assert writer.finish_episode(task="stack boxes", is_success=True) == 6
    finally:
        writer.finalize()

    assert not list(root.rglob("*.png"))
    assert len(list(root.rglob("*.mp4"))) == 3
    with (root / "meta" / "info.json").open() as handle:
        info = json.load(handle)
    assert info["codebase_version"] == "v3.0"
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 6
    assert info["rlinf_streaming_video"]["protocol_id"] == PROTOCOL_ID
    assert info["rlinf_streaming_video"]["encoding"] == ENCODING_CONTRACT

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id="local/stream-test", root=root)
    assert len(dataset) == 6
    assert dataset.meta.total_episodes == 1
    assert (root / dataset.meta.get_video_file_path(0, "image")).is_file()
    assert tuple(dataset[0]["image"].shape) == (3, 16, 16)


def test_ready_transaction_recovery_is_idempotent(tmp_path):
    root = tmp_path / "rank_0" / "id_0"
    writer = StreamingLeRobotDatasetWriter(queue_size=8)
    writer.create(
        repo_id=str(root),
        robot_type="dual_FR3",
        fps=30,
        image_shape=(16, 16, 3),
        state_dim=2,
        action_dim=2,
        has_image=True,
        wrist_image_keys={"wrist_image": (16, 16, 3)},
        extra_view_image_keys={"extra_view_image": (16, 16, 3)},
        has_intervene_flag=True,
        has_segment_id=True,
        has_observation_timestamp=True,
    )
    try:
        for index in range(3):
            writer.append_frame(_frame(index))
        writer.finish_episode(task="stack boxes", is_success=True)
    finally:
        writer.finalize()

    StreamingLeRobotDatasetWriter.recover_rank(str(tmp_path), rank=0)
    StreamingLeRobotDatasetWriter.recover_rank(str(tmp_path), rank=0)
    with (root / "meta" / "info.json").open() as handle:
        assert json.load(handle)["total_episodes"] == 1


def test_multiple_episodes_use_independent_v3_files(tmp_path):
    root = tmp_path / "dataset"
    writer = StreamingLeRobotDatasetWriter(queue_size=8)
    writer.create(
        repo_id=str(root),
        robot_type="dual_FR3",
        fps=30,
        image_shape=(16, 16, 3),
        state_dim=2,
        action_dim=2,
        has_image=True,
        wrist_image_keys={"wrist_image": (16, 16, 3)},
        extra_view_image_keys={"extra_view_image": (16, 16, 3)},
        has_intervene_flag=True,
        has_segment_id=True,
        has_observation_timestamp=True,
    )
    try:
        for count in (4, 5):
            for index in range(count):
                writer.append_frame(_frame(index))
            writer.finish_episode(task="stack boxes", is_success=True)
    finally:
        writer.finalize()

    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(repo_id="local/multiple-stream-test", root=root)
    assert len(dataset) == 9
    assert dataset.meta.total_episodes == 2
    assert len(list(root.rglob("*.mp4"))) == 6
    assert len(list((root / "data").rglob("*.parquet"))) == 2
    assert len(list((root / "meta" / "episodes").rglob("*.parquet"))) == 2
    assert dataset[0]["episode_index"] == 0
    assert dataset[4]["episode_index"] == 1


def test_publish_failure_preserves_ready_transaction(monkeypatch, tmp_path):
    root = tmp_path / "rank_0" / "id_0"
    writer = StreamingLeRobotDatasetWriter(queue_size=8)
    writer.create(
        repo_id=str(root),
        robot_type="dual_FR3",
        fps=30,
        image_shape=(16, 16, 3),
        state_dim=2,
        action_dim=2,
        has_image=True,
        wrist_image_keys={"wrist_image": (16, 16, 3)},
        extra_view_image_keys={"extra_view_image": (16, 16, 3)},
        has_intervene_flag=True,
        has_segment_id=True,
        has_observation_timestamp=True,
    )
    for index in range(3):
        writer.append_frame(_frame(index))
    monkeypatch.setattr(
        StreamingLeRobotDatasetWriter,
        "_publish_ready",
        MagicMock(side_effect=RuntimeError("injected publish crash")),
    )
    with pytest.raises(RuntimeError, match="injected publish crash"):
        writer.finish_episode(task="stack boxes", is_success=True)
    writer.finalize()

    ready = root / ".streaming" / "partial" / "episode_000000" / "manifest.ready.json"
    assert ready.is_file()
    monkeypatch.undo()
    StreamingLeRobotDatasetWriter.recover_rank(str(tmp_path), rank=0)
    assert not ready.exists()
    assert (root / ".streaming" / "committed" / "episode_000000.json").is_file()
    with (root / "meta" / "info.json").open() as handle:
        assert json.load(handle)["total_episodes"] == 1


def test_streaming_writer_fails_closed_on_version_drift(monkeypatch):
    monkeypatch.setattr(
        "rlinf.data.storage.lerobot.streaming_writer.importlib.metadata.version",
        lambda package: "9.9.9",
    )
    with pytest.raises(RuntimeError, match="requires lerobot=="):
        StreamingLeRobotDatasetWriter()


def test_recovery_checks_version_before_touching_partial_data(monkeypatch, tmp_path):
    root = tmp_path / "rank_0" / "id_0"
    partial = root / ".streaming" / "partial" / "episode_000000"
    partial.mkdir(parents=True)
    sentinel = partial / "unfinished.mp4"
    sentinel.write_bytes(b"do not touch")
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(
        json.dumps({"rlinf_streaming_video": {"protocol_id": PROTOCOL_ID}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "rlinf.data.storage.lerobot.streaming_writer.importlib.metadata.version",
        lambda package: "9.9.9",
    )

    with pytest.raises(RuntimeError, match="requires lerobot=="):
        StreamingLeRobotDatasetWriter.recover_rank(str(tmp_path), rank=0)
    assert sentinel.read_bytes() == b"do not touch"


def _created_writer(root):
    writer = StreamingLeRobotDatasetWriter(queue_size=8)
    writer.create(
        repo_id=str(root),
        robot_type="dual_FR3",
        fps=30,
        image_shape=(16, 16, 3),
        state_dim=2,
        action_dim=2,
        has_image=True,
        wrist_image_keys={"wrist_image": (16, 16, 3)},
        extra_view_image_keys={"extra_view_image": (16, 16, 3)},
        has_intervene_flag=True,
        has_segment_id=True,
        has_observation_timestamp=True,
    )
    return writer


def test_queue_overflow_fails_closed_without_accepting_frame(monkeypatch, tmp_path):
    writer = _created_writer(tmp_path / "dataset")
    try:
        writer.append_frame(_frame(0))
        accepted = len(writer._frames)
        original = writer._queue.put_nowait
        monkeypatch.setattr(
            writer._queue, "put_nowait", MagicMock(side_effect=queue.Full)
        )
        with pytest.raises(RuntimeError, match="queue exceeded"):
            writer.append_frame(_frame(1))
        assert len(writer._frames) == accepted
        monkeypatch.setattr(writer._queue, "put_nowait", original)
    finally:
        writer.finalize()


def test_dead_encoder_is_detected_before_accepting_frame(tmp_path):
    writer = _created_writer(tmp_path / "dataset")
    writer._process.terminate()
    writer._process.join(timeout=5)
    try:
        with pytest.raises(RuntimeError, match="encoder exited unexpectedly"):
            writer.append_frame(_frame(0))
        assert writer._frames == []
    finally:
        writer.finalize()


def test_camera_group_mismatch_aborts_episode(tmp_path):
    writer = _created_writer(tmp_path / "dataset")
    try:
        writer.append_frame(_frame(0))
        malformed = _frame(1)
        del malformed["wrist_image"]
        with pytest.raises(ValueError, match="camera group mismatch"):
            writer.append_frame(malformed)
        assert len(writer._frames) == 1
    finally:
        writer.finalize()

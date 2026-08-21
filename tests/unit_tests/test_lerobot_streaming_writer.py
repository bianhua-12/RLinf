# Copyright 2026 The RLinf Authors.

import json
import os
import queue
import time
from contextlib import suppress
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from rlinf.data.storage.lerobot.streaming_writer import (
    ENCODING_CONTRACT,
    PROTOCOL_ID,
    StreamingLeRobotDatasetWriter,
    _sampled_image_stats,
)


def _commit_result(episode_index, *, success, error=None, traceback_text=None):
    now = time.monotonic()
    return {
        "kind": "commit_result",
        "episode_index": episode_index,
        "started_at": now,
        "completed_at": now,
        "success": success,
        "error": error,
        "traceback": traceback_text,
    }


def _blocking_commit_worker(commands, results):
    while True:
        command = commands.get()
        if command[0] == "close":
            results.put({"kind": "closed"})
            return
        _, root, partial_dir, manifest = command
        root = Path(root)
        Path(partial_dir, "commit-worker-entered").write_text("entered")
        while not (root / "release-commit-worker").exists():
            time.sleep(0.01)
        StreamingLeRobotDatasetWriter._publish_ready(
            root, Path(partial_dir), manifest, verify_sources=False
        )
        StreamingLeRobotDatasetWriter._append_manifest_metadata(root, manifest)
        results.put(_commit_result(manifest["episode_index"], success=True))


def _permanently_blocked_commit_worker(commands, results):
    command = commands.get()
    _, _, partial_dir, _ = command
    Path(partial_dir, "commit-worker-entered").write_text("entered")
    while True:
        time.sleep(1.0)


def _failing_commit_worker(commands, results):
    command = commands.get()
    manifest = command[3]
    results.put(
        _commit_result(
            manifest["episode_index"],
            success=False,
            error="RuntimeError: injected publish crash",
            traceback_text="injected worker traceback",
        )
    )


def _exiting_commit_worker(commands, results):
    commands.get()
    os._exit(17)


def _partial_move_commit_worker(commands, results):
    _, root, partial_dir, manifest = commands.get()
    root = Path(root)
    partial_dir = Path(partial_dir)
    item = next(iter(manifest["files"].values()))
    source = partial_dir / item["source"]
    destination = root / item["destination"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(source, destination)
    os._exit(19)


def _committed_before_metadata_worker(commands, results):
    _, root, partial_dir, manifest = commands.get()
    StreamingLeRobotDatasetWriter._publish_ready(
        Path(root), Path(partial_dir), manifest, verify_sources=False
    )
    os._exit(23)


def _wait_for(path: Path, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while not path.exists():
        if time.monotonic() >= deadline:
            raise AssertionError(f"timed out waiting for {path}")
        time.sleep(0.01)


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
    monkeypatch.setattr(
        StreamingLeRobotDatasetWriter,
        "_COMMIT_WORKER_TARGET",
        staticmethod(_failing_commit_worker),
    )
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
    writer.finish_episode(task="stack boxes", is_success=True)
    with pytest.raises(RuntimeError, match="injected worker traceback"):
        writer.finalize()

    ready = root / ".streaming" / "partial" / "episode_000000" / "manifest.ready.json"
    assert ready.is_file()
    monkeypatch.undo()
    StreamingLeRobotDatasetWriter.recover_rank(str(tmp_path), rank=0)
    assert not ready.exists()
    assert (root / ".streaming" / "committed" / "episode_000000.json").is_file()
    with (root / "meta" / "info.json").open() as handle:
        assert json.load(handle)["total_episodes"] == 1


def test_finish_returns_after_ready_manifest_before_async_publish(
    monkeypatch, tmp_path
):
    root = tmp_path / "rank_0" / "id_0"
    monkeypatch.setattr(
        StreamingLeRobotDatasetWriter,
        "_COMMIT_WORKER_TARGET",
        staticmethod(_blocking_commit_worker),
    )
    writer = _created_writer(root)
    for index in range(3):
        writer.append_frame(_frame(index))

    assert writer.finish_episode(task="stack boxes", is_success=True) == 3
    ready = root / ".streaming" / "partial" / "episode_000000" / "manifest.ready.json"
    _wait_for(ready.parent / "commit-worker-entered")
    assert ready.is_file()
    with (root / "meta" / "info.json").open() as handle:
        assert json.load(handle)["total_episodes"] == 0

    (root / "release-commit-worker").write_text("release")
    writer.finalize()
    with (root / "meta" / "info.json").open() as handle:
        assert json.load(handle)["total_episodes"] == 1


def test_next_episode_can_finish_while_previous_publish_is_blocked(
    monkeypatch, tmp_path
):
    root = tmp_path / "rank_0" / "id_0"
    monkeypatch.setattr(
        StreamingLeRobotDatasetWriter,
        "_COMMIT_WORKER_TARGET",
        staticmethod(_blocking_commit_worker),
    )
    writer = _created_writer(root)

    for index in range(3):
        writer.append_frame(_frame(index))
    assert writer.finish_episode(task="stack boxes", is_success=True) == 3
    _wait_for(
        root / ".streaming" / "partial" / "episode_000000" / "commit-worker-entered"
    )

    for index in range(2):
        writer.append_frame(_frame(index + 3))
    assert writer.finish_episode(task="stack boxes", is_success=True) == 2

    first_ready = root / ".streaming" / "partial" / "episode_000000"
    second_ready = root / ".streaming" / "partial" / "episode_000001"
    assert (first_ready / "manifest.ready.json").is_file()
    assert (second_ready / "manifest.ready.json").is_file()
    with (second_ready / "manifest.ready.json").open() as handle:
        second_manifest = json.load(handle)
    assert second_manifest["episode_index"] == 1
    import pandas as pd

    second_data = pd.read_parquet(second_ready / "data.parquet")
    assert second_data["index"].tolist() == [3, 4]

    (root / "release-commit-worker").write_text("release")
    writer.finalize()
    with (root / "meta" / "info.json").open() as handle:
        info = json.load(handle)
    assert info["total_episodes"] == 2
    assert info["total_frames"] == 5


def test_normal_async_publish_does_not_rehash_staged_sources(monkeypatch, tmp_path):
    writer = _created_writer(tmp_path / "dataset")
    for index in range(3):
        writer.append_frame(_frame(index))

    original_sha256 = __import__(
        "rlinf.data.storage.lerobot.streaming_writer", fromlist=["_sha256"]
    )._sha256
    calls = []

    def counting_sha256(path):
        calls.append(path)
        return original_sha256(path)

    monkeypatch.setattr(
        "rlinf.data.storage.lerobot.streaming_writer._sha256", counting_sha256
    )
    writer.finish_episode(task="stack boxes", is_success=True)
    writer.finalize()

    assert len(calls) == 4


def test_permanent_commit_block_is_bounded_and_preserves_ready_manifests(
    monkeypatch, tmp_path
):
    root = tmp_path / "rank_0" / "id_0"
    monkeypatch.setattr(
        StreamingLeRobotDatasetWriter,
        "_COMMIT_WORKER_TARGET",
        staticmethod(_permanently_blocked_commit_worker),
    )
    writer = _created_writer(root, max_pending_commits=2, commit_watchdog_timeout=0.3)
    for episode in range(2):
        for index in range(2):
            writer.append_frame(_frame(episode * 2 + index))
        writer.finish_episode(task="stack boxes", is_success=True)
    _wait_for(
        root / ".streaming" / "partial" / "episode_000000" / "commit-worker-entered"
    )

    with pytest.raises(RuntimeError, match="pending commit limit reached"):
        writer.append_frame(_frame(4))
    assert writer.pending_commit_count == 2
    assert all(
        (
            root
            / ".streaming"
            / "partial"
            / f"episode_{episode:06d}"
            / "manifest.ready.json"
        ).is_file()
        for episode in range(2)
    )

    started = time.monotonic()
    with pytest.raises(
        (RuntimeError, TimeoutError), match=r"watchdog|finalizing stream commits"
    ):
        writer.finalize()
    assert time.monotonic() - started < 1.0
    assert not writer._commit_process.is_alive()


def test_commit_worker_error_propagates_traceback_without_reusing_index(
    monkeypatch, tmp_path
):
    root = tmp_path / "rank_0" / "id_0"
    monkeypatch.setattr(
        StreamingLeRobotDatasetWriter,
        "_COMMIT_WORKER_TARGET",
        staticmethod(_failing_commit_worker),
    )
    writer = _created_writer(root, commit_watchdog_timeout=1.0)
    for index in range(2):
        writer.append_frame(_frame(index))
    writer.finish_episode(task="stack boxes", is_success=True)
    try:
        writer._commit_process.join(timeout=2.0)
        with pytest.raises(RuntimeError, match="injected worker traceback"):
            writer.append_frame(_frame(2))
        assert writer._next_episode == 1
        with pytest.raises(RuntimeError, match="injected worker traceback"):
            writer.finalize()
    finally:
        if not writer._closed:
            with suppress(BaseException):
                writer.finalize()

    monkeypatch.undo()
    StreamingLeRobotDatasetWriter.recover_rank(str(tmp_path), rank=0)
    with (root / "meta" / "info.json").open() as handle:
        info = json.load(handle)
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 2


def test_unexpected_commit_worker_exit_fails_fast(monkeypatch, tmp_path):
    root = tmp_path / "dataset"
    monkeypatch.setattr(
        StreamingLeRobotDatasetWriter,
        "_COMMIT_WORKER_TARGET",
        staticmethod(_exiting_commit_worker),
    )
    writer = _created_writer(root, commit_watchdog_timeout=1.0)
    for index in range(2):
        writer.append_frame(_frame(index))
    writer.finish_episode(task="stack boxes", is_success=True)
    writer._commit_process.join(timeout=2.0)
    with pytest.raises(
        RuntimeError, match=r"exited unexpectedly.*pending episodes=\[0\]"
    ):
        writer.append_frame(_frame(2))
    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        writer.finalize()


@pytest.mark.parametrize(
    ("worker_target", "exit_code"),
    [(_partial_move_commit_worker, 19), (_committed_before_metadata_worker, 23)],
)
def test_recovery_repairs_worker_crash_points(
    monkeypatch, tmp_path, worker_target, exit_code
):
    root = tmp_path / "rank_0" / "id_0"
    monkeypatch.setattr(
        StreamingLeRobotDatasetWriter,
        "_COMMIT_WORKER_TARGET",
        staticmethod(worker_target),
    )
    writer = _created_writer(root, commit_watchdog_timeout=1.0)
    for index in range(2):
        writer.append_frame(_frame(index))
    writer.finish_episode(task="stack boxes", is_success=True)
    writer._commit_process.join(timeout=2.0)
    assert writer._commit_process.exitcode == exit_code
    with pytest.raises(RuntimeError, match="exited unexpectedly"):
        writer.finalize()

    monkeypatch.undo()
    StreamingLeRobotDatasetWriter.recover_rank(str(tmp_path), rank=0)
    StreamingLeRobotDatasetWriter.recover_rank(str(tmp_path), rank=0)
    with (root / "meta" / "info.json").open() as handle:
        info = json.load(handle)
    assert info["total_episodes"] == 1
    assert info["total_frames"] == 2
    assert len(list((root / "data").rglob("*.parquet"))) == 1
    assert len(list((root / "videos").rglob("*.mp4"))) == 3


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


def test_recovery_discards_partial_episode_without_ready_marker(tmp_path):
    root = tmp_path / "rank_0" / "id_0"
    writer = _created_writer(root)
    writer.finalize()
    partial = root / ".streaming" / "partial" / "episode_000000"
    partial.mkdir(parents=True)
    (partial / "unfinished.mp4").write_bytes(b"incomplete")

    StreamingLeRobotDatasetWriter.recover_rank(str(tmp_path), rank=0)

    assert not partial.exists()


def test_recovery_fails_closed_on_destination_hash_conflict(monkeypatch, tmp_path):
    root = tmp_path / "rank_0" / "id_0"
    monkeypatch.setattr(
        StreamingLeRobotDatasetWriter,
        "_COMMIT_WORKER_TARGET",
        staticmethod(_permanently_blocked_commit_worker),
    )
    writer = _created_writer(root, commit_watchdog_timeout=0.2)
    for index in range(2):
        writer.append_frame(_frame(index))
    writer.finish_episode(task="stack boxes", is_success=True)
    ready = root / ".streaming" / "partial" / "episode_000000"
    _wait_for(ready / "commit-worker-entered")
    with (ready / "manifest.ready.json").open() as handle:
        manifest = json.load(handle)
    first = next(iter(manifest["files"].values()))
    destination = root / first["destination"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(b"conflicting destination")
    with pytest.raises((RuntimeError, TimeoutError)):
        writer.finalize()

    monkeypatch.undo()
    with pytest.raises(RuntimeError, match="stream transaction conflict"):
        StreamingLeRobotDatasetWriter.recover_rank(str(tmp_path), rank=0)
    assert (ready / "manifest.ready.json").is_file()


def _created_writer(root, **kwargs):
    writer = StreamingLeRobotDatasetWriter(queue_size=8, **kwargs)
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

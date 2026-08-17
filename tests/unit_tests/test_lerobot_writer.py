# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from rlinf.data.storage.lerobot.writer import LeRobotDatasetWriter


def _fake_video_manager_modules(manager):
    lerobot_module = ModuleType("lerobot")
    lerobot_module.__path__ = []
    datasets_module = ModuleType("lerobot.datasets")
    datasets_module.__path__ = []
    video_utils_module = ModuleType("lerobot.datasets.video_utils")
    video_utils_module.VideoEncodingManager = manager
    return {
        "lerobot": lerobot_module,
        "lerobot.datasets": datasets_module,
        "lerobot.datasets.video_utils": video_utils_module,
    }


@pytest.mark.parametrize(
    ("use_videos", "expected_dtype"),
    [(False, "image"), (True, "video")],
)
def test_auto_generated_camera_features_follow_config(use_videos, expected_dtype):
    lerobot_module = ModuleType("lerobot")
    lerobot_module.__path__ = []
    datasets_module = ModuleType("lerobot.datasets")
    datasets_module.__path__ = []
    dataset_module = ModuleType("lerobot.datasets.lerobot_dataset")
    dataset_cls = MagicMock()
    dataset_cls.create.return_value = MagicMock()
    dataset_module.LeRobotDataset = dataset_cls

    modules = {
        "lerobot": lerobot_module,
        "lerobot.datasets": datasets_module,
        "lerobot.datasets.lerobot_dataset": dataset_module,
    }
    with (
        patch.dict(sys.modules, modules),
        patch("rlinf.data.storage.lerobot.writer._silence_hf_datasets_progress_bars"),
    ):
        writer = LeRobotDatasetWriter()
        writer.create(
            repo_id="test_dataset",
            image_shape=(224, 224, 3),
            wrist_image_keys={"wrist_image": (224, 224, 3)},
            extra_view_image_keys={
                "extra_view_image-0": (224, 224, 3),
                "extra_view_image-1": (224, 224, 3),
            },
            has_observation_timestamp=True,
            use_videos=use_videos,
        )

    features = dataset_cls.create.call_args.kwargs["features"]
    camera_keys = [
        "image",
        "wrist_image",
        "extra_view_image-0",
        "extra_view_image-1",
    ]
    assert all(features[key]["dtype"] == expected_dtype for key in camera_keys)
    assert features["observation_timestamp_ns"] == {
        "dtype": "int64",
        "shape": (1,),
        "names": ["observation_timestamp_ns"],
    }
    assert dataset_cls.create.call_args.kwargs["use_videos"] is use_videos


def test_deferred_video_encoding_uses_unreachable_batch_threshold():
    lerobot_module = ModuleType("lerobot")
    lerobot_module.__path__ = []
    datasets_module = ModuleType("lerobot.datasets")
    datasets_module.__path__ = []
    dataset_module = ModuleType("lerobot.datasets.lerobot_dataset")
    dataset_cls = MagicMock()
    dataset_cls.create.return_value = MagicMock()
    dataset_module.LeRobotDataset = dataset_cls

    modules = {
        "lerobot": lerobot_module,
        "lerobot.datasets": datasets_module,
        "lerobot.datasets.lerobot_dataset": dataset_module,
    }
    with (
        patch.dict(sys.modules, modules),
        patch("rlinf.data.storage.lerobot.writer._silence_hf_datasets_progress_bars"),
    ):
        writer = LeRobotDatasetWriter()
        writer.create(
            repo_id="test_dataset",
            use_videos=True,
            defer_video_encoding_until_finalize=True,
        )

    assert dataset_cls.create.call_args.kwargs["batch_encoding_size"] == sys.maxsize
    assert writer.defer_video_encoding_until_finalize is True


def test_episode_stats_are_submitted_outside_collector_process():
    writer = LeRobotDatasetWriter()
    writer.dataset = MagicMock()
    executor = MagicMock()
    executor.submit.return_value.result.return_value = {"image": "stats"}
    writer._episode_stats_executor = executor
    original_compute_stats = MagicMock()
    dataset_module = SimpleNamespace(compute_episode_stats=original_compute_stats)

    def save_episode():
        writer.dataset.stats = dataset_module.compute_episode_stats(
            {"image": ["frame.png"]}, {"image": {"dtype": "video"}}
        )

    writer.dataset.save_episode.side_effect = save_episode
    with patch(
        "rlinf.data.storage.lerobot.writer.importlib.import_module",
        return_value=dataset_module,
    ):
        writer._save_episode()

    submitted = executor.submit.call_args.args
    assert submitted[1] == {"image": ["frame.png"]}
    assert submitted[2] == {"image": {"dtype": "video"}}
    assert writer.dataset.stats == {"image": "stats"}
    assert dataset_module.compute_episode_stats is original_compute_stats


def test_add_episode_can_consume_transferred_frames():
    writer = LeRobotDatasetWriter()
    writer.dataset = MagicMock()
    episode = [
        {"task": "test task", "image": object()},
        {"task": "test task", "image": object()},
    ]

    with patch("rlinf.data.storage.lerobot.writer.add_frame_to_dataset") as add:
        writer.add_episode(episode, consume=True)

    assert add.call_count == 2
    assert episode == []
    writer.dataset.save_episode.assert_called_once_with()


def test_add_episode_failure_preserves_unaccepted_frames():
    writer = LeRobotDatasetWriter()
    writer.dataset = MagicMock()
    episode = [
        {"task": "test task", "image": "first"},
        {"task": "test task", "image": "second"},
        {"task": "test task", "image": "third"},
    ]

    with patch(
        "rlinf.data.storage.lerobot.writer.add_frame_to_dataset",
        side_effect=[None, RuntimeError("add failed")],
    ):
        with pytest.raises(RuntimeError, match="add failed"):
            writer.add_episode(episode, consume=True)

    assert [frame["image"] for frame in episode] == ["second", "third"]
    writer.dataset.save_episode.assert_not_called()


def test_finalize_encodes_deferred_videos_and_stops_stats_process():
    writer = LeRobotDatasetWriter()
    writer.defer_video_encoding_until_finalize = True
    stats_executor = MagicMock()
    writer._episode_stats_executor = stats_executor
    dataset = MagicMock()
    dataset.episodes_since_last_encoding = 3
    dataset.image_writer = None
    writer.dataset = dataset

    manager = MagicMock()
    with patch.dict(sys.modules, _fake_video_manager_modules(manager)):
        writer.finalize()

    manager.assert_called_once_with(dataset)
    stats_executor.shutdown.assert_called_once_with(wait=True)
    assert dataset.episodes_since_last_encoding == 0
    assert writer.dataset is None


def test_finalize_stops_resources_and_keeps_dataset_after_encoding_error():
    writer = LeRobotDatasetWriter()
    writer.defer_video_encoding_until_finalize = True
    stats_executor = MagicMock()
    writer._episode_stats_executor = stats_executor
    dataset = MagicMock()
    dataset.episodes_since_last_encoding = 2
    image_writer = dataset.image_writer
    writer.dataset = dataset

    manager = MagicMock()
    with patch.dict(sys.modules, _fake_video_manager_modules(manager)):
        manager.return_value.__exit__.side_effect = RuntimeError("encode failed")
        with pytest.raises(RuntimeError, match="encode failed"):
            writer.finalize()

    image_writer.wait_until_done.assert_called_once_with()
    image_writer.stop.assert_called_once_with()
    stats_executor.shutdown.assert_called_once_with(wait=True)
    assert dataset.image_writer is None
    assert dataset.episodes_since_last_encoding == 2
    assert writer.dataset is dataset


def test_finalize_can_retry_after_encoding_error():
    writer = LeRobotDatasetWriter()
    writer.defer_video_encoding_until_finalize = True
    dataset = MagicMock()
    dataset.episodes_since_last_encoding = 2
    dataset.image_writer = None
    writer.dataset = dataset
    manager = MagicMock()
    manager.return_value.__exit__.side_effect = [RuntimeError("encode failed"), None]

    with patch.dict(sys.modules, _fake_video_manager_modules(manager)):
        with pytest.raises(RuntimeError, match="encode failed"):
            writer.finalize()
        writer.finalize()

    assert manager.call_count == 2
    assert dataset.episodes_since_last_encoding == 0
    assert writer.dataset is None


def test_finalize_passes_writer_failure_to_video_cleanup_manager():
    writer = LeRobotDatasetWriter()
    writer.use_videos = True
    dataset = MagicMock()
    dataset.episodes_since_last_encoding = 0
    dataset.image_writer = None
    writer.dataset = dataset
    prior_error = RuntimeError("save failed")
    prior_traceback = prior_error.__traceback__
    manager = MagicMock()

    with patch.dict(sys.modules, _fake_video_manager_modules(manager)):
        with pytest.raises(RuntimeError, match="save failed"):
            writer.finalize(prior_error=prior_error)

    manager.assert_called_once_with(dataset)
    manager.return_value.__exit__.assert_called_once_with(
        RuntimeError, prior_error, prior_traceback
    )
    assert writer.dataset is dataset

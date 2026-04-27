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

import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

from rlinf.models.embodiment.reward.vlm_reward_utils.input_builder import (
    RobochanallengeInputBuilder,
    SimpleDualViewTernaryInputBuilder,
    VideoVLMInputBuilder,
)


def _make_frame(
    value: int,
    singleton_batch: bool = False,
    num_views: int = 1,
) -> torch.Tensor:
    frame = torch.zeros((num_views, 2, 2, 3), dtype=torch.uint8)
    frame[..., 0] = value
    if not singleton_batch and num_views == 1:
        return frame.squeeze(0)
    return frame


def _make_history_buffer(
    batch_size: int,
    env_ids: list[int],
    step_offset: int = 0,
    singleton_batch: bool = False,
    num_views: int = 1,
) -> dict[str, list[list[torch.Tensor]]]:
    history = {"main_images": [[] for _ in range(batch_size)]}
    for env_id in env_ids:
        history["main_images"][env_id] = [
            _make_frame(
                env_id * 10 + step_offset + step_idx,
                singleton_batch=singleton_batch,
                num_views=num_views,
            )
            for step_idx in range(2)
        ]
    return history


def _merge_history_buffers(
    *segments: dict[str, list[list[torch.Tensor]]],
) -> dict[str, list[list[torch.Tensor]]]:
    batch_size = len(segments[0]["main_images"])
    merged = {"main_images": [[] for _ in range(batch_size)]}
    for segment in segments:
        for env_id, frames in enumerate(segment["main_images"]):
            merged["main_images"][env_id].extend(frames)
    return merged


def test_extract_videos_keeps_env_and_timestep_axes():
    builder = VideoVLMInputBuilder(
        _processor=None,
        history_buffer_names=["history_window"],
    )
    history = _make_history_buffer(batch_size=4, env_ids=[0, 1, 2, 3])

    videos = builder.extract_videos(history)

    assert len(videos) == 4
    assert [len(env_videos) for env_videos in videos] == [1, 1, 1, 1]
    assert [len(env_videos[0]) for env_videos in videos] == [2, 2, 2, 2]
    assert [frame.getpixel((0, 0))[0] for frame in videos[0][0]] == [0, 1]
    assert [frame.getpixel((0, 0))[0] for frame in videos[3][0]] == [30, 31]


def test_extract_videos_squeezes_singleton_frame_axes():
    builder = VideoVLMInputBuilder(
        _processor=None,
        history_buffer_names=["history_window"],
    )
    history = _make_history_buffer(
        batch_size=1,
        env_ids=[0],
        singleton_batch=True,
    )

    videos = builder.extract_videos(history)

    assert len(videos) == 1
    assert len(videos[0]) == 1
    assert len(videos[0][0]) == 2
    assert videos[0][0][0].size == (2, 2)


def test_extract_videos_rejects_unsupported_multiview_frame_shapes():
    builder = VideoVLMInputBuilder(
        _processor=None,
        history_buffer_names=["history_window"],
    )
    history = _make_history_buffer(batch_size=1, env_ids=[0], num_views=2)

    with pytest.raises(TypeError, match="Cannot handle this data type"):
        builder.extract_videos(history)


def test_extract_videos_accepts_numpy_history_frames():
    builder = VideoVLMInputBuilder(
        _processor=None,
        history_buffer_names=["history_window"],
    )
    history = {
        "main_images": [
            [
                np.zeros((2, 2, 3), dtype=np.uint8),
                np.ones((2, 2, 3), dtype=np.uint8),
            ]
        ]
    }

    videos = builder.extract_videos(history)

    assert len(videos) == 1
    assert len(videos[0][0]) == 2
    assert [frame.getpixel((0, 0))[0] for frame in videos[0][0]] == [0, 1]


def test_robochallenge_prepare_inputs_aligns_selected_envs_and_video_order():
    builder = RobochanallengeInputBuilder(
        _processor=None,
        history_buffer_names=["full_history", "history_window"],
    )
    observations = {
        "task_descriptions": ["task-0", "task-1", "task-2", "task-3"],
    }
    valid_input_ids = [1, 3]
    history_window = _make_history_buffer(
        batch_size=4,
        env_ids=valid_input_ids,
        step_offset=2,
    )
    full_history = _merge_history_buffers(
        _make_history_buffer(batch_size=4, env_ids=valid_input_ids, step_offset=0),
        history_window,
    )

    prepared = builder.prepare_inputs(
        observations=observations,
        history_input={
            "full_history": full_history,
            "history_window": history_window,
        },
        valid_input_ids=valid_input_ids,
    )

    videos_list = prepared["videos_list"]
    prompt_texts_list = prepared["prompt_texts_list"]

    assert len(videos_list) == 2
    assert len(prompt_texts_list) == 2
    assert "Task: task-1." in prompt_texts_list[0][0]
    assert "Task: task-3." in prompt_texts_list[1][0]

    full_video_env_1, clip_video_env_1 = videos_list[0]
    full_video_env_3, clip_video_env_3 = videos_list[1]

    assert [frame.getpixel((0, 0))[0] for frame in full_video_env_1] == [10, 11, 12, 13]
    assert [frame.getpixel((0, 0))[0] for frame in clip_video_env_1] == [12, 13]
    assert [frame.getpixel((0, 0))[0] for frame in full_video_env_3] == [30, 31, 32, 33]
    assert [frame.getpixel((0, 0))[0] for frame in clip_video_env_3] == [32, 33]


def test_simple_dualview_builder_accepts_main_and_extra_view_history():
    builder = SimpleDualViewTernaryInputBuilder(
        _processor=None,
        history_buffer_names=["history_window"],
    )
    observations = {
        "task_descriptions": ["task-0", "task-1"],
        "extra_view_images": torch.zeros((2, 2, 2, 3), dtype=torch.uint8),
    }
    history_input = {
        "history_window": {
            "main_images": [
                [_make_frame(0), _make_frame(1)],
                [_make_frame(10), _make_frame(11)],
            ],
            "extra_view_images": [
                [_make_frame(100), _make_frame(101)],
                [_make_frame(110), _make_frame(111)],
            ],
        }
    }

    valid_input_ids = builder.get_valid_input_ids(observations, history_input)
    prepared = builder.prepare_inputs(observations, history_input, valid_input_ids)

    assert valid_input_ids == [0, 1]
    videos_env_0 = prepared["videos_list"][0]
    videos_env_1 = prepared["videos_list"][1]
    assert [frame.getpixel((0, 0))[0] for frame in videos_env_0[0]] == [0, 1]
    assert [frame.getpixel((0, 0))[0] for frame in videos_env_0[1]] == [100, 101]
    assert [frame.getpixel((0, 0))[0] for frame in videos_env_1[0]] == [10, 11]
    assert [frame.getpixel((0, 0))[0] for frame in videos_env_1[1]] == [110, 111]


def test_simple_dualview_builder_logs_missing_extra_view_when_valid_ids_empty(caplog):
    builder = SimpleDualViewTernaryInputBuilder(
        _processor=None,
        history_buffer_names=["history_window"],
    )
    observations = {
        "task_descriptions": ["task-0"],
        "extra_view_images": None,
    }
    history_input = {
        "history_window": {
            "main_images": [[_make_frame(0), _make_frame(1)]],
            "extra_view_images": [[]],
        }
    }

    with caplog.at_level("WARNING"):
        valid_input_ids = builder.get_valid_input_ids(observations, history_input)

    assert valid_input_ids == []
    assert "observations.extra_view_images is None" in caplog.text

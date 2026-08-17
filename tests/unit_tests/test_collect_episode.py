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

from unittest.mock import MagicMock

import gymnasium as gym
import numpy as np
import pytest
import torch

from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.envs.wrappers.collect_episode import CollectEpisode


class _FreshObservationEnv(gym.Env):
    def reset(self, *, seed=None, options=None):
        del seed, options
        return _observation(0), {"observation_timestamp_ns": np.array([100])}

    def close(self):
        pass


def _observation(value: int) -> dict:
    return {
        "main_images": np.full((1, 2, 2, 3), value, dtype=np.uint8),
        "states": torch.full((1, 2), value, dtype=torch.float32),
        "task_descriptions": ["test task"],
    }


def test_pre_record_skips_observation_ownership(tmp_path):
    wrapper = CollectEpisode(
        _FreshObservationEnv(), str(tmp_path), copy_observations=False
    )
    wrapper.reset()
    wrapper._slice_observation = MagicMock(
        side_effect=AssertionError("pre-record observation must not be retained")
    )

    wrapper._record_step(
        np.zeros((1, 2)),
        _observation(1),
        torch.zeros(1),
        torch.tensor([False]),
        torch.tensor([False]),
        {"pre_record": np.array([True])},
    )

    wrapper._slice_observation.assert_not_called()
    assert wrapper._buffers[0]["actions"] == []
    assert len(wrapper._buffers[0]["observations"]) == 1
    wrapper.close()


def test_fresh_observation_mode_reuses_storage_and_records_timestamp(tmp_path):
    wrapper = CollectEpisode(
        _FreshObservationEnv(),
        str(tmp_path),
        export_format="lerobot",
        copy_observations=False,
    )
    wrapper.reset()
    obs = _observation(1)

    wrapper._record_step(
        np.array([[0.25, 0.5]], dtype=np.float32),
        obs,
        torch.tensor([0.0]),
        torch.tensor([False]),
        torch.tensor([False]),
        {"observation_timestamp_ns": np.array([200])},
    )

    recorded = wrapper._buffers[0]["observations"][1]
    assert recorded is not obs
    assert np.shares_memory(recorded["main_images"], obs["main_images"])
    assert recorded["states"].data_ptr() == obs["states"].data_ptr()
    assert wrapper._buffers[0]["observation_timestamps_ns"] == [100, 200]

    episode = wrapper._buffer_to_lerobot_ep(wrapper._buffers[0], 0, False)
    assert episode is not None
    np.testing.assert_array_equal(
        episode[0]["observation_timestamp_ns"], np.array([100], dtype=np.int64)
    )
    wrapper.close()


def test_default_observation_mode_copies_storage(tmp_path):
    wrapper = CollectEpisode(_FreshObservationEnv(), str(tmp_path))
    obs, _ = wrapper.reset()
    recorded = wrapper._buffers[0]["observations"][0]

    assert not np.shares_memory(recorded["main_images"], obs["main_images"])
    assert recorded["states"].data_ptr() != obs["states"].data_ptr()
    wrapper.close()


def test_realworld_timestamp_uses_monotonic_clock(monkeypatch):
    env = RealWorldEnv.__new__(RealWorldEnv)
    env.num_envs = 2
    infos = {}
    monkeypatch.setattr(
        "rlinf.envs.realworld.realworld_env.time.monotonic_ns", lambda: 123456
    )

    env._stamp_observation_info(infos)

    np.testing.assert_array_equal(
        infos["observation_timestamp_ns"],
        np.array([123456, 123456], dtype=np.int64),
    )


def test_realworld_numpy_observations_own_reused_vector_storage():
    env = RealWorldEnv.__new__(RealWorldEnv)
    env.main_image_key = "base"
    env.task_descriptions = ["test task"]
    env.return_numpy = True
    raw_obs = {
        "state": {"joint": np.zeros((1, 2), dtype=np.float32)},
        "frames": {
            "base": np.zeros((1, 2, 2, 3), dtype=np.uint8),
            "wrist": np.zeros((1, 2, 2, 3), dtype=np.uint8),
        },
    }

    obs = env._wrap_obs(raw_obs)
    raw_obs["frames"]["base"].fill(1)
    raw_obs["frames"]["wrist"].fill(1)

    assert isinstance(obs["main_images"], np.ndarray)
    assert isinstance(obs["extra_view_images"], np.ndarray)
    assert not obs["main_images"].any()
    assert not obs["extra_view_images"].any()


def test_realworld_numpy_step_never_tensorizes(monkeypatch):
    class FakeVectorEnv:
        def step(self, actions):
            del actions
            return (
                {
                    "state": {"joint": np.zeros((1, 2), dtype=np.float32)},
                    "frames": {
                        "base": np.zeros((1, 2, 2, 3), dtype=np.uint8),
                        "wrist": np.zeros((1, 2, 2, 3), dtype=np.uint8),
                    },
                },
                np.array([0.0], dtype=np.float32),
                np.array([False]),
                np.array([False]),
                {"rlt_switch_flags": np.array([False])},
            )

    env = RealWorldEnv.__new__(RealWorldEnv)
    env.env = FakeVectorEnv()
    env.cfg = type("Cfg", (), {"max_episode_steps": None})()
    env.num_envs = 1
    env.main_image_key = "base"
    env.task_descriptions = ["test task"]
    env.return_numpy = True
    env.manual_episode_control_only = False
    env.ignore_terminations = False
    env.auto_reset = False
    env._elapsed_steps = np.zeros(1, dtype=np.int32)
    env._init_metrics()
    monkeypatch.setattr(
        "rlinf.envs.realworld.realworld_env.to_tensor",
        MagicMock(side_effect=AssertionError("NumPy collection must not tensorize")),
    )

    obs, reward, terminated, truncated, infos = env.step(
        np.zeros((1, 2), dtype=np.float32)
    )

    for value in (obs["main_images"], obs["states"], reward, terminated, truncated):
        assert isinstance(value, np.ndarray)
    assert isinstance(infos["episode"]["return"], np.ndarray)
    assert isinstance(infos["intervene_action"], np.ndarray)
    assert isinstance(infos["intervene_flag"], np.ndarray)
    assert isinstance(infos["rlt_switch_flags"], np.ndarray)


def test_realworld_numpy_chunk_step_stays_numpy():
    env = RealWorldEnv.__new__(RealWorldEnv)
    env.num_envs = 1
    env.return_numpy = True
    env.auto_reset = False
    env.ignore_terminations = False
    env.step = MagicMock(
        return_value=(
            {"main_images": np.zeros((1, 2, 2, 3), dtype=np.uint8)},
            np.array([0.0], dtype=np.float32),
            np.array([False]),
            np.array([False]),
            {
                "intervene_action": np.zeros((1, 2), dtype=np.float32),
                "intervene_flag": np.array([False]),
            },
        )
    )

    _, rewards, terminated, truncated, infos = env.chunk_step(
        np.zeros((1, 3, 2), dtype=np.float32)
    )

    assert rewards.shape == (1, 3)
    assert terminated.shape == (1, 3)
    assert truncated.shape == (1, 3)
    assert infos[-1]["intervene_action"].shape == (1, 6)
    assert infos[-1]["intervene_flag"].shape == (1, 3)
    assert all(
        isinstance(value, np.ndarray) for value in (rewards, terminated, truncated)
    )


def test_realworld_default_output_stays_torch():
    env = RealWorldEnv.__new__(RealWorldEnv)
    env.return_numpy = False

    output = env._format_output(np.array([1.0], dtype=np.float32))

    assert isinstance(output, torch.Tensor)


def test_maybe_flush_does_not_scan_unfinished_episode(tmp_path):
    wrapper = CollectEpisode(_FreshObservationEnv(), str(tmp_path))
    wrapper._get_episode_success = MagicMock(
        side_effect=AssertionError("unfinished episodes must not be scanned")
    )

    wrapper._maybe_flush(np.array([False]), np.array([False]))

    wrapper._get_episode_success.assert_not_called()
    wrapper.close()


def test_deferred_video_encoding_requires_no_periodic_finalize(tmp_path):
    with pytest.raises(ValueError, match="requires finalize_interval=0"):
        CollectEpisode(
            _FreshObservationEnv(),
            str(tmp_path),
            use_videos=True,
            defer_video_encoding_until_finalize=True,
        )


def test_lerobot_writer_receives_deferred_and_isolated_options(tmp_path):
    wrapper = CollectEpisode(
        _FreshObservationEnv(),
        str(tmp_path),
        export_format="lerobot",
        use_videos=True,
        finalize_interval=0,
        defer_video_encoding_until_finalize=True,
        isolate_episode_stats=True,
        image_writer_threads=12,
        image_writer_processes=0,
    )
    writer = MagicMock()
    writer.dataset = None
    wrapper._lerobot_writer = writer
    frame = {
        "image": np.zeros((2, 2, 3), dtype=np.uint8),
        "state": np.zeros(2, dtype=np.float32),
        "actions": np.zeros(2, dtype=np.float32),
        "intervene_flag": np.array([False]),
        "segment_id": np.array([0], dtype=np.uint8),
        "observation_timestamp_ns": np.array([1], dtype=np.int64),
        "task": "test task",
    }

    wrapper._ensure_lerobot_writer([frame])
    kwargs = writer.create.call_args.kwargs
    assert kwargs["defer_video_encoding_until_finalize"] is True
    assert kwargs["isolate_episode_stats"] is True
    assert kwargs["image_writer_threads"] == 12
    assert kwargs["image_writer_processes"] == 0

    writer.dataset = object()
    episode = [frame]
    wrapper._write_lerobot_episode(episode)
    writer.add_episode.assert_called_once_with(episode, consume=True)
    wrapper.close()


def test_close_waits_all_futures_and_closes_env_after_failure(tmp_path):
    wrapper = CollectEpisode(_FreshObservationEnv(), str(tmp_path))
    first = MagicMock()
    first.result.side_effect = RuntimeError("first failed")
    second = MagicMock()
    wrapper._futures = [first, second]
    wrapper.env.close = MagicMock()

    with pytest.raises(RuntimeError, match="first failed"):
        wrapper.close()

    first.result.assert_called_once_with()
    second.result.assert_called_once_with()
    wrapper.env.close.assert_called_once_with()
    assert wrapper._executor is None


def test_close_retains_writer_when_finalization_fails(tmp_path):
    wrapper = CollectEpisode(
        _FreshObservationEnv(), str(tmp_path), export_format="lerobot"
    )
    writer = MagicMock()
    writer.dataset = object()
    writer.finalize.side_effect = RuntimeError("finalize failed")
    wrapper._lerobot_writer = writer

    with pytest.raises(RuntimeError, match="finalize failed"):
        wrapper.close()

    assert wrapper._lerobot_writer is writer
    writer.finalize.side_effect = None
    wrapper.close()


def test_close_retries_writer_without_reclosing_environment(tmp_path):
    wrapper = CollectEpisode(
        _FreshObservationEnv(), str(tmp_path), export_format="lerobot"
    )
    wrapper.env.close = MagicMock()
    writer = MagicMock()
    writer.dataset = object()
    writer.finalize.side_effect = [RuntimeError("finalize failed"), None]
    wrapper._lerobot_writer = writer

    with pytest.raises(RuntimeError, match="finalize failed"):
        wrapper.close()
    wrapper.close()
    wrapper.close()

    assert writer.finalize.call_count == 2
    assert wrapper._lerobot_writer is None
    wrapper.env.close.assert_called_once_with()
    assert wrapper._executor is None


def test_pickle_close_remains_idempotent(tmp_path):
    wrapper = CollectEpisode(_FreshObservationEnv(), str(tmp_path))
    wrapper.env.close = MagicMock()

    wrapper.close()
    wrapper.close()

    wrapper.env.close.assert_called_once_with()

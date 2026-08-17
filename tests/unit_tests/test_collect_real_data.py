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

import importlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))


def _load_collect_real_data_module():
    module_path = _REPO_ROOT / "examples" / "embodiment" / "collect_real_data.py"
    spec = importlib.util.spec_from_file_location(
        "_collect_real_data_under_test", module_path
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


collect_real_data = _load_collect_real_data_module()


def test_initialize_collection_storage_skips_demos_when_disabled(monkeypatch, tmp_path):
    collector = collect_real_data.DataCollector.__new__(collect_real_data.DataCollector)
    collector.cfg = OmegaConf.create(
        {
            "runner": {
                "logger": {"log_path": str(tmp_path)},
                "save_demos": False,
            },
            "env": {
                "eval": {
                    "data_collection": {
                        "enabled": True,
                        "save_dir": str(tmp_path / "collected_data"),
                    }
                }
            },
        }
    )
    collector.env = SimpleNamespace(action_space=SimpleNamespace(shape=(16,)))
    collector.log_info = MagicMock()
    replay_buffer = MagicMock()
    collect_episode = MagicMock(side_effect=lambda env, **_: env)
    wrappers = importlib.import_module("rlinf.envs.wrappers")
    monkeypatch.setattr(collect_real_data, "TrajectoryReplayBuffer", replay_buffer)
    monkeypatch.setattr(wrappers, "CollectEpisode", collect_episode)

    collector._initialize_collection_storage()

    collect_episode.assert_called_once()
    assert collect_episode.call_args.kwargs["copy_observations"] is False
    replay_buffer.assert_not_called()
    assert collector.buffer is None
    assert not (tmp_path / "demos").exists()


def test_run_collection_skips_trajectory_work_when_demos_disabled(monkeypatch):
    class FakeEnv:
        def reset(self):
            return {"unused": torch.zeros(1)}, {}

        def step(self, action):
            return (
                {"unused": torch.ones(1)},
                torch.tensor([1.0]),
                torch.tensor([True]),
                torch.tensor([False]),
                {},
            )

    collector = collect_real_data.DataCollector.__new__(collect_real_data.DataCollector)
    collector.cfg = OmegaConf.create(
        {
            "runner": {"logger": {"log_path": "/tmp"}},
            "env": {"eval": {"max_episode_steps": 10}},
        }
    )
    collector.env = FakeEnv()
    collector.save_demos = False
    collector.num_data_episodes = 1
    collector._preexisting_success = 0
    collector.total_cnt = 0
    collector.manual_episode_control_only = False
    collector.action_dim = 7
    collector._target_step_period = None
    collector.buffer = None
    collector.log_info = MagicMock()
    collector._process_obs = MagicMock(
        side_effect=AssertionError("_process_obs must not run")
    )

    builder = MagicMock(side_effect=AssertionError("builder must not be created"))
    chunk_step_result = MagicMock(
        side_effect=AssertionError("ChunkStepResult must not be created")
    )
    progress_bar = MagicMock()
    monkeypatch.setattr(collect_real_data, "EmbodiedTrajectoryBuilder", builder)
    monkeypatch.setattr(collect_real_data, "ChunkStepResult", chunk_step_result)
    monkeypatch.setattr(collect_real_data, "tqdm", MagicMock(return_value=progress_bar))

    collector._run_collection()

    collector._process_obs.assert_not_called()
    builder.assert_not_called()
    chunk_step_result.assert_not_called()
    progress_bar.update.assert_called_once_with(1)


def test_step_deadline_compensates_for_previous_oversleep(monkeypatch):
    collector = collect_real_data.DataCollector.__new__(collect_real_data.DataCollector)
    collector._target_step_period = 1.0 / 30.0
    sleep = MagicMock()
    perf_counter = MagicMock(side_effect=[0.010, 0.034])
    monkeypatch.setattr(collect_real_data.time, "sleep", sleep)
    monkeypatch.setattr(collect_real_data.time, "perf_counter", perf_counter)

    deadline = collector._wait_for_step_deadline(0.0)
    deadline = collector._wait_for_step_deadline(deadline)

    assert deadline == pytest.approx(2.0 / 30.0)
    assert [args[0] for args, _ in sleep.call_args_list] == pytest.approx(
        [1.0 / 30.0 - 0.010, 2.0 / 30.0 - 0.034]
    )

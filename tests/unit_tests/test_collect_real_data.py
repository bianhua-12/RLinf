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
    replay_buffer.assert_not_called()
    assert collector.buffer is None
    assert not (tmp_path / "demos").exists()

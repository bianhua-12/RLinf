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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryCache, TrajectoryReplayBuffer
from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


def test_mlp_policy_replay_projection_keeps_states_only():
    model = MLPPolicy(
        obs_dim=42,
        action_dim=8,
        num_action_chunks=1,
        add_value_head=False,
        add_q_head=True,
    )

    assert model.get_replay_buffer_projection() == {
        "curr_obs": ("states",),
        "next_obs": ("states",),
    }


def test_replay_buffer_projection_drops_image_obs():
    buffer = TrajectoryReplayBuffer(
        enable_cache=False,
        trajectory_projection={
            "curr_obs": ("states",),
            "next_obs": ("states",),
        },
    )
    trajectory = Trajectory(
        rewards=torch.zeros((2, 1), dtype=torch.float32),
        actions=torch.zeros((2, 1, 8), dtype=torch.float32),
        curr_obs={
            "states": torch.zeros((2, 1, 42), dtype=torch.float32),
            "main_images": torch.zeros((2, 1, 2, 2, 3), dtype=torch.uint8),
        },
        next_obs={
            "states": torch.ones((2, 1, 42), dtype=torch.float32),
            "extra_view_images": torch.zeros((2, 1, 2, 2, 3), dtype=torch.uint8),
        },
        forward_inputs={
            "states": torch.zeros((2, 1, 42), dtype=torch.float32),
            "action": torch.zeros((2, 1, 8), dtype=torch.float32),
        },
    )

    flat = buffer._flatten_trajectory(trajectory)

    assert set(flat["curr_obs"].keys()) == {"states"}
    assert set(flat["next_obs"].keys()) == {"states"}
    assert "forward_inputs" not in flat


def test_replay_cache_fails_fast_when_allocation_exceeds_available_memory(
    monkeypatch,
):
    cache = TrajectoryCache(max_size=8)
    monkeypatch.setattr(
        "rlinf.data.replay_buffer.psutil.virtual_memory",
        lambda: SimpleNamespace(available=1),
    )

    with pytest.raises(MemoryError, match="Replay buffer allocation exceeds"):
        cache.put(0, {"states": torch.zeros((4, 42), dtype=torch.float32)})


def test_sac_checkpoint_flags_can_disable_replay_buffer_resume():
    worker = object.__new__(EmbodiedSACFSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {
                "replay_buffer": {
                    "save_checkpoint": False,
                    "load_checkpoint": False,
                }
            }
        }
    )

    assert worker._should_save_replay_buffer_checkpoint() is False
    assert worker._should_load_replay_buffer_checkpoint() is False

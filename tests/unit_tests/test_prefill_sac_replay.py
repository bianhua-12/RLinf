# Copyright 2025 The RLinf Authors.
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

import pytest

from examples.embodiment.prefill_sac_replay import (
    compute_local_prefill_target_samples,
    extract_min_replay_stats,
)


def test_compute_local_prefill_target_samples_uses_local_episode_share():
    assert compute_local_prefill_target_samples(64, 50, 1) == 3200
    assert compute_local_prefill_target_samples(65, 50, 4) == 850


@pytest.mark.parametrize(
    ("num_episodes", "max_episode_steps", "actor_world_size"),
    [
        (0, 50, 1),
        (64, 0, 1),
        (64, 50, 0),
    ],
)
def test_compute_local_prefill_target_samples_rejects_invalid_inputs(
    num_episodes, max_episode_steps, actor_world_size
):
    with pytest.raises(ValueError):
        compute_local_prefill_target_samples(
            num_episodes=num_episodes,
            max_episode_steps=max_episode_steps,
            actor_world_size=actor_world_size,
        )


def test_extract_min_replay_stats_returns_minima_and_defaults():
    assert extract_min_replay_stats([]) == {
        "num_trajectories": 0.0,
        "total_samples": 0.0,
        "recv_queue": 0.0,
    }
    assert extract_min_replay_stats(
        [
            {"num_trajectories": 2, "total_samples": 3200, "recv_queue": 0},
            {"num_trajectories": 3, "total_samples": 4800, "recv_queue": 1},
        ]
    ) == {
        "num_trajectories": 2.0,
        "total_samples": 3200.0,
        "recv_queue": 0.0,
    }

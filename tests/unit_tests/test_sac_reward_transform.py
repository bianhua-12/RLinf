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

import torch
from omegaconf import OmegaConf

from rlinf.data.embodied_io_struct import Trajectory
from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy
from rlinf.workers.actor.reward_transform import (
    add_success_bonus_from_info_successes,
    apply_reward_transform_to_trajectory,
    broadcast_success_once_to_episode_steps,
    transform_rewards_with_gae_delta_sign,
    transform_trajectory_rewards_with_gae_delta_sign,
)


def test_transform_rewards_with_gae_delta_sign_matches_expected_pattern():
    rewards = torch.ones((6, 1, 1), dtype=torch.float32)
    dones = torch.zeros((7, 1, 1), dtype=torch.bool)

    transformed = transform_rewards_with_gae_delta_sign(
        rewards=rewards,
        dones=dones,
        gamma=1.0,
        gae_lambda=1.0,
        history_steps=2,
    )

    expected = torch.tensor([[[0.0]], [[0.0]], [[-1.0]], [[-1.0]], [[-1.0]], [[-1.0]]])
    assert torch.equal(transformed, expected)


def test_prepare_rollout_trajectories_for_replay_transforms_rewards_before_buffer_add():
    worker = object.__new__(EmbodiedSACFSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {
                "reward_transform": "gae_delta_sign_5_plus_success10",
                "gamma": 1.0,
                "gae_lambda": 1.0,
            }
        }
    )
    trajectory = Trajectory(
        rewards=torch.ones((6, 1, 1), dtype=torch.float32),
        dones=torch.zeros((6, 1, 1), dtype=torch.bool),
        successes=torch.tensor(
            [[[False]], [[False]], [[False]], [[False]], [[True]], [[True]]]
        ),
    )

    transformed = worker._prepare_rollout_trajectories_for_replay([trajectory])[0]

    base = torch.tensor([[[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[-1.0]]])
    expected = add_success_bonus_from_info_successes(
        base,
        torch.tensor([[[False]], [[False]], [[False]], [[False]], [[True]], [[True]]]),
        success_bonus=10.0,
    )
    assert torch.equal(transformed.rewards, expected)
    assert transformed.successes is None


def test_broadcast_success_once_marks_full_successful_episodes():
    successes = torch.tensor(
        [[[False]], [[True]], [[False]], [[False]], [[True]], [[False]]]
    )
    dones = torch.tensor(
        [[[False]], [[True]], [[False]], [[True]], [[False]], [[False]]]
    )

    mask = broadcast_success_once_to_episode_steps(successes=successes, dones=dones)

    expected = torch.tensor(
        [[[True]], [[True]], [[False]], [[False]], [[True]], [[True]]]
    )
    assert torch.equal(mask, expected)


def test_broadcast_success_once_accepts_t_plus_one_dones_layout():
    successes = torch.tensor(
        [[[False]], [[True]], [[False]], [[False]], [[True]], [[False]]]
    )
    dones = torch.tensor(
        [[[False]], [[False]], [[True]], [[False]], [[True]], [[False]], [[False]]]
    )

    mask = broadcast_success_once_to_episode_steps(successes=successes, dones=dones)

    expected = torch.tensor(
        [[[True]], [[True]], [[False]], [[False]], [[True]], [[True]]]
    )
    assert torch.equal(mask, expected)


def test_episode_success_once_only_overwrites_rewards_with_episode_mask():
    trajectory = Trajectory(
        rewards=torch.tensor(
            [[[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]], [[2.0]]],
            dtype=torch.float32,
        ),
        dones=torch.tensor(
            [[[False]], [[True]], [[False]], [[True]], [[False]], [[False]]]
        ),
        successes=torch.tensor(
            [[[False]], [[True]], [[False]], [[False]], [[True]], [[False]]]
        ),
    )

    transformed = apply_reward_transform_to_trajectory(
        trajectory,
        "episode_success_once_only",
        gamma=1.0,
        gae_lambda=1.0,
        clear_successes=True,
    )

    expected = torch.tensor([[[1.0]], [[1.0]], [[0.0]], [[0.0]], [[1.0]], [[1.0]]])
    assert torch.equal(transformed.rewards, expected)
    assert transformed.successes is None


def test_sign1_plus_episode_success10_uses_episode_broadcast_not_step_success():
    trajectory = Trajectory(
        rewards=torch.tensor(
            [[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]], [[6.0]]],
            dtype=torch.float32,
        ),
        dones=torch.tensor(
            [[[False]], [[True]], [[False]], [[True]], [[False]], [[False]]]
        ),
        successes=torch.tensor(
            [[[False]], [[True]], [[False]], [[False]], [[True]], [[False]]]
        ),
    )

    transformed = apply_reward_transform_to_trajectory(
        trajectory,
        "gae_delta_sign_1_plus_episode_success10",
        gamma=0.8,
        gae_lambda=0.9,
        clear_successes=True,
    )

    base = transform_trajectory_rewards_with_gae_delta_sign(
        rewards=torch.tensor(
            [[[1.0]], [[2.0]], [[3.0]], [[4.0]], [[5.0]], [[6.0]]],
            dtype=torch.float32,
        ),
        dones=torch.tensor(
            [[[False]], [[True]], [[False]], [[True]], [[False]], [[False]]]
        ),
        gamma=0.8,
        gae_lambda=0.9,
        history_steps=1,
    )
    episode_success_mask = torch.tensor(
        [[[True]], [[True]], [[False]], [[False]], [[True]], [[True]]]
    )
    expected = add_success_bonus_from_info_successes(
        base,
        episode_success_mask,
        success_bonus=10.0,
    )

    assert torch.equal(transformed.rewards, expected)
    assert transformed.successes is None


def test_prepare_rollout_trajectories_for_replay_supports_sign5_episode_success10():
    worker = object.__new__(EmbodiedSACFSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {
                "reward_transform": "gae_delta_sign_5_plus_episode_success10",
                "gamma": 1.0,
                "gae_lambda": 1.0,
            }
        }
    )
    trajectory = Trajectory(
        rewards=torch.ones((6, 1, 1), dtype=torch.float32),
        dones=torch.tensor(
            [[[False]], [[False]], [[False]], [[True]], [[False]], [[False]]]
        ),
        successes=torch.tensor(
            [[[False]], [[False]], [[True]], [[False]], [[True]], [[False]]]
        ),
    )

    transformed = worker._prepare_rollout_trajectories_for_replay([trajectory])[0]

    base = transform_trajectory_rewards_with_gae_delta_sign(
        rewards=torch.ones((6, 1, 1), dtype=torch.float32),
        dones=torch.tensor(
            [[[False]], [[False]], [[False]], [[True]], [[False]], [[False]]]
        ),
        gamma=1.0,
        gae_lambda=1.0,
        history_steps=5,
    )
    episode_success_mask = torch.tensor(
        [[[True]], [[True]], [[True]], [[True]], [[True]], [[True]]]
    )
    expected = add_success_bonus_from_info_successes(
        base,
        episode_success_mask,
        success_bonus=10.0,
    )
    assert torch.equal(transformed.rewards, expected)
    assert transformed.successes is None


def test_ppo_rollout_batch_reward_transform_applies_sign5_plus_success10():
    worker = object.__new__(EmbodiedFSDPActor)
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {
                "reward_transform": "gae_delta_sign_5_plus_success10",
                "gamma": 1.0,
                "gae_lambda": 1.0,
            }
        }
    )
    worker.rollout_batch = {
        "rewards": torch.ones((6, 1, 1), dtype=torch.float32),
        "dones": torch.zeros((7, 1, 1), dtype=torch.bool),
        "successes": torch.tensor(
            [[[False]], [[False]], [[False]], [[False]], [[True]], [[True]]]
        ),
    }

    worker._apply_reward_transform_to_rollout_batch()

    base = torch.tensor([[[0.0]], [[0.0]], [[0.0]], [[0.0]], [[0.0]], [[-1.0]]])
    expected = add_success_bonus_from_info_successes(
        base,
        torch.tensor([[[False]], [[False]], [[False]], [[False]], [[True]], [[True]]]),
        success_bonus=10.0,
    )
    assert torch.equal(worker.rollout_batch["rewards"], expected)


def test_ppo_rollout_batch_reward_transform_is_noop_when_unset():
    worker = object.__new__(EmbodiedFSDPActor)
    worker.cfg = OmegaConf.create({"algorithm": {"reward_transform": None}})
    original = torch.tensor([[[1.0]], [[2.0]]], dtype=torch.float32)
    worker.rollout_batch = {"rewards": original.clone()}

    worker._apply_reward_transform_to_rollout_batch()

    assert torch.equal(worker.rollout_batch["rewards"], original)

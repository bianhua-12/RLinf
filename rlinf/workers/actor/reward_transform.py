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

from __future__ import annotations

from collections.abc import Iterable

import torch

from rlinf.data.embodied_io_struct import Trajectory

SUPPORTED_REWARD_TRANSFORMS = {
    "episode_success_once_only",
    "gae_delta_sign_1_plus_episode_success10",
    "gae_delta_sign_5",
    "gae_delta_sign_5_plus_episode_success10",
    "gae_delta_sign_5_plus_success10",
}


def _flatten_time_major(tensor: torch.Tensor) -> torch.Tensor:
    num_chunk, bsz, chunk_size = tensor.shape
    return tensor.transpose(1, 2).reshape(num_chunk * chunk_size, bsz)


def _restore_time_major(
    tensor: torch.Tensor, *, num_chunk: int, bsz: int, chunk_size: int
) -> torch.Tensor:
    return tensor.reshape(num_chunk, chunk_size, bsz).transpose(1, 2).contiguous()


def _align_step_dones(
    successes: torch.Tensor,
    dones: torch.Tensor,
) -> torch.Tensor:
    if dones.shape == successes.shape:
        return dones
    if (
        dones.shape[0] == successes.shape[0] + 1
        and dones.shape[1] == successes.shape[1]
        and dones.shape[2] == successes.shape[2]
    ):
        num_chunk, bsz, chunk_size = successes.shape
        n_steps = num_chunk * chunk_size
        dones_full_flat = dones.transpose(1, 2).reshape(
            (num_chunk + 1) * chunk_size, bsz
        )
        dones_flat = dones_full_flat[-(n_steps + 1) :]
        aligned_flat = dones_flat[1:]
        return _restore_time_major(
            aligned_flat,
            num_chunk=num_chunk,
            bsz=bsz,
            chunk_size=chunk_size,
        )
    raise ValueError(
        "Unsupported dones layout for episode-success broadcast: "
        f"{successes.shape=} {dones.shape=}"
    )


def broadcast_success_once_to_episode_steps(
    successes: torch.Tensor,
    dones: torch.Tensor,
) -> torch.Tensor:
    """Broadcast any step-level success to the whole episode segment.

    Args:
        successes: Per-step success flags with shape ``[num_chunk, bsz, chunk_size]``.
        dones: Per-step episode-end flags aligned with ``successes``.

    Returns:
        Bool tensor with the same shape as ``successes``. For each environment,
        any episode segment that contains at least one successful step is marked
        ``True`` for every step in that segment.
    """
    dones = _align_step_dones(successes, dones)

    num_chunk, bsz, chunk_size = successes.shape
    successes_flat = _flatten_time_major(successes).bool()
    dones_flat = _flatten_time_major(dones).bool()
    success_mask_flat = torch.zeros_like(successes_flat)

    for env_idx in range(bsz):
        episode_start = 0
        episode_has_success = False
        for step_idx in range(successes_flat.shape[0]):
            if successes_flat[step_idx, env_idx]:
                episode_has_success = True
            if dones_flat[step_idx, env_idx]:
                if episode_has_success:
                    success_mask_flat[episode_start : step_idx + 1, env_idx] = True
                episode_start = step_idx + 1
                episode_has_success = False
        if episode_start < successes_flat.shape[0] and episode_has_success:
            success_mask_flat[episode_start:, env_idx] = True

    return _restore_time_major(
        success_mask_flat,
        num_chunk=num_chunk,
        bsz=bsz,
        chunk_size=chunk_size,
    )


def _get_history_steps_for_reward_transform(
    reward_transform: str, default_history_steps: int
) -> int:
    if reward_transform == "gae_delta_sign_1_plus_episode_success10":
        return 1
    return default_history_steps


def transform_rewards_with_gae_delta_sign(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    history_steps: int = 5,
) -> torch.Tensor:
    """Convert rewards to sign(gae[t] - gae[t-history_steps]) for each env.

    Args:
        rewards: Reward tensor with shape ``[num_chunk, bsz, chunk_size]``.
        dones: Done tensor with shape ``[num_chunk + 1, bsz, chunk_size]``.
        gamma: Discount factor.
        gae_lambda: GAE lambda.
        history_steps: Frame gap used to compute the GAE delta.

    Returns:
        Tensor with the same shape as ``rewards`` where each timestep reward is
        mapped to ``{-1, 0, 1}`` by the sign of the historical GAE delta.
    """
    num_chunk, bsz, chunk_size = rewards.shape
    n_steps = num_chunk * chunk_size

    rewards_flat = _flatten_time_major(rewards)
    flattened_dones_full = dones.transpose(1, 2).reshape(
        (num_chunk + 1) * chunk_size, bsz
    )
    dones_flat = flattened_dones_full[-(n_steps + 1) :]

    gae = torch.zeros_like(rewards_flat)
    running = torch.zeros(bsz, device=rewards.device, dtype=rewards.dtype)
    gamma_lambda = float(gamma) * float(gae_lambda)
    for step in reversed(range(n_steps)):
        not_done = (~dones_flat[step + 1]).to(rewards.dtype)
        running = rewards_flat[step] + gamma_lambda * not_done * running
        gae[step] = running

    transformed = torch.zeros_like(rewards_flat)
    if history_steps <= 0:
        transformed = torch.sign(gae)
    elif n_steps > history_steps:
        delta = gae[history_steps:] - gae[:-history_steps]
        transformed[history_steps:] = torch.sign(delta)

    return _restore_time_major(
        transformed,
        num_chunk=num_chunk,
        bsz=bsz,
        chunk_size=chunk_size,
    )


def add_success_bonus_from_info_successes(
    rewards: torch.Tensor,
    successes: torch.Tensor,
    success_bonus: float = 10.0,
) -> torch.Tensor:
    """Add a fixed bonus to each successful action step."""
    if rewards.shape != successes.shape:
        raise ValueError(
            f"Shape mismatch for rewards and successes: {rewards.shape=} vs {successes.shape=}"
        )
    return rewards + float(success_bonus) * successes.to(rewards.dtype)


def _transform_rewards_with_aligned_dones(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    history_steps: int,
) -> torch.Tensor:
    """Trajectory variant where ``dones`` is aligned 1:1 with ``rewards``."""
    if rewards.shape != dones.shape:
        raise ValueError(
            f"Expected rewards and dones to share shape, got {rewards.shape=} and {dones.shape=}"
        )

    num_chunk, bsz, chunk_size = rewards.shape
    n_steps = num_chunk * chunk_size
    rewards_flat = _flatten_time_major(rewards)
    dones_flat = _flatten_time_major(dones)

    gae = torch.zeros_like(rewards_flat)
    running = torch.zeros(bsz, device=rewards.device, dtype=rewards.dtype)
    gamma_lambda = float(gamma) * float(gae_lambda)
    for step in reversed(range(n_steps)):
        not_done = (~dones_flat[step]).to(rewards.dtype)
        running = rewards_flat[step] + gamma_lambda * not_done * running
        gae[step] = running

    transformed = torch.zeros_like(rewards_flat)
    if history_steps <= 0:
        transformed = torch.sign(gae)
    elif n_steps > history_steps:
        delta = gae[history_steps:] - gae[:-history_steps]
        transformed[history_steps:] = torch.sign(delta)

    return _restore_time_major(
        transformed,
        num_chunk=num_chunk,
        bsz=bsz,
        chunk_size=chunk_size,
    )


def transform_trajectory_rewards_with_gae_delta_sign(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    history_steps: int = 5,
) -> torch.Tensor:
    """Apply PPO-style sign(delta-GAE) transform to trajectory tensors.

    PPO rollout batches store ``dones`` with one extra step. SAC replay
    trajectories store per-action ``dones`` aligned with rewards. This helper
    accepts either layout and applies the matching recurrence.
    """
    if dones is None:
        raise ValueError("Trajectory reward transform requires dones.")

    if dones.shape[0] == rewards.shape[0]:
        return _transform_rewards_with_aligned_dones(
            rewards=rewards,
            dones=dones,
            gamma=gamma,
            gae_lambda=gae_lambda,
            history_steps=history_steps,
        )
    if dones.shape[0] == rewards.shape[0] + 1:
        return transform_rewards_with_gae_delta_sign(
            rewards=rewards,
            dones=dones,
            gamma=gamma,
            gae_lambda=gae_lambda,
            history_steps=history_steps,
        )
    raise ValueError(
        "Unsupported dones layout for trajectory reward transform: "
        f"{rewards.shape=} {dones.shape=}"
    )


def apply_reward_transform_to_trajectory(
    trajectory: Trajectory,
    reward_transform: str | None,
    *,
    gamma: float,
    gae_lambda: float,
    history_steps: int = 5,
    success_bonus: float = 10.0,
    clear_successes: bool = True,
) -> Trajectory:
    """Mutate one trajectory in-place before it enters SAC replay."""
    if reward_transform is None:
        if clear_successes:
            trajectory.successes = None
        return trajectory
    if reward_transform not in SUPPORTED_REWARD_TRANSFORMS:
        raise ValueError(
            f"Unsupported reward_transform={reward_transform!r}. "
            f"Supported: {sorted(SUPPORTED_REWARD_TRANSFORMS)}"
        )
    if trajectory.rewards is None:
        raise ValueError("reward_transform requires trajectory.rewards.")

    episode_success_mask = None
    if reward_transform in {
        "episode_success_once_only",
        "gae_delta_sign_1_plus_episode_success10",
        "gae_delta_sign_5_plus_episode_success10",
    }:
        if trajectory.successes is None:
            raise ValueError(
                f"reward_transform={reward_transform} requires trajectory.successes."
            )
        if trajectory.dones is None:
            raise ValueError(
                f"reward_transform={reward_transform} requires trajectory.dones."
            )
        episode_success_mask = broadcast_success_once_to_episode_steps(
            successes=trajectory.successes,
            dones=trajectory.dones,
        )

    if reward_transform == "episode_success_once_only":
        trajectory.rewards = episode_success_mask.to(trajectory.rewards.dtype)
    else:
        rewards = transform_trajectory_rewards_with_gae_delta_sign(
            rewards=trajectory.rewards,
            dones=trajectory.dones,
            gamma=gamma,
            gae_lambda=gae_lambda,
            history_steps=_get_history_steps_for_reward_transform(
                reward_transform, history_steps
            ),
        )
        if reward_transform == "gae_delta_sign_5_plus_success10":
            if trajectory.successes is None:
                raise ValueError(
                    "reward_transform=gae_delta_sign_5_plus_success10 requires "
                    "trajectory.successes."
                )
            rewards = add_success_bonus_from_info_successes(
                rewards=rewards,
                successes=trajectory.successes,
                success_bonus=success_bonus,
            )
        elif reward_transform in {
            "gae_delta_sign_1_plus_episode_success10",
            "gae_delta_sign_5_plus_episode_success10",
        }:
            rewards = add_success_bonus_from_info_successes(
                rewards=rewards,
                successes=episode_success_mask,
                success_bonus=success_bonus,
            )
        trajectory.rewards = rewards

    if clear_successes:
        trajectory.successes = None
    return trajectory


def apply_reward_transform_to_trajectories(
    trajectories: Iterable[Trajectory],
    reward_transform: str | None,
    *,
    gamma: float,
    gae_lambda: float,
    history_steps: int = 5,
    success_bonus: float = 10.0,
    clear_successes: bool = True,
) -> list[Trajectory]:
    """Apply a reward transform to a list of trajectories in-place."""
    transformed = []
    for trajectory in trajectories:
        transformed.append(
            apply_reward_transform_to_trajectory(
                trajectory,
                reward_transform,
                gamma=gamma,
                gae_lambda=gae_lambda,
                history_steps=history_steps,
                success_bonus=success_bonus,
                clear_successes=clear_successes,
            )
        )
    return transformed

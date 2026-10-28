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

"""Direct policy gradients through learned dynamics and a state reward function."""

import random
from contextlib import contextmanager
from types import SimpleNamespace

import numpy as np
import torch


@contextmanager
def isolated_rng(device):
    """Keep validation and checkpoint probes out of the training RNG stream."""
    device = torch.device(device)
    devices = [] if device.type == "cpu" else [device.index]
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices, device_type=device.type):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def ring_start_dict(start):
    """Adapt an original ring start without resampling or changing its context."""
    return {
        "state": start.state[None],
        "previous_state": start.xi[None, :, :48],
        "previous_action": start.xi[None, :, 48:56],
        "reset": start.xi[None, :, 56],
    }


@contextmanager
def frozen_parameters(module):
    """Freeze model weights while preserving derivatives with respect to inputs."""
    flags = [p.requires_grad for p in module.parameters()]
    try:
        module.requires_grad_(False)
        yield
    finally:
        for parameter, flag in zip(module.parameters(), flags):
            parameter.requires_grad_(flag)


def score_native_reward(native_reward, rule):
    """Convert bounded native progress rewards into a versioned training score."""
    if rule == "native":
        return native_reward
    if rule != "completion_cost_v1":
        raise ValueError(f"Unknown ReFL reward rule: {rule}")
    valid = torch.isfinite(native_reward) & (native_reward >= 0) & (native_reward <= 1)
    if not valid.all():
        raise ValueError("completion_cost_v1 requires native rewards in [0, 1]")
    # Native success is exactly one; all unfinished PickCube states score below it.
    # Squaring changes pre-success pathwise gradients; subtracting one alone cannot.
    return native_reward.square() - 1


def imagined_objective(
    actor,
    dynamics,
    reward,
    start,
    horizon,
    gamma,
    reward_rule="native",
    critic=None,
    return_rollout=False,
):
    """Return discounted post-action rewards without a value bootstrap.

    Terminal states are absorbing with zero subsequent score. Native scoring
    preserves the original objective. completion_cost_v1 charges unfinished
    transitions r**2-1 and scores native success as zero. Hard success masks
    remain detached. An optional frozen critic bootstraps each success boundary
    once and each surviving horizon boundary once, retaining state derivatives.
    """
    if horizon < 1 or not 0 < gamma <= 1:
        raise ValueError("Expected a positive horizon and gamma in (0, 1]")
    if critic is not None and reward_rule != "native":
        raise ValueError("Critic mode requires native rewards")
    state = start["state"][0].detach()
    context = {
        k: start[k][0].detach() for k in ("previous_state", "previous_action", "reset")
    }
    alive = torch.ones(state.shape[0], device=state.device)
    total = torch.zeros_like(alive)
    bootstrap = torch.zeros_like(alive)
    states, rewards, stops, active = [state], [], [], []
    for step in range(horizon):
        action = actor(state)
        predicted = dynamics.step(state, action, **context)
        next_state = torch.where(alive[:, None].bool(), predicted, state)
        native_reward = reward(next_state)
        score = score_native_reward(native_reward, reward_rule)
        total = total + gamma**step * alive * score
        stop = reward.terminated(next_state).detach()
        if critic is not None:
            bootstrap = bootstrap + gamma ** (step + 1) * alive * stop * critic(
                next_state
            )
        if return_rollout:
            states.append(next_state)
            rewards.append(native_reward)
            stops.append(stop)
            active.append(alive)
        alive = alive * (~stop).float()
        context = {
            "previous_state": state,
            "previous_action": action,
            "reset": torch.zeros_like(alive),
        }
        state = next_state
    if critic is not None:
        bootstrap = bootstrap + gamma**horizon * alive * critic(state)
    objective = (total + bootstrap).mean()
    if not return_rollout:
        return objective
    return objective, SimpleNamespace(
        states=torch.stack(states),
        rewards=torch.stack(rewards),
        stops=torch.stack(stops),
        active=torch.stack(active),
        bootstrap=bootstrap.mean(),
        reward_objective=total.mean(),
    )


@torch.no_grad()
def bootstrapped_lambda_targets(rewards, stops, values, gamma, gae_lambda):
    """Cut lambda traces at success while retaining V(final_observation), as PPO does."""
    if not 0 < gamma <= 1 or not 0 <= gae_lambda <= 1:
        raise ValueError("Invalid gamma or GAE lambda")
    if (
        values.shape != (rewards.shape[0] + 1, *rewards.shape[1:])
        or stops.shape != rewards.shape
    ):
        raise ValueError("Expected H rewards/stops and H+1 value rows")
    result = []
    target = values[-1]
    for step in range(len(rewards) - 1, -1, -1):
        continuation = (1 - gae_lambda) * values[step + 1] + gae_lambda * target
        target = rewards[step] + gamma * torch.where(
            stops[step], values[step + 1], continuation
        )
        result.append(target)
    return torch.stack(result[::-1])


def checkpoint_uses_critic(bundle):
    """Reject contradictory value-network checkpoint contracts before loading."""
    version = bundle.get("format")
    config = bundle.get("config", {})
    if version == "rlinf-refl-pickcube-v1":
        enabled = False
        if bundle.get("config", {}).get("use_critic", False):
            raise ValueError("Historical v1 checkpoints cannot enable a critic")
    elif version == "rlinf-refl-pickcube-v2":
        enabled = bundle.get("config", {}).get("use_critic")
        if not isinstance(enabled, bool):
            raise ValueError("v2 checkpoint requires explicit use_critic")
    else:
        raise ValueError("Unknown ReFL checkpoint format")
    rule = config.get("reward_rule", "native" if version.endswith("v1") else None)
    if rule not in ("native", "completion_cost_v1"):
        raise ValueError("Unknown checkpoint reward semantics")
    if enabled and (
        rule != "native" or config.get("critic_bootstrap") != "success_and_horizon"
    ):
        raise ValueError("Unknown checkpoint critic semantics")
    if any((name in bundle) != enabled for name in ("critic", "critic_optimizer")):
        raise ValueError("Critic payload/config mismatch")
    if ("value" in bundle.get("probe", {})) != enabled:
        raise ValueError("Critic probe/config mismatch")
    return enabled


class TransitionWindows:
    """Sample contiguous live transitions from explicitly selected environment IDs."""

    def __init__(self, data, env_ids, horizon=3):
        if horizon < 1 or len(data["state"]) < horizon:
            raise ValueError("Not enough transitions for the requested horizon")
        self.data, self.horizon = data, horizon
        count = len(data["state"]) - horizon + 1
        valid = torch.ones(
            (count, len(env_ids)), dtype=torch.bool, device=env_ids.device
        )
        for offset in range(horizon):
            valid &= data["alive"][offset : offset + count, env_ids]
            valid &= (
                data["episode_id"][offset : offset + count, env_ids]
                == data["episode_id"][:count, env_ids]
            )
            if offset < horizon - 1:
                valid &= ~(data["terminated"] | data["truncated"])[
                    offset : offset + count, env_ids
                ]
            if offset:
                valid &= ~data["reset"][offset : offset + count, env_ids].bool()
        self.starts, local_envs = torch.nonzero(valid, as_tuple=True)
        self.envs = env_ids[local_envs]
        if not self.starts.numel():
            raise ValueError("No live reset-free windows in selected environments")

    def sample(self, count, generator=None):
        picks = torch.randint(
            len(self.starts), (count,), device=self.starts.device, generator=generator
        )
        times = (
            self.starts[picks][None]
            + torch.arange(self.horizon, device=self.starts.device)[:, None]
        )
        envs = self.envs[picks][None].expand_as(times)
        return {key: value[times, envs] for key, value in self.data.items()}


def validate_source_columns(source_env, num_envs):
    """Bind column-based splits to recorded simulator source identities."""
    if source_env.ndim != 2 or source_env.shape[1] != num_envs:
        raise ValueError("Unexpected source_env layout")
    expected = torch.arange(num_envs, device=source_env.device).expand_as(source_env)
    if not torch.equal(source_env, expected):
        raise ValueError("Dataset source_env columns do not match the declared split")


def world_model_batch(batch):
    """Adapt executed-action windows to the preserved physical48 WM loss."""
    return SimpleNamespace(
        states=torch.cat([batch["state"][:1], batch["next_state"]], 0),
        a_exec=batch["action"],
        xi=torch.cat(
            [
                batch["previous_state"][0],
                batch["previous_action"][0],
                batch["reset"][0, :, None].float(),
            ],
            -1,
        ),
    )


def mixed_windows(fixed, online, batch_size):
    """Use equal fixed/online populations with identical transition semantics."""
    if batch_size < 2 or batch_size % 2:
        raise ValueError("Expected an even batch size of at least two")
    a, b = fixed.sample(batch_size // 2), online.sample(batch_size // 2)
    if a.keys() != b.keys():
        raise ValueError("Fixed and online transition fields differ")
    return {key: torch.cat([a[key], b[key]], 1) for key in a}

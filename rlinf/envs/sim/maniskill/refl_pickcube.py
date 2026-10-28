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

"""Native PickCube rewards and WoVR-style keyframe-initialized collection.

Adapted from SuperMimic under the MIT license in
rlinf/models/embodiment/refl_pickcube.LICENSE.
"""

from __future__ import annotations

import hashlib
import inspect
from types import MethodType
from types import SimpleNamespace as NS
from typing import Any

import torch

from rlinf.models.embodiment.refl_pickcube import neural_forward


def _vel(actor: Any, name: str) -> torch.Tensor:
    value = getattr(actor, name, None)
    if value is None:
        value = getattr(actor, f"get_{name}")()
    return value


def tree_map(fn, x):
    if isinstance(x, dict):
        return {k: tree_map(fn, v) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return fn(x)
    raise TypeError(f"Unsupported snapshot leaf: {type(x)}")


def assert_layout(env, s):
    b = env.unwrapped
    a = b.agent
    info = b.evaluate()
    expected = torch.cat(
        [
            a.robot.get_qpos(),
            a.robot.get_qvel(),
            info["is_grasped"][:, None].float(),
            a.tcp.pose.raw_pose,
            b.goal_site.pose.p,
            b.cube.pose.raw_pose,
            b.cube.pose.p - a.tcp.pose.p,
            b.goal_site.pose.p - b.cube.pose.p,
        ],
        -1,
    )
    torch.testing.assert_close(s, expected, rtol=0, atol=1e-6)


def restore(env, snap):
    b = env.unwrapped
    env.reset(options={"reset_to_env_states": {"env_states": snap["sim"]}})
    if "controller" in snap["sim"]:
        b.agent.set_controller_state(snap["sim"]["controller"])
    for key, c in b.agent.controller.controllers.items():
        for name, value in snap["controller"][key].items():
            setattr(c, name, value.clone())
        if "_target_qpos" in snap["controller"][key]:
            c.set_drive_targets(c._target_qpos)
    b._elapsed_steps.zero_()
    b.scene._gpu_apply_all()
    b.scene.px.gpu_update_articulation_kinematics()
    b.scene._gpu_fetch_all()
    return b.get_obs().detach().clone()


def phases(s, info, previous):
    near = s[:, 36:39].norm(dim=-1) < 0.06
    grasp = info["is_grasped"]
    goal = s[:, 39:42].norm(dim=-1) < 0.05
    return {
        "approach": ~near & ~grasp,
        "near_ungrasped": near & ~grasp,
        "grasped": grasp,
        "transport": grasp & ~goal,
        "near_goal": goal,
        "success": info["success"],
        "dropped": previous[:, 18].bool() & ~grasp & ~info["success"],
    }


@torch.no_grad()
def physical_state(env, observation):
    cube = env.unwrapped.cube
    return torch.cat(
        [observation, _vel(cube, "linear_velocity"), _vel(cube, "angular_velocity")], -1
    ).clone()


@torch.no_grad()
def collect(env, policy, bank, sampler, iteration, seed, train_ids):
    n = env.unwrapped.num_envs
    device = train_ids.device
    native = iteration % 5 == 0
    if n != 4096:
        raise ValueError("Reference bank requires its original4096-environment layout")
    if native:
        obs, _ = env.reset(seed=seed + iteration)
        length = 50
        bins = None
    else:
        frames, bins = sampler.sample()
        held_frames, held_bins = sampler.sample(adaptive=False)
        frames[3584:] = held_frames[3584:]
        bins[3584:] = held_bins[3584:]
        ids = torch.arange(n, device=device)
        obs = restore(env, tree_map(lambda x: x[frames, ids], bank["states"]))
        length = 32
    assert_layout(env, obs)
    previous = physical_state(env, obs)
    previous_action = torch.zeros(n, 8, device=device)
    alive = torch.ones(n, dtype=torch.bool, device=device)
    returns = torch.zeros(n, device=device)
    success = torch.zeros_like(alive)
    grasp = success.clone()
    parts = {}
    for t in range(length):
        s = physical_state(env, obs)
        clean = policy(obs)
        action = (clean + 0.1 * torch.randn_like(clean)).clamp(-1, 1)
        nxt, reward, terminated, truncated, info = env.step(action)
        next_state = physical_state(env, nxt)
        cut = truncated | ~alive
        if t == length - 1:
            cut = cut | ~terminated
        row = dict(
            state=s,
            next_state=next_state,
            action=action,
            clean_action=clean,
            reward=reward,
            terminated=terminated,
            truncated=cut,
            alive=alive,
            previous_state=previous,
            previous_action=previous_action,
            reset=torch.full_like(alive, t == 0),
            native=torch.full_like(alive, native),
            source_env=torch.arange(n, device=device),
            episode_id=torch.arange(n, device=device) + iteration * n,
            **phases(nxt, info, obs),
        )
        for key, value in row.items():
            parts.setdefault(key, []).append(value.detach().clone())
        returns += (sampler.gamma**t) * reward * alive
        success |= info["success"] & alive
        grasp |= info["is_grasped"] & alive
        alive = alive & ~(terminated | truncated)
        previous, previous_action = s, action
        obs = nxt
    if bins is not None:
        sampler.update(bins, returns, torch.arange(n, device=device) < 3584)
    data = {k: torch.stack(v) for k, v in parts.items()}
    return data, {
        "origin": "native" if native else "reference",
        "vector_steps": length,
        "transitions": int(data["alive"][:, train_ids].sum()),
        "success_rate": float(success[train_ids].float().mean()),
        "grasp_rate": float(grasp[train_ids].float().mean()),
        "sampler_return": float(returns[train_ids].mean()),
    }


@torch.no_grad()
def evaluate_policy(env, policy, seed, episodes=512):
    n = env.unwrapped.num_envs
    if episodes % n:
        raise ValueError("Evaluation population must be a multiple of scene size")
    successes = []
    returns = []
    grasps = []
    saturations = []
    for batch in range(episodes // n):
        obs, _ = env.reset(seed=seed + batch)
        total = torch.zeros(n, device=obs.device)
        success = torch.zeros(n, dtype=torch.bool, device=obs.device)
        grasp = success.clone()
        for _ in range(50):
            raw = neural_forward(
                policy.net, obs, precision=getattr(policy, "precision", "float32")
            )
            obs, r, _, _, info = env.step(raw.clamp(-1, 1))
            total += r
            success |= info["success"]
            grasp |= info["is_grasped"]
            saturations.append((raw.abs() > 1).float().mean())
        successes.append(success)
        returns.append(total)
        grasps.append(grasp)
    successes = torch.cat(successes)
    returns = torch.cat(returns)
    grasps = torch.cat(grasps)
    metrics = {
        "episodes": episodes,
        "successes": int(successes.sum()),
        "success_rate": float(successes.float().mean()),
        "mean_return": float(returns.mean()),
        "grasp_rate": float(grasps.float().mean()),
        "action_saturation": float(torch.stack(saturations).mean()),
        "seed_start": seed,
        "horizon": 50,
    }
    return metrics, {
        "success": successes.cpu(),
        "returns": returns.cpu(),
        "grasp": grasps.cpu(),
    }


class NativeRewardOracle:
    def __init__(self):
        env = make_env("PickCube-v1", 1)
        base = env.unwrapped
        self.task = type(base)
        self.static = type(base.agent).is_static
        self.goal_thresh = base.goal_thresh
        self.robot_uids = base.robot_uids
        self.provenance = {
            name: hashlib.sha256(inspect.getsource(fn).encode()).hexdigest()
            for name, fn in [
                ("evaluate", self.task.evaluate),
                ("dense_reward", self.task.compute_dense_reward),
                ("normalized_dense_reward", self.task.compute_normalized_dense_reward),
                ("is_static", self.static),
            ]
        }
        env.close()

    def _proxy(self, state):
        s = state.reshape(-1, state.shape[-1])[..., :42]
        agent = NS(
            robot=NS(get_qvel=lambda: s[:, 9:18]), tcp=NS(pose=NS(p=s[:, 19:22]))
        )
        agent.tcp_pose = agent.tcp.pose
        agent.is_static = lambda threshold: self.static(agent, threshold)
        agent.is_grasping = lambda cube: s[:, 18].clamp(0, 1)
        proxy = NS(
            agent=agent,
            cube=NS(pose=NS(p=s[:, 29:32])),
            goal_site=NS(pose=NS(p=s[:, 26:29])),
            goal_thresh=self.goal_thresh,
            robot_uids=self.robot_uids,
        )
        proxy.compute_dense_reward = MethodType(self.task.compute_dense_reward, proxy)
        return proxy

    @torch.no_grad()
    def terminated(self, state):
        # PickCube's native BaseEnv.step terminates exactly on evaluate.success.
        # This task adapter uses a hard, detached mask; no learned done head.
        return self.task.evaluate(self._proxy(state))["success"].reshape(
            state.shape[:-1]
        )

    def __call__(self, state):
        proxy = self._proxy(state)
        info = self.task.evaluate(proxy)
        r = self.task.compute_normalized_dense_reward(proxy, None, None, info)
        return r.reshape(state.shape[:-1])


def make_env(env_id, num_envs):
    """Use the PPO790 control, observation, and reward contract."""
    import gymnasium as gym
    import mani_skill.envs  # noqa: F401

    if env_id != "PickCube-v1":
        raise ValueError("Expected PickCube-v1")
    env = gym.make(
        env_id,
        num_envs=num_envs,
        obs_mode="state",
        control_mode="pd_joint_delta_pos",
        reward_mode="normalized_dense",
        sim_backend="gpu",
        render_backend="none",
        max_episode_steps=50,
        reconfiguration_freq=0,
        sim_config={"sim_freq": 100, "control_freq": 20},
    )
    if env.unwrapped.single_action_space.shape != (8,):
        env.close()
        raise ValueError("Expected eight-action Panda control")
    return env


@torch.no_grad()
def validate_reference_bank(env, bank):
    """Recheck restored frames against cached next-step physics on this runtime.

    Invalid rows are excluded in a run-local mask; the source bank is immutable.
    Tolerances match the original bank collector.
    """
    valid = []
    pose_ids = list(range(9)) + list(range(19, 36))
    for frame in range(len(bank["valid"])):
        obs = restore(env, tree_map(lambda value: value[frame], bank["states"]))
        assert_layout(env, obs)
        pose_error = (
            (obs[:, pose_ids] - bank["observations"][frame, :, pose_ids]).abs().amax(-1)
        )
        nxt, reward, _, _, info = env.step(bank["actions"][frame])
        expected = bank["next_states"][frame]
        accepted = (
            (pose_error < 1e-4)
            & ((nxt[:, 29:32] - expected[:, 29:32]).norm(dim=-1) < 0.003)
            & ((nxt[:, :9] - expected[:, :9]).abs().amax(-1) < 0.01)
            & ((reward - bank["rewards"][frame]).abs() < 0.05)
            & (info["is_grasped"] == expected[:, 18].bool())
            & bank["valid"][frame]
        )
        valid.append(accepted)
    result = torch.stack(valid)
    if not result.any(0).all():
        raise ValueError("Reference bank has environments without a reproducible frame")
    return result


class KIRSampler:
    """Sample restorable frames from four task-critical state groups."""

    gamma = 0.99

    def __init__(self, bank):
        self.observations = bank["observations"]
        self.valid = bank["valid"]
        self.times = bank["times"]
        obs = self.observations
        near = obs[..., 36:39].norm(dim=-1) < 0.06
        grasp = obs[..., 18] >= 0.5
        goal = obs[..., 39:42].norm(dim=-1) < 0.05
        self.masks = [
            self.valid & (~near) & (~grasp),
            self.valid & grasp & (~goal),
            self.valid & goal,
            self.valid & near & (~grasp),
        ]
        self.group_counts = [int(m.sum()) for m in self.masks]

    def sample(self, adaptive=True):
        n = self.valid.shape[1]
        group = torch.arange(n, device=self.valid.device) % len(self.masks)
        group = group[torch.randperm(n, device=self.valid.device)]
        eligible = torch.stack(
            [self.masks[g][:, i] for i, g in enumerate(group)], dim=1
        )
        fallback = ~eligible.any(0)
        if fallback.any():
            eligible[:, fallback] = self.valid[:, fallback]
        frames = torch.multinomial(eligible.T.float(), 1).squeeze(-1)
        return frames, group

    def update(self, group, returns, env_mask=None):
        # KIR deliberately keeps the phase mixture fixed; it does not turn into
        # a return-bin sampler after policy performance changes.
        return None


class BinSampler:
    """One-second bins; native discounted return, fixed Gmax=1/(1-gamma).

    A bin refers to time within each source environment's reference trajectory.
    Environment source IDs never cross the training/validation partition.
    """

    def __init__(self, valid, frame_times, gamma=0.99):
        self.valid, self.times, self.gamma = valid, frame_times, gamma
        self.bins = (frame_times // 20).long()
        self.nbins = int(self.bins.max()) + 1
        self.returns = torch.zeros(self.nbins, valid.shape[1], device=valid.device)
        self.memory = torch.ones_like(self.returns)
        self.available = torch.stack(
            [valid[self.bins == b].any(0) for b in range(self.nbins)]
        )
        if not self.available.any(0).all():
            raise ValueError("An environment has no restorable reference frames")
        self.last = {}

    def weights(self):
        uniform = self.available.float() / self.available.sum(0)
        logits = -25 * self.returns * (1 - self.gamma)
        hard = torch.softmax(logits.masked_fill(~self.available, -torch.inf), dim=0)
        hist = self.memory * self.available
        hist = hist / hist.sum(0).clamp_min(1e-12)
        return 0.5 * uniform + 0.4 * hard + 0.1 * hist, hard

    def sample(self, generator=None, adaptive=True):
        w, hard = self.weights()
        if not adaptive:
            w = self.available.float() / self.available.sum(0)
        selected = torch.multinomial(w.T, 1, generator=generator).squeeze(-1)
        eligible = self.valid & (self.bins[:, None] == selected[None])
        frames = torch.multinomial(eligible.T.float(), 1, generator=generator).squeeze(
            -1
        )
        self.last = {
            "bin_ids": selected,
            "frames": frames,
            "probabilities": w,
            "hard": hard,
        }
        return frames, selected

    def update(self, bin_ids, returns, env_mask=None):
        _, hard = self.weights()
        ids = torch.arange(self.valid.shape[1], device=self.valid.device)
        if env_mask is not None:
            ids = ids[env_mask]
        b = bin_ids[ids]
        self.returns[b, ids] = 0.8 * self.returns[b, ids] + 0.2 * returns[ids]
        # Keep validation environments' historical weights untouched.
        self.memory[:, ids] = 0.9 * self.memory[:, ids] + 0.1 * hard[:, ids]

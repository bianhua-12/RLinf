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

"""Pinned ManiSkill update equations from SuperMimic56814f6.

MIT copyright2026 Yuxi Liu, license in models/embodiment/refl_pickcube.LICENSE.
Only the native explicit-rollout/loss adapters are supported here.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import TYPE_CHECKING

import torch
import torch.distributed as dist
from torch import nn

from rlinf.data.refl_replay import ReplayBuffer, WindowBatch

if TYPE_CHECKING:
    from typing import Any as MotionLibrary

    from torch.nn import Module as Critic
    from torch.nn import Module as Policy
    from torch.nn import Module as WorldModel


def world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


@torch.no_grad()
def allreduce_grads(module: torch.nn.Module) -> None:
    """Average gradients across ranks (flattened into one buffer per dtype)."""
    if world_size() == 1:
        return
    grads = [p.grad for p in module.parameters() if p.grad is not None]
    if not grads:
        return
    flat = torch.cat([g.reshape(-1) for g in grads])
    dist.all_reduce(flat, op=dist.ReduceOp.AVG)
    offset = 0
    for g in grads:
        n = g.numel()
        g.copy_(flat[offset : offset + n].view_as(g))
        offset += n


@dataclass
class ImagRollout:
    states: torch.Tensor  # (H+1, n, state_dim)
    actions: torch.Tensor  # (H, n, action_dim)
    rewards: torch.Tensor  # (H, n)      r_{t+1}
    dhat: torch.Tensor  # (H+1, n)    predicted termination prob of s_t (d_0 = 0)
    survival: torch.Tensor  # (H+1, n)    c_t
    goals: torch.Tensor  # (H+1, n, goal_dim)
    e_root: torch.Tensor  # (H+1, n, 4)
    frames: torch.Tensor  # (H+1, n)
    reward_terms: dict  # mean of each reward term over the rollout (diagnostics)


def policy_objective(
    ro: ImagRollout, terminal_values: torch.Tensor | None, gamma: float
) -> torch.Tensor:
    """J = mean_n [ sum_t gamma^t c_t r_{t+1} + gamma^H c_H V_H ] (SHAC-style; V_H keeps the gradient w.r.t. s_H).

    terminal_values: (n,) critic estimate at s_H, or None for the critic-free variant.
    """
    H = ro.rewards.shape[0]
    disc = gamma ** torch.arange(H, device=ro.rewards.device, dtype=ro.rewards.dtype)
    J = (disc[:, None] * ro.survival[:-1] * ro.rewards).sum(0)
    if terminal_values is not None:
        J = J + (gamma**H) * ro.survival[-1] * terminal_values
    return J.mean()


@torch.no_grad()
def gae_targets(
    rewards: torch.Tensor,
    dhat_next: torch.Tensor,
    values: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """R_t = r_{t+1} + gamma (1 - d_{t+1}) [ (1-lam) V_{t+1} + lam R_{t+1} ],  R_H = V_H.

    rewards (H, n), dhat_next (H, n) = d_{t+1}, values (H+1, n) = V_0..V_H. Returns (H, n) targets.
    """
    H = rewards.shape[0]
    R = values[H].clone()
    out = torch.empty_like(rewards)
    for t in range(H - 1, -1, -1):
        R = rewards[t] + gamma * (1.0 - dhat_next[t]) * (
            (1.0 - lam) * values[t + 1] + lam * R
        )
        out[t] = R
    return out


@dataclass
class PolicyTrainerCfg:
    horizon: int = 32
    batch_size: int = 4096
    history: int = 10
    future_frames: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.95)
    grad_clip_norm: float = 1.0
    grad_clip_value: float = 0.1
    critic_lr: float = 2e-4
    critic_grad_clip: float = 1.0
    critic_epochs: int = 5
    critic_minibatches: int = 4
    use_critic: bool = True
    obs_noise: bool = True
    compile: str | None = "default"
    anchor_obs: bool = (
        False  # None | "default" | "reduce-overhead" | "max-autotune-no-cudagraphs"
    )


class PolicyTrainer:
    def __init__(
        self,
        wm: WorldModel,
        policy: Policy,
        critic: Critic | None,
        lib: MotionLibrary,
        cfg: PolicyTrainerCfg,
        *,
        rollout_fn=None,
        critic_inputs_fn=None,
    ):
        if cfg.use_critic and critic is None:
            raise ValueError("use_critic=True requires a critic")
        self.wm, self.policy, self.critic, self.lib, self.cfg = (
            wm,
            policy,
            critic,
            lib,
            cfg,
        )
        self.opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr, betas=cfg.betas)
        self.critic_opt = (
            torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr)
            if cfg.use_critic
            else None
        )
        self.rollout_fn = rollout_fn
        self.critic_inputs_fn = critic_inputs_fn
        if rollout_fn is None:
            raise ValueError("Native ManiSkill requires an explicit rollout adapter")

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr
        if self.critic_opt is not None:
            for g in self.critic_opt.param_groups:
                g["lr"] = lr

    def _critic_inputs(self, ro: ImagRollout):
        if self.critic_inputs_fn is not None:
            return self.critic_inputs_fn(ro)
        H1, n, _ = ro.states.shape
        feat = self.wm.spec.canonical(ro.states.reshape(H1 * n, -1)).reshape(H1, n, -1)
        return feat, ro.goals, ro.e_root

    def update(self, buffer: ReplayBuffer, xi_norm_fn=None) -> dict[str, float]:
        cfg = self.cfg
        start = buffer.sample_starts(cfg.batch_size, cfg.history)

        # ---- actor: BPTT through the frozen world model ------------------------------------
        self.wm.requires_grad_(False)
        if cfg.use_critic:
            self.critic.requires_grad_(False)
        if self.rollout_fn is None:
            raise ValueError("Native ManiSkill requires an explicit rollout adapter")
        else:
            ro = self.rollout_fn(start, cfg.horizon)
        if cfg.use_critic:
            xi_norm = self.wm.spec.normalize_xi(start.xi)
            feat, goals, e_root = self._critic_inputs(ro)
            v_H = self.critic(feat[-1], goals[-1], xi_norm, e_root[-1])
        else:
            v_H = None
        J = policy_objective(ro, v_H, cfg.gamma)
        loss = -J / cfg.horizon
        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        allreduce_grads(self.policy)
        gn = torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), cfg.grad_clip_norm, error_if_nonfinite=True
        )
        torch.nn.utils.clip_grad_value_(self.policy.parameters(), cfg.grad_clip_value)
        self.opt.step()
        self.wm.requires_grad_(True)
        if cfg.use_critic:
            self.critic.requires_grad_(True)

        metrics = {
            "imag_objective": float(J.detach()),
            "imag_reward_discounted": float(
                policy_objective(ro, None, cfg.gamma).detach()
            ),
            "imag_bootstrap": float(
                (J - policy_objective(ro, None, cfg.gamma)).detach()
            ),
            "imag_return_undisc": float(
                (ro.survival[:-1] * ro.rewards).sum(0).mean().detach()
            ),
            "imag_reward_mean": float(ro.rewards.mean().detach()),
            "imag_survival_H": float(ro.survival[-1].mean().detach()),
            "imag_dhat_mean": float(ro.dhat[1:].mean().detach()),
            "policy_grad_norm": float(gn),
            "action_abs_mean": float(ro.actions.abs().mean().detach()),
        }
        for k, v in ro.reward_terms.items():
            metrics[f"imag_r/{k}"] = v

        # ---- critic: fit to GAE targets of this rollout (targets fixed) ---------------------
        if cfg.use_critic:
            with torch.no_grad():
                feat_d, goals_d, e_root_d = (
                    feat.detach(),
                    goals.detach(),
                    e_root.detach(),
                )
                H1, n, _ = feat_d.shape
                xi_rep = xi_norm[None].expand(H1, n, -1)
                values = self.critic(
                    feat_d.reshape(H1 * n, -1),
                    goals_d.reshape(H1 * n, -1),
                    xi_rep.reshape(H1 * n, -1),
                    e_root_d.reshape(H1 * n, -1),
                ).reshape(H1, n)
                targets = gae_targets(
                    ro.rewards.detach(),
                    ro.dhat[1:].detach(),
                    values,
                    cfg.gamma,
                    cfg.gae_lambda,
                )  # (H, n)
                X_feat = feat_d[:-1].reshape(-1, feat_d.shape[-1])
                X_goal = goals_d[:-1].reshape(-1, goals_d.shape[-1])
                X_xi = xi_rep[:-1].reshape(-1, xi_rep.shape[-1])
                X_er = e_root_d[:-1].reshape(-1, 4)
                Y = targets.reshape(-1)
            N = Y.shape[0]
            mb = N // cfg.critic_minibatches
            last = torch.zeros((), device=Y.device)
            for _ in range(cfg.critic_epochs):
                perm = torch.randperm(N, device=Y.device)
                for i in range(cfg.critic_minibatches):
                    idx = perm[i * mb : (i + 1) * mb]
                    pred = self.critic(X_feat[idx], X_goal[idx], X_xi[idx], X_er[idx])
                    closs = torch.nn.functional.mse_loss(pred, Y[idx])
                    self.critic_opt.zero_grad(set_to_none=True)
                    closs.backward()
                    allreduce_grads(self.critic)
                    torch.nn.utils.clip_grad_norm_(
                        self.critic.parameters(),
                        cfg.critic_grad_clip,
                        error_if_nonfinite=True,
                    )
                    self.critic_opt.step()
                    last = closs.detach()
            metrics["critic_loss"] = float(last)
            metrics["critic_value_mean"] = float(values.mean())
            metrics["critic_target_mean"] = float(targets.mean())
        return metrics


@dataclass
class WMTrainerCfg:
    window: int = 8
    batches_per_update: int = 10
    batch_size: int = 2048
    lr: float = 2e-4
    grad_clip: float = 1.0
    weight_decay: float = 0.0
    compile: str | None = "default"


class WorldModelTrainer:
    def __init__(
        self, wm: WorldModel, lib: MotionLibrary, cfg: WMTrainerCfg, *, loss_fn=None
    ):
        self.wm = wm
        self.lib = lib
        self.cfg = cfg
        self.loss_fn = loss_fn
        self.opt = torch.optim.Adam(
            wm.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        wm.enable_compile(cfg.compile)

    def set_lr(self, lr: float) -> None:
        for g in self.opt.param_groups:
            g["lr"] = lr

    def loss_on(self, w: WindowBatch):
        if self.loss_fn is not None:
            return self.loss_fn(w)
        raise ValueError("Native ManiSkill requires an explicit world-model loss")

    def update(self, buffer: ReplayBuffer) -> dict[str, float]:
        agg: dict[str, torch.Tensor] = {}
        for _ in range(self.cfg.batches_per_update):
            w = buffer.sample_windows(self.cfg.batch_size, self.cfg.window)
            loss, m = self.loss_on(w)
            self.opt.zero_grad(set_to_none=True)
            loss.backward()
            allreduce_grads(self.wm)
            gn = torch.nn.utils.clip_grad_norm_(
                self.wm.parameters(), self.cfg.grad_clip, error_if_nonfinite=True
            )
            self.opt.step()
            m["loss"] = loss.detach()
            m["grad_norm"] = gn.detach()
            for k, v in m.items():
                agg[k] = agg.get(k, 0.0) + v
        # Single host sync per update (keeps the CPU queuing GPU work across minibatches).
        return {k: float(v) / self.cfg.batches_per_update for k, v in agg.items()}

    @torch.no_grad()
    def online_error(
        self, buffer: ReplayBuffer, n: int = 1024, k: int = 8
    ) -> torch.Tensor:
        """k-step open-loop error on (recent) replay windows; (k,) tensor."""
        w = buffer.sample_windows(n, k)
        return self.wm.rollout_error(w.states, w.a_exec, w.xi)


def context_from_xi(xi):
    return {
        "previous_state": xi[..., :48],
        "previous_action": xi[..., 48:56],
        "reset": xi[..., 56],
    }


class SingleModel(nn.Module):
    def __init__(self, dynamics, reward):
        super().__init__()
        self.dynamics = dynamics
        self.reward = reward
        self.spec = SimpleNamespace(normalize_xi=lambda xi: xi)


class SingleImagination:
    def __init__(self, model, policy, terminated):
        self.model, self.policy, self.terminated = model, policy, terminated
        self.last_saturation = 0.0

    def __call__(self, start, horizon):
        s = start.state
        ctx = context_from_xi(start.xi)
        alive = torch.ones(len(s), device=s.device)
        states = [s]
        actions = []
        rewards = []
        done = [torch.zeros_like(alive)]
        survival = [alive]
        saturation = []
        for _ in range(horizon):
            raw = self.policy.raw(s)
            a = raw.clamp(-1, 1)
            predicted = self.model.dynamics.step(s, a, **ctx)
            nxt = torch.where(alive[:, None].bool(), predicted, s)
            r = self.model.reward(nxt)[0] * alive
            terminal = self.terminated(nxt).detach() | ~alive.bool()
            actions.append(a)
            rewards.append(r)
            states.append(nxt)
            done.append(terminal.float())
            alive = alive * (~terminal).float()
            survival.append(alive)
            saturation.append((raw.detach().abs() > 1).float().mean())
            ctx = {
                "previous_state": s,
                "previous_action": a,
                "reset": torch.zeros_like(alive),
            }
            s = nxt
        states = torch.stack(states)
        shape = states.shape[:2]
        self.last_saturation = float(torch.stack(saturation).mean())
        return ImagRollout(
            states,
            torch.stack(actions),
            torch.stack(rewards),
            torch.stack(done),
            torch.stack(survival),
            states[..., 26:29],
            torch.zeros(*shape, 4, device=s.device),
            start.frame[None].expand(horizon + 1, -1),
            {},
        )

    def critic_inputs(self, rollout):
        return rollout.states[..., :42], rollout.goals, rollout.e_root


class NativeRewardModule(nn.Module):
    """Differentiable state reward adapter used by the direct-reward ablation."""

    def __init__(self, oracle):
        super().__init__()
        self.oracle = oracle
        self.geometry = True

    def forward(self, state):
        return (self.oracle(state),)

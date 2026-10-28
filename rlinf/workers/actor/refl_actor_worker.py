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

"""Colocated policy, differentiable dynamics, and native PickCube collection."""

import hashlib
from collections import deque
from contextlib import nullcontext
from pathlib import Path

import torch

from rlinf.algorithms.refl import (
    TransitionWindows,
    bootstrapped_lambda_targets,
    checkpoint_uses_critic,
    frozen_parameters,
    imagined_objective,
    isolated_rng,
    mixed_windows,
    ring_start_dict,
    validate_source_columns,
    world_model_batch,
)
from rlinf.data.refl_replay import (
    MixedReplay,
    ReplayBuffer,
    WarmupReplay,
    Windows,
    record_online,
)
from rlinf.scheduler import Worker


class ReFLActorWorker(Worker):
    def __init__(self, config):
        super().__init__()
        self.cfg = config
        self.env = self.evaluation = None
        self.iteration = self.actor_updates = self.wm_updates = 0
        self.critic_updates = 0

    def initialize(self):
        from rlinf.envs.sim.maniskill.refl_pickcube import (
            BinSampler,
            KIRSampler,
            NativeRewardOracle,
            make_env,
            validate_reference_bank,
        )
        from rlinf.models.embodiment.refl_pickcube import (
            SingleActor,
            SingleCritic,
            load_world_model,
            set_precision,
            validate_warmstart,
        )
        from rlinf.runners.refl_runner import assert_clean_source

        assert_clean_source(Path(self.cfg["repository"]), self.cfg["commit"])
        if (
            Path(__file__).resolve().parents[3]
            != Path(self.cfg["repository"]).resolve()
        ):
            raise ValueError("Worker imported a different RLinf checkout")
        torch.manual_seed(self.cfg["seed"])
        torch.set_float32_matmul_precision("highest")
        self.torch_platform.manual_seed_all(self.cfg["seed"])
        self.device = torch.device("cuda", self.torch_platform.current_device())
        self.root = Path(self.cfg["output_dir"])
        self.inputs = {
            key: torch.load(value["path"], map_location=self.device, weights_only=False)
            for key, value in self.cfg["inputs"].items()
        }
        warm = self.inputs["actor"]
        if warm.get("format") != "single-pickcube-ppo-warmstart-v1":
            raise ValueError("Expected the PPO790 warmstart format")
        validate_warmstart(warm, self.cfg["gamma"])
        self.actor = SingleActor().to(self.device)
        self.actor.net.load_state_dict(warm["actor_mean"], strict=True)
        self.wm = load_world_model(self.inputs["world_model"], self.device)
        set_precision(self.actor, self.cfg["precision"])
        set_precision(self.wm, self.cfg["precision"])
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.cfg["actor_lr"], betas=(0.9, 0.95)
        )
        self.wm_optimizer = torch.optim.Adam(self.wm.parameters(), lr=self.cfg["wm_lr"])
        self.critic = self.critic_optimizer = None
        if self.cfg["use_critic"]:
            with isolated_rng(self.device):
                self.critic = SingleCritic().to(self.device)
                self.critic.net.load_state_dict(warm["critic"], strict=True)
                set_precision(self.critic, self.cfg["precision"])
            self.critic_optimizer = torch.optim.Adam(
                self.critic.parameters(), lr=self.cfg["critic_lr"]
            )
            self.initial_critic = {
                k: p.detach().clone() for k, p in self.critic.named_parameters()
            }
            self.critic_generator = torch.Generator(device=self.device).manual_seed(
                41000 + self.cfg["seed"]
            )
        packed = self.inputs["dataset"]
        if packed["coverage"].get("splits") != {
            "train": [0, 3583],
            "val": [3584, 3839],
            "test": [3840, 4095],
        }:
            raise ValueError("Expected the dataset793 disjoint environment split")
        self.train_ids = torch.arange(3584, device=self.device)
        self.val_ids = torch.arange(3584, 3840, device=self.device)
        validate_source_columns(packed["data"]["source_env"], 4096)
        self.ring = self.cfg["replay_backend"] == "ring"
        self.fixed = (Windows if self.ring else TransitionWindows)(
            packed["data"], self.train_ids, 3
        )
        self.diagnostic_generator = torch.Generator(device=self.device).manual_seed(
            31000
        )
        self.fixed_val = TransitionWindows(packed["data"], self.val_ids, 3).sample(
            512, self.diagnostic_generator
        )
        self.bank = self.inputs["bank"]
        if tuple(self.bank["observations"].shape[1:]) != (4096, 42):
            raise ValueError("Expected the original 4096-environment state42 bank")
        if self.ring:
            self.replay = ReplayBuffer(3584, 1000, 48, 8, 57, self.device)
            self.mixed_replay = MixedReplay(WarmupReplay(self.fixed), self.replay)
        else:
            self.online = deque(maxlen=self.cfg["online_iterations"])
        self.oracle = NativeRewardOracle()
        self.env = make_env("PickCube-v1", 4096)
        self.evaluation = make_env("PickCube-v1", 128)
        source_valid_count = int(self.bank["valid"].sum())
        self.bank = dict(self.bank)
        if "bank_mask" in self.inputs:
            mask = self.inputs["bank_mask"]
            if (
                mask.dtype != torch.bool
                or mask.shape != self.bank["valid"].shape
                or (mask & ~self.bank["valid"]).any()
                or not mask.any(0).all()
            ):
                raise ValueError(
                    "Cached bank validity mask is incompatible with source bank"
                )
            self.bank["valid"] = mask
        else:
            with isolated_rng(self.device):
                self.bank["valid"] = validate_reference_bank(self.env, self.bank)
        torch.save(self.bank["valid"].cpu(), self.root / "runtime_bank_valid.pt")
        self.sampler = (
            BinSampler(self.bank["valid"], self.bank["times"], gamma=0.99)
            if self.cfg["collection_sampler"] == "bins"
            else KIRSampler(self.bank)
        )
        self.save_checkpoint("initial.pt")
        return {
            "collection_sampler": self.cfg["collection_sampler"],
            "replay_backend": self.cfg["replay_backend"],
            "reward_rule": self.cfg["reward_rule"],
            "kir_group_counts": getattr(self.sampler, "group_counts", None),
            "reward_source_sha256": self.oracle.provenance,
            "source_bank_valid_count": source_valid_count,
            "runtime_bank_valid_count": int(self.bank["valid"].sum()),
            "reward_model_parameters": 0,
            "critic_parameters": sum(p.numel() for p in self.critic.parameters())
            if self.critic is not None
            else 0,
            "initial_training_rng_sha256": hashlib.sha256(
                torch.get_rng_state().cpu().numpy().tobytes()
                + self.torch_platform.get_rng_state().cpu().numpy().tobytes()
            ).hexdigest(),
        }

    def train_iteration(self):
        from rlinf.envs.sim.maniskill.refl_pickcube import collect

        self.iteration += 1
        data, collection = collect(
            self.env,
            self.actor,
            self.bank,
            self.sampler,
            self.iteration,
            self.cfg["seed"],
            self.train_ids,
        )
        live = data["alive"][:, self.train_ids]
        truth = data["next_state"][:, self.train_ids][live][:4096]
        native_reward = data["reward"][:, self.train_ids][live][:4096]
        native_done = data["terminated"][:, self.train_ids][live][:4096]
        with torch.no_grad():
            parity = float((self.oracle(truth) - native_reward).abs().max())
            if parity > 1e-6 or not torch.equal(
                self.oracle.terminated(truth), native_done
            ):
                raise ValueError("Native state reward/termination parity failed")
        if self.ring:
            record_online(self.replay, data, self.train_ids)
        else:
            self.online.append(data)
            merged = {k: torch.cat([chunk[k] for chunk in self.online]) for k in data}
            online = TransitionWindows(merged, self.train_ids, 3)
            actor_starts = TransitionWindows(merged, self.train_ids, 1)
        batch_size = self.cfg["batch_size"]
        for _ in range(self.cfg["actor_updates_per_iteration"]):
            for _ in range(self.cfg["wm_updates_per_actor"]):
                batch = (
                    self.mixed_replay.sample_windows(batch_size, 3)
                    if self.ring
                    else world_model_batch(
                        mixed_windows(self.fixed, online, batch_size)
                    )
                )
                loss, _ = self.wm.replay_loss(batch)
                self.wm_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                wm_norm = torch.nn.utils.clip_grad_norm_(
                    self.wm.parameters(), 1.0, error_if_nonfinite=True
                )
                self.wm_optimizer.step()
                self.wm_updates += 1
            self.wm_optimizer.zero_grad(set_to_none=True)
            start = (
                ring_start_dict(self.replay.sample_starts(batch_size, 1))
                if self.ring
                else actor_starts.sample(batch_size)
            )
            if self.critic_optimizer is not None:
                self.critic_optimizer.zero_grad(set_to_none=True)
            with (
                frozen_parameters(self.wm),
                (
                    frozen_parameters(self.critic)
                    if self.critic is not None
                    else nullcontext()
                ),
            ):
                result = imagined_objective(
                    self.actor,
                    self.wm,
                    self.oracle,
                    start,
                    self.cfg["imagination_horizon"],
                    self.cfg["gamma"],
                    reward_rule=self.cfg["reward_rule"],
                    critic=self.critic,
                    return_rollout=self.critic is not None,
                )
                if self.critic is not None:
                    objective, trajectory = result
                    bootstrap = float(trajectory.bootstrap.detach())
                else:
                    objective, bootstrap = result, 0.0
                self.actor_optimizer.zero_grad(set_to_none=True)
                (-objective / self.cfg["imagination_horizon"]).backward()
                actor_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), 1.0, error_if_nonfinite=True
                )
                if not float(actor_norm) > 0:
                    raise ValueError("Policy gradient through WM is zero")
                if any(p.grad is not None for p in self.wm.parameters()):
                    raise ValueError("Actor update accumulated world-model gradients")
                if self.critic is not None and any(
                    p.grad is not None for p in self.critic.parameters()
                ):
                    raise ValueError(
                        "Actor update accumulated critic parameter gradients"
                    )
                torch.nn.utils.clip_grad_value_(self.actor.parameters(), 0.1)
                self.actor_optimizer.step()
            self.actor_updates += 1
            critic_metrics = (
                self._fit_critic(trajectory) if self.critic is not None else {}
            )
        result = {
            "iteration": self.iteration,
            "updates/actor": self.actor_updates,
            "updates/wm": self.wm_updates,
            "updates/rm": 0,
            "updates/critic": self.critic_updates,
            "actor/bootstrap": bootstrap,
            **critic_metrics,
            "actor/gradient_norm": float(actor_norm),
            "actor/imagined_objective": float(objective.detach()),
            "wm/loss": float(loss.detach()),
            "wm/gradient_norm": float(wm_norm),
            "reward/native_parity_max": parity,
            **{"collection/" + key: value for key, value in collection.items()},
            **self.wm_diagnostics(self.fixed_val, "fixed_val"),
            **self.wm_diagnostics(
                TransitionWindows(data, self.val_ids, 3).sample(
                    512, self.diagnostic_generator
                ),
                "current_val",
            ),
        }
        self.save_checkpoint("latest.pt")
        return result

    def _fit_critic(self, rollout):
        with torch.no_grad():
            states = rollout.states.detach()
            values = self.critic(states)
            targets = bootstrapped_lambda_targets(
                rollout.rewards.detach(),
                rollout.stops,
                values,
                self.cfg["gamma"],
                self.cfg["critic_gae_lambda"],
            )
            active = rollout.active.bool()
            inputs, targets = states[:-1][active], targets[active]
        n = len(inputs)
        if n < self.cfg["critic_minibatches"]:
            raise ValueError("Insufficient live critic samples")
        for _ in range(self.cfg["critic_epochs"]):
            order = torch.randperm(
                n, device=self.device, generator=self.critic_generator
            )
            for ids in order.tensor_split(self.cfg["critic_minibatches"]):
                prediction = self.critic(inputs[ids])
                loss = torch.nn.functional.mse_loss(prediction, targets[ids])
                self.critic_optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), 1.0, error_if_nonfinite=True
                )
                self.critic_optimizer.step()
                self.critic_updates += 1
        delta = sum(
            (p.detach() - self.initial_critic[k]).square().sum()
            for k, p in self.critic.named_parameters()
        ).sqrt()
        return {
            "critic/loss": float(loss.detach()),
            "critic/gradient_norm": float(grad_norm),
            "critic/target_mean": float(targets.mean()),
            "critic/live_samples": n,
            "critic/parameter_delta_l2": float(delta),
        }

    @torch.no_grad()
    def wm_diagnostics(self, batch, label):
        context = {
            key: batch[key][0] for key in ("previous_state", "previous_action", "reset")
        }
        pred = self.wm.rollout(batch["state"][0], batch["action"], **context)
        truth = batch["next_state"]
        metrics = {
            f"wm/{label}/h3_normalized_mse": float(
                self.wm.residual(pred[-1], truth[-1]).square().mean()
            )
        }
        for name, mask in {
            "all": torch.ones(len(truth[0]), dtype=torch.bool, device=truth.device),
            "grasped": truth[..., 18].ge(0.5).any(0),
            "near_object": truth[..., 36:39].norm(dim=-1).lt(0.06).any(0),
            "near_goal": truth[..., 39:42].norm(dim=-1).lt(0.05).any(0),
        }.items():
            metrics[f"wm/{label}/{name}/count"] = int(mask.sum())
            if mask.any():
                metrics[f"wm/{label}/{name}/cube_error_m"] = float(
                    (pred[-1, mask, 29:32] - truth[-1, mask, 29:32]).norm(dim=-1).mean()
                )
                metrics[f"wm/{label}/{name}/reward_error"] = float(
                    (self.oracle(pred[-1, mask]) - self.oracle(truth[-1, mask]))
                    .abs()
                    .mean()
                )
        return metrics

    def save_checkpoint(self, name):
        state = self.fixed_val["state"][0, :8]
        context = {
            k: self.fixed_val[k][0, :8]
            for k in ("previous_state", "previous_action", "reset")
        }
        with torch.no_grad():
            action = self.actor(state)
            next_state = self.wm.step(state, action, **context)
        checkpoint = {
            "format": "rlinf-refl-pickcube-v2",
            "config": self.cfg,
            "actor": self.actor.state_dict(),
            "world_model": self.wm.state_dict(),
            "world_model_constructor": {
                k: v for k, v in self.inputs["world_model"].items() if k != "model"
            },
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "wm_optimizer": self.wm_optimizer.state_dict(),
            "iteration": self.iteration,
            "actor_updates": self.actor_updates,
            "wm_updates": self.wm_updates,
            "critic_updates": self.critic_updates,
            "torch_rng": torch.get_rng_state(),
            "accelerator_rng": self.torch_platform.get_rng_state_all(),
            "resume_supported": False,
            "probe": {
                "state": state,
                "context": context,
                "action": action,
                "next_state": next_state,
            },
        }
        if self.critic is not None:
            checkpoint["critic"] = self.critic.state_dict()
            checkpoint["critic_optimizer"] = self.critic_optimizer.state_dict()
            with torch.no_grad():
                checkpoint["probe"]["value"] = self.critic(state)
        destination = self.root / name
        temporary = destination.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(destination)

    def evaluate(self, checkpoint=None, seed=30000):
        with isolated_rng(self.device):
            return self._evaluate(checkpoint, seed)

    def _evaluate(self, checkpoint=None, seed=30000):
        from rlinf.envs.sim.maniskill.refl_pickcube import evaluate_policy
        from rlinf.models.embodiment.refl_pickcube import (
            SingleActor,
            SingleCritic,
            load_world_model,
            set_precision,
        )

        actor = self.actor
        if checkpoint is not None:
            bundle = torch.load(
                self.root / checkpoint, map_location=self.device, weights_only=False
            )
            has_critic = checkpoint_uses_critic(bundle)
            actor = SingleActor().to(self.device)
            actor.load_state_dict(bundle["actor"], strict=True)
            set_precision(actor, self.cfg["precision"])
            restored_wm = load_world_model(
                {**bundle["world_model_constructor"], "model": bundle["world_model"]},
                self.device,
            )
            set_precision(restored_wm, self.cfg["precision"])
            probe = bundle["probe"]
            with torch.no_grad():
                action = actor(probe["state"])
                prediction = restored_wm.step(
                    probe["state"], action, **probe["context"]
                )
            torch.testing.assert_close(action, probe["action"], rtol=0, atol=1e-6)
            torch.testing.assert_close(
                prediction, probe["next_state"], rtol=0, atol=1e-6
            )
            del restored_wm
            if has_critic:
                restored_critic = SingleCritic().to(self.device)
                restored_critic.load_state_dict(bundle["critic"], strict=True)
                set_precision(restored_critic, bundle["config"]["precision"])
                with torch.no_grad():
                    torch.testing.assert_close(
                        restored_critic(probe["state"]),
                        probe["value"],
                        rtol=0,
                        atol=1e-6,
                    )
                del restored_critic
        result, episodes = evaluate_policy(self.evaluation, actor, seed)
        if checkpoint:
            torch.save(episodes, self.root / (checkpoint + ".episodes.pt"))
        return result

    def close(self):
        for env in (self.env, self.evaluation):
            if env is not None:
                env.close()

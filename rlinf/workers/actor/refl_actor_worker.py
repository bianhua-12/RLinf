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
from pathlib import Path

import torch

from rlinf.algorithms.refl import (
    TransitionWindows,
    checkpoint_uses_critic,
    isolated_rng,
    validate_source_columns,
)
from rlinf.algorithms.refl_maniskill import (
    NativeRewardModule,
    PolicyTrainer,
    PolicyTrainerCfg,
    SingleImagination,
    SingleModel,
    WMTrainerCfg,
    WorldModelTrainer,
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
            NativeRewardOracle,
            make_env,
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
        self.oracle = NativeRewardOracle()
        self.wm = load_world_model(self.inputs["world_model"], self.device)
        self.actor = SingleActor().to(self.device)
        self.actor.net.load_state_dict(warm["actor_mean"], strict=True)
        self.critic = None
        if self.cfg["use_critic"]:
            self.critic = SingleCritic().to(self.device)
            self.critic.net.load_state_dict(warm["critic"], strict=True)
            self.initial_critic = {
                k: p.detach().clone() for k, p in self.critic.named_parameters()
            }
        self.model = SingleModel(self.wm, NativeRewardModule(self.oracle))
        self.imagination = SingleImagination(
            self.model, self.actor, self.oracle.terminated
        )
        policy_cfg = PolicyTrainerCfg(
            horizon=self.cfg["imagination_horizon"],
            batch_size=self.cfg["batch_size"],
            history=1,
            gamma=self.cfg["gamma"],
            gae_lambda=self.cfg["critic_gae_lambda"],
            lr=self.cfg["actor_lr"],
            critic_lr=self.cfg["critic_lr"],
            use_critic=self.cfg["use_critic"],
            critic_epochs=self.cfg["critic_epochs"] if self.cfg["use_critic"] else 0,
            critic_minibatches=self.cfg["critic_minibatches"]
            if self.cfg["use_critic"]
            else 0,
            obs_noise=False,
            compile=None,
        )
        self.policy_trainer = PolicyTrainer(
            self.model,
            self.actor,
            self.critic,
            None,
            policy_cfg,
            rollout_fn=self.imagination,
            critic_inputs_fn=self.imagination.critic_inputs,
        )
        wm_cfg = WMTrainerCfg(
            window=3,
            batches_per_update=self.cfg["wm_updates_per_actor"],
            batch_size=self.cfg["batch_size"],
            lr=self.cfg["wm_lr"],
            compile=None,
        )
        self.wm_trainer = WorldModelTrainer(
            self.wm, None, wm_cfg, loss_fn=self.wm.replay_loss
        )
        self.actor_optimizer = self.policy_trainer.opt
        self.critic_optimizer = self.policy_trainer.critic_opt
        self.wm_optimizer = self.wm_trainer.opt
        for module in [self.model, self.actor] + (
            [self.critic] if self.critic is not None else []
        ):
            set_precision(module, self.cfg["precision"])
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
        self.fixed = Windows(packed["data"], self.train_ids, 3)
        self.diagnostic_generator = torch.Generator(device=self.device).manual_seed(
            31000
        )
        self.fixed_val = TransitionWindows(packed["data"], self.val_ids, 3).sample(
            512, self.diagnostic_generator
        )
        self.bank = self.inputs["bank"]
        if tuple(self.bank["observations"].shape[1:]) != (4096, 42):
            raise ValueError("Expected the original 4096-environment state42 bank")
        self.replay = ReplayBuffer(3584, 1000, 48, 8, 57, self.device)
        self.mixed_replay = MixedReplay(WarmupReplay(self.fixed), self.replay)
        self.env = make_env("PickCube-v1", 4096)
        self.evaluation = make_env("PickCube-v1", 128)
        source_valid_count = int(self.bank["valid"].sum())
        if "bank_mask" in self.inputs:
            raise ValueError("ManiSkill alignment uses the original source bank mask")
        self.sampler = BinSampler(self.bank["valid"], self.bank["times"], gamma=0.99)
        torch.save(self.bank["valid"].cpu(), self.root / "runtime_bank_valid.pt")
        self.save_checkpoint("initial.pt")
        return {
            "collection_sampler": self.cfg["collection_sampler"],
            "replay_backend": self.cfg["replay_backend"],
            "reward_rule": self.cfg["reward_rule"],
            "kir_group_counts": getattr(self.sampler, "group_counts", None),
            "reward_source_sha256": self.oracle.provenance,
            "source_bank_valid_count": source_valid_count,
            "training_bank_valid_count": int(self.bank["valid"].sum()),
            "bank_validity": "source",
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
        record_online(self.replay, data, self.train_ids)
        for _ in range(self.cfg["actor_updates_per_iteration"]):
            wm_metrics = self.wm_trainer.update(self.mixed_replay)
            self.wm_updates += self.cfg["wm_updates_per_actor"]
            actor_metrics = self.policy_trainer.update(self.replay)
            self.actor_updates += 1
            if self.critic is not None:
                self.critic_updates += (
                    self.cfg["critic_epochs"] * self.cfg["critic_minibatches"]
                )
        critic_metrics = {}
        if self.critic is not None:
            delta = sum(
                (p.detach() - self.initial_critic[k]).square().sum()
                for k, p in self.critic.named_parameters()
            ).sqrt()
            grad_norm = sum(
                p.grad.detach().square().sum()
                for p in self.critic.parameters()
                if p.grad is not None
            ).sqrt()
            critic_metrics = {
                "critic/loss": actor_metrics["critic_loss"],
                "critic/gradient_norm": float(grad_norm),
                "critic/parameter_delta_l2": float(delta),
                "critic/value_mean": actor_metrics["critic_value_mean"],
                "critic/target_mean": actor_metrics["critic_target_mean"],
            }
        result = {
            "iteration": self.iteration,
            "updates/actor": self.actor_updates,
            "updates/wm": self.wm_updates,
            "updates/rm": 0,
            "updates/critic": self.critic_updates,
            "actor/bootstrap": actor_metrics["imag_bootstrap"],
            **critic_metrics,
            "actor/gradient_norm": actor_metrics["policy_grad_norm"],
            "actor/imagined_objective": actor_metrics["imag_objective"],
            "wm/loss": wm_metrics["loss"],
            "wm/gradient_norm": wm_metrics["grad_norm"],
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
            "format": "rlinf-refl-pickcube-v3",
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

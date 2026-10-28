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

"""Measure reward/value actor gradients without updating any model."""

import argparse
import json
from pathlib import Path

import torch

from rlinf.algorithms.refl_maniskill import (
    NativeRewardModule,
    SingleImagination,
    SingleModel,
    policy_objective,
)
from rlinf.data.refl_replay import StartBatch, Windows
from rlinf.envs.sim.maniskill.refl_pickcube import (
    NativeRewardOracle,
    evaluate_policy,
    make_env,
)
from rlinf.models.embodiment.refl_pickcube import (
    SingleActor,
    SingleCritic,
    load_world_model,
    set_precision,
)
from rlinf.runners.refl_runner import prepare_run, sha256, write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = prepare_run(
        json.loads(args.config.read_text()), Path(__file__).resolve().parents[2]
    )
    output = Path(cfg["output_dir"])
    passed = False
    try:
        torch.manual_seed(cfg["seed"])
        torch.set_float32_matmul_precision("highest")
        if set(cfg["diagnostic_checkpoints"]) != {"initial", "final"}:
            raise ValueError("Expected initial and final diagnostic checkpoints")
        bundles = {}
        for name, item in cfg["diagnostic_checkpoints"].items():
            if sha256(item["path"]) != item["sha256"]:
                raise ValueError("Diagnostic checkpoint digest mismatch")
            bundles[name] = torch.load(
                item["path"], map_location="cuda", weights_only=False
            )
            bundle = bundles[name]
            if (
                bundle.get("format") != "rlinf-refl-pickcube-v3"
                or bundle.get("config", {}).get("critic_bootstrap") != "horizon_only"
                or not bundle.get("config", {}).get("use_critic")
            ):
                raise ValueError("Expected a v3 horizon-only critic checkpoint")
        data = torch.load(
            cfg["inputs"]["dataset"]["path"], map_location="cuda", weights_only=False
        )
        windows = Windows(data["data"], torch.arange(3584, 3840, device="cuda"), 3)
        batch = windows.sample(256, torch.Generator(device="cuda").manual_seed(31415))
        state, action = batch["state"][0], batch["previous_action"][0]
        xi = torch.cat(
            [batch["previous_state"][0], action, batch["reset"][0, :, None].float()], -1
        )
        start = StartBatch(
            state,
            xi,
            torch.zeros(256, dtype=torch.long, device="cuda"),
            action,
            state[None],
            action[None],
        )
        oracle = NativeRewardOracle()
        results = []
        for policy_name, bundle in bundles.items():
            actor = SingleActor().cuda()
            actor.load_state_dict(bundle["actor"])
            wm = load_world_model(
                {**bundle["world_model_constructor"], "model": bundle["world_model"]},
                "cuda",
            )
            model = SingleModel(wm, NativeRewardModule(oracle))
            for module in (actor, model):
                set_precision(module, cfg["precision"])
            model.requires_grad_(False)
            rollout = SingleImagination(model, actor, oracle.terminated)(start, 13)
            reward = policy_objective(rollout, None, 0.8)
            parameters = tuple(actor.parameters())

            def gradient(value):
                return torch.cat(
                    [
                        x.flatten()
                        for x in torch.autograd.grad(
                            value, parameters, retain_graph=True
                        )
                    ]
                )

            gr = gradient(reward)
            for value_name, value_bundle in bundles.items():
                critic = SingleCritic().cuda()
                critic.load_state_dict(value_bundle["critic"])
                set_precision(critic, cfg["precision"])
                critic.requires_grad_(False)
                bootstrap = (
                    0.8**13 * rollout.survival[-1] * critic(rollout.states[-1])
                ).mean()
                gv = gradient(bootstrap)
                cosine = torch.dot(gr, gv) / (gr.norm() * gv.norm()).clamp_min(1e-20)
                results.append(
                    {
                        "actor_wm": policy_name,
                        "critic": value_name,
                        "reward_objective": float(reward.detach()),
                        "bootstrap": float(bootstrap.detach()),
                        "reward_gradient_norm": float(gr.norm()),
                        "value_gradient_norm": float(gv.norm()),
                        "value_to_reward_norm_ratio": float(
                            gv.norm() / gr.norm().clamp_min(1e-20)
                        ),
                        "gradient_cosine": float(cosine),
                        "combined_dot_reward": float(torch.dot(gr + gv, gr)),
                        "survival_horizon": float(rollout.survival[-1].mean()),
                        "sample_count": 256,
                    }
                )
            del rollout, model, wm
        env = make_env("PickCube-v1", 128)
        try:
            cpu_rng, gpu_rng = torch.get_rng_state(), torch.cuda.get_rng_state()
            evaluate_policy(env, actor, 30000, episodes=512)
            rng = {
                "cpu_unchanged": torch.equal(cpu_rng, torch.get_rng_state()),
                "cuda_unchanged": torch.equal(gpu_rng, torch.cuda.get_rng_state()),
            }
        finally:
            env.close()
        write_json(
            output / "diagnosis.json",
            {
                "gradient_decomposition": results,
                "native_evaluation_rng": rng,
                "scope": "fixed held-out real starts; raw objective gradients before clipping and Adam; not causal training ablation",
            },
        )
        passed = True
    finally:
        write_json(
            output / "completion.json",
            {
                "completed": passed,
                "exit_code": 0 if passed else 1,
                "start_sha256": sha256(output / "start.json"),
                "evidence": "GPU diagnostic, no updates",
            },
        )


if __name__ == "__main__":
    main()

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

"""Launch one colocated ReFL worker on an allocated NVIDIA GPU."""

import argparse
import os
import sys
from pathlib import Path

sys.dont_write_bytecode = True
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"


def main():
    from omegaconf import OmegaConf

    from rlinf.runners.refl_runner import ReFLRunner, prepare_run
    from rlinf.scheduler import Cluster, PackedPlacementStrategy
    from rlinf.workers.actor.refl_actor_worker import ReFLActorWorker

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    for field in (
        "iterations",
        "batch_size",
        "imagination_horizon",
        "eval_every",
        "actor_updates_per_iteration",
        "wm_updates_per_actor",
        "online_iterations",
    ):
        if not isinstance(cfg[field], int) or cfg[field] < 1:
            parser.error(f"{field} must be a positive integer")
    if cfg["batch_size"] % 2:
        parser.error("batch_size must be even")
    if cfg.get("collection_sampler") not in ("bins", "kir"):
        parser.error("collection_sampler must explicitly select bins or kir")
    if cfg.get("replay_backend") not in ("ring", "trajectory"):
        parser.error("replay_backend must explicitly select ring or trajectory")
    if cfg.get("reward_rule") not in ("native", "completion_cost_v1"):
        parser.error("reward_rule must explicitly select native or completion_cost_v1")
    if not isinstance(cfg.get("use_critic"), bool):
        parser.error("use_critic must explicitly select true or false")
    if cfg["use_critic"]:
        if (
            cfg["reward_rule"] != "native"
            or cfg.get("critic_bootstrap") != "success_and_horizon"
        ):
            parser.error(
                "Critic requires native rewards and success_and_horizon bootstrap"
            )
        for field in ("critic_epochs", "critic_minibatches"):
            if not isinstance(cfg.get(field), int) or cfg[field] < 1:
                parser.error(f"{field} must be positive")
        if cfg["batch_size"] < cfg["critic_minibatches"] or cfg["critic_lr"] <= 0:
            parser.error("Invalid critic batch count or learning rate")
        if not 0 <= cfg["critic_gae_lambda"] <= 1:
            parser.error("critic_gae_lambda must be in [0, 1]")
    if cfg["imagination_horizon"] != 13:
        parser.error("This PickCube migration preserves the H13 imagination contract")
    if cfg["wandb_mode"] not in ("online", "offline"):
        parser.error("wandb_mode must be online or offline")
    if not 0 < cfg["gamma"] <= 1 or min(cfg["actor_lr"], cfg["wm_lr"]) <= 0:
        parser.error("Invalid gamma or optimizer learning rate")
    repo = Path(__file__).resolve().parents[2]
    cfg = prepare_run(cfg, repo)
    cluster = Cluster(num_nodes=1)
    worker = ReFLActorWorker.create_group(cfg).launch(
        cluster, name="ReFLActor", placement_strategy=PackedPlacementStrategy(0, 0)
    )
    ReFLRunner(cfg, worker).run()


if __name__ == "__main__":
    main()

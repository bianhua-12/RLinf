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

"""Training script for CFG SFT using FSDPCfgWorker.

This script trains CFG models using a unified data loading approach that
supports both SFT data (expert trajectories) and collected data (mixed success/failure).
"""

import json
import os

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.cfg_sft_runner import CfgSFTRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.cfg.fsdp_cfg_worker import FSDPCfgWorker
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)


@hydra.main(version_base="1.1", config_path="config", config_name="libero_cfg_openpi")
def main(cfg) -> None:
    # Set environment variables for data loading
    data_cfg = cfg.get("data", {})

    # Support multiple data path options (new datasets format and legacy)
    datasets = data_cfg.get("train_data_paths")
    if datasets and len(datasets) > 0:
        os.environ["HF_LEROBOT_HOME"] = datasets[0].get("path", "")
    elif data_cfg.get("sft_data_path"):
        os.environ["HF_LEROBOT_HOME"] = data_cfg.sft_data_path
    elif data_cfg.get("data_path"):
        os.environ["HF_LEROBOT_HOME"] = data_cfg.data_path

    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create actor worker group using FSDPCfgWorker
    actor_placement = component_placement.get_strategy("actor")
    actor_group = FSDPCfgWorker.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )

    rollout_group = None
    if hasattr(cfg, "rollout") and cfg.runner.val_check_interval > 0:
        rollout_placement = component_placement.get_strategy("rollout")
        rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
            cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
        )

    env_group = None
    if hasattr(cfg, "env") and cfg.runner.val_check_interval > 0:
        env_placement = component_placement.get_strategy("env")
        env_group = EnvWorker.create_group(cfg).launch(
            cluster, name=cfg.env.group_name, placement_strategy=env_placement
        )

    runner = CfgSFTRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()

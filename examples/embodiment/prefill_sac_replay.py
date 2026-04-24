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

import json
import math
import time
from typing import Any

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.scheduler import Channel, Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.async_env_worker import AsyncEnvWorker
from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)

mp.set_start_method("spawn", force=True)


def compute_local_prefill_target_samples(
    num_episodes: int, max_episode_steps: int, actor_world_size: int
) -> int:
    """Compute the local replay-buffer sample target for a prefill run."""
    if num_episodes <= 0:
        raise ValueError("num_episodes must be > 0")
    if max_episode_steps <= 0:
        raise ValueError("max_episode_steps must be > 0")
    if actor_world_size <= 0:
        raise ValueError("actor_world_size must be > 0")
    local_episodes = math.ceil(num_episodes / actor_world_size)
    return local_episodes * max_episode_steps


def extract_min_replay_stats(stats_list: list[dict[str, Any]]) -> dict[str, float]:
    """Take the minimum local replay stats across actor ranks."""
    valid_stats = [stats for stats in stats_list if stats]
    if not valid_stats:
        return {"num_trajectories": 0.0, "total_samples": 0.0, "recv_queue": 0.0}
    keys = ("num_trajectories", "total_samples", "recv_queue")
    return {
        key: min(float(stats.get(key, 0.0)) for stats in valid_stats) for key in keys
    }


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="maniskill_sac_mlp_dualview_env_reward_async",
)
def main(cfg) -> None:
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    if cfg.algorithm.loss_type != "embodied_sac":
        raise ValueError(
            "prefill_sac_replay.py only supports embodied_sac configurations."
        )
    if cfg.runner.get("resume_dir", None) is None:
        raise ValueError("prefill_sac_replay.py requires runner.resume_dir.")
    if cfg.algorithm.replay_buffer.get("load_checkpoint", False):
        raise ValueError(
            "Prefill must not restore an old replay buffer. "
            "Set algorithm.replay_buffer.load_checkpoint=false."
        )

    prefill_cfg = cfg.get("prefill", {})
    target_episodes = int(prefill_cfg.get("num_episodes", 64))
    poll_interval_s = float(prefill_cfg.get("poll_interval_s", 1.0))
    max_wait_seconds = float(prefill_cfg.get("max_wait_seconds", 1800.0))
    channel_drain_timeout_s = float(prefill_cfg.get("channel_drain_timeout_s", 30.0))
    log_interval_s = float(prefill_cfg.get("log_interval_s", 10.0))

    cluster = Cluster(
        cluster_cfg=cfg.cluster, distributed_log_dir=cfg.runner.per_worker_log_path
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
    from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
        AsyncEmbodiedSACFSDPPolicy,
    )

    actor_placement = component_placement.get_strategy("actor")
    actor_group = AsyncEmbodiedSACFSDPPolicy.create_group(cfg).launch(
        cluster, name=cfg.actor.group_name, placement_strategy=actor_placement
    )
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_group = AsyncMultiStepRolloutWorker.create_group(cfg).launch(
        cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
    )
    env_placement = component_placement.get_strategy("env")
    env_group = AsyncEnvWorker.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    reward_group = None
    if cfg.get("reward", {}).get("use_reward_model", False) and not cfg.get(
        "reward", {}
    ).get("standalone_realworld", False):
        reward_placement = component_placement.get_strategy("reward")
        reward_group = EmbodiedRewardWorker.create_group(cfg).launch(
            cluster, name=cfg.reward.group_name, placement_strategy=reward_placement
        )

    runner = AsyncEmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
        reward=reward_group,
    )

    actor_handle = None
    env_handle = None
    rollout_handle = None
    reward_handle = None
    try:
        runner.init_workers()
        runner.update_rollout_weights(no_wait=runner.sync_weight_no_wait)
        if runner.reward is not None:
            runner.reward_channel = Channel.create("Reward")

        env_handle = runner.env.interact(
            input_channel=runner.env_channel,
            rollout_channel=runner.rollout_channel,
            reward_channel=runner.reward_channel,
            actor_channel=runner.actor_channel,
            metric_channel=runner.env_metric_channel,
        )
        rollout_handle = runner.rollout.generate(
            input_channel=runner.rollout_channel,
            output_channel=runner.env_channel,
            metric_channel=runner.rollout_metric_channel,
        )
        if runner.reward is not None:
            reward_handle = runner.reward.compute_rewards_async(
                input_channel=runner.reward_channel,
                output_channel=runner.env_channel,
            )
        actor_handle = runner.actor.recv_rollout_trajectories(
            input_channel=runner.actor_channel
        )
        actor_handle.wait()

        actor_world_size = component_placement.get_world_size("actor")
        target_local_samples = compute_local_prefill_target_samples(
            num_episodes=target_episodes,
            max_episode_steps=int(cfg.env.train.max_episode_steps),
            actor_world_size=actor_world_size,
        )
        runner.logger.info(
            "Starting SAC replay prefill: "
            f"target_episodes={target_episodes}, actor_world_size={actor_world_size}, "
            f"target_local_samples={target_local_samples}."
        )

        start_time = time.monotonic()
        last_log_time = 0.0
        while True:
            stats_list = runner.actor.get_replay_buffer_stats().wait()
            min_stats = extract_min_replay_stats(stats_list)
            if min_stats["total_samples"] >= target_local_samples:
                runner.logger.info(
                    f"Prefill target reached with local replay stats: {min_stats}."
                )
                break

            now = time.monotonic()
            if now - start_time > max_wait_seconds:
                raise TimeoutError(
                    "Timed out while prefilling SAC replay buffer: "
                    f"target_local_samples={target_local_samples}, "
                    f"current_stats={min_stats}."
                )
            if now - last_log_time >= log_interval_s:
                runner.logger.info(
                    "Waiting for SAC replay prefill: "
                    f"target_local_samples={target_local_samples}, "
                    f"current_stats={min_stats}, "
                    f"actor_channel_qsize={runner.actor_channel.qsize()}."
                )
                last_log_time = now
            time.sleep(poll_interval_s)

        runner.env.stop().wait()
        runner.rollout.stop().wait()
        if runner.reward is not None:
            runner.reward.stop().wait()

        env_handle.wait()
        rollout_handle.wait()
        if reward_handle is not None:
            reward_handle.wait()

        drain_deadline = time.monotonic() + channel_drain_timeout_s
        while time.monotonic() < drain_deadline:
            stats_list = runner.actor.get_replay_buffer_stats().wait()
            min_stats = extract_min_replay_stats(stats_list)
            if runner.actor_channel.qsize() == 0 and min_stats["recv_queue"] == 0:
                break
            time.sleep(poll_interval_s)
        final_stats = extract_min_replay_stats(
            runner.actor.get_replay_buffer_stats().wait()
        )
        runner.logger.info(f"Saving prefilled SAC checkpoint with stats {final_stats}.")
        runner._save_checkpoint()
    finally:
        if env_handle is not None:
            runner.env.stop().wait()
            env_handle.wait()
        if rollout_handle is not None:
            runner.rollout.stop().wait()
            rollout_handle.wait()
        if runner.reward is not None:
            runner.reward.stop().wait()
            if reward_handle is not None:
                reward_handle.wait()
        runner.actor.stop().wait()
        if actor_handle is not None:
            actor_handle.wait()
        runner.metric_logger.finish()
        runner.stop_logging = True
        runner.log_queue.join()
        runner.log_thread.join(timeout=1.0)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
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

"""Audit dual-view Qwen ternary reward accuracy against oracle delta-GAE labels."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from rlinf.envs.maniskill.maniskill_env import ManiskillEnv
from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy
from rlinf.models.embodiment.reward import get_reward_model_class
from rlinf.workers.env.reward_postprocess import (
    OracleDeltaGAEConfig,
    OracleDeltaGAERewardProcessor,
)

DEFAULT_SOURCE600 = (
    "logs/async_sac_dualview_control_aligned_source600_20260419-180620/"
    "maniskill_sac_mlp_dualview_env_reward_async/checkpoints/global_step_600"
)
DEFAULT_PPO_VALUE_CKPT = (
    "/mnt/public/shchen/codes/ManiSkill/runs/"
    "ppo_pickcube_state_h100_10m_20260415_074150/final_ckpt.pt"
)
DEFAULT_QWEN_MODEL = "/mnt/public/ztx/RLinf/Qwen3-VL-4B-Instruct"
DEFAULT_QWEN_LORA = (
    "logs/qwen3_vl_4b_5frame_dualview_gae_delta_judge_sft/"
    "qwen3_vl_4b_5frame_dualview_gae_delta_judge_sft/checkpoints/global_step_1000"
)
LABELS = ("positive", "unchanged", "negative")


@dataclass
class AuditRecord:
    batch_index: int
    episode_index: int
    env_id: int
    step: int
    task: str
    reward: float
    done: bool
    episode_success: bool
    oracle_delta_gae: float
    oracle_label: str
    qwen_score: float
    qwen_label: str
    qwen_output: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-ckpt", default=DEFAULT_SOURCE600)
    parser.add_argument("--ppo-value-ckpt", default=DEFAULT_PPO_VALUE_CKPT)
    parser.add_argument("--qwen-model-path", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-lora-path", default=DEFAULT_QWEN_LORA)
    parser.add_argument(
        "--output-dir",
        default="logs/qwen_dualview_ternary_accuracy_source600",
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-batches", type=int, default=4)
    parser.add_argument("--steps-per-episode", type=int, default=50)
    parser.add_argument("--history-size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--actor-device", default="cuda:0")
    parser.add_argument("--qwen-device", default="cuda:0")
    parser.add_argument("--oracle-device", default="cpu")
    parser.add_argument("--qwen-micro-batch-size", type=int, default=64)
    parser.add_argument("--delta-threshold", type=float, default=0.2)
    parser.add_argument(
        "--oracle-label-mode",
        choices=("threshold", "batch_tercile"),
        default="threshold",
    )
    return parser.parse_args()


def abs_path(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def resolve_actor_weights(path: str) -> str:
    checkpoint_path = Path(path).expanduser().resolve()
    if checkpoint_path.is_dir():
        candidate = checkpoint_path / "actor" / "model_state_dict" / "full_weights.pt"
        if candidate.exists():
            return str(candidate)
    return str(checkpoint_path)


def build_env_cfg(args: argparse.Namespace):
    return OmegaConf.create(
        {
            "seed": args.seed,
            "wrap_obs_mode": "simple_prompt",
            "use_full_state": True,
            "auto_reset": False,
            "ignore_terminations": False,
            "use_rel_reward": False,
            "reward_mode": "raw",
            "group_size": 1,
            "use_fixed_reset_state_ids": False,
            "max_steps_per_rollout_epoch": args.steps_per_episode,
            "max_episode_steps": args.steps_per_episode,
            "video_cfg": {
                "save_video": False,
                "info_on_video": False,
                "video_base_dir": "",
            },
            "init_params": {
                "id": "PickCubeDualView-v1",
                "num_envs": None,
                "obs_mode": "rgb",
                "control_mode": "pd_ee_delta_pos",
                "sim_backend": "gpu",
                "sim_config": {"sim_freq": 100, "control_freq": 20},
                "sensor_configs": {
                    "shader_pack": "default",
                    "width": 224,
                    "height": 224,
                },
                "human_render_camera_configs": {
                    "shader_pack": "default",
                    "width": 224,
                    "height": 224,
                },
                "render_mode": "all",
            },
        }
    )


def build_qwen_cfg(args: argparse.Namespace):
    return OmegaConf.create(
        {
            "model_type": "history_vlm",
            "model_path": abs_path(args.qwen_model_path),
            "lora_path": abs_path(args.qwen_lora_path),
            "precision": "bf16",
            "input_builder_name": "simple_dualview_ternary_input_builder",
            "reward_parser_name": "weighted_ternary_reward_parser",
            "reward_parser_params": {
                "positive_reward": 1.0,
                "unchanged_reward": 0.0,
                "negative_reward": -0.5,
            },
            "history_buffers": {
                "history_window": {
                    "history_size": args.history_size,
                    "input_interval": 1,
                    "history_keys": ["main_images", "extra_view_images"],
                    "input_on_done": False,
                }
            },
            "enable_history_offload": True,
            "infer_micro_batch_size": args.qwen_micro_batch_size,
            "subprocessor_kwargs": {
                "video_processor": {"do_sample_frames": True},
            },
            "max_new_tokens": 8,
            "do_sample": False,
            "temperature": 0.0,
            "use_chat_template": True,
        }
    )


def load_actor(path: str, device: str) -> MLPPolicy:
    model = MLPPolicy(
        obs_dim=42,
        action_dim=4,
        num_action_chunks=1,
        add_value_head=False,
        add_q_head=True,
        q_head_type="default",
    )
    state = torch.load(abs_path(path), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def empty_frame_history(num_envs: int) -> list[list[torch.Tensor]]:
    return [[] for _ in range(num_envs)]


def append_frames(
    history: list[list[torch.Tensor]],
    images: torch.Tensor,
    max_len: int,
) -> None:
    images = images.detach().cpu()
    for env_id in range(images.shape[0]):
        history[env_id].append(images[env_id].clone())
        if len(history[env_id]) > max_len:
            del history[env_id][0 : len(history[env_id]) - max_len]


@torch.inference_mode()
def qwen_generate_with_text(
    model,
    reward_input: dict[str, Any],
) -> tuple[torch.Tensor, list[str]]:
    history_input = reward_input["history_input"]
    observations = {
        key: value for key, value in reward_input.items() if key != "history_input"
    }
    batch_size = len(next(iter(next(iter(history_input.values())).values())))
    micro_batch_size = int(getattr(model, "infer_micro_batch_size", 0)) or batch_size

    all_rewards: list[torch.Tensor] = []
    all_outputs: list[str] = []
    for start in range(0, batch_size, micro_batch_size):
        end = min(start + micro_batch_size, batch_size)
        micro_obs = model.slice_observations(observations, start, end)
        micro_history = model.slice_history_input(history_input, start, end)
        rewards = torch.zeros((end - start,), dtype=torch.float32)
        outputs = [""] * (end - start)

        batched_inputs, valid_ids = model.input_builder.build_inputs(
            micro_obs, model._model.device, micro_history
        )
        if valid_ids:
            prompt_length = batched_inputs["input_ids"].shape[-1]
            output_ids = model._model.generate(**batched_inputs, **model.gen_kwargs)
            decoded = model._processor.batch_decode(
                output_ids[..., prompt_length:], skip_special_tokens=True
            )
            parsed = model.reward_parser.parse_rewards(decoded).float().cpu()
            for local_idx, reward, output in zip(valid_ids, parsed, decoded):
                rewards[local_idx] = reward
                outputs[local_idx] = output

        all_rewards.append(rewards)
        all_outputs.extend(outputs)

    return torch.cat(all_rewards, dim=0), all_outputs


def label_from_delta(delta: float, threshold: float) -> str:
    if delta > threshold:
        return "positive"
    if delta < -threshold:
        return "negative"
    return "unchanged"


def label_from_qwen_score(score: float) -> str:
    if score > 0.5:
        return "positive"
    if score < -0.25:
        return "negative"
    return "unchanged"


def _confusion_from_records(records: list[AuditRecord]) -> dict[str, dict[str, int]]:
    matrix = {label: dict.fromkeys(LABELS, 0) for label in LABELS}
    for record in records:
        matrix[record.oracle_label][record.qwen_label] += 1
    return matrix


def _metrics_from_records(records: list[AuditRecord]) -> dict[str, Any]:
    confusion = _confusion_from_records(records)
    total = len(records)
    correct = sum(confusion[label][label] for label in LABELS)
    per_class = {}
    f1_values = []
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(confusion[label][other] for other in LABELS if other != label)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2.0 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[label].values()),
        }
        f1_values.append(f1)

    return {
        "count": total,
        "accuracy": correct / total if total > 0 else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "confusion_matrix": confusion,
        "per_class": per_class,
    }


def write_outputs(
    output_dir: Path, records: list[AuditRecord], summary: dict[str, Any]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "records.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    with (output_dir / "records.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()

    actor = load_actor(resolve_actor_weights(args.source_ckpt), args.actor_device)
    reward_cls = get_reward_model_class("history_vlm")
    qwen_reward_model = reward_cls(build_qwen_cfg(args)).to(args.qwen_device).eval()
    oracle = OracleDeltaGAERewardProcessor(
        OracleDeltaGAEConfig(
            value_checkpoint_path=abs_path(args.ppo_value_ckpt),
            gamma=0.8,
            gae_lambda=0.9,
            label_mode=args.oracle_label_mode,
            delta_threshold=args.delta_threshold,
            positive_reward=1.0,
            unchanged_reward=0.0,
            negative_reward=-0.5,
            device=args.oracle_device,
        )
    )

    env = ManiskillEnv(
        cfg=build_env_cfg(args),
        num_envs=args.num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=True,
    )

    records: list[AuditRecord] = []
    all_success_records: list[AuditRecord] = []
    all_fail_records: list[AuditRecord] = []

    for batch_index in range(args.num_batches):
        obs, _ = env.reset()
        main_history = empty_frame_history(args.num_envs)
        extra_history = empty_frame_history(args.num_envs)
        curr_states = []
        next_states = []
        rewards = []
        dones = []
        qwen_scores_by_step: list[torch.Tensor] = []
        qwen_outputs_by_step: list[list[str]] = []
        task_descriptions = list(map(str, obs["task_descriptions"]))
        episode_success = torch.zeros((args.num_envs,), dtype=torch.bool)

        for step in range(args.steps_per_episode):
            states = obs["states"].to(device=args.actor_device, dtype=torch.float32)
            actions, _, _, _ = actor._generate_actions(
                states, mode="eval", calculate_values=False
            )
            next_obs, reward, terminations, truncations, infos = env.step(
                actions.to(env.device)
            )
            curr_states.append(obs["states"].detach().cpu().float())
            next_states.append(next_obs["states"].detach().cpu().float())
            rewards.append(reward.detach().cpu().float())
            done = torch.logical_or(terminations, truncations).detach().cpu().bool()
            dones.append(done)
            success = (
                infos.get("success", terminations).detach().cpu().bool().reshape(-1)
            )
            episode_success |= success

            append_frames(main_history, next_obs["main_images"], args.history_size)
            append_frames(
                extra_history, next_obs["extra_view_images"], args.history_size
            )
            reward_input = {
                "task_descriptions": next_obs["task_descriptions"],
                "history_input": {
                    "history_window": {
                        "main_images": main_history,
                        "extra_view_images": extra_history,
                    }
                },
            }
            qwen_scores, qwen_outputs = qwen_generate_with_text(
                qwen_reward_model,
                reward_input,
            )
            qwen_scores_by_step.append(qwen_scores)
            qwen_outputs_by_step.append(qwen_outputs)

            obs = next_obs

        curr_state_tensor = torch.stack(curr_states, dim=0)
        next_state_tensor = torch.stack(next_states, dim=0)
        reward_tensor = torch.stack(rewards, dim=0)
        done_tensor = torch.stack(dones, dim=0)
        reward_assign_lengths = torch.stack(
            [
                torch.full(
                    (args.num_envs,),
                    fill_value=min(step + 1, args.history_size),
                    dtype=torch.long,
                )
                for step in range(args.steps_per_episode)
            ],
            dim=0,
        )
        _, delta_gae, _ = oracle.compute_oracle_rewards(
            curr_states=curr_state_tensor,
            next_states=next_state_tensor,
            rewards=reward_tensor,
            dones=done_tensor,
            reward_assign_lengths=reward_assign_lengths,
        )

        for env_id in range(args.num_envs):
            episode_index = batch_index * args.num_envs + env_id
            task = task_descriptions[env_id]
            for step in range(args.steps_per_episode):
                qwen_score = float(qwen_scores_by_step[step][env_id].item())
                qwen_output = qwen_outputs_by_step[step][env_id]
                oracle_delta = float(delta_gae[step, env_id].item())
                record = AuditRecord(
                    batch_index=batch_index,
                    episode_index=episode_index,
                    env_id=env_id,
                    step=step,
                    task=task,
                    reward=float(reward_tensor[step, env_id].item()),
                    done=bool(done_tensor[step, env_id].item()),
                    episode_success=bool(episode_success[env_id].item()),
                    oracle_delta_gae=oracle_delta,
                    oracle_label=label_from_delta(oracle_delta, args.delta_threshold),
                    qwen_score=qwen_score,
                    qwen_label=label_from_qwen_score(qwen_score),
                    qwen_output=qwen_output,
                )
                records.append(record)
                if record.episode_success:
                    all_success_records.append(record)
                else:
                    all_fail_records.append(record)

    summary = {
        "metadata": {
            "source_ckpt": abs_path(args.source_ckpt),
            "ppo_value_ckpt": abs_path(args.ppo_value_ckpt),
            "qwen_model_path": abs_path(args.qwen_model_path),
            "qwen_lora_path": abs_path(args.qwen_lora_path),
            "num_episodes": args.num_envs * args.num_batches,
            "steps_per_episode": args.steps_per_episode,
            "history_size": args.history_size,
            "delta_threshold": args.delta_threshold,
            "oracle_label_mode": args.oracle_label_mode,
        },
        "overall": _metrics_from_records(records),
        "successful_episodes": _metrics_from_records(all_success_records),
        "unsuccessful_episodes": _metrics_from_records(all_fail_records),
    }
    write_outputs(output_dir, records, summary)


if __name__ == "__main__":
    main()

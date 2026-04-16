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

"""Compare a trained MLP value head against Qwen-VL reward on ManiSkill rollouts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image, ImageDraw

from rlinf.envs.maniskill.maniskill_env import ManiskillEnv
from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy
from rlinf.models.embodiment.reward import get_reward_model_class

DEFAULT_ACTOR_CKPT = (
    "logs/20260411-20:03:49-maniskill_ppo_mlp/maniskill_ppo_mlp/"
    "checkpoints/global_step_100/actor/model_state_dict/full_weights.pt"
)
DEFAULT_VALUE_CKPT = (
    "logs/20260411-20:03:49-maniskill_ppo_mlp/maniskill_ppo_mlp/"
    "checkpoints/global_step_1000/actor/model_state_dict/full_weights.pt"
)
DEFAULT_QWEN_MODEL = "Qwen3-VL-4B-Instruct"
DEFAULT_QWEN_LORA = (
    "logs/20260406-21:34:05-qwen3_vl_4b_sft/qwen3_vl_4b_video_judge_sft/"
    "checkpoints/global_step_1000"
)


@dataclass
class StepRecord:
    step: int
    env_id: int
    task: str
    env_reward: float
    success: bool
    terminated: bool
    truncated: bool
    value_1000: float
    value_window_start_1000: float | None
    value_window_end_1000: float | None
    value_step_delta_1000: float | None
    value_delta_1000: float | None
    env_reward_window_sum: float | None
    env_reward_window_mean: float | None
    qwen_reward: float | None
    qwen_output: str | None
    clip_path: str | None
    rendered_clip_path: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-ckpt", default=DEFAULT_ACTOR_CKPT)
    parser.add_argument("--value-ckpt", default=DEFAULT_VALUE_CKPT)
    parser.add_argument("--qwen-model-path", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-lora-path", default=DEFAULT_QWEN_LORA)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--history-size", type=int, default=5)
    parser.add_argument("--qwen-interval", type=int, default=1)
    parser.add_argument("--qwen-micro-batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--qwen-device", default="cuda:0")
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--sim-backend", default="gpu")
    parser.add_argument("--save-clips", action="store_true")
    parser.add_argument("--clip-fps", type=int, default=10)
    parser.add_argument("--render-frame-repeat", type=int, default=1)
    return parser.parse_args()


def abs_path(path: str) -> str:
    return str(Path(path).expanduser().resolve())


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
            "max_steps_per_rollout_epoch": args.steps,
            "max_episode_steps": args.steps,
            "video_cfg": {
                "save_video": False,
                "info_on_video": False,
                "video_base_dir": "",
            },
            "init_params": {
                "id": "PickCube-v1",
                "num_envs": None,
                "obs_mode": "rgb",
                "control_mode": "pd_joint_delta_pos",
                "sim_backend": args.sim_backend,
                "sim_config": {"sim_freq": 100, "control_freq": 20},
                "max_episode_steps": args.steps,
                "sensor_configs": {
                    "shader_pack": "default",
                    "width": args.width,
                    "height": args.height,
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
            "input_builder_name": "simple_robochallenge_input_builder",
            "reward_parser_name": "simple_robochallenge_reward_parser",
            "history_buffers": {
                "history_window": {
                    "history_size": args.history_size,
                    "input_interval": 1,
                    "history_keys": ["main_images"],
                    "input_on_done": False,
                }
            },
            "enable_history_offload": True,
            "infer_micro_batch_size": args.qwen_micro_batch_size,
            "subprocessor_kwargs": {
                "video_processor": {"do_sample_frames": True},
            },
            "max_new_tokens": 16,
            "do_sample": False,
            "temperature": 0.0,
            "use_chat_template": True,
        }
    )


def load_mlp(path: str, device: str) -> MLPPolicy:
    model = MLPPolicy(
        obs_dim=42,
        action_dim=8,
        num_action_chunks=1,
        add_value_head=True,
        add_q_head=False,
    )
    state = torch.load(abs_path(path), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def clone_images_by_env(images: torch.Tensor) -> list[list[torch.Tensor]]:
    images = images.detach().cpu()
    return [[images[i].clone()] for i in range(images.shape[0])]


def clone_values_by_env(values: torch.Tensor) -> list[list[float]]:
    values = values.detach().float().cpu().reshape(-1)
    return [[float(values[i].item())] for i in range(values.shape[0])]


def zero_rewards_by_env(num_envs: int) -> list[list[float]]:
    return [[0.0] for _ in range(num_envs)]


def append_images(
    history: list[list[torch.Tensor]], images: torch.Tensor, max_len: int
) -> None:
    images = images.detach().cpu()
    for env_id in range(images.shape[0]):
        history[env_id].append(images[env_id].clone())
        if len(history[env_id]) > max_len:
            del history[env_id][0 : len(history[env_id]) - max_len]


def append_values(
    history: list[list[float]], values: torch.Tensor, max_len: int
) -> None:
    values = values.detach().float().cpu().reshape(-1)
    for env_id in range(values.shape[0]):
        history[env_id].append(float(values[env_id].item()))
        if len(history[env_id]) > max_len:
            del history[env_id][0 : len(history[env_id]) - max_len]


def append_rewards(
    history: list[list[float]], rewards: list[Any], max_len: int
) -> None:
    for env_id, reward in enumerate(rewards):
        history[env_id].append(float(reward))
        if len(history[env_id]) > max_len:
            del history[env_id][0 : len(history[env_id]) - max_len]


def tensor_to_list(x: torch.Tensor | Any) -> list[Any]:
    if torch.is_tensor(x):
        return x.detach().cpu().reshape(-1).tolist()
    return list(x)


def to_uint8_rgb(image: torch.Tensor | Any) -> Any:
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    if str(image.dtype) != "uint8":
        image = image.clip(0, 255).astype("uint8")
    return image[..., :3]


def draw_overlay(frame: Any, lines: list[str]) -> Any:
    image = Image.fromarray(to_uint8_rgb(frame)).convert("RGB")
    draw = ImageDraw.Draw(image)
    line_height = 14
    pad = 5
    width = max(draw.textlength(line) for line in lines) + 2 * pad
    height = len(lines) * line_height + 2 * pad
    draw.rectangle((0, 0, width, height), fill=(0, 0, 0))
    for idx, line in enumerate(lines):
        draw.text((pad, pad + idx * line_height), line, fill=(255, 255, 255))
    return np.asarray(image)


def save_history_clip(
    output_dir: Path,
    image_history: list[list[torch.Tensor]],
    *,
    step: int,
    env_id: int,
    history_size: int,
    fps: int,
    render_frame_repeat: int,
    overlay_lines: list[str] | None = None,
    clip_subdir: str = "clips",
) -> str:
    frames = list(image_history[env_id])
    if not frames:
        raise ValueError(f"No frames available for env_id={env_id} at step={step}.")
    while len(frames) < history_size:
        frames.insert(0, frames[0])
    frames = frames[-history_size:]

    rel_path = Path(clip_subdir) / f"step_{step:04d}_env_{env_id:03d}.mp4"
    clip_path = output_dir / rel_path
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(clip_path, fps=fps) as writer:
        for frame_idx, frame in enumerate(frames):
            if overlay_lines:
                frame_lines = [f"input_frame={frame_idx + 1}/{history_size}"]
                frame_lines.extend(overlay_lines)
                rendered_frame = draw_overlay(frame, frame_lines)
            else:
                rendered_frame = to_uint8_rgb(frame)
            for _ in range(render_frame_repeat):
                writer.append_data(rendered_frame)
    return str(rel_path)


def window_stats(
    values: list[float], rewards: list[float], window_size: int
) -> dict[str, float]:
    value_window = list(values)
    reward_window = list(rewards)
    while len(value_window) < window_size:
        value_window.insert(0, value_window[0])
    while len(reward_window) < window_size:
        reward_window.insert(0, 0.0)
    value_window = value_window[-window_size:]
    reward_window = reward_window[-window_size:]
    reward_sum = float(sum(reward_window))
    return {
        "value_start": float(value_window[0]),
        "value_end": float(value_window[-1]),
        "value_delta": float(value_window[-1] - value_window[0]),
        "reward_sum": reward_sum,
        "reward_mean": reward_sum / float(window_size),
    }


@torch.inference_mode()
def qwen_generate_with_text(
    model, reward_input: dict[str, Any]
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


def write_outputs(
    output_dir: Path, records: list[StepRecord], metadata: dict[str, Any]
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "records": [asdict(record) for record in records]}
    with (output_dir / "value_qwen_compare.pkl").open("wb") as f:
        pickle.dump(payload, f)
    with (output_dir / "value_qwen_compare.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    with (output_dir / "value_qwen_compare.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def main() -> None:
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(
            "logs",
            "maniskill_actor100_value1000_qwen_compare",
        )
    output_dir = Path(args.output_dir).resolve()

    device = torch.device(args.device)
    actor = load_mlp(args.actor_ckpt, str(device))
    value_model = load_mlp(args.value_ckpt, str(device))

    reward_cls = get_reward_model_class("history_vlm")
    qwen_reward_model = reward_cls(build_qwen_cfg(args)).to(args.qwen_device).eval()

    env = ManiskillEnv(
        cfg=build_env_cfg(args),
        num_envs=args.num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
        record_metrics=True,
    )
    obs, _ = env.reset()
    initial_states = obs["states"].to(device=device, dtype=torch.float32)
    initial_values = (
        value_model.value_head(initial_states).detach().float().cpu().reshape(-1)
    )
    image_history = clone_images_by_env(obs["main_images"])
    value_history = clone_values_by_env(initial_values)
    reward_history = zero_rewards_by_env(args.num_envs)
    prev_values: torch.Tensor | None = None
    records: list[StepRecord] = []

    for step in range(args.steps):
        states = obs["states"].to(device=device, dtype=torch.float32)
        actions, _, _, values = actor._generate_actions(
            states, mode="eval", calculate_values=True
        )
        value_1000 = value_model.value_head(states).detach().float().cpu().reshape(-1)
        value_delta = (
            torch.full_like(value_1000, float("nan"))
            if prev_values is None
            else value_1000 - prev_values
        )
        prev_values = value_1000.clone()

        obs, reward, terminations, truncations, infos = env.step(actions.to(env.device))
        next_states = obs["states"].to(device=device, dtype=torch.float32)
        value_after_step = (
            value_model.value_head(next_states).detach().float().cpu().reshape(-1)
        )
        append_images(image_history, obs["main_images"], args.history_size)
        append_values(value_history, value_after_step, args.history_size)

        qwen_rewards: torch.Tensor | None = None
        qwen_outputs: list[str] | None = None
        if step % args.qwen_interval == 0:
            reward_input = {
                "task_descriptions": obs["task_descriptions"],
                "history_input": {
                    "history_window": {
                        "main_images": image_history,
                    }
                },
            }
            qwen_rewards, qwen_outputs = qwen_generate_with_text(
                qwen_reward_model, reward_input
            )

        success = tensor_to_list(
            infos.get("success", torch.zeros(args.num_envs, dtype=torch.bool))
        )
        reward_values = tensor_to_list(reward)
        append_rewards(reward_history, reward_values, args.history_size)
        term_values = tensor_to_list(terminations)
        trunc_values = tensor_to_list(truncations)
        for env_id in range(args.num_envs):
            stats = window_stats(
                value_history[env_id],
                reward_history[env_id],
                args.history_size,
            )
            qwen_output = None if qwen_outputs is None else qwen_outputs[env_id]
            qwen_reward = (
                None if qwen_rewards is None else float(qwen_rewards[env_id].item())
            )
            value_label = "positive" if stats["value_delta"] >= 0.0 else "negative"
            overlay_lines = [
                f"step={step} env={env_id}",
                f"env_reward={float(reward_values[env_id]):.4f}",
                f"env_reward_5f_sum={stats['reward_sum']:.4f}",
                f"value_5f_delta={stats['value_delta']:+.4f} ({value_label})",
                f"qwen={qwen_output} reward={qwen_reward}",
            ]
            clip_path = None
            rendered_clip_path = None
            if args.save_clips:
                clip_path = save_history_clip(
                    output_dir,
                    image_history,
                    step=step,
                    env_id=env_id,
                    history_size=args.history_size,
                    fps=args.clip_fps,
                    render_frame_repeat=args.render_frame_repeat,
                    overlay_lines=None,
                    clip_subdir="clean_clips",
                )
                rendered_clip_path = save_history_clip(
                    output_dir,
                    image_history,
                    step=step,
                    env_id=env_id,
                    history_size=args.history_size,
                    fps=args.clip_fps,
                    render_frame_repeat=args.render_frame_repeat,
                    overlay_lines=overlay_lines,
                    clip_subdir="rendered_clips",
                )
            records.append(
                StepRecord(
                    step=step,
                    env_id=env_id,
                    task=str(obs["task_descriptions"][env_id]),
                    env_reward=float(reward_values[env_id]),
                    success=bool(success[env_id]),
                    terminated=bool(term_values[env_id]),
                    truncated=bool(trunc_values[env_id]),
                    value_1000=float(value_after_step[env_id].item()),
                    value_window_start_1000=stats["value_start"],
                    value_window_end_1000=stats["value_end"],
                    value_step_delta_1000=None
                    if torch.isnan(value_delta[env_id])
                    else float(value_delta[env_id].item()),
                    value_delta_1000=stats["value_delta"],
                    env_reward_window_sum=stats["reward_sum"],
                    env_reward_window_mean=stats["reward_mean"],
                    qwen_reward=qwen_reward,
                    qwen_output=qwen_output,
                    clip_path=clip_path,
                    rendered_clip_path=rendered_clip_path,
                )
            )

        if (step + 1) % 10 == 0 or step == 0:
            latest = records[-args.num_envs :]
            mean_value = sum(r.value_1000 for r in latest) / len(latest)
            mean_qwen = [r.qwen_reward for r in latest if r.qwen_reward is not None]
            mean_qwen_text = (
                "nan" if not mean_qwen else f"{sum(mean_qwen) / len(mean_qwen):.4f}"
            )
            print(
                f"step={step + 1}/{args.steps} "
                f"value1000_mean={mean_value:.4f} qwen_mean={mean_qwen_text}",
                flush=True,
            )

    metadata = {
        "actor_ckpt": abs_path(args.actor_ckpt),
        "value_ckpt": abs_path(args.value_ckpt),
        "qwen_model_path": abs_path(args.qwen_model_path),
        "qwen_lora_path": abs_path(args.qwen_lora_path),
        "num_envs": args.num_envs,
        "steps": args.steps,
        "seed": args.seed,
        "history_size": args.history_size,
        "qwen_interval": args.qwen_interval,
        "save_clips": args.save_clips,
        "clip_fps": args.clip_fps,
        "render_frame_repeat": args.render_frame_repeat,
    }
    write_outputs(output_dir, records, metadata)

    valid_pairs = [
        (r.value_delta_1000, r.qwen_reward)
        for r in records
        if r.value_delta_1000 is not None and r.qwen_reward is not None
    ]
    agreement = None
    if valid_pairs:
        agreement = sum(
            (value_delta >= 0.0) == (qwen_reward >= 0.5)
            for value_delta, qwen_reward in valid_pairs
        ) / len(valid_pairs)
    print(f"wrote: {output_dir / 'value_qwen_compare.pkl'}")
    print(f"records={len(records)} value_delta_qwen_sign_agreement={agreement}")


if __name__ == "__main__":
    main()

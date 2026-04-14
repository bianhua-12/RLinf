#!/usr/bin/env python3
"""Compare actor100 rollouts with dual-view ternary Qwen reward and value1000."""

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

from rlinf.data.datasets.vlm import RoboChallengeProgressSFTDataset
from rlinf.envs.maniskill.maniskill_env import ManiskillEnv
from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy
from rlinf.models.embodiment.reward import get_reward_model_class
from rlinf.models.embodiment.reward.vlm_reward_utils.reward_parser import (
    _parse_ternary_output,
)


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
    "logs/qwen3_vl_4b_5frame_dualview_value_judge_v6_sft/"
    "qwen3_vl_4b_5frame_dualview_value_judge_v6_sft/checkpoints/global_step_1000"
)


@dataclass
class CompareRecord:
    step: int
    env_id: int
    task: str
    env_reward: float
    success: bool
    terminated: bool
    truncated: bool
    value_start_1000: float
    value_end_1000: float
    value_delta_1000: float
    value_label_1000: str
    reward_model_score: float | None
    reward_model_label: str | None
    reward_model_output: str | None
    direct_score: float | None
    direct_label: str | None
    direct_output: str | None
    clean_clip_path: str | None
    rendered_clip_path: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor-ckpt", default=DEFAULT_ACTOR_CKPT)
    parser.add_argument("--value-ckpt", default=DEFAULT_VALUE_CKPT)
    parser.add_argument("--qwen-model-path", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-lora-path", default=DEFAULT_QWEN_LORA)
    parser.add_argument("--output-dir", default="logs/maniskill_actor100_dualview_reward_compare")
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
    parser.add_argument("--delta-threshold", type=float, default=0.2)
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
            "input_builder_name": "simple_dualview_ternary_input_builder",
            "reward_parser_name": "simple_ternary_reward_parser",
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


def _extract_extra_view(obs: dict[str, Any]) -> torch.Tensor | None:
    extra = obs.get("extra_view_images")
    if extra is None:
        return None
    if extra.ndim == 5:
        return extra[:, 0]
    if extra.ndim == 4:
        return extra
    return None


def _to_numpy(array_like: Any) -> np.ndarray | None:
    if array_like is None:
        return None
    if torch.is_tensor(array_like):
        return array_like.detach().cpu().numpy()
    return np.asarray(array_like)


def _render_extra_view_batch(env: Any) -> torch.Tensor | None:
    render_targets = []
    seen_ids = set()
    for target in (
        getattr(env, "env", None),
        getattr(getattr(env, "env", None), "unwrapped", None),
        getattr(getattr(getattr(env, "unwrapped", None), "env", None), "unwrapped", None),
        getattr(env, "unwrapped", None),
        env,
    ):
        if target is None:
            continue
        target_id = id(target)
        if target_id in seen_ids:
            continue
        seen_ids.add(target_id)
        render_targets.append(target)

    rendered = None
    for target in render_targets:
        scene = getattr(target, "scene", None)
        if scene is not None:
            get_human_render_camera_images_fn = getattr(
                scene, "get_human_render_camera_images", None
            )
            update_render_fn = getattr(scene, "update_render", None)
            if callable(get_human_render_camera_images_fn):
                try:
                    if callable(update_render_fn):
                        update_render_fn(
                            update_sensors=False,
                            update_human_render_cameras=True,
                        )
                    render_images = get_human_render_camera_images_fn("render_camera")
                    if isinstance(render_images, dict):
                        rendered = render_images.get("render_camera")
                    else:
                        rendered = render_images
                except Exception:
                    rendered = None
                if rendered is not None:
                    break

        render_rgb_array_fn = getattr(target, "render_rgb_array", None)
        if callable(render_rgb_array_fn):
            try:
                rendered = render_rgb_array_fn("render_camera")
            except Exception:
                rendered = None
            if rendered is not None:
                break

    if rendered is None:
        return None
    rendered_np = _to_numpy(rendered)
    if rendered_np is None:
        return None
    if rendered_np.ndim == 3:
        rendered_np = np.expand_dims(rendered_np, axis=0)
    return torch.from_numpy(rendered_np)


def clone_frames_by_env(frames: torch.Tensor) -> list[list[torch.Tensor]]:
    frames = frames.detach().cpu()
    return [[frames[i].clone()] for i in range(frames.shape[0])]


def clone_values_by_env(values: torch.Tensor) -> list[list[float]]:
    values = values.detach().float().cpu().reshape(-1)
    return [[float(values[i].item())] for i in range(values.shape[0])]


def append_frames(history: list[list[torch.Tensor]], frames: torch.Tensor, max_len: int) -> None:
    frames = frames.detach().cpu()
    for env_id in range(frames.shape[0]):
        history[env_id].append(frames[env_id].clone())
        if len(history[env_id]) > max_len:
            del history[env_id][0 : len(history[env_id]) - max_len]


def append_values(history: list[list[float]], values: torch.Tensor, max_len: int) -> None:
    values = values.detach().float().cpu().reshape(-1)
    for env_id in range(values.shape[0]):
        history[env_id].append(float(values[env_id].item()))
        if len(history[env_id]) > max_len:
            del history[env_id][0 : len(history[env_id]) - max_len]


def to_uint8_rgb(image: torch.Tensor | Any) -> np.ndarray:
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    if image.dtype != np.uint8:
        image = image.clip(0, 255).astype(np.uint8)
    return image[..., :3]


def render_dualview_frame(main_frame: Any, third_frame: Any, lines: list[str] | None = None) -> np.ndarray:
    main_np = to_uint8_rgb(main_frame)
    third_np = to_uint8_rgb(third_frame)
    combined = np.concatenate([main_np, third_np], axis=1)
    image = Image.fromarray(combined).convert("RGB")
    if lines:
        draw = ImageDraw.Draw(image)
        line_height = 14
        pad = 5
        width = max(draw.textlength(line) for line in lines) + 2 * pad
        height = len(lines) * line_height + 2 * pad
        draw.rectangle((0, 0, width, height), fill=(0, 0, 0))
        for idx, line in enumerate(lines):
            draw.text((pad, pad + idx * line_height), line, fill=(255, 255, 255))
    return np.asarray(image)


def save_dualview_clip(
    output_dir: Path,
    main_history: list[list[torch.Tensor]],
    third_history: list[list[torch.Tensor]],
    *,
    step: int,
    env_id: int,
    history_size: int,
    fps: int,
    render_frame_repeat: int,
    overlay_lines: list[str] | None = None,
    clip_subdir: str = "clips",
) -> str:
    main_frames = list(main_history[env_id])
    third_frames = list(third_history[env_id])
    while len(main_frames) < history_size:
        main_frames.insert(0, main_frames[0])
    while len(third_frames) < history_size:
        third_frames.insert(0, third_frames[0])
    main_frames = main_frames[-history_size:]
    third_frames = third_frames[-history_size:]

    rel_path = Path(clip_subdir) / f"step_{step:04d}_env_{env_id:03d}.mp4"
    clip_path = output_dir / rel_path
    clip_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(clip_path, fps=fps) as writer:
        for frame_idx, (main_frame, third_frame) in enumerate(zip(main_frames, third_frames)):
            frame_lines = None
            if overlay_lines is not None:
                frame_lines = [f"input_frame={frame_idx + 1}/{history_size}"]
                frame_lines.extend(overlay_lines)
            rendered = render_dualview_frame(main_frame, third_frame, frame_lines)
            for _ in range(render_frame_repeat):
                writer.append_data(rendered)
    return str(rel_path)


def ternary_label(delta: float, threshold: float) -> str:
    if delta > threshold:
        return "positive"
    if delta < -threshold:
        return "negative"
    return "unchanged"


def build_direct_inputs(
    processor,
    tasks: list[str],
    main_history: list[list[torch.Tensor]],
    third_history: list[list[torch.Tensor]],
    valid_ids: list[int],
):
    prompt_texts = [
        [
            f"You are currently performing the task: {tasks[env_id]}. "
            "Please judge whether the operation shown in these two video views "
            "makes the task better, worse, or unchanged. "
            "Answer with exactly one word: positive, negative, or unchanged."
        ]
        for env_id in valid_ids
    ]
    videos = []
    for env_id in valid_ids:
        main_frames = [Image.fromarray(to_uint8_rgb(frame)) for frame in main_history[env_id]]
        third_frames = [Image.fromarray(to_uint8_rgb(frame)) for frame in third_history[env_id]]
        videos.append([main_frames, third_frames])

    _, full_inputs, _ = RoboChallengeProgressSFTDataset.process_inputs(
        processor=processor,
        system_prompt=None,
        use_chat_template=True,
        prompt_texts=prompt_texts,
        videos=videos,
        answer_text=None,
    )
    return full_inputs


@torch.inference_mode()
def qwen_generate_with_text(model, reward_input: dict[str, Any]) -> tuple[torch.Tensor, list[str]]:
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


@torch.inference_mode()
def direct_generate_with_text(
    model,
    tasks: list[str],
    main_history: list[list[torch.Tensor]],
    third_history: list[list[torch.Tensor]],
) -> tuple[list[float | None], list[str | None], list[str]]:
    batch_size = len(tasks)
    micro_batch_size = int(getattr(model, "infer_micro_batch_size", 0)) or batch_size
    all_scores: list[float | None] = []
    all_labels: list[str | None] = []
    all_outputs: list[str] = []

    for start in range(0, batch_size, micro_batch_size):
        end = min(start + micro_batch_size, batch_size)
        valid_ids = list(range(start, end))
        inputs = build_direct_inputs(
            model._processor,
            tasks,
            main_history,
            third_history,
            valid_ids,
        )
        inputs = {
            key: value.to(model._model.device) if isinstance(value, torch.Tensor) else value
            for key, value in inputs.items()
        }
        prompt_length = inputs["input_ids"].shape[-1]
        output_ids = model._model.generate(**inputs, **model.gen_kwargs)
        decoded = model._processor.batch_decode(
            output_ids[..., prompt_length:], skip_special_tokens=True
        )
        for output in decoded:
            score, label = _parse_ternary_output(output)
            all_scores.append(score)
            all_labels.append(label)
            all_outputs.append(output)
    return all_scores, all_labels, all_outputs


def write_outputs(output_dir: Path, records: list[CompareRecord], metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "records": [asdict(record) for record in records]}
    with (output_dir / "dualview_reward_compare.pkl").open("wb") as f:
        pickle.dump(payload, f)
    with (output_dir / "dualview_reward_compare.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    with (output_dir / "dualview_reward_compare.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        for record in records:
            writer.writerow(asdict(record))


def summarize_records(records: list[CompareRecord]) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        return {"total": 0}

    reward_value_match = sum(
        r.reward_model_label == r.value_label_1000
        for r in records
        if r.reward_model_label is not None
    )
    direct_value_match = sum(
        r.direct_label == r.value_label_1000
        for r in records
        if r.direct_label is not None
    )
    reward_direct_match = sum(
        r.reward_model_label == r.direct_label
        for r in records
        if r.reward_model_label is not None and r.direct_label is not None
    )

    return {
        "total": total,
        "reward_value_match": reward_value_match / total,
        "direct_value_match": direct_value_match / total,
        "reward_direct_match": reward_direct_match / total,
        "value_counts": {
            label: sum(r.value_label_1000 == label for r in records)
            for label in ("positive", "unchanged", "negative")
        },
        "reward_counts": {
            label: sum(r.reward_model_label == label for r in records)
            for label in ("positive", "unchanged", "negative")
        },
        "direct_counts": {
            label: sum(r.direct_label == label for r in records)
            for label in ("positive", "unchanged", "negative")
        },
    }


def main() -> None:
    args = parse_args()
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
    initial_values = value_model.value_head(initial_states).detach().float().cpu().reshape(-1)
    extra_view_batch = _extract_extra_view(obs)
    if extra_view_batch is None:
        extra_view_batch = _render_extra_view_batch(env)
    if extra_view_batch is None:
        raise ValueError("Unable to obtain dual-view render_camera frames from environment.")
    main_history = clone_frames_by_env(obs["main_images"])
    third_history = clone_frames_by_env(extra_view_batch)
    value_history = clone_values_by_env(initial_values)
    records: list[CompareRecord] = []

    for step in range(args.steps):
        states = obs["states"].to(device=device, dtype=torch.float32)
        actions, _, _, _ = actor._generate_actions(states, mode="eval", calculate_values=True)
        obs, reward, terminations, truncations, infos = env.step(actions.to(env.device))
        next_states = obs["states"].to(device=device, dtype=torch.float32)
        value_after_step = value_model.value_head(next_states).detach().float().cpu().reshape(-1)
        extra_view_batch = _extract_extra_view(obs)
        if extra_view_batch is None:
            extra_view_batch = _render_extra_view_batch(env)
        if extra_view_batch is None:
            raise ValueError("Unable to obtain dual-view render_camera frames from environment.")
        append_frames(main_history, obs["main_images"], args.history_size)
        append_frames(third_history, extra_view_batch, args.history_size)
        append_values(value_history, value_after_step, args.history_size)

        qwen_scores: list[float | None] = [None] * args.num_envs
        qwen_labels: list[str | None] = [None] * args.num_envs
        qwen_outputs: list[str | None] = [None] * args.num_envs
        direct_scores: list[float | None] = [None] * args.num_envs
        direct_labels: list[str | None] = [None] * args.num_envs
        direct_outputs: list[str | None] = [None] * args.num_envs

        if step % args.qwen_interval == 0:
            reward_input = {
                "task_descriptions": obs["task_descriptions"],
                "history_input": {
                    "history_window": {
                        "main_images": main_history,
                        "extra_view_images": third_history,
                    }
                },
            }
            reward_scores_tensor, reward_outputs = qwen_generate_with_text(
                qwen_reward_model, reward_input
            )
            direct_score_list, direct_label_list, direct_output_list = direct_generate_with_text(
                qwen_reward_model,
                list(map(str, obs["task_descriptions"])),
                main_history,
                third_history,
            )
            for env_id in range(args.num_envs):
                qwen_scores[env_id] = float(reward_scores_tensor[env_id].item())
                if qwen_scores[env_id] > 0.5:
                    qwen_labels[env_id] = "positive"
                elif qwen_scores[env_id] < -0.5:
                    qwen_labels[env_id] = "negative"
                else:
                    qwen_labels[env_id] = "unchanged"
                qwen_outputs[env_id] = reward_outputs[env_id]
                direct_scores[env_id] = direct_score_list[env_id]
                direct_labels[env_id] = direct_label_list[env_id]
                direct_outputs[env_id] = direct_output_list[env_id]

        success = (
            infos.get("success", torch.zeros(args.num_envs, dtype=torch.bool))
            .detach()
            .cpu()
            .reshape(-1)
            .tolist()
        )
        reward_values = reward.detach().cpu().reshape(-1).tolist()
        term_values = terminations.detach().cpu().reshape(-1).tolist()
        trunc_values = truncations.detach().cpu().reshape(-1).tolist()

        for env_id in range(args.num_envs):
            value_window = list(value_history[env_id])[-args.history_size :]
            while len(value_window) < args.history_size:
                value_window.insert(0, value_window[0])
            start_value = float(value_window[0])
            end_value = float(value_window[-1])
            value_delta = float(end_value - start_value)
            value_label = ternary_label(value_delta, args.delta_threshold)

            overlay_lines = [
                f"step={step} env={env_id}",
                f"env_reward={float(reward_values[env_id]):.4f}",
                f"value_delta={value_delta:+.4f} ({value_label})",
                f"reward_model={qwen_labels[env_id]} score={qwen_scores[env_id]}",
                f"direct={direct_labels[env_id]}",
            ]
            clean_clip_path = None
            rendered_clip_path = None
            if args.save_clips:
                clean_clip_path = save_dualview_clip(
                    output_dir,
                    main_history,
                    third_history,
                    step=step,
                    env_id=env_id,
                    history_size=args.history_size,
                    fps=args.clip_fps,
                    render_frame_repeat=args.render_frame_repeat,
                    overlay_lines=None,
                    clip_subdir="clean_clips",
                )
                rendered_clip_path = save_dualview_clip(
                    output_dir,
                    main_history,
                    third_history,
                    step=step,
                    env_id=env_id,
                    history_size=args.history_size,
                    fps=args.clip_fps,
                    render_frame_repeat=args.render_frame_repeat,
                    overlay_lines=overlay_lines,
                    clip_subdir="rendered_clips",
                )

            records.append(
                CompareRecord(
                    step=step,
                    env_id=env_id,
                    task=str(obs["task_descriptions"][env_id]),
                    env_reward=float(reward_values[env_id]),
                    success=bool(success[env_id]),
                    terminated=bool(term_values[env_id]),
                    truncated=bool(trunc_values[env_id]),
                    value_start_1000=start_value,
                    value_end_1000=end_value,
                    value_delta_1000=value_delta,
                    value_label_1000=value_label,
                    reward_model_score=qwen_scores[env_id],
                    reward_model_label=qwen_labels[env_id],
                    reward_model_output=qwen_outputs[env_id],
                    direct_score=direct_scores[env_id],
                    direct_label=direct_labels[env_id],
                    direct_output=direct_outputs[env_id],
                    clean_clip_path=clean_clip_path,
                    rendered_clip_path=rendered_clip_path,
                )
            )

        if (step + 1) % 10 == 0 or step == 0:
            latest = records[-args.num_envs :]
            value_mean = sum(r.value_delta_1000 for r in latest) / len(latest)
            reward_mean = [
                r.reward_model_score for r in latest if r.reward_model_score is not None
            ]
            reward_text = "nan" if not reward_mean else f"{sum(reward_mean) / len(reward_mean):.4f}"
            print(
                f"step={step + 1}/{args.steps} value_delta_mean={value_mean:+.4f} "
                f"reward_model_mean={reward_text}",
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
        "delta_threshold": args.delta_threshold,
    }
    write_outputs(output_dir, records, metadata)
    summary = summarize_records(records)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"wrote: {output_dir / 'dualview_reward_compare.pkl'}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

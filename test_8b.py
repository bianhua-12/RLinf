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

from __future__ import annotations

import argparse
import csv
import glob
import json
import pickle
import sys
import textwrap
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf, open_dict
from PIL import Image, ImageDraw, ImageFont

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from rlinf.models.embodiment.reward import get_reward_model_class  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--episode-glob", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-episodes", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def load_episode(path: str) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)


def get_task_description(observations: list[Any]) -> str:
    for obs in observations:
        if not isinstance(obs, dict):
            continue
        if "task_descriptions" not in obs:
            continue
        task_desc = obs["task_descriptions"]
        if isinstance(task_desc, (list, tuple)):
            if len(task_desc) == 0:
                continue
            task_desc = task_desc[0]
        if task_desc is None:
            continue
        task_desc = str(task_desc).strip()
        if task_desc:
            return task_desc
    return ""


def get_step_observations(episode: dict[str, Any]) -> list[dict[str, Any]]:
    observations = list(episode.get("observations", []))
    actions = list(episode.get("actions", []))
    if len(observations) >= len(actions) + 1:
        observations = observations[1 : len(actions) + 1]
    return [obs for obs in observations if isinstance(obs, dict)]


def get_step_rewards(episode: dict[str, Any], num_steps: int) -> list[float]:
    rewards = list(episode.get("rewards", []))
    if len(rewards) >= num_steps + 1:
        rewards = rewards[1 : num_steps + 1]
    rewards = rewards[:num_steps]
    return [float(r) for r in rewards]


def get_main_image(obs: dict[str, Any]) -> Any | None:
    if "main_images" not in obs:
        return None
    image = obs["main_images"]
    if isinstance(image, (list, tuple)) and len(image) > 0:
        image = image[0]
    return image


def get_history_frames(
    step_observations: list[dict[str, Any]], start: int, end: int
) -> list[Any]:
    frames: list[Any] = []
    for obs in step_observations[start:end]:
        if not isinstance(obs, dict):
            continue
        image = get_main_image(obs)
        if image is not None:
            frames.append(image)
    return frames


def build_sample_ranges(num_steps: int, interval: int) -> list[tuple[int, int]]:
    return [
        (max(0, end_step - interval), end_step)
        for end_step in range(interval, num_steps + 1, interval)
    ]


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def inspect_robochallenge_text(text: str) -> dict[str, Any]:
    obj = extract_json_object(text)
    if obj is None:
        return {
            "parse_ok": False,
            "judgement": None,
            "confidence": None,
        }

    judgement = obj.get("judgement", obj.get("answer", obj.get("label")))
    confidence = obj.get("confidence")
    if judgement is None:
        return {
            "parse_ok": False,
            "judgement": None,
            "confidence": None,
        }

    judgement = str(judgement).strip().lower()
    if judgement not in {"positive", "negative"}:
        return {
            "parse_ok": False,
            "judgement": judgement,
            "confidence": confidence,
        }

    try:
        conf = None if confidence is None else float(confidence)
    except Exception:
        conf = None

    return {
        "parse_ok": conf is not None,
        "judgement": judgement,
        "confidence": conf,
    }


def compute_proxy_gt(
    step_rewards: list[float],
    segment_start: int,
    end_step: int,
    success: bool,
) -> dict[str, float]:
    segment_rewards = step_rewards[segment_start:end_step]
    prefix_rewards = step_rewards[:end_step]
    segment_return = float(sum(segment_rewards))
    prefix_return = float(sum(prefix_rewards))
    return {
        "gt_reward": float(segment_return > 0.0),
        "segment_return": segment_return,
        "prefix_return": prefix_return,
        "episode_success": float(success),
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    var_x = sum((x - mean_x) ** 2 for x in xs)
    var_y = sum((y - mean_y) ** 2 for y in ys)
    if var_x <= 0 or var_y <= 0:
        return None
    cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    return cov / (var_x**0.5 * var_y**0.5)


def build_model(config_path: str, device: str):
    full_cfg = OmegaConf.load(config_path)
    model_cfg = full_cfg.reward.model
    with open_dict(model_cfg):
        model_cfg.num_envs = 1
    reward_cls = get_reward_model_class(model_cfg.model_type)
    model = reward_cls(model_cfg)
    model = model.to(device)
    model.eval()
    return model, full_cfg


def run_single_sample(
    model,
    task_description: str,
    full_frames: list[Any],
    clip_frames: list[Any],
) -> dict[str, Any]:
    observations = {"task_descriptions": [task_description]}
    history_input = {
        "full_history": {"main_images": [full_frames]},
        "history_window": {"main_images": [clip_frames]},
    }

    with torch.inference_mode():
        batched_inputs, valid_input_ids = model.input_builder.build_inputs(
            observations,
            model._model.device,
            history_input,
        )
        if len(valid_input_ids) == 0:
            return {
                "pred_reward": 0.0,
                "raw_output": "",
                "parse_ok": False,
                "judgement": None,
                "confidence": None,
                "skip_reason": "invalid_input",
            }

        prompt_length = batched_inputs["input_ids"].shape[-1]
        output_ids = model._model.generate(**batched_inputs, **model.gen_kwargs)
        texts = model._processor.batch_decode(
            output_ids[..., prompt_length:],
            skip_special_tokens=True,
        )
        pred_rewards = model.reward_parser.parse_rewards(texts)

    text = texts[0]
    parsed = inspect_robochallenge_text(text)
    return {
        "pred_reward": float(pred_rewards[0].item()),
        "raw_output": text,
        "parse_ok": bool(parsed["parse_ok"]),
        "judgement": parsed["judgement"],
        "confidence": parsed["confidence"],
        "skip_reason": None,
    }


def normalize_frame(frame: Any) -> np.ndarray:
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    frame = np.asarray(frame)
    if frame.ndim == 4 and frame.shape[0] == 1:
        frame = frame[0]
    if frame.ndim == 3 and frame.shape[0] in {1, 3} and frame.shape[-1] not in {1, 3}:
        frame = np.transpose(frame, (1, 2, 0))
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.ndim == 3 and frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)
    if np.issubdtype(frame.dtype, np.floating):
        if frame.max() <= 1.0:
            frame = frame * 255.0
    return np.clip(frame, 0, 255).astype(np.uint8)


def get_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def build_footer(width: int, lines: list[str]) -> np.ndarray:
    font = get_font(18)
    wrapped_lines: list[str] = []
    for line in lines:
        wrapped_lines.extend(textwrap.wrap(line, width=84) or [""])
    line_height = 28
    footer_height = 24 + line_height * len(wrapped_lines) + 16
    image = Image.new("RGB", (width, footer_height), "white")
    draw = ImageDraw.Draw(image)
    y = 16
    for line in wrapped_lines:
        draw.text((16, y), line, fill="black", font=font)
        y += line_height
    return np.asarray(image)


def resize_frame(frame: np.ndarray, scale: int = 3) -> np.ndarray:
    image = Image.fromarray(frame)
    width, height = image.size
    return np.asarray(image.resize((width * scale, height * scale)))


def pad_frame(frame: np.ndarray, width: int, height: int) -> np.ndarray:
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    y = (height - frame.shape[0]) // 2
    x = (width - frame.shape[1]) // 2
    canvas[y : y + frame.shape[0], x : x + frame.shape[1]] = frame
    return canvas


def render_debug_video(
    history_window_frames: list[Any],
    full_history_frames: list[Any],
    footer_lines: list[str],
    output_path: Path,
    fps: int,
) -> None:
    left_seed = resize_frame(normalize_frame(full_history_frames[0]), scale=3)
    right_seed = resize_frame(normalize_frame(history_window_frames[0]), scale=3)
    pane_height = max(left_seed.shape[0], right_seed.shape[0])
    pane_width = max(left_seed.shape[1], right_seed.shape[1])
    total_frames = max(len(full_history_frames), len(history_window_frames))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(output_path, fps=fps) as writer:
        for frame_idx in range(total_frames):
            left_frame = pad_frame(
                resize_frame(
                    normalize_frame(
                        full_history_frames[
                            min(frame_idx, len(full_history_frames) - 1)
                        ]
                    ),
                    scale=3,
                ),
                pane_width,
                pane_height,
            )
            right_frame = pad_frame(
                resize_frame(
                    normalize_frame(
                        history_window_frames[
                            min(frame_idx, len(history_window_frames) - 1)
                        ]
                    ),
                    scale=3,
                ),
                pane_width,
                pane_height,
            )

            video_frame = np.concatenate([left_frame, right_frame], axis=1)
            footer = build_footer(video_frame.shape[1], footer_lines)
            writer.append_data(np.concatenate([video_frame, footer], axis=0))


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fps = 4

    model, cfg = build_model(args.config, args.device)

    history_buffers = cfg.reward.model.history_buffers
    full_history_size = int(history_buffers.full_history.history_size)
    interval = int(history_buffers.history_window.input_interval)
    if bool(history_buffers.history_window.get("input_on_done", False)):
        raise ValueError(
            "this visual debugger assumes history_window.input_on_done=False"
        )

    episode_paths = sorted(glob.glob(args.episode_glob))[: args.max_episodes]
    rows: list[dict[str, Any]] = []

    for episode_path in episode_paths:
        episode = load_episode(episode_path)
        step_observations = get_step_observations(episode)
        sample_ranges = build_sample_ranges(len(step_observations), interval)
        if not sample_ranges:
            continue

        task_description = get_task_description(step_observations)
        if not task_description:
            continue

        step_rewards = get_step_rewards(episode, len(step_observations))
        success = bool(episode.get("success", False))
        rank = int(episode.get("rank", -1))
        env_idx = int(episode.get("env_idx", -1))
        episode_id = int(episode.get("episode_id", -1))

        for sample_idx, (segment_start, end_step) in enumerate(sample_ranges):
            full_input_start = max(0, end_step - full_history_size)
            full_input_frames = get_history_frames(
                step_observations, full_input_start, end_step
            )
            history_window_frames = get_history_frames(
                step_observations, segment_start, end_step
            )
            if len(full_input_frames) == 0 or len(history_window_frames) == 0:
                continue

            pred = run_single_sample(
                model,
                task_description,
                full_input_frames,
                history_window_frames,
            )
            gt = compute_proxy_gt(step_rewards, segment_start, end_step, success)

            raw_output = pred["raw_output"].strip() or "<empty>"
            footer_lines = [
                f"predict_reward: {pred['pred_reward']:.4f} | gt_reward: {gt['gt_reward']:.4f}",
                f"segment_return: {gt['segment_return']:.4f} | prefix_return: {gt['prefix_return']:.4f} | success: {int(success)}",
                f"task: {task_description}",
                f"raw_output: {raw_output}",
            ]

            video_rel_path = Path(
                f"videos/rank_{rank}_env_{env_idx}_episode_{episode_id}/sample_{sample_idx:04d}_end_{end_step}.mp4"
            )
            render_debug_video(
                history_window_frames,
                full_input_frames,
                footer_lines,
                output_dir / video_rel_path,
                fps=fps,
            )

            rows.append(
                {
                    "episode_path": episode_path,
                    "rank": rank,
                    "env_idx": env_idx,
                    "episode_id": episode_id,
                    "success": success,
                    "sample_idx": sample_idx,
                    "segment_start": segment_start,
                    "end_step": end_step,
                    "history_window_len": len(history_window_frames),
                    "full_history_input_len": len(full_input_frames),
                    "task_description": task_description,
                    "video_path": str(video_rel_path),
                    **gt,
                    **pred,
                }
            )

    rows_path = output_dir / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\\n")

    parse_failures = [row for row in rows if not row["parse_ok"]]
    with (output_dir / "top_parse_failures.jsonl").open("w", encoding="utf-8") as f:
        for row in parse_failures[:100]:
            f.write(json.dumps(row, ensure_ascii=False) + "\\n")

    pred_rewards = [row["pred_reward"] for row in rows]
    segment_returns = [row["segment_return"] for row in rows]
    prefix_returns = [row["prefix_return"] for row in rows]
    success_labels = [row["episode_success"] for row in rows]

    summary = {
        "num_episodes": len(episode_paths),
        "num_rows": len(rows),
        "num_videos": len(rows),
        "parse_ok_rate": None
        if len(rows) == 0
        else sum(row["parse_ok"] for row in rows) / len(rows),
        "pred_reward_mean": None
        if len(rows) == 0
        else sum(pred_rewards) / len(pred_rewards),
        "pred_positive_rate_at_0_5": None
        if len(rows) == 0
        else sum(r >= 0.5 for r in pred_rewards) / len(pred_rewards),
        "corr_pred_vs_segment_return": pearson(pred_rewards, segment_returns),
        "corr_pred_vs_prefix_return": pearson(pred_rewards, prefix_returns),
        "corr_pred_vs_episode_success": pearson(pred_rewards, success_labels),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    with (output_dir / "summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary.keys()))
        writer.writeheader()
        writer.writerow(summary)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"videos saved to: {output_dir / 'videos'}")


if __name__ == "__main__":
    main()

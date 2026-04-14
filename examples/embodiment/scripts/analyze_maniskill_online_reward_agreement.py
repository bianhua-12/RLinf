#!/usr/bin/env python3
"""Analyze online ManiSkill trajectories with env/value/Qwen reward agreement."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from PIL import Image, ImageDraw
from transformers import AutoModelForVision2Seq, AutoProcessor

from rlinf.data.datasets.vlm import SimpleRobochallengeSFTDataset
from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy
from rlinf.models.embodiment.reward import get_reward_model_class


DEFAULT_VALUE_CKPT = (
    "logs/20260411-20:03:49-maniskill_ppo_mlp/maniskill_ppo_mlp/"
    "checkpoints/global_step_1000/actor/model_state_dict/full_weights.pt"
)
DEFAULT_QWEN_MODEL = "Qwen3-VL-4B-Instruct"
DEFAULT_QWEN_LORA = (
    "logs/20260413-21:38:14/qwen3_vl_4b_5frame_value_judge_sft/"
    "checkpoints/global_step_1000"
)


@dataclass
class WindowRecord:
    episode_path: str
    train_step: int
    step_bin_start: int
    env_idx: int
    episode_id: int
    window_index: int
    frame_start: int
    frame_end: int
    success: bool
    task: str
    env_reward_window_sum: float
    env_reward_window_mean: float
    env_reward_label: str
    value_window_start: float
    value_window_end: float
    value_delta: float
    value_label: str
    rlinf_qwen_reward: float
    rlinf_qwen_label: str
    rlinf_qwen_output: str
    direct_qwen_label: str
    direct_qwen_output: str
    value_matches_env: bool
    rlinf_qwen_matches_env: bool
    direct_qwen_matches_env: bool
    direct_qwen_matches_value: bool
    direct_qwen_matches_rlinf_qwen: bool
    clip_path: str | None
    rendered_clip_path: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--value-ckpt", default=DEFAULT_VALUE_CKPT)
    parser.add_argument("--qwen-model-path", default=DEFAULT_QWEN_MODEL)
    parser.add_argument("--qwen-lora-path", default=DEFAULT_QWEN_LORA)
    parser.add_argument("--value-device", default="cuda:0")
    parser.add_argument("--wrapper-device", default="cuda:0")
    parser.add_argument("--direct-device", default="cuda:1")
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--window-stride", type=int, default=1)
    parser.add_argument("--step-bin-size", type=int, default=50)
    parser.add_argument("--max-episodes", type=int, default=-1)
    parser.add_argument("--max-windows", type=int, default=-1)
    parser.add_argument("--save-clips", action="store_true")
    parser.add_argument("--clip-fps", type=int, default=10)
    parser.add_argument("--render-frame-repeat", type=int, default=1)
    parser.add_argument("--qwen-micro-batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    return parser.parse_args()


def abs_path(path: str) -> str:
    return str(Path(path).expanduser().resolve())


def torch_dtype(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def resolve_device(device: str) -> str:
    if not device.startswith("cuda"):
        return device
    if not torch.cuda.is_available():
        return "cpu"
    if ":" not in device:
        return device
    try:
        index = int(device.split(":", maxsplit=1)[1])
    except ValueError:
        return device
    if index < torch.cuda.device_count():
        return device
    return "cuda:0"


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


def build_wrapper_qwen_cfg(args: argparse.Namespace):
    return OmegaConf.create(
        {
            "model_type": "history_vlm",
            "model_path": abs_path(args.qwen_model_path),
            "lora_path": abs_path(args.qwen_lora_path),
            "precision": args.dtype,
            "input_builder_name": "simple_robochallenge_input_builder",
            "reward_parser_name": "simple_robochallenge_reward_parser",
            "history_buffers": {
                "history_window": {
                    "history_size": args.window_size,
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
            "max_new_tokens": args.max_new_tokens,
            "do_sample": False,
            "temperature": 0.0,
            "use_chat_template": True,
        }
    )


def load_direct_qwen(
    model_path: str,
    lora_path: str,
    device: str,
    dtype: torch.dtype,
):
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
    )

    checkpoint = Path(lora_path)
    if checkpoint.is_dir():
        checkpoint = checkpoint / "actor" / "model_state_dict" / "full_weights.pt"
    checkpoint_state_dict = torch.load(
        str(checkpoint),
        map_location="cpu",
        weights_only=True,
    )
    checkpoint_state_dict = {
        key.removeprefix("module."): value
        for key, value in checkpoint_state_dict.items()
    }
    if checkpoint_state_dict and all(key.startswith("model.") for key in checkpoint_state_dict):
        checkpoint_state_dict = {
            key.removeprefix("model."): value
            for key, value in checkpoint_state_dict.items()
        }
    lora_state_dict = {
        key: value
        for key, value in checkpoint_state_dict.items()
        if "lora_" in key
    }

    if lora_state_dict:
        lora_rank = next(
            int(value.shape[0])
            for key, value in lora_state_dict.items()
            if "lora_A" in key
        )
        target_modules = sorted(
            {
                key.split(".lora_")[0].split(".")[-1]
                for key in lora_state_dict
                if ".lora_" in key
            }
        )
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora_rank,
                lora_alpha=lora_rank,
                lora_dropout=0.0,
                target_modules=target_modules,
                init_lora_weights="gaussian",
            ),
        )
        set_peft_model_state_dict(model, lora_state_dict)
    else:
        model.load_state_dict(checkpoint_state_dict, strict=True)
    del checkpoint_state_dict
    del lora_state_dict

    return processor, model.to(device).eval()


def list_episode_paths(root: Path) -> list[Path]:
    return sorted(root.rglob("*.pkl"))


def load_episode(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return pickle.load(f)


def to_float(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().float().cpu().item())
    return float(value)


def to_uint8_rgb(image: Any):
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()
    image = image[..., :3]
    if str(image.dtype) != "uint8":
        image = image.clip(0, 255).astype("uint8")
    return image


def build_prompt(task: str) -> str:
    return (
        f"You are currently performing the task: {task}. "
        "Please judge whether the operation shown in this video segment "
        "makes the task better or worse. Answer with exactly one word: "
        "positive or negative."
    )


def parse_output_label(text: str) -> str:
    lowered = str(text).strip().lower()
    if "negative" in lowered:
        return "negative"
    if "positive" in lowered:
        return "positive"
    return "unknown"


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
def generate_direct_qwen(
    processor,
    model,
    *,
    frames: list[Any],
    prompt: str,
    max_new_tokens: int,
) -> str:
    pil_frames = [Image.fromarray(to_uint8_rgb(frame)).convert("RGB") for frame in frames]
    _, inputs, _ = SimpleRobochallengeSFTDataset.process_inputs(
        processor=processor,
        system_prompt=None,
        use_chat_template=True,
        prompt_texts=[[prompt]],
        videos=[[pil_frames]],
        answer_text=None,
    )
    inputs = {key: value.to(model.device) for key, value in inputs.items()}
    prompt_length = inputs["input_ids"].shape[-1]
    output_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
    )
    decoded = processor.batch_decode(
        output_ids[..., prompt_length:],
        skip_special_tokens=True,
    )
    return decoded[0]


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
    return image


def save_clip(
    output_dir: Path,
    frames: list[Any],
    *,
    relative_path: Path,
    fps: int,
    render_frame_repeat: int,
    overlay_lines: list[str] | None = None,
) -> str:
    path = output_dir / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps) as writer:
        for frame_idx, frame in enumerate(frames):
            if overlay_lines is None:
                rendered = to_uint8_rgb(frame)
            else:
                rendered = to_uint8_rgb(
                    draw_overlay(
                    frame,
                    [f"input_frame={frame_idx + 1}/{len(frames)}"] + overlay_lines,
                    )
                )
            for _ in range(render_frame_repeat):
                writer.append_data(rendered)
    return str(relative_path)


def summarize_records(records: list[WindowRecord]) -> dict[str, Any]:
    total = len(records)
    if total == 0:
        return {"total": 0}

    counts = Counter()
    by_step_bin: dict[int, Counter] = defaultdict(Counter)
    for record in records:
        counts["total"] += 1
        counts[f"env_{record.env_reward_label}"] += 1
        counts[f"value_{record.value_label}"] += 1
        counts[f"rlinf_qwen_{record.rlinf_qwen_label}"] += 1
        counts[f"direct_qwen_{record.direct_qwen_label}"] += 1
        counts["value_matches_env"] += int(record.value_matches_env)
        counts["rlinf_qwen_matches_env"] += int(record.rlinf_qwen_matches_env)
        counts["direct_qwen_matches_env"] += int(record.direct_qwen_matches_env)
        counts["direct_qwen_matches_value"] += int(record.direct_qwen_matches_value)
        counts["direct_qwen_matches_rlinf_qwen"] += int(
            record.direct_qwen_matches_rlinf_qwen
        )

        bin_counts = by_step_bin[record.step_bin_start]
        bin_counts["total"] += 1
        bin_counts["value_matches_env"] += int(record.value_matches_env)
        bin_counts["rlinf_qwen_matches_env"] += int(record.rlinf_qwen_matches_env)
        bin_counts["direct_qwen_matches_env"] += int(record.direct_qwen_matches_env)
        bin_counts["direct_qwen_matches_value"] += int(record.direct_qwen_matches_value)
        bin_counts["direct_qwen_matches_rlinf_qwen"] += int(
            record.direct_qwen_matches_rlinf_qwen
        )

    summary = {
        "total": total,
        "counts": dict(counts),
        "rates": {
            "value_matches_env": counts["value_matches_env"] / total,
            "rlinf_qwen_matches_env": counts["rlinf_qwen_matches_env"] / total,
            "direct_qwen_matches_env": counts["direct_qwen_matches_env"] / total,
            "direct_qwen_matches_value": counts["direct_qwen_matches_value"] / total,
            "direct_qwen_matches_rlinf_qwen": (
                counts["direct_qwen_matches_rlinf_qwen"] / total
            ),
        },
        "by_step_bin": {
            str(step_bin): {
                "total": counter["total"],
                "value_matches_env": counter["value_matches_env"] / counter["total"],
                "rlinf_qwen_matches_env": (
                    counter["rlinf_qwen_matches_env"] / counter["total"]
                ),
                "direct_qwen_matches_env": (
                    counter["direct_qwen_matches_env"] / counter["total"]
                ),
                "direct_qwen_matches_value": (
                    counter["direct_qwen_matches_value"] / counter["total"]
                ),
                "direct_qwen_matches_rlinf_qwen": (
                    counter["direct_qwen_matches_rlinf_qwen"] / counter["total"]
                ),
            }
            for step_bin, counter in sorted(by_step_bin.items())
            if counter["total"] > 0
        },
    }
    return summary


def write_outputs(output_dir: Path, records: list[WindowRecord], metadata: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {"metadata": metadata, "records": [asdict(record) for record in records]}
    with (output_dir / "online_reward_agreement.pkl").open("wb") as f:
        pickle.dump(payload, f)
    with (output_dir / "online_reward_agreement.jsonl").open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    if records:
        with (output_dir / "online_reward_agreement.csv").open(
            "w", newline="", encoding="utf-8"
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(records[0]).keys()))
            writer.writeheader()
            for record in records:
                writer.writerow(asdict(record))
    summary = summarize_records(records)
    summary["metadata"] = metadata
    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    args = parse_args()
    args.value_device = resolve_device(args.value_device)
    args.wrapper_device = resolve_device(args.wrapper_device)
    args.direct_device = resolve_device(args.direct_device)
    episode_root = Path(args.episode_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    value_model = load_mlp(args.value_ckpt, args.value_device)
    reward_cls = get_reward_model_class("history_vlm")
    wrapper_qwen = reward_cls(build_wrapper_qwen_cfg(args)).to(args.wrapper_device).eval()
    direct_processor, direct_model = load_direct_qwen(
        abs_path(args.qwen_model_path),
        abs_path(args.qwen_lora_path),
        args.direct_device,
        torch_dtype(args.dtype),
    )

    episode_paths = list_episode_paths(episode_root)
    if args.max_episodes > 0:
        episode_paths = episode_paths[: args.max_episodes]

    records: list[WindowRecord] = []
    global_window_count = 0

    for episode_idx, episode_path in enumerate(episode_paths):
        episode = load_episode(episode_path)
        observations = list(episode.get("observations", []))
        rewards = list(episode.get("rewards", []))
        if len(observations) < args.window_size:
            continue

        task = str(
            episode.get("task")
            or episode.get("task_name")
            or observations[0].get("task_descriptions", "")
        )
        train_step = int(episode.get("step", 0))
        step_bin_start = (train_step // args.step_bin_size) * args.step_bin_size

        states = torch.stack(
            [
                obs["states"].detach().float().cpu()
                if isinstance(obs["states"], torch.Tensor)
                else torch.as_tensor(obs["states"], dtype=torch.float32)
                for obs in observations
            ],
            dim=0,
        )
        with torch.inference_mode():
            values = (
                value_model.value_head(states.to(args.value_device))
                .detach()
                .float()
                .cpu()
                .reshape(-1)
            )

        window_payloads: list[dict[str, Any]] = []
        for start in range(0, len(observations) - args.window_size + 1, args.window_stride):
            if args.max_windows > 0 and global_window_count >= args.max_windows:
                break
            end = start + args.window_size
            frames = [observations[idx]["main_images"] for idx in range(start, end)]
            reward_window = [to_float(rewards[idx]) for idx in range(start, end)]
            value_start = float(values[start].item())
            value_end = float(values[end - 1].item())
            value_delta = value_end - value_start
            window_payloads.append(
                {
                    "start": start,
                    "end": end,
                    "frames": frames,
                    "reward_sum": float(sum(reward_window)),
                    "reward_mean": float(sum(reward_window) / args.window_size),
                    "value_start": value_start,
                    "value_end": value_end,
                    "value_delta": value_delta,
                    "prompt": build_prompt(task),
                }
            )
            global_window_count += 1
        if not window_payloads:
            continue

        reward_input = {
            "task_descriptions": [task for _ in window_payloads],
            "history_input": {
                "history_window": {
                    "main_images": [payload["frames"] for payload in window_payloads],
                }
            },
        }
        wrapper_rewards, wrapper_outputs = qwen_generate_with_text(wrapper_qwen, reward_input)

        for local_idx, payload in enumerate(window_payloads):
            direct_output = generate_direct_qwen(
                direct_processor,
                direct_model,
                frames=payload["frames"],
                prompt=payload["prompt"],
                max_new_tokens=args.max_new_tokens,
            )

            env_label = "positive" if payload["reward_sum"] > 0.0 else "negative"
            value_label = "positive" if payload["value_delta"] >= 0.0 else "negative"
            wrapper_reward = float(wrapper_rewards[local_idx].item())
            wrapper_output = wrapper_outputs[local_idx]
            wrapper_label = "positive" if wrapper_reward >= 0.5 else "negative"
            direct_label = parse_output_label(direct_output)

            clip_path = None
            rendered_clip_path = None
            if args.save_clips:
                episode_stem = episode_path.stem
                safe_episode_stem = episode_stem.replace("/", "_")
                rank_str = "00"
                if "rank_" in episode_stem:
                    rank_str = episode_stem.split("rank_", maxsplit=1)[1].split("_", maxsplit=1)[0]
                base_name = (
                    f"{safe_episode_stem}"
                    f"_rank_{int(rank_str):02d}"
                    f"_env_{int(episode.get('env_idx', 0)):02d}"
                    f"_step_{train_step:06d}"
                    f"_episode_{int(episode.get('episode_id', episode_idx)):05d}"
                    f"_window_{local_idx:04d}"
                )
                clip_path = save_clip(
                    output_dir,
                    payload["frames"],
                    relative_path=Path("clips") / f"{base_name}.mp4",
                    fps=args.clip_fps,
                    render_frame_repeat=args.render_frame_repeat,
                    overlay_lines=None,
                )
                overlay_lines = [
                    f"train_step={train_step} frame={payload['start']}->{payload['end'] - 1}",
                    f"env_sum={payload['reward_sum']:+.4f} ({env_label})",
                    f"value_delta={payload['value_delta']:+.4f} ({value_label})",
                    f"rlinf_qwen={wrapper_label} reward={wrapper_reward:.4f}",
                    f"direct_qwen={direct_label}",
                ]
                rendered_clip_path = save_clip(
                    output_dir,
                    payload["frames"],
                    relative_path=Path("rendered_clips") / f"{base_name}.mp4",
                    fps=args.clip_fps,
                    render_frame_repeat=args.render_frame_repeat,
                    overlay_lines=overlay_lines,
                )

            records.append(
                WindowRecord(
                    episode_path=str(episode_path),
                    train_step=train_step,
                    step_bin_start=step_bin_start,
                    env_idx=int(episode.get("env_idx", 0)),
                    episode_id=int(episode.get("episode_id", episode_idx)),
                    window_index=local_idx,
                    frame_start=int(payload["start"]),
                    frame_end=int(payload["end"] - 1),
                    success=bool(episode.get("success", False)),
                    task=task,
                    env_reward_window_sum=float(payload["reward_sum"]),
                    env_reward_window_mean=float(payload["reward_mean"]),
                    env_reward_label=env_label,
                    value_window_start=float(payload["value_start"]),
                    value_window_end=float(payload["value_end"]),
                    value_delta=float(payload["value_delta"]),
                    value_label=value_label,
                    rlinf_qwen_reward=wrapper_reward,
                    rlinf_qwen_label=wrapper_label,
                    rlinf_qwen_output=wrapper_output,
                    direct_qwen_label=direct_label,
                    direct_qwen_output=direct_output,
                    value_matches_env=(value_label == env_label),
                    rlinf_qwen_matches_env=(wrapper_label == env_label),
                    direct_qwen_matches_env=(direct_label == env_label),
                    direct_qwen_matches_value=(direct_label == value_label),
                    direct_qwen_matches_rlinf_qwen=(direct_label == wrapper_label),
                    clip_path=clip_path,
                    rendered_clip_path=rendered_clip_path,
                )
            )

        print(
            f"episode={episode_idx + 1}/{len(episode_paths)} "
            f"windows={len(window_payloads)} total_records={len(records)}",
            flush=True,
        )
        if args.max_windows > 0 and global_window_count >= args.max_windows:
            break

    metadata = {
        "episode_root": str(episode_root),
        "value_ckpt": abs_path(args.value_ckpt),
        "qwen_model_path": abs_path(args.qwen_model_path),
        "qwen_lora_path": abs_path(args.qwen_lora_path),
        "window_size": args.window_size,
        "window_stride": args.window_stride,
        "step_bin_size": args.step_bin_size,
        "max_episodes": args.max_episodes,
        "max_windows": args.max_windows,
        "save_clips": args.save_clips,
        "clip_fps": args.clip_fps,
        "render_frame_repeat": args.render_frame_repeat,
    }
    write_outputs(output_dir, records, metadata)

    summary = summarize_records(records)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

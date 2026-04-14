#!/usr/bin/env python3
"""Export dual-view 5-frame value-judge samples from collected episodes."""

from __future__ import annotations

import argparse
import json
import pickle
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import numpy as np
import torch

from rlinf.models.embodiment.mlp_policy.mlp_policy import MLPPolicy


DEFAULT_VALUE_CKPT = (
    "/mnt/project_rlinf/ztx/RLinf/logs/20260411-20:03:49-maniskill_ppo_mlp/"
    "maniskill_ppo_mlp/checkpoints/global_step_1000/actor/model_state_dict/full_weights.pt"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-episode-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--value-ckpt", default=DEFAULT_VALUE_CKPT)
    parser.add_argument("--task-description", default=None)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--delta-threshold", type=float, default=0.2)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--obs-dim", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--balance",
        choices=("none", "downsample", "posneg_downsample"),
        default="posneg_downsample",
        help="Balance all labels or only positive/negative separately in train/eval.",
    )
    parser.add_argument("--max-per-label", type=int, default=None)
    parser.add_argument(
        "--reverse-positive-as-negative",
        action="store_true",
        help="Reverse both videos of each positive sample and relabel it as negative.",
    )
    return parser.parse_args()


def prompt_for_task(task: str) -> str:
    return (
        f"You are currently performing the task: {task}. "
        "Please judge whether the operation shown in these two video views "
        "makes the task better, worse, or unchanged. "
        "Answer with exactly one word: positive, negative, or unchanged."
    )


def to_numpy_rgb(image: Any) -> np.ndarray:
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image[..., :3]


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


def infer_obs_dim(episode_paths: list[Path]) -> int:
    for episode_path in episode_paths:
        with episode_path.open("rb") as f:
            episode = pickle.load(f)
        observations = episode.get("observations", [])
        for obs in observations:
            state = obs.get("states")
            if state is None:
                continue
            if torch.is_tensor(state):
                return int(state.numel())
            return int(np.asarray(state).size)
    raise ValueError("Failed to infer obs_dim from raw episodes.")


def load_value_model(path: str, obs_dim: int, device: str) -> MLPPolicy:
    model = MLPPolicy(
        obs_dim=obs_dim,
        action_dim=8,
        num_action_chunks=1,
        add_value_head=True,
        add_q_head=False,
    )
    state = torch.load(str(Path(path).resolve()), map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def extract_dual_view_frames(observations: list[dict[str, Any]], start_idx: int, end_idx: int) -> tuple[list[Any], list[Any]] | None:
    main_frames: list[Any] = []
    third_frames: list[Any] = []
    for idx in range(start_idx, end_idx + 1):
        obs = observations[idx]
        main_image = obs.get("main_images")
        extra_view = obs.get("extra_view_images")
        if main_image is None or extra_view is None:
            return None
        if torch.is_tensor(extra_view):
            if extra_view.ndim < 4 or extra_view.shape[0] < 1:
                return None
            third_image = extra_view[0]
        else:
            extra_array = np.asarray(extra_view)
            if extra_array.ndim < 4 or extra_array.shape[0] < 1:
                return None
            third_image = extra_array[0]
        main_frames.append(main_image)
        third_frames.append(third_image)
    return main_frames, third_frames


def write_clip(path: Path, frames: list[Any], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(to_numpy_rgb(frame))


def build_messages(prompt: str, main_rel_path: str, third_rel_path: str, label: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": main_rel_path},
                {"type": "video", "video": third_rel_path},
                {"type": "text", "text": prompt},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": label}],
        },
    ]


def split_name(
    episode_path: Path, eval_ratio: float, seed: int, explicit_eval: bool | None
) -> str:
    if explicit_eval is not None:
        return "eval" if explicit_eval else "train"
    rng = random.Random(f"{seed}:{episode_path.name}")
    return "eval" if rng.random() < eval_ratio else "train"


def maybe_balance(
    samples: list[dict[str, Any]],
    *,
    balance: str,
    max_per_label: int | None,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        grouped[sample["answer"]].append(sample)

    rng = random.Random(seed)
    for values in grouped.values():
        rng.shuffle(values)

    if balance == "downsample" and grouped:
        keep = min(len(values) for values in grouped.values())
        if max_per_label is not None:
            keep = min(keep, max_per_label)
        grouped = {label: values[:keep] for label, values in grouped.items()}
    elif balance == "posneg_downsample":
        if "positive" in grouped and "negative" in grouped:
            keep = min(len(grouped["positive"]), len(grouped["negative"]))
            if max_per_label is not None:
                keep = min(keep, max_per_label)
            grouped["positive"] = grouped["positive"][:keep]
            grouped["negative"] = grouped["negative"][:keep]
        elif max_per_label is not None:
            grouped = {label: values[:max_per_label] for label, values in grouped.items()}
        if max_per_label is not None and "unchanged" in grouped:
            grouped["unchanged"] = grouped["unchanged"][:max_per_label]
    elif max_per_label is not None:
        grouped = {label: values[:max_per_label] for label, values in grouped.items()}

    merged: list[dict[str, Any]] = []
    for values in grouped.values():
        merged.extend(values)
    rng.shuffle(merged)
    return merged


def write_manifest(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            clean_sample = {
                key: value for key, value in sample.items() if not key.startswith("_")
            }
            f.write(json.dumps(clean_sample, ensure_ascii=False) + "\n")


def materialize_clips(
    output_dir: Path,
    samples_by_split: dict[str, list[dict[str, Any]]],
    *,
    fps: int,
) -> None:
    cache: dict[Path, list[dict[str, Any]]] = {}
    for samples in samples_by_split.values():
        for sample in samples:
            episode_path = Path(sample["_episode_path"])
            if episode_path not in cache:
                with episode_path.open("rb") as f:
                    episode = pickle.load(f)
                cache[episode_path] = episode.get("observations", [])
            observations = cache[episode_path]
            start_idx = int(sample["_start_idx"])
            end_idx = int(sample["_end_idx"])
            frames = extract_dual_view_frames(observations, start_idx, end_idx)
            if frames is None:
                continue
            main_frames, third_frames = frames
            if sample.get("_reverse_frames"):
                main_frames = list(reversed(main_frames))
                third_frames = list(reversed(third_frames))
            write_clip(output_dir / sample["main_clip_path"], main_frames, fps=fps)
            write_clip(output_dir / sample["third_clip_path"], third_frames, fps=fps)


def label_from_delta(delta: float, threshold: float) -> str:
    if delta > threshold:
        return "positive"
    if delta < -threshold:
        return "negative"
    return "unchanged"


def build_reverse_positive_sample(sample: dict[str, Any]) -> dict[str, Any]:
    reversed_sample = dict(sample)
    reversed_sample["_reverse_frames"] = True
    reversed_sample["answer"] = "negative"
    main_path = Path(sample["main_clip_path"])
    third_path = Path(sample["third_clip_path"])
    reversed_main_path = main_path.with_name(f"negative_reverse_{main_path.name}")
    reversed_third_path = third_path.with_name(f"negative_reverse_{third_path.name}")
    reversed_sample["main_clip_path"] = str(reversed_main_path)
    reversed_sample["third_clip_path"] = str(reversed_third_path)
    reversed_sample["messages"] = build_messages(
        str(sample["prompt"]),
        str(reversed_main_path),
        str(reversed_third_path),
        "negative",
    )
    reversed_sample["segment_metadata"] = dict(sample["segment_metadata"])
    reversed_sample["segment_metadata"]["augmentation"] = "reverse_positive"
    reversed_sample["supervision"] = dict(sample["supervision"])
    reversed_sample["supervision"]["label"] = "negative"
    reversed_sample["supervision"]["source_label"] = "positive"
    reversed_sample["supervision"]["source_score"] = float(sample["supervision"]["score"])
    reversed_sample["supervision"]["score"] = -abs(float(sample["supervision"]["score"]))
    reversed_sample["supervision"]["score_name"] = "reversed_positive_value_delta_window"
    return reversed_sample


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_episode_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    episode_paths = sorted(raw_root.glob("*.pkl"))
    print(
        json.dumps(
            {
                "stage": "start",
                "raw_episode_root": str(raw_root),
                "output_dir": str(output_dir),
                "num_episodes": len(episode_paths),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )
    obs_dim = args.obs_dim or infer_obs_dim(episode_paths)
    device = resolve_device(args.device)
    value_model = load_value_model(args.value_ckpt, obs_dim=obs_dim, device=device)

    candidates_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    skipped = Counter()

    total_episodes = len(episode_paths)
    for episode_idx, episode_path in enumerate(episode_paths, start=1):
        with episode_path.open("rb") as f:
            episode = pickle.load(f)
        observations = episode.get("observations", [])
        if len(observations) < args.window_size:
            skipped["too_short"] += 1
            continue

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
                value_model.value_head(states.to(device))
                .detach()
                .float()
                .cpu()
                .reshape(-1)
            )

        split = split_name(
            episode_path,
            eval_ratio=args.eval_ratio,
            seed=args.seed,
            explicit_eval=None,
        )
        task = str(
            episode.get("task")
            or episode.get("task_description")
            or args.task_description
            or "Pick up the red cube and place it on the green spot on the table."
        )
        prompt = prompt_for_task(task)

        for start_idx in range(0, len(observations) - args.window_size + 1, args.stride):
            end_idx = start_idx + args.window_size - 1
            if extract_dual_view_frames(observations, start_idx, end_idx) is None:
                skipped["missing_dual_view"] += 1
                continue
            start_value = float(values[start_idx].item())
            end_value = float(values[end_idx].item())
            delta = end_value - start_value
            label = label_from_delta(delta, args.delta_threshold)
            base_name = (
                f"{episode_path.stem}_frames_{start_idx:04d}_{end_idx:04d}"
            )
            main_rel_path = Path(split) / "main_clips" / f"{label}_{base_name}.mp4"
            third_rel_path = Path(split) / "third_clips" / f"{label}_{base_name}.mp4"
            sample = {
                "_episode_path": str(episode_path),
                "_start_idx": start_idx,
                "_end_idx": end_idx,
                "task": task,
                "prompt": prompt,
                "answer": label,
                "messages": build_messages(
                    str(prompt),
                    str(main_rel_path),
                    str(third_rel_path),
                    label,
                ),
                "main_clip_path": str(main_rel_path),
                "third_clip_path": str(third_rel_path),
                "source_episode_path": str(episode_path),
                "segment_metadata": {
                    "start_step": start_idx,
                    "end_step": end_idx,
                    "window_size": args.window_size,
                    "episode_id": episode.get("episode_id"),
                    "env_idx": episode.get("env_idx"),
                    "success": episode.get("success"),
                    "views": ["main_images", "extra_view_images[0]"],
                },
                "supervision": {
                    "label": label,
                    "score": delta,
                    "score_name": "value_delta_window",
                    "delta_threshold": args.delta_threshold,
                    "start_value": start_value,
                    "end_value": end_value,
                    "value_ckpt": str(Path(args.value_ckpt).resolve()),
                },
            }
            candidates_by_split[split].append(sample)
            if args.reverse_positive_as_negative and label == "positive":
                candidates_by_split[split].append(build_reverse_positive_sample(sample))

        if episode_idx == 1 or episode_idx % 20 == 0 or episode_idx == total_episodes:
            partial_info = {
                "stage": "scan",
                "episode_idx": episode_idx,
                "total_episodes": total_episodes,
                "current_episode": episode_path.name,
                "skipped": dict(skipped),
                "raw_counts": {
                    split_name: dict(Counter(sample["answer"] for sample in samples))
                    for split_name, samples in candidates_by_split.items()
                },
            }
            with (output_dir / "dataset_info.partial.json").open(
                "w", encoding="utf-8"
            ) as f:
                json.dump(partial_info, f, ensure_ascii=False, indent=2)
            print(json.dumps(partial_info, ensure_ascii=False), flush=True)

    raw_counts = {
        split: dict(Counter(sample["answer"] for sample in samples))
        for split, samples in candidates_by_split.items()
    }
    final_samples_by_split = {
        split: maybe_balance(
            samples,
            balance=args.balance,
            max_per_label=args.max_per_label,
            seed=args.seed + (0 if split == "train" else 1),
        )
        for split, samples in candidates_by_split.items()
    }
    final_counts = {
        split: dict(Counter(sample["answer"] for sample in samples))
        for split, samples in final_samples_by_split.items()
    }

    print(
        json.dumps(
            {
                "stage": "materialize",
                "final_counts": final_counts,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    materialize_clips(output_dir, final_samples_by_split, fps=args.fps)

    for split, samples in final_samples_by_split.items():
        write_manifest(output_dir / split / "segments.jsonl", samples)

    info = {
        "raw_episode_root": str(raw_root),
        "value_ckpt": str(Path(args.value_ckpt).resolve()),
        "output_dir": str(output_dir),
        "task_description": args.task_description,
        "window_size": args.window_size,
        "stride": args.stride,
        "delta_threshold": args.delta_threshold,
        "eval_ratio": args.eval_ratio,
        "seed": args.seed,
        "balance": args.balance,
        "max_per_label": args.max_per_label,
        "reverse_positive_as_negative": args.reverse_positive_as_negative,
        "num_episodes": len(episode_paths),
        "obs_dim": obs_dim,
        "device": device,
        "skipped": dict(skipped),
        "raw_counts": raw_counts,
        "final_counts": final_counts,
        "train_manifest": str(output_dir / "train" / "segments.jsonl"),
        "eval_manifest": str(output_dir / "eval" / "segments.jsonl"),
    }
    with (output_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(json.dumps(info, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

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

"""Export 5-frame value-delta video judge samples from collected episodes."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-episode-root", required=True)
    parser.add_argument("--value-scores-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--task-description",
        default="pick the red cube to the green point",
    )
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--balance",
        choices=("none", "downsample"),
        default="downsample",
        help="Downsample the majority class separately in train/eval. No duplication.",
    )
    parser.add_argument("--max-per-label", type=int, default=None)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--reverse-positive-as-negative",
        action="store_true",
        help=(
            "Add a reversed copy of each positive window and label it as negative. "
            "This creates stronger negative examples than tiny value drops."
        ),
    )
    return parser.parse_args()


def prompt_for_task(task: str) -> str:
    return (
        f"You are currently performing the task: {task}. "
        "Please judge whether the operation shown in these 5 consecutive frames "
        "makes the task better or worse. Answer with exactly one word: positive "
        "or negative."
    )


def to_numpy_rgb(image: Any) -> np.ndarray:
    if torch.is_tensor(image):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image[..., :3]


def load_value_scores(path: Path) -> dict[str, list[float]]:
    scores_by_path: dict[str, list[float]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            episode_path = str(Path(record["episode_path"]).resolve())
            scores = [float(value) for value in record["value_scores"]]
            scores_by_path[episode_path] = scores
            scores_by_path[Path(episode_path).name] = scores
    return scores_by_path


def get_scores_for_episode(
    scores_by_path: dict[str, list[float]], episode_path: Path
) -> list[float] | None:
    return scores_by_path.get(str(episode_path.resolve())) or scores_by_path.get(
        episode_path.name
    )


def write_clip(path: Path, frames: list[Any], fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(path, fps=fps) as writer:
        for frame in frames:
            writer.append_data(to_numpy_rgb(frame))


def build_messages(prompt: str, clip_rel_path: str, label: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": clip_rel_path},
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
    if set(grouped) != {"positive", "negative"}:
        return samples

    rng = random.Random(seed)
    pos = grouped["positive"]
    neg = grouped["negative"]
    for values in (pos, neg):
        rng.shuffle(values)

    if balance == "downsample":
        keep = min(len(pos), len(neg))
        if max_per_label is not None:
            keep = min(keep, max_per_label)
        pos = pos[:keep]
        neg = neg[:keep]
    elif max_per_label is not None:
        pos = pos[:max_per_label]
        neg = neg[:max_per_label]

    merged = pos + neg
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
            frames = [
                observations[idx]["main_images"]
                for idx in range(start_idx, end_idx + 1)
            ]
            if sample.get("_reverse_frames"):
                frames = list(reversed(frames))
            write_clip(output_dir / sample["clip_path"], frames, fps=fps)


def build_reverse_positive_sample(
    sample: dict[str, Any], prompt: str
) -> dict[str, Any]:
    """Create a negative sample by reversing a value-increasing clip."""
    clip_path = Path(sample["clip_path"])
    reversed_clip_path = clip_path.with_name(f"negative_reverse_{clip_path.name}")
    reversed_sample = dict(sample)
    reversed_sample["_reverse_frames"] = True
    reversed_sample["answer"] = "negative"
    reversed_sample["messages"] = build_messages(
        str(prompt), str(reversed_clip_path), "negative"
    )
    reversed_sample["clip_path"] = str(reversed_clip_path)
    reversed_sample["segment_metadata"] = dict(sample["segment_metadata"])
    reversed_sample["segment_metadata"]["augmentation"] = "reverse_positive"
    reversed_sample["supervision"] = dict(sample["supervision"])
    reversed_sample["supervision"]["label"] = "negative"
    reversed_sample["supervision"]["score"] = -abs(
        float(sample["supervision"]["score"])
    )
    reversed_sample["supervision"]["score_name"] = (
        "reversed_positive_value_delta_window"
    )
    reversed_sample["supervision"]["source_label"] = "positive"
    reversed_sample["supervision"]["source_score"] = float(
        sample["supervision"]["score"]
    )
    return reversed_sample


def main() -> None:
    args = parse_args()
    raw_root = Path(args.raw_episode_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    scores_by_path = load_value_scores(Path(args.value_scores_jsonl).resolve())
    prompt = prompt_for_task(args.task_description)

    candidates_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "eval": []}
    skipped = Counter()
    episode_paths = sorted(raw_root.glob("*.pkl"))
    for episode_path in episode_paths:
        scores = get_scores_for_episode(scores_by_path, episode_path)
        if scores is None:
            skipped["missing_scores"] += 1
            continue
        with episode_path.open("rb") as f:
            episode = pickle.load(f)
        observations = episode.get("observations", [])
        max_end = min(len(scores), len(observations)) - 1
        if max_end + 1 < args.window_size:
            skipped["too_short"] += 1
            continue

        split = split_name(
            episode_path,
            eval_ratio=args.eval_ratio,
            seed=args.seed,
            explicit_eval=None,
        )
        for start_idx in range(0, max_end - args.window_size + 2, args.stride):
            end_idx = start_idx + args.window_size - 1
            delta = float(scores[end_idx] - scores[start_idx])
            if abs(delta) < args.score_threshold:
                skipped["small_abs_delta"] += 1
                continue
            label = "positive" if delta > 0 else "negative"
            clip_rel_path = (
                Path(split)
                / "clips"
                / (
                    f"{label}_{episode_path.stem}"
                    f"_frames_{start_idx:04d}_{end_idx:04d}.mp4"
                )
            )
            sample = {
                "_episode_path": str(episode_path),
                "_start_idx": start_idx,
                "_end_idx": end_idx,
                "task": args.task_description,
                "prompt": prompt,
                "answer": label,
                "messages": build_messages(str(prompt), str(clip_rel_path), label),
                "clip_path": str(clip_rel_path),
                "source_episode_path": str(episode_path),
                "segment_metadata": {
                    "start_step": start_idx,
                    "end_step": end_idx,
                    "window_size": args.window_size,
                    "episode_id": episode.get("episode_id"),
                    "env_idx": episode.get("env_idx"),
                    "success": episode.get("success"),
                },
                "supervision": {
                    "label": label,
                    "score": delta,
                    "score_name": "value_delta_window",
                    "start_value": float(scores[start_idx]),
                    "end_value": float(scores[end_idx]),
                    "value_scores_jsonl": str(Path(args.value_scores_jsonl).resolve()),
                },
            }
            candidates_by_split[split].append(sample)
            if args.reverse_positive_as_negative and label == "positive":
                candidates_by_split[split].append(
                    build_reverse_positive_sample(sample, prompt)
                )

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

    materialize_clips(output_dir, final_samples_by_split, fps=args.fps)

    for split, samples in final_samples_by_split.items():
        write_manifest(output_dir / split / "segments.jsonl", samples)

    info = {
        "raw_episode_root": str(raw_root),
        "value_scores_jsonl": str(Path(args.value_scores_jsonl).resolve()),
        "output_dir": str(output_dir),
        "task_description": args.task_description,
        "window_size": args.window_size,
        "stride": args.stride,
        "score_threshold": args.score_threshold,
        "eval_ratio": args.eval_ratio,
        "seed": args.seed,
        "balance": args.balance,
        "max_per_label": args.max_per_label,
        "reverse_positive_as_negative": args.reverse_positive_as_negative,
        "num_episodes": len(episode_paths),
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

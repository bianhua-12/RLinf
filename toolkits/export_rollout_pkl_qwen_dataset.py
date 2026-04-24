#!/usr/bin/env python3
"""Export Qwen dual-view value-judge data from RLinf rollout dump pkl files."""

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
import json
import pickle
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

DEFAULT_TASK = "Pick up the red cube and place it on the green spot on the table."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--task-description", default=DEFAULT_TASK)
    parser.add_argument("--window-size", type=int, default=5)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--delta-threshold", type=float, default=0.2)
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=None,
        help="Optional positive threshold; defaults to delta-threshold.",
    )
    parser.add_argument(
        "--negative-threshold",
        type=float,
        default=None,
        help=(
            "Optional negative threshold magnitude; defaults to delta-threshold. "
            "A larger value widens the unchanged band on the negative side."
        ),
    )
    parser.add_argument(
        "--score-source",
        choices=("returns", "gae_target", "prev_values", "cumulative_reward"),
        default="returns",
        help="Signal used to label a window by end-start delta.",
    )
    parser.add_argument("--eval-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument(
        "--balance",
        choices=("none", "downsample", "posneg_downsample"),
        default="downsample",
    )
    parser.add_argument("--max-per-label", type=int, default=None)
    parser.add_argument(
        "--max-eval-per-label",
        type=int,
        default=None,
        help="Optional per-label cap for eval split; defaults to max-per-label.",
    )
    parser.add_argument("--reverse-positive-as-negative", action="store_true")
    parser.add_argument(
        "--max-dumps",
        type=int,
        default=None,
        help="Optional limit for debugging large rollout directories.",
    )
    parser.add_argument("--min-global-step", type=int, default=None)
    parser.add_argument("--max-global-step", type=int, default=None)
    parser.add_argument(
        "--global-step-stride",
        type=int,
        default=None,
        help="Only scan rollout files whose global_step is divisible by this value.",
    )
    parser.add_argument(
        "--early-stop-when-balanced",
        action="store_true",
        help="Stop scanning once train/eval all have max-per-label samples per class.",
    )
    parser.add_argument(
        "--split-on-done",
        action="store_true",
        help="Split rollout chunks by dones and never create windows across reset boundaries.",
    )
    parser.add_argument(
        "--output-format",
        choices=("mp4", "pkl"),
        default="mp4",
        help="Store each selected sample as encoded mp4 clips or one pkl with raw RGB frames.",
    )
    return parser.parse_args()


def to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)


def prompt_for_task(task: str, window_size: int) -> str:
    return (
        f"You are currently performing the task: {task}. "
        f"Please judge whether the operation shown in these two {window_size}-frame "
        "video views makes the task better, worse, or unchanged. "
        "Answer with exactly one word: positive, negative, or unchanged."
    )


def label_from_delta(
    delta: float, positive_threshold: float, negative_threshold: float
) -> str:
    if delta > positive_threshold:
        return "positive"
    if delta < -negative_threshold:
        return "negative"
    return "unchanged"


def write_clip(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (int(frames.shape[2]), int(frames.shape[1])),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame[..., :3], cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def build_messages(
    prompt: str,
    main_rel_path: str,
    third_rel_path: str,
    label: str,
    window_size: int,
) -> list[dict[str, Any]]:
    return [
        {
            "role": "user",
            "content": [
                {"type": "video", "video": main_rel_path, "nframes": window_size},
                {"type": "video", "video": third_rel_path, "nframes": window_size},
                {"type": "text", "text": prompt},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": label}]},
    ]


def split_name(key: str, eval_ratio: float, seed: int) -> str:
    rng = random.Random(f"{seed}:{key}")
    return "eval" if rng.random() < eval_ratio else "train"


def score_series(batch: dict[str, Any], source: str, env_idx: int) -> np.ndarray:
    if source in ("returns", "gae_target"):
        return to_numpy(batch["returns"][:, env_idx, 0]).astype(np.float32)
    if source == "prev_values":
        return to_numpy(batch["prev_values"][:-1, env_idx, 0]).astype(np.float32)
    rewards = to_numpy(batch["rewards"][:, env_idx, 0]).astype(np.float32)
    return np.cumsum(rewards)


def get_extra_view(extra: np.ndarray) -> np.ndarray:
    # Dumped ManiSkill data is usually [T, E, H, W, C]. Keep support for
    # [T, E, V, H, W, C] in case a future collector stores multiple extra views.
    if extra.ndim == 6:
        return extra[:, :, 0]
    return extra


def episode_segments_from_dones(
    dones: np.ndarray, num_steps: int, env_idx: int
) -> list[tuple[int, int]]:
    """Return inclusive [start, end] frame segments that do not cross resets."""
    env_dones = dones[:num_steps, env_idx, 0].astype(bool)
    boundaries = [int(x) for x in np.where(env_dones)[0] if 0 < int(x) < num_steps]
    segments: list[tuple[int, int]] = []
    start = 0
    for boundary in boundaries:
        if boundary > start:
            segments.append((start, boundary - 1))
        start = boundary
    if start < num_steps:
        segments.append((start, num_steps - 1))
    return segments


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
        if max_per_label is not None and "unchanged" in grouped:
            grouped["unchanged"] = grouped["unchanged"][:max_per_label]
    elif max_per_label is not None:
        grouped = {label: values[:max_per_label] for label, values in grouped.items()}

    merged: list[dict[str, Any]] = []
    for values in grouped.values():
        merged.extend(values)
    rng.shuffle(merged)
    return merged


def build_reverse_sample(sample: dict[str, Any]) -> dict[str, Any]:
    reversed_sample = dict(sample)
    reversed_sample["_reverse_frames"] = True
    reversed_sample["answer"] = "negative"
    main_path = Path(sample["main_clip_path"])
    third_path = Path(sample["third_clip_path"])
    reversed_main = main_path.with_name(f"negative_reverse_{main_path.name}")
    reversed_third = third_path.with_name(f"negative_reverse_{third_path.name}")
    reversed_sample["main_clip_path"] = str(reversed_main)
    reversed_sample["third_clip_path"] = str(reversed_third)
    reversed_sample["messages"] = build_messages(
        str(sample["prompt"]),
        str(reversed_main),
        str(reversed_third),
        "negative",
        int(sample["segment_metadata"]["window_size"]),
    )
    reversed_sample["segment_metadata"] = dict(sample["segment_metadata"])
    reversed_sample["segment_metadata"]["augmentation"] = "reverse_positive"
    reversed_sample["supervision"] = dict(sample["supervision"])
    reversed_sample["supervision"]["label"] = "negative"
    reversed_sample["supervision"]["source_label"] = "positive"
    reversed_sample["supervision"]["source_score"] = float(
        sample["supervision"]["score"]
    )
    reversed_sample["supervision"]["score"] = -abs(
        float(sample["supervision"]["score"])
    )
    reversed_sample["supervision"]["score_name"] = (
        f"reversed_positive_{sample['supervision']['score_name']}"
    )
    return reversed_sample


def write_manifest(path: Path, samples: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for sample in samples:
            clean = {k: v for k, v in sample.items() if not k.startswith("_")}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")


def materialize_clips(
    output_dir: Path, samples: list[dict[str, Any]], fps: int
) -> None:
    by_dump: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_dump[Path(sample["_dump_path"])].append(sample)

    written = 0
    for dump_idx, (dump_path, dump_samples) in enumerate(
        sorted(by_dump.items()), start=1
    ):
        with dump_path.open("rb") as f:
            payload = pickle.load(f)
        obs = payload["batch"]["curr_obs"]
        main = to_numpy(obs["main_images"])
        third = get_extra_view(to_numpy(obs["extra_view_images"]))
        for sample in dump_samples:
            env_idx = int(sample["_env_idx"])
            start = int(sample["_start_idx"])
            end = int(sample["_end_idx"])
            main_frames = main[start : end + 1, env_idx]
            third_frames = third[start : end + 1, env_idx]
            if sample.get("_reverse_frames"):
                main_frames = main_frames[::-1]
                third_frames = third_frames[::-1]
            write_clip(output_dir / sample["main_clip_path"], main_frames, fps)
            write_clip(output_dir / sample["third_clip_path"], third_frames, fps)
            written += 1
            if written % 1000 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "write_clips",
                            "written": written,
                            "dump_idx": dump_idx,
                            "num_dumps": len(by_dump),
                        }
                    ),
                    flush=True,
                )


def pkl_rel_path_for_sample(sample: dict[str, Any]) -> str:
    main_path = Path(sample["main_clip_path"])
    split = main_path.parts[0] if main_path.parts else "train"
    return str(Path(split) / "pkl_samples" / main_path.with_suffix(".pkl").name)


def materialize_pkl_samples(output_dir: Path, samples: list[dict[str, Any]]) -> None:
    by_dump: dict[Path, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        sample["pkl_path"] = pkl_rel_path_for_sample(sample)
        by_dump[Path(sample["_dump_path"])].append(sample)

    written = 0
    for dump_idx, (dump_path, dump_samples) in enumerate(
        sorted(by_dump.items()), start=1
    ):
        with dump_path.open("rb") as f:
            payload = pickle.load(f)
        obs = payload["batch"]["curr_obs"]
        main = to_numpy(obs["main_images"])
        third = get_extra_view(to_numpy(obs["extra_view_images"]))
        for sample in dump_samples:
            env_idx = int(sample["_env_idx"])
            start = int(sample["_start_idx"])
            end = int(sample["_end_idx"])
            main_frames = main[start : end + 1, env_idx]
            third_frames = third[start : end + 1, env_idx]
            if sample.get("_reverse_frames"):
                main_frames = main_frames[::-1]
                third_frames = third_frames[::-1]

            pkl_path = output_dir / sample["pkl_path"]
            pkl_path.parent.mkdir(parents=True, exist_ok=True)
            with pkl_path.open("wb") as f:
                pickle.dump(
                    {
                        "main_frames": np.asarray(main_frames, dtype=np.uint8),
                        "third_frames": np.asarray(third_frames, dtype=np.uint8),
                        "prompt": sample["prompt"],
                        "answer": sample["answer"],
                        "task": sample["task"],
                        "segment_metadata": sample["segment_metadata"],
                        "supervision": sample["supervision"],
                    },
                    f,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
            written += 1
            if written % 1000 == 0:
                print(
                    json.dumps(
                        {
                            "stage": "write_pkls",
                            "written": written,
                            "dump_idx": dump_idx,
                            "num_dumps": len(by_dump),
                        }
                    ),
                    flush=True,
                )


def collect_candidates(args: argparse.Namespace) -> dict[str, list[dict[str, Any]]]:
    rollout_dir = Path(args.rollout_dir).resolve()
    dump_paths = sorted(rollout_dir.glob("*.pkl"))
    step_pattern = re.compile(r"global_step_(\d+)_rank_\d+\.pkl")
    if args.min_global_step is not None or args.max_global_step is not None:
        filtered_paths: list[Path] = []
        for dump_path in dump_paths:
            match = step_pattern.match(dump_path.name)
            if match is None:
                filtered_paths.append(dump_path)
                continue
            file_global_step = int(match.group(1))
            if (
                args.min_global_step is not None
                and file_global_step < args.min_global_step
            ):
                continue
            if (
                args.max_global_step is not None
                and file_global_step > args.max_global_step
            ):
                continue
            if (
                args.global_step_stride is not None
                and file_global_step % int(args.global_step_stride) != 0
            ):
                continue
            filtered_paths.append(dump_path)
        dump_paths = filtered_paths
    if args.max_dumps is not None:
        dump_paths = dump_paths[: args.max_dumps]
    if not dump_paths:
        raise FileNotFoundError(f"No pkl files found under {rollout_dir}")

    prompt = prompt_for_task(args.task_description, args.window_size)
    positive_threshold = (
        float(args.positive_threshold)
        if args.positive_threshold is not None
        else float(args.delta_threshold)
    )
    negative_threshold = (
        float(args.negative_threshold)
        if args.negative_threshold is not None
        else float(args.delta_threshold)
    )
    rng = random.Random(args.seed)
    retained: dict[str, dict[str, list[dict[str, Any]]]] = {
        "train": defaultdict(list),
        "eval": defaultdict(list),
    }
    raw_counts: dict[str, Counter[str]] = {"train": Counter(), "eval": Counter()}
    required_labels = ("positive", "negative", "unchanged")
    split_caps = {
        "train": args.max_per_label,
        "eval": args.max_eval_per_label
        if args.max_eval_per_label is not None
        else args.max_per_label,
    }
    # Keep memory bounded when one class (usually unchanged) dominates. We still
    # count every raw label, but only retain enough candidates to materialize the
    # requested balanced dataset.
    store_caps = {
        split: (
            None
            if split_caps[split] is None
            else max(int(split_caps[split]) * 2, int(split_caps[split]) + 1024)
        )
        for split in ("train", "eval")
    }
    stored_counts: dict[str, Counter[str]] = {"train": Counter(), "eval": Counter()}

    def retain_sample(split: str, label: str, sample: dict[str, Any]) -> None:
        cap = store_caps[split]
        if cap is None:
            retained[split][label].append(sample)
            stored_counts[split][label] += 1
            return
        seen = raw_counts[split][label]
        label_samples = retained[split][label]
        if len(label_samples) < cap:
            label_samples.append(sample)
            stored_counts[split][label] = len(label_samples)
            return
        # Reservoir sampling keeps retained samples distributed across the
        # whole 0..N scan instead of over-representing early random rollouts.
        replace_idx = rng.randrange(seen)
        if replace_idx < cap:
            label_samples[replace_idx] = sample

    for dump_i, dump_path in enumerate(dump_paths, start=1):
        with dump_path.open("rb") as f:
            payload = pickle.load(f)
        batch = payload["batch"]
        main = to_numpy(batch["curr_obs"]["main_images"])
        third = get_extra_view(to_numpy(batch["curr_obs"]["extra_view_images"]))
        if main.shape[:2] != third.shape[:2]:
            raise ValueError(
                f"View shape mismatch in {dump_path}: {main.shape} vs {third.shape}"
            )
        num_steps, num_envs = int(main.shape[0]), int(main.shape[1])
        rank = int(payload.get("rank", -1))
        global_step = int(payload.get("global_step", -1))
        if args.min_global_step is not None and global_step < args.min_global_step:
            continue
        if args.max_global_step is not None and global_step > args.max_global_step:
            continue

        for env_idx in range(num_envs):
            scores = score_series(batch, args.score_source, env_idx)
            if args.split_on_done:
                dones = to_numpy(batch["dones"])
                segments = episode_segments_from_dones(dones, num_steps, env_idx)
            else:
                segments = [(0, num_steps - 1)]
            for episode_segment_idx, (segment_start, segment_end) in enumerate(
                segments
            ):
                if segment_end - segment_start + 1 < args.window_size:
                    continue
                max_start = segment_end - args.window_size + 1
                for start in range(segment_start, max_start + 1, args.stride):
                    end = start + args.window_size - 1
                    delta = float(scores[end] - scores[start])
                    label = label_from_delta(
                        delta, positive_threshold, negative_threshold
                    )
                    key = f"{dump_path.name}:env{env_idx}:s{start}:e{end}"
                    split = split_name(key, args.eval_ratio, args.seed)
                    base = (
                        f"{label}_g{global_step:06d}_r{rank:02d}_env{env_idx:03d}_"
                        f"frames_{start:04d}_{end:04d}_win_{args.window_size:02d}.mp4"
                    )
                    main_rel = Path(split) / "main_clips" / base
                    third_rel = Path(split) / "third_clips" / base
                    sample = {
                        "_dump_path": str(dump_path),
                        "_env_idx": env_idx,
                        "_start_idx": start,
                        "_end_idx": end,
                        "task": args.task_description,
                        "prompt": prompt,
                        "answer": label,
                        "messages": build_messages(
                            prompt,
                            str(main_rel),
                            str(third_rel),
                            label,
                            args.window_size,
                        ),
                        "main_clip_path": str(main_rel),
                        "third_clip_path": str(third_rel),
                        "source_rollout_path": str(dump_path),
                        "segment_metadata": {
                            "global_step": global_step,
                            "rank": rank,
                            "env_idx": env_idx,
                            "start_step": start,
                            "end_step": end,
                            "window_size": args.window_size,
                            "episode_segment_idx": episode_segment_idx,
                            "episode_segment_start": segment_start,
                            "episode_segment_end": segment_end,
                            "split_on_done": bool(args.split_on_done),
                            "views": ["main_images", "extra_view_images"],
                        },
                        "supervision": {
                            "label": label,
                            "score": delta,
                            "score_name": f"{args.score_source}_delta_window",
                            "score_source": args.score_source,
                            "delta_threshold": args.delta_threshold,
                            "positive_threshold": positive_threshold,
                            "negative_threshold": negative_threshold,
                            "start_score": float(scores[start]),
                            "end_score": float(scores[end]),
                        },
                    }
                    raw_counts[split][label] += 1
                    retain_sample(split, label, sample)
                    if args.reverse_positive_as_negative and label == "positive":
                        reversed_sample = build_reverse_sample(sample)
                        raw_counts[split]["negative"] += 1
                        retain_sample(split, "negative", reversed_sample)

        print(
            json.dumps(
                {
                    "stage": "scan",
                    "dump_idx": dump_i,
                    "num_dumps": len(dump_paths),
                    "dump": str(dump_path),
                    "raw_counts": {k: dict(v) for k, v in raw_counts.items()},
                    "stored_counts": {k: dict(v) for k, v in stored_counts.items()},
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if args.early_stop_when_balanced and args.max_per_label is not None:
            if all(
                split_caps[split] is not None
                and raw_counts[split][label] >= split_caps[split]
                for split in ("train", "eval")
                for label in required_labels
            ):
                print(
                    json.dumps(
                        {
                            "stage": "early_stop",
                            "dump_idx": dump_i,
                            "num_dumps": len(dump_paths),
                            "raw_counts": {k: dict(v) for k, v in raw_counts.items()},
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
                break

    return {
        split: [
            sample
            for label_samples in retained[split].values()
            for sample in label_samples
        ]
        for split in ("train", "eval")
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = collect_candidates(args)
    raw_counts = {
        split: dict(Counter(sample["answer"] for sample in samples))
        for split, samples in candidates.items()
    }
    final_samples = {
        split: maybe_balance(
            samples,
            balance=args.balance,
            max_per_label=(
                args.max_eval_per_label
                if split == "eval" and args.max_eval_per_label is not None
                else args.max_per_label
            ),
            seed=args.seed + (0 if split == "train" else 1),
        )
        for split, samples in candidates.items()
    }
    final_counts = {
        split: dict(Counter(sample["answer"] for sample in samples))
        for split, samples in final_samples.items()
    }
    print(
        json.dumps(
            {
                "stage": "materialize",
                "raw_counts": raw_counts,
                "final_counts": final_counts,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    for split, samples in final_samples.items():
        if args.output_format == "pkl":
            materialize_pkl_samples(output_dir, samples)
        else:
            materialize_clips(output_dir, samples, args.fps)
        write_manifest(output_dir / split / "segments.jsonl", samples)

    info = {
        "rollout_dir": str(Path(args.rollout_dir).resolve()),
        "output_dir": str(output_dir),
        "task_description": args.task_description,
        "window_size": args.window_size,
        "stride": args.stride,
        "delta_threshold": args.delta_threshold,
        "positive_threshold": (
            args.positive_threshold
            if args.positive_threshold is not None
            else args.delta_threshold
        ),
        "negative_threshold": (
            args.negative_threshold
            if args.negative_threshold is not None
            else args.delta_threshold
        ),
        "score_source": args.score_source,
        "eval_ratio": args.eval_ratio,
        "seed": args.seed,
        "balance": args.balance,
        "max_per_label": args.max_per_label,
        "max_eval_per_label": args.max_eval_per_label,
        "global_step_stride": args.global_step_stride,
        "reverse_positive_as_negative": args.reverse_positive_as_negative,
        "output_format": args.output_format,
        "raw_counts": raw_counts,
        "final_counts": final_counts,
        "train_manifest": str(output_dir / "train" / "segments.jsonl"),
        "eval_manifest": str(output_dir / "eval" / "segments.jsonl"),
    }
    with (output_dir / "dataset_info.json").open("w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

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

"""Build a dual-view GAE-delta ternary dataset by reusing existing clip videos."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset-dir", required=True)
    parser.add_argument("--traj-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--label-mode",
        choices=("threshold", "global_tercile"),
        default="threshold",
    )
    parser.add_argument("--delta-threshold", type=float, default=0.2)
    parser.add_argument("--primary-view-index", type=int, default=0)
    parser.add_argument(
        "--secondary-view-indices",
        default="1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ternary_label(delta: float, threshold: float) -> str:
    if delta > threshold:
        return "positive"
    if delta < -threshold:
        return "negative"
    return "unchanged"


def tercile_label(
    delta: float,
    *,
    lower_cutoff: float,
    upper_cutoff: float,
) -> str:
    if delta >= upper_cutoff:
        return "positive"
    if delta <= lower_cutoff:
        return "negative"
    return "unchanged"


def absolutize_clip_path(source_dataset_dir: Path, clip_path: str) -> str:
    clip = Path(clip_path)
    if clip.is_absolute():
        return str(clip)
    return os.path.abspath(str(source_dataset_dir / clip))


def load_episode_gae(group: h5py.File, episode_id: int) -> list[float]:
    dataset = group[f"traj_{episode_id}"]["offline_value_labels"]["pre_action"][
        "ppo_gae_target"
    ]
    return [float(value) for value in dataset[:]]


def parse_secondary_indices(raw: str, primary_view_index: int) -> list[int]:
    values = [int(token) for token in raw.split(",") if token.strip()]
    if not values:
        raise ValueError("secondary-view-indices must not be empty")
    if primary_view_index in values:
        raise ValueError("secondary-view-indices must not include primary-view-index")
    if len(values) != len(set(values)):
        raise ValueError("secondary-view-indices contains duplicates")
    return sorted(values)


def group_rows_by_window(
    rows: list[dict[str, Any]],
) -> dict[tuple[int, int, int], dict[int, dict[str, Any]]]:
    grouped: dict[tuple[int, int, int], dict[int, dict[str, Any]]] = {}
    for row in rows:
        meta = row["segment_metadata"]
        key = (int(meta["episode_id"]), int(meta["start_step"]), int(meta["end_step"]))
        view_index = int(meta["view_index"])
        grouped.setdefault(key, {})
        if view_index in grouped[key]:
            raise ValueError(f"Duplicate view {view_index} for window {key}")
        grouped[key][view_index] = row
    return grouped


def compute_window_delta(
    *,
    key: tuple[int, int, int],
    gae_values: list[float],
) -> dict[str, float]:
    _, start_step, end_step = key
    start_gae = float(gae_values[start_step])
    end_gae = float(gae_values[end_step])
    gae_delta = end_gae - start_gae
    return {
        "start_gae": start_gae,
        "end_gae": end_gae,
        "gae_delta": gae_delta,
    }


def compute_global_tercile_cutoffs(
    delta_by_key: dict[tuple[int, int, int], float],
) -> tuple[float, float]:
    if not delta_by_key:
        raise ValueError("No unique windows were found for quantile labeling")
    deltas = np.asarray(list(delta_by_key.values()), dtype=np.float64)
    lower_cutoff = float(np.quantile(deltas, 1.0 / 3.0))
    upper_cutoff = float(np.quantile(deltas, 2.0 / 3.0))
    return lower_cutoff, upper_cutoff


def build_dualview_prompt(task: str, window_size: int) -> str:
    return (
        f"You are currently performing the task: {task}. "
        f"Please judge whether the operation shown in these two {window_size}-frame "
        "videos, which capture the same time window from two different views, makes "
        "the task better, worse, or unchanged. Answer with exactly one word: "
        "positive, negative, or unchanged."
    )


def build_dualview_record(
    *,
    primary_row: dict[str, Any],
    secondary_row: dict[str, Any],
    source_dataset_dir: Path,
    start_gae: float,
    end_gae: float,
    gae_delta: float,
    label: str,
    score_metadata: dict[str, Any],
) -> dict[str, Any]:
    primary_meta = primary_row["segment_metadata"]
    secondary_meta = secondary_row["segment_metadata"]
    episode_id = int(primary_meta["episode_id"])
    start_step = int(primary_meta["start_step"])
    end_step = int(primary_meta["end_step"])
    window_size = int(primary_meta["window_size"])
    success = bool(primary_meta["success"])

    primary_clip_path = absolutize_clip_path(
        source_dataset_dir, primary_row["clip_path"]
    )
    secondary_clip_path = absolutize_clip_path(
        source_dataset_dir, secondary_row["clip_path"]
    )
    task = str(primary_row["task"])
    prompt = build_dualview_prompt(task, window_size)
    base_supervision = primary_row.get("supervision", {})

    return {
        "task": task,
        "prompt": prompt,
        "answer": label,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": primary_clip_path},
                    {"type": "video", "video": secondary_clip_path},
                    {"type": "text", "text": prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": label}],
            },
        ],
        "clip_path": primary_clip_path,
        "secondary_clip_path": secondary_clip_path,
        "clip_paths": [primary_clip_path, secondary_clip_path],
        "source_traj_id": primary_row["source_traj_id"],
        "segment_metadata": {
            "episode_id": episode_id,
            "start_step": start_step,
            "end_step": end_step,
            "window_size": window_size,
            "success": success,
            "primary_view_index": int(primary_meta["view_index"]),
            "primary_azimuth_deg": float(primary_meta["azimuth_deg"]),
            "secondary_view_index": int(secondary_meta["view_index"]),
            "secondary_azimuth_deg": float(secondary_meta["azimuth_deg"]),
        },
        "supervision": {
            "label": label,
            "score": gae_delta,
            "score_name": "gae_delta_window",
            "score_source": "ppo_gae_target",
            "start_gae": start_gae,
            "end_gae": end_gae,
            "start_value": start_gae,
            "end_value": end_gae,
            "alignment": base_supervision.get("alignment", "pre_action"),
            "checkpoint_path": base_supervision.get("checkpoint_path"),
            **score_metadata,
        },
    }


def summarize_labels(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(row["answer"] for row in rows)
    return {label: int(counter[label]) for label in sorted(counter)}


def main() -> None:
    args = parse_args()
    source_dataset_dir = Path(os.path.abspath(args.source_dataset_dir))
    traj_path = Path(os.path.abspath(args.traj_path))
    output_dir = Path(os.path.abspath(args.output_dir))
    primary_view_index = int(args.primary_view_index)
    secondary_view_indices = parse_secondary_indices(
        args.secondary_view_indices, primary_view_index
    )

    split_rows: dict[str, list[dict[str, Any]]] = {}
    split_counts: dict[str, int] = {}
    label_counts: dict[str, dict[str, int]] = {}
    unique_window_counts: dict[str, int] = {}
    unique_window_label_counts: dict[str, dict[str, int]] = {}
    grouped_rows_by_split: dict[
        str, dict[tuple[int, int, int], dict[int, dict[str, Any]]]
    ] = {}
    delta_stats_by_key: dict[tuple[int, int, int], dict[str, float]] = {}
    window_label_by_key: dict[tuple[int, int, int], str] = {}
    quantile_info: dict[str, Any] = {}

    with h5py.File(traj_path, "r") as traj_file:
        gae_cache: dict[int, list[float]] = {}
        for split in ("train", "eval"):
            input_manifest = source_dataset_dir / split / "segments.jsonl"
            rows = load_jsonl(input_manifest)
            grouped_rows = group_rows_by_window(rows)
            grouped_rows_by_split[split] = grouped_rows
            unique_window_counts[split] = len(grouped_rows)

            for key in sorted(grouped_rows):
                episode_id, _, _ = key
                if episode_id not in gae_cache:
                    gae_cache[episode_id] = load_episode_gae(traj_file, episode_id)
                if key not in delta_stats_by_key:
                    delta_stats_by_key[key] = compute_window_delta(
                        key=key,
                        gae_values=gae_cache[episode_id],
                    )

        if args.label_mode == "global_tercile":
            lower_cutoff, upper_cutoff = compute_global_tercile_cutoffs(
                {key: stats["gae_delta"] for key, stats in delta_stats_by_key.items()}
            )
            quantile_info = {
                "label_mode": "global_tercile_quantile",
                "quantile_population": "unique_windows",
                "lower_quantile": 1.0 / 3.0,
                "upper_quantile": 2.0 / 3.0,
                "lower_cutoff": lower_cutoff,
                "upper_cutoff": upper_cutoff,
            }
            for key, stats in delta_stats_by_key.items():
                window_label_by_key[key] = tercile_label(
                    stats["gae_delta"],
                    lower_cutoff=lower_cutoff,
                    upper_cutoff=upper_cutoff,
                )
        else:
            quantile_info = {
                "label_mode": "threshold",
                "delta_threshold": float(args.delta_threshold),
            }
            for key, stats in delta_stats_by_key.items():
                window_label_by_key[key] = ternary_label(
                    stats["gae_delta"], args.delta_threshold
                )

        for split in ("train", "eval"):
            grouped_rows = grouped_rows_by_split[split]
            converted: list[dict[str, Any]] = []
            window_labels: list[str] = []
            for key in sorted(grouped_rows):
                views = grouped_rows[key]
                if primary_view_index not in views:
                    raise ValueError(
                        f"Missing primary view {primary_view_index} for {key}"
                    )
                for secondary_view_index in secondary_view_indices:
                    if secondary_view_index not in views:
                        raise ValueError(
                            f"Missing secondary view {secondary_view_index} for {key}"
                        )

                score_stats = delta_stats_by_key[key]
                label = window_label_by_key[key]
                score_metadata = (
                    {"delta_threshold": float(args.delta_threshold)}
                    if args.label_mode == "threshold"
                    else {
                        "label_mode": "global_tercile_quantile",
                        "quantile_population": "unique_windows",
                        "lower_cutoff": quantile_info["lower_cutoff"],
                        "upper_cutoff": quantile_info["upper_cutoff"],
                    }
                )
                primary_row = views[primary_view_index]
                for secondary_view_index in secondary_view_indices:
                    converted.append(
                        build_dualview_record(
                            primary_row=primary_row,
                            secondary_row=views[secondary_view_index],
                            source_dataset_dir=source_dataset_dir,
                            start_gae=score_stats["start_gae"],
                            end_gae=score_stats["end_gae"],
                            gae_delta=score_stats["gae_delta"],
                            label=label,
                            score_metadata=score_metadata,
                        )
                    )
                window_labels.append(label)

            split_rows[split] = converted
            split_counts[split] = len(converted)
            label_counts[split] = summarize_labels(converted)
            unique_window_label_counts[split] = {
                label: int(count)
                for label, count in sorted(Counter(window_labels).items())
            }

    for split, rows in split_rows.items():
        write_jsonl(output_dir / split / "segments.jsonl", rows)

    info = {
        "source_dataset_dir": str(source_dataset_dir),
        "source_traj_h5": str(traj_path),
        "output_dir": str(output_dir),
        "target_field": "ppo_gae_target",
        "score_name": "gae_delta_window",
        "video_mode": "reuse_absolute_paths_dualview",
        "primary_view_index": primary_view_index,
        "secondary_view_indices": secondary_view_indices,
        "unique_window_counts": unique_window_counts,
        "unique_window_label_counts": unique_window_label_counts,
        "split_counts": split_counts,
        "label_counts": label_counts,
        **quantile_info,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

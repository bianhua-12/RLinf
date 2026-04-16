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

"""Relabel a clip-level ternary dataset with GAE deltas from a ManiSkill H5."""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import h5py


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dataset-dir", required=True)
    parser.add_argument("--traj-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--delta-threshold", type=float, default=0.2)
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


def load_episode_gae(group: h5py.File, episode_id: int) -> list[float]:
    dataset = group[f"traj_{episode_id}"]["offline_value_labels"]["pre_action"][
        "ppo_gae_target"
    ]
    return [float(value) for value in dataset[:]]


def absolutize_clip_path(source_dataset_dir: Path, clip_path: str) -> str:
    clip = Path(clip_path)
    if clip.is_absolute():
        return str(clip)
    return os.path.abspath(str(source_dataset_dir / clip))


def relabel_sample(
    sample: dict[str, Any],
    *,
    source_dataset_dir: Path,
    gae_values: list[float],
    delta_threshold: float,
) -> dict[str, Any]:
    relabeled = json.loads(json.dumps(sample))
    segment_metadata = relabeled["segment_metadata"]
    supervision = relabeled["supervision"]

    start_step = int(segment_metadata["start_step"])
    end_step = int(segment_metadata["end_step"])
    start_gae = float(gae_values[start_step])
    end_gae = float(gae_values[end_step])
    gae_delta = end_gae - start_gae
    label = ternary_label(gae_delta, delta_threshold)

    clip_path = absolutize_clip_path(source_dataset_dir, relabeled["clip_path"])
    relabeled["clip_path"] = clip_path
    relabeled["answer"] = label

    for message in relabeled.get("messages", []):
        if message.get("role") == "assistant":
            content = message.get("content", [])
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    item["text"] = label
        content = message.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "video":
                    item["video"] = clip_path

    supervision["label"] = label
    supervision["score"] = gae_delta
    supervision["score_name"] = "gae_delta_window"
    supervision["score_source"] = "ppo_gae_target"
    supervision["delta_threshold"] = float(delta_threshold)
    supervision["start_gae"] = start_gae
    supervision["end_gae"] = end_gae
    supervision["start_value"] = start_gae
    supervision["end_value"] = end_gae
    return relabeled


def summarize_labels(rows: list[dict[str, Any]]) -> dict[str, int]:
    counter = Counter(row["answer"] for row in rows)
    return {label: int(counter[label]) for label in sorted(counter)}


def main() -> None:
    args = parse_args()
    source_dataset_dir = Path(os.path.abspath(args.source_dataset_dir))
    traj_path = Path(os.path.abspath(args.traj_path))
    output_dir = Path(os.path.abspath(args.output_dir))

    split_rows: dict[str, list[dict[str, Any]]] = {}
    counts: dict[str, int] = {}
    label_counts: dict[str, dict[str, int]] = {}

    with h5py.File(traj_path, "r") as traj_file:
        gae_cache: dict[int, list[float]] = {}
        for split in ("train", "eval"):
            input_manifest = source_dataset_dir / split / "segments.jsonl"
            rows = load_jsonl(input_manifest)
            converted: list[dict[str, Any]] = []
            for row in rows:
                episode_id = int(row["segment_metadata"]["episode_id"])
                if episode_id not in gae_cache:
                    gae_cache[episode_id] = load_episode_gae(traj_file, episode_id)
                converted.append(
                    relabel_sample(
                        row,
                        source_dataset_dir=source_dataset_dir,
                        gae_values=gae_cache[episode_id],
                        delta_threshold=args.delta_threshold,
                    )
                )
            split_rows[split] = converted
            counts[split] = len(converted)
            label_counts[split] = summarize_labels(converted)

    for split, rows in split_rows.items():
        write_jsonl(output_dir / split / "segments.jsonl", rows)

    info = {
        "source_dataset_dir": str(source_dataset_dir),
        "source_traj_h5": str(traj_path),
        "output_dir": str(output_dir),
        "target_field": "ppo_gae_target",
        "score_name": "gae_delta_window",
        "delta_threshold": float(args.delta_threshold),
        "split_counts": counts,
        "label_counts": label_counts,
        "video_mode": "reuse_absolute_paths",
    }
    (output_dir / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

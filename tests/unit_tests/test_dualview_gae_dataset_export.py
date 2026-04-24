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

import importlib.util
import json
import sys
from pathlib import Path

import h5py


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "toolkits"
    / "export_dualview_gae_delta_judge_dataset.py"
)


def _load_script_module():
    spec = importlib.util.spec_from_file_location(
        "export_dualview_gae_delta_judge_dataset",
        SCRIPT_PATH,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _build_view_row(*, episode_id: int, view_index: int, clip_path: str) -> dict:
    return {
        "task": "Pick up the red cube and place it on the green spot on the table.",
        "prompt": "dummy prompt",
        "answer": "unchanged",
        "messages": [],
        "clip_path": clip_path,
        "source_traj_id": f"traj_{episode_id}",
        "segment_metadata": {
            "episode_id": episode_id,
            "start_step": 0,
            "end_step": 1,
            "window_size": 2,
            "success": False,
            "view_index": view_index,
            "azimuth_deg": float(view_index * 10),
        },
        "supervision": {
            "alignment": "pre_action",
            "checkpoint_path": "/tmp/value_head.pt",
        },
    }


def test_dualview_export_supports_global_terciles(tmp_path: Path, monkeypatch) -> None:
    module = _load_script_module()
    source_dataset_dir = tmp_path / "source_dataset"
    output_dir = tmp_path / "output_dataset"
    traj_path = tmp_path / "trajectory.h5"

    train_rows = [
        _build_view_row(episode_id=0, view_index=0, clip_path="train/0_view0.mp4"),
        _build_view_row(episode_id=0, view_index=1, clip_path="train/0_view1.mp4"),
        _build_view_row(episode_id=1, view_index=0, clip_path="train/1_view0.mp4"),
        _build_view_row(episode_id=1, view_index=1, clip_path="train/1_view1.mp4"),
    ]
    eval_rows = [
        _build_view_row(episode_id=2, view_index=0, clip_path="eval/2_view0.mp4"),
        _build_view_row(episode_id=2, view_index=1, clip_path="eval/2_view1.mp4"),
    ]
    _write_jsonl(source_dataset_dir / "train" / "segments.jsonl", train_rows)
    _write_jsonl(source_dataset_dir / "eval" / "segments.jsonl", eval_rows)

    with h5py.File(traj_path, "w") as h5_file:
        for episode_id, values in {
            0: [0.0, 3.0],
            1: [0.0, 1.0],
            2: [0.0, -1.0],
        }.items():
            group = h5_file.create_group(f"traj_{episode_id}")
            pre_action = group.create_group("offline_value_labels").create_group(
                "pre_action"
            )
            pre_action.create_dataset("ppo_gae_target", data=values)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "export_dualview_gae_delta_judge_dataset.py",
            "--source-dataset-dir",
            str(source_dataset_dir),
            "--traj-path",
            str(traj_path),
            "--output-dir",
            str(output_dir),
            "--label-mode",
            "global_tercile",
            "--primary-view-index",
            "0",
            "--secondary-view-indices",
            "1",
        ],
    )
    module.main()

    train_output = [
        json.loads(line)
        for line in (output_dir / "train" / "segments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    eval_output = [
        json.loads(line)
        for line in (output_dir / "eval" / "segments.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    labels_by_episode = {
        row["segment_metadata"]["episode_id"]: row["answer"]
        for row in train_output + eval_output
    }

    assert labels_by_episode == {
        0: "positive",
        1: "unchanged",
        2: "negative",
    }

    info = json.loads((output_dir / "dataset_info.json").read_text(encoding="utf-8"))
    assert info["label_mode"] == "global_tercile_quantile"
    assert info["quantile_population"] == "unique_windows"
    assert info["unique_window_counts"] == {"train": 2, "eval": 1}
    assert info["unique_window_label_counts"] == {
        "train": {"positive": 1, "unchanged": 1},
        "eval": {"negative": 1},
    }

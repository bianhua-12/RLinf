#!/usr/bin/env python3
"""Export rollout dump env videos with reward/return/GAE curves."""

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
import pickle
from pathlib import Path

import cv2
import numpy as np
import torch


def _to_numpy(x) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def load_env_series(dump_path: Path, env_idx: int) -> dict[str, np.ndarray]:
    with dump_path.open("rb") as f:
        payload = pickle.load(f)
    batch = payload["batch"]
    curr_obs = batch["curr_obs"]
    out = {
        "main": _to_numpy(curr_obs["main_images"][:, env_idx]),
        "extra": _to_numpy(curr_obs["extra_view_images"][:, env_idx]),
        "reward": _to_numpy(batch["rewards"][:, env_idx, 0]).astype(np.float32),
        "gae_target": _to_numpy(batch["returns"][:, env_idx, 0]).astype(np.float32),
        "value": _to_numpy(batch["prev_values"][:-1, env_idx, 0]).astype(np.float32),
        "done": _to_numpy(batch["dones"][1:, env_idx, 0]).astype(bool),
    }
    out["return"] = np.cumsum(out["reward"])
    return out


def draw_curve_panel(
    series: dict[str, np.ndarray], t: int, width: int, height: int
) -> np.ndarray:
    panel = np.full((height, width, 3), 255, dtype=np.uint8)
    margin_l, margin_r, margin_t, margin_b = 55, 20, 28, 42
    x0, y0 = margin_l, margin_t
    x1, y1 = width - margin_r, height - margin_b

    keys = [
        ("reward", (40, 160, 40), "reward"),
        ("return", (220, 120, 30), "return"),
        ("gae_target", (40, 80, 220), "GAE target"),
        ("value", (160, 60, 160), "value"),
    ]
    values = np.concatenate([series[k] for k, _, _ in keys])
    y_min = float(np.min(values))
    y_max = float(np.max(values))
    if abs(y_max - y_min) < 1e-6:
        y_max = y_min + 1.0
    n = len(series["reward"])

    cv2.rectangle(panel, (x0, y0), (x1, y1), (220, 220, 220), 1)
    cv2.putText(
        panel,
        f"step={t} reward={series['reward'][t]:.3f} return={series['return'][t]:.3f} "
        f"gae={series['gae_target'][t]:.3f} value={series['value'][t]:.3f}",
        (12, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )

    def xy(idx: int, val: float) -> tuple[int, int]:
        x = int(x0 + (x1 - x0) * idx / max(n - 1, 1))
        y = int(y1 - (y1 - y0) * (float(val) - y_min) / (y_max - y_min))
        return x, y

    for key, color, label in keys:
        pts = np.array([xy(i, v) for i, v in enumerate(series[key])], dtype=np.int32)
        cv2.polylines(panel, [pts], False, color, 2, cv2.LINE_AA)
        lx = x0 + 8 + 135 * keys.index((key, color, label))
        cv2.line(panel, (lx, height - 18), (lx + 24, height - 18), color, 2)
        cv2.putText(
            panel,
            label,
            (lx + 30, height - 13),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (40, 40, 40),
            1,
            cv2.LINE_AA,
        )

    cx, _ = xy(t, y_min)
    cv2.line(panel, (cx, y0), (cx, y1), (0, 0, 0), 1)
    for done_idx in np.where(series["done"])[0]:
        dx, _ = xy(int(done_idx), y_min)
        cv2.line(panel, (dx, y0), (dx, y1), (120, 120, 120), 1)
    cv2.putText(
        panel,
        f"{y_max:.2f}",
        (5, y0 + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        (80, 80, 80),
        1,
    )
    cv2.putText(
        panel, f"{y_min:.2f}", (5, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (80, 80, 80), 1
    )
    return panel


def export_video(dump_path: Path, env_idx: int, out_path: Path, fps: int) -> None:
    series = load_env_series(dump_path, env_idx)
    main = series["main"]
    extra = series["extra"]
    n, h, w, _ = main.shape
    curve_h = 260
    frame_w = w * 2
    frame_h = h + curve_h
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (frame_w, frame_h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer: {out_path}")

    for t in range(n):
        top = np.concatenate([main[t], extra[t]], axis=1)
        top = np.ascontiguousarray(top)
        cv2.putText(
            top,
            "main view",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            top,
            "extra/third view",
            (w + 10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        panel = draw_curve_panel(series, t, frame_w, curve_h)
        frame = np.concatenate([top, panel], axis=0)
        writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path, required=True)
    parser.add_argument("--env-idx", type=int, action="append", required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    out_dir = args.out_dir or args.dump.parent.parent / "analysis" / "rollout_videos"
    for env_idx in args.env_idx:
        out_path = out_dir / f"{args.dump.stem}_env_{env_idx:03d}.mp4"
        export_video(args.dump, env_idx, out_path, args.fps)
        print(out_path)


if __name__ == "__main__":
    main()

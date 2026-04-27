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
import json
from pathlib import Path
from statistics import mean, median

PROFILE_PREFIXES = ("time/", "reward/", "rollout/", "train/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize async PPO profiling scalars from TensorBoard events."
    )
    parser.add_argument(
        "log_path",
        type=Path,
        help="Run log path or tensorboard directory containing event files.",
    )
    parser.add_argument("--warmup-steps", type=int, default=2)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Output JSON path. Defaults to <log_path>/async_ppo_profile_summary.json.",
    )
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=None,
        help="Output CSV path. Defaults to <json-output>.csv.",
    )
    return parser.parse_args()


def find_event_dirs(log_path: Path) -> list[Path]:
    candidates = []
    if any(log_path.glob("events.out.tfevents.*")):
        candidates.append(log_path)
    tensorboard_path = log_path / "tensorboard"
    if tensorboard_path.exists():
        candidates.extend(
            path
            for path in [tensorboard_path, *tensorboard_path.rglob("*")]
            if path.is_dir() and any(path.glob("events.out.tfevents.*"))
        )
    return sorted(set(candidates))


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def load_scalars(event_dirs: list[Path]) -> dict[str, list[tuple[int, float]]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError as exc:
        raise SystemExit(
            "tensorboard is required to summarize profile results. "
            "Install tensorboard or run in the RLinf training environment."
        ) from exc

    scalars: dict[str, list[tuple[int, float]]] = {}
    for event_dir in event_dirs:
        accumulator = EventAccumulator(str(event_dir), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags().get("scalars", []):
            if not tag.startswith(PROFILE_PREFIXES):
                continue
            for event in accumulator.Scalars(tag):
                scalars.setdefault(tag, []).append(
                    (int(event.step), float(event.value))
                )
    return scalars


def summarize(
    scalars: dict[str, list[tuple[int, float]]],
    warmup_steps: int,
    measure_steps: int,
) -> dict[str, dict[str, float]]:
    summary = {}
    for tag, points in sorted(scalars.items()):
        if not points:
            continue
        ordered_points = sorted(points)
        first_step = ordered_points[0][0]
        min_step = first_step + warmup_steps
        max_step = min_step + measure_steps
        values = [
            value for step, value in ordered_points if min_step <= step < max_step
        ]
        if not values:
            continue
        summary[tag] = {
            "count": float(len(values)),
            "mean": float(mean(values)),
            "p50": float(median(values)),
            "p95": float(percentile(values, 0.95)),
            "max": float(max(values)),
            "min_step": float(min_step),
            "max_step_exclusive": float(max_step),
        }
    return summary


def write_csv(summary: dict[str, dict[str, float]], csv_output: Path) -> None:
    csv_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "metric",
        "count",
        "mean",
        "p50",
        "p95",
        "max",
        "min_step",
        "max_step_exclusive",
    ]
    with csv_output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for metric, stats in summary.items():
            writer.writerow({"metric": metric, **stats})


def main() -> None:
    args = parse_args()
    event_dirs = find_event_dirs(args.log_path)
    if not event_dirs:
        raise SystemExit(f"No TensorBoard event files found under {args.log_path}")

    scalars = load_scalars(event_dirs)
    summary = summarize(
        scalars,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
    )

    json_output = args.json_output or (args.log_path / "async_ppo_profile_summary.json")
    csv_output = args.csv_output or json_output.with_suffix(".csv")

    json_output.parent.mkdir(parents=True, exist_ok=True)
    json_output.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_csv(summary, csv_output)

    print(f"Wrote {json_output}")
    print(f"Wrote {csv_output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot env return, env reward, and PPO GAE target from RLinf logs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


def _between(text: str, start: str, end: str | None = None) -> str:
    start_idx = text.find(start)
    if start_idx < 0:
        return ""
    section = text[start_idx + len(start) :]
    if end is not None:
        end_idx = section.find(end)
        if end_idx >= 0:
            section = section[:end_idx]
    return section


def _metric(section: str, key: str) -> float | None:
    # Avoid matching "reward" inside "reward_model_output".
    pattern = rf"(?<![A-Za-z0-9_/]){re.escape(key)}=([-+]?(?:\d+\.\d+|\d+|\.\d+)(?:e[-+]?\d+)?)"
    matches = re.findall(pattern, section)
    if not matches:
        return None
    return float(matches[-1])


def parse_log(log_path: Path) -> list[dict[str, float | int]]:
    text = log_path.read_text(errors="ignore")
    rows: list[dict[str, float | int]] = []
    for block in text.split("Metric Table"):
        step_match = re.search(r"Global Step:\s+(\d+)/", block)
        if step_match is None:
            continue

        env_section = _between(block, "Environment", "Rollout")
        rollout_section = _between(block, "Rollout", "Training/Actor")
        row = {
            "step": int(step_match.group(1)),
            "env_return": _metric(env_section, "return"),
            "env_reward": _metric(env_section, "reward"),
            "success_once": _metric(env_section, "success_once"),
            "gae_target_mean": _metric(rollout_section, "returns_mean"),
            "gae_target_min": _metric(rollout_section, "returns_min"),
            "gae_target_max": _metric(rollout_section, "returns_max"),
            "rollout_reward_mean": _metric(rollout_section, "rewards"),
        }
        if (
            row["env_return"] is not None
            and row["env_reward"] is not None
            and row["gae_target_mean"] is not None
        ):
            rows.append(row)
    return rows


def write_csv(rows: list[dict[str, float | int]], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "step",
        "env_return",
        "env_reward",
        "success_once",
        "gae_target_mean",
        "gae_target_min",
        "gae_target_max",
        "rollout_reward_mean",
    ]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _zscore(values: list[float]) -> list[float]:
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / max(len(values) - 1, 1)
    std = var**0.5
    if std == 0:
        return [0.0 for _ in values]
    return [(v - mean) / std for v in values]


def plot(rows: list[dict[str, float | int]], png_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    png_path.parent.mkdir(parents=True, exist_ok=True)
    steps = [int(row["step"]) for row in rows]
    env_returns = [float(row["env_return"]) for row in rows]
    env_rewards = [float(row["env_reward"]) for row in rows]
    gae_targets = [float(row["gae_target_mean"]) for row in rows]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    ax = axes[0]
    ax.plot(steps, env_returns, label="Environment/return", linewidth=1.8)
    ax.plot(steps, gae_targets, label="Rollout/returns_mean (GAE target)", linewidth=1.8)
    ax.set_ylabel("return / GAE target")
    ax.grid(True, alpha=0.25)
    ax2 = ax.twinx()
    ax2.plot(steps, env_rewards, label="Environment/reward", color="tab:green", linewidth=1.5)
    ax2.set_ylabel("env reward")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    ax.set_title(title)

    ax = axes[1]
    ax.plot(steps, _zscore(env_returns), label="z(Environment/return)", linewidth=1.5)
    ax.plot(steps, _zscore(env_rewards), label="z(Environment/reward)", linewidth=1.5)
    ax.plot(steps, _zscore(gae_targets), label="z(GAE target)", linewidth=1.5)
    ax.axhline(0.0, color="black", linewidth=0.8, alpha=0.4)
    ax.set_xlabel("Global Step")
    ax.set_ylabel("z-score")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, required=True, help="Path to run_embodiment.log")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or args.log.parent / "analysis"
    rows = parse_log(args.log)
    if not rows:
        raise SystemExit(f"No aligned return/reward/GAE target metrics found in {args.log}")

    csv_path = out_dir / "return_reward_gae_target.csv"
    png_path = out_dir / "return_reward_gae_target.png"
    write_csv(rows, csv_path)
    plot(rows, png_path, title=args.log.parent.name)

    print(f"rows={len(rows)}")
    print(f"csv={csv_path}")
    print(f"png={png_path}")
    print(f"step_range={rows[0]['step']}..{rows[-1]['step']}")
    print(
        "last="
        f"return={rows[-1]['env_return']}, "
        f"reward={rows[-1]['env_reward']}, "
        f"gae_target_mean={rows[-1]['gae_target_mean']}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot per-step reward, cumulative return, and GAE target from rollout dumps."""

from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path

import torch


def load_rollout_series(dump_dir: Path, env_idx: int) -> dict[str, torch.Tensor]:
    paths = sorted(dump_dir.glob("global_step_*_rank_*.pkl"))
    if not paths:
        raise FileNotFoundError(f"No rollout dump pkl found under {dump_dir}")

    parts: dict[str, list[torch.Tensor]] = {
        "reward": [],
        "gae_target": [],
        "advantage": [],
        "value": [],
        "done": [],
    }
    for path in paths:
        with path.open("rb") as f:
            payload = pickle.load(f)
        batch = payload["batch"]
        parts["reward"].append(batch["rewards"][:, env_idx, 0].float())
        parts["gae_target"].append(batch["returns"][:, env_idx, 0].float())
        parts["advantage"].append(batch["advantages"][:, env_idx, 0].float())
        parts["value"].append(batch["prev_values"][:-1, env_idx, 0].float())
        parts["done"].append(batch["dones"][1:, env_idx, 0].bool().float())

    out = {key: torch.cat(value, dim=0) for key, value in parts.items()}
    out["return"] = torch.cumsum(out["reward"], dim=0)
    return out


def write_csv(series: dict[str, torch.Tensor], csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["t", "return", "reward", "gae_target", "advantage", "value", "done"]
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        n = int(series["reward"].numel())
        for t in range(n):
            writer.writerow(
                {
                    "t": t,
                    "return": float(series["return"][t]),
                    "reward": float(series["reward"][t]),
                    "gae_target": float(series["gae_target"][t]),
                    "advantage": float(series["advantage"][t]),
                    "value": float(series["value"][t]),
                    "done": int(series["done"][t].item()),
                }
            )


def _zscore(values: torch.Tensor) -> torch.Tensor:
    std = values.std(unbiased=False)
    if float(std) == 0.0:
        return torch.zeros_like(values)
    return (values - values.mean()) / std


def plot(series: dict[str, torch.Tensor], png_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    png_path.parent.mkdir(parents=True, exist_ok=True)
    x = torch.arange(series["reward"].numel()).numpy()
    done_x = x[series["done"].bool().numpy()]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    ax = axes[0]
    ax.plot(x, series["return"].numpy(), label="cumulative return", linewidth=1.8)
    ax.plot(x, series["gae_target"].numpy(), label="GAE target (returns)", linewidth=1.5)
    ax.plot(x, series["value"].numpy(), label="old value", linewidth=1.2, alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(x, series["reward"].numpy(), label="step reward", color="tab:green", linewidth=1.2)
    ax.set_ylabel("return / GAE target / value")
    ax2.set_ylabel("step reward")
    for pos in done_x:
        ax.axvline(pos, color="black", alpha=0.12, linewidth=0.8)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="best")
    ax.grid(True, alpha=0.25)
    ax.set_title(title)

    ax = axes[1]
    for key, label in [
        ("return", "z(cumulative return)"),
        ("reward", "z(step reward)"),
        ("gae_target", "z(GAE target)"),
        ("value", "z(old value)"),
    ]:
        ax.plot(x, _zscore(series[key]).numpy(), label=label, linewidth=1.2)
    for pos in done_x:
        ax.axvline(pos, color="black", alpha=0.12, linewidth=0.8)
    ax.axhline(0, color="black", alpha=0.4, linewidth=0.8)
    ax.set_xlabel("Collected env step")
    ax.set_ylabel("z-score")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(png_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump-dir", type=Path, required=True)
    parser.add_argument("--env-idx", type=int, default=0)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()

    series = load_rollout_series(args.dump_dir, args.env_idx)
    out_dir = args.out_dir or args.dump_dir.parent / "analysis"
    stem = f"env_{args.env_idx:02d}_return_reward_gae"
    csv_path = out_dir / f"{stem}.csv"
    png_path = out_dir / f"{stem}.png"
    write_csv(series, csv_path)
    plot(series, png_path, f"{args.dump_dir.parent.name} env={args.env_idx}")
    print(f"steps={series['reward'].numel()}")
    print(f"csv={csv_path}")
    print(f"png={png_path}")


if __name__ == "__main__":
    main()

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

"""Serve a Franka-fold CFGRL checkpoint through one positive condition."""

from __future__ import annotations

import argparse
import logging
import pathlib
from typing import Any

import numpy as np

RUNTIME_CONFIG_NAME = "pi05_franka_fold_full_rtc"
CHECKPOINT_CONFIG_NAME = "pi05_franka_fold_recap_cfgrl"
ASSET_ID = "franka_fold_offline_v1"
DEFAULT_TASK = "fold the clothes"
POSITIVE_SUFFIX = "\nAdvantage: positive"


def positive_condition_prompt(prompt: str) -> str:
    """Return the CFGRL positive-condition prompt without double-appending it."""
    return prompt if prompt.endswith(POSITIVE_SUFFIX) else prompt + POSITIVE_SUFFIX


class PositiveConditionPolicy:
    """Run a CFGRL actor as a single positive-conditioned SFT-style policy."""

    def __init__(self, policy: Any, default_task: str = DEFAULT_TASK):
        self._policy = policy
        self._default_task = default_task

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        conditioned = dict(observation)
        prompt = conditioned.get("prompt", self._default_task)
        if isinstance(prompt, bytes):
            prompt = prompt.decode("utf-8")
        elif not isinstance(prompt, str):
            prompt = prompt.item()
        conditioned["prompt"] = positive_condition_prompt(prompt)
        return self._policy.infer(conditioned)

    @property
    def metadata(self) -> dict[str, Any]:
        return self._policy.metadata


def validate_checkpoint(checkpoint_dir: pathlib.Path) -> pathlib.Path:
    """Validate and resolve the JAX checkpoint and its offline norm stats."""
    checkpoint_dir = checkpoint_dir.expanduser().resolve()
    required = (
        checkpoint_dir / "params" / "_METADATA",
        checkpoint_dir / "assets" / ASSET_ID / "norm_stats.json",
    )
    missing = [path for path in required if not path.is_file()]
    if missing:
        formatted = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            f"Incomplete {CHECKPOINT_CONFIG_NAME} checkpoint. Missing:\n{formatted}"
        )
    return checkpoint_dir


def create_policy(checkpoint_dir: pathlib.Path, task: str):
    """Load CFGRL weights into the equivalent single-condition RTC runtime."""
    from openpi.policies import policy_config
    from openpi.shared import normalize
    from openpi.training import config as training_config

    config = training_config.get_config(RUNTIME_CONFIG_NAME)
    if config.model.training_rtc_max_delay <= 0:
        raise RuntimeError(f"{RUNTIME_CONFIG_NAME} does not enable Training RTC")

    norm_stats = normalize.load(checkpoint_dir / "assets" / ASSET_ID)
    policy = policy_config.create_trained_policy(
        config,
        checkpoint_dir,
        default_prompt=positive_condition_prompt(task),
        norm_stats=norm_stats,
    )
    return PositiveConditionPolicy(policy, default_task=task), config


def warmup_policy(policy: PositiveConditionPolicy) -> None:
    """Compile and verify regular and Training RTC inference before serving."""
    from openpi.policies import franka_fold_policy

    observation = franka_fold_policy.make_franka_fold_example()
    regular = policy.infer(observation)
    actions = np.asarray(regular["actions"], dtype=np.float32)
    expected = (
        franka_fold_policy.ACTION_HORIZON,
        franka_fold_policy.ACTION_DIM,
    )
    if actions.shape != expected:
        raise ValueError(f"Warmup returned actions shape {actions.shape}, expected {expected}")

    rtc = policy.infer(
        {
            **observation,
            "training_rtc_action_prefix": actions.copy(),
            "training_rtc_delay_steps": 4,
        }
    )
    if rtc.get("training_rtc_delay_steps") != 4:
        raise ValueError("Training RTC warmup did not preserve the requested delay")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=pathlib.Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--task", default=DEFAULT_TASK)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        raise SystemExit("port must be between 1 and 65535")

    checkpoint_dir = validate_checkpoint(args.checkpoint_dir)
    policy, config = create_policy(checkpoint_dir, args.task)
    logging.info("Warming up positive-condition and Training RTC inference")
    warmup_policy(policy)

    from openpi.serving import websocket_policy_server

    metadata = {
        **policy.metadata,
        "dataset": ASSET_ID,
        "method": "recap_cfgrl",
        "policy_family": "pi0.5",
        "config_name": CHECKPOINT_CONFIG_NAME,
        "runtime_config_name": RUNTIME_CONFIG_NAME,
        "inference_mode": "positive_condition_only",
        "condition_prompt": positive_condition_prompt(args.task),
        "asset_id": ASSET_ID,
        "action_horizon": config.model.action_horizon,
        "action_dim": config.model.action_dim,
    }
    logging.info("Serving %s at ws://%s:%d", checkpoint_dir, args.host, args.port)
    websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host=args.host,
        port=args.port,
        metadata=metadata,
    ).serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main()

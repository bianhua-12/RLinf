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

"""ReFL scorer: a frozen world model rolls the policy's action chunk forward from
the simulator state, and the analytic task reward of the imagined states is the
score. Differentiable from the score back to the model-space action sample."""

import json
from typing import Any

import torch
from torch import nn

from rlinf.models.embodiment.physical_wm.model import (
    ACTION_DIM,
    Carry,
    imagined_return,
    load_wm,
)

# Observation keys produced by :class:`CarryTracker` in the env; Pi0RL copies
# them into ``forward_inputs``.
CARRY_KEYS = {
    "state": "refl_state",
    "previous": "refl_previous",
    "previous_action": "refl_previous_action",
    "reset": "refl_reset",
    "context": "refl_context",
}


class CarryTracker:
    """Env-side record of the world-model carry of every sub-env.

    The env calls :meth:`reset` after a full or partial reset and :meth:`step`
    after each control step, with the physical state at that point; then
    :meth:`observation` gives the ``refl_*`` observation keys.
    """

    def __init__(self, episode_steps: int):
        self.episode_steps = int(episode_steps)
        self.carry: dict[str, torch.Tensor] | None = None

    def reset(self, state: torch.Tensor, env_idx: torch.Tensor | None = None) -> None:
        """Start the history of the sub-envs in ``env_idx`` (all when None)."""
        num_envs = state.shape[0]
        if self.carry is None or env_idx is None:
            self.carry = {
                "state": state,
                "previous": state.clone(),
                "previous_action": state.new_zeros(num_envs, ACTION_DIM),
                "reset": torch.ones(num_envs, dtype=torch.bool, device=state.device),
            }
            return
        mask = torch.zeros(num_envs, dtype=torch.bool, device=state.device)
        mask[torch.as_tensor(env_idx, device=state.device)] = True
        carry = self.carry
        carry["state"] = state
        carry["previous"] = torch.where(mask[:, None], state, carry["previous"])
        carry["previous_action"] = torch.where(
            mask[:, None],
            torch.zeros_like(carry["previous_action"]),
            carry["previous_action"],
        )
        carry["reset"] = carry["reset"] | mask

    def step(self, state: torch.Tensor, actions: torch.Tensor) -> None:
        """Record a control step that executed ``actions`` and reached ``state``."""
        if self.carry is None:
            raise RuntimeError("CarryTracker.step called before reset")
        actions = torch.as_tensor(actions, device=state.device, dtype=state.dtype)
        self.carry = {
            "state": state,
            "previous": self.carry["state"],
            # The normalized delta-pose controllers clip actions to [-1, 1].
            "previous_action": actions.reshape(state.shape[0], -1).clamp(-1, 1),
            "reset": torch.zeros(state.shape[0], dtype=torch.bool, device=state.device),
        }

    def observation(self, elapsed_steps: torch.Tensor) -> dict[str, torch.Tensor]:
        """``refl_*`` keys; ``elapsed_steps`` [num_envs] gives the episode progress."""
        if self.carry is None:
            raise RuntimeError("CarryTracker.observation called before reset")
        carry = self.carry
        elapsed = torch.as_tensor(elapsed_steps, device=carry["state"].device)
        return {
            CARRY_KEYS["state"]: carry["state"].clone(),
            CARRY_KEYS["previous"]: carry["previous"].clone(),
            CARRY_KEYS["previous_action"]: carry["previous_action"].clone(),
            CARRY_KEYS["reset"]: carry["reset"].float(),
            CARRY_KEYS["context"]: elapsed.reshape(-1, 1).float() / self.episode_steps,
        }


def decode_env_actions(
    samples: torch.Tensor, q01: torch.Tensor, q99: torch.Tensor
) -> torch.Tensor:
    """Model-space actions -> normalized env actions, as the OpenPI output transform.

    Quantile unnormalization of the first 6 dims; the gripper keeps the hard
    threshold of the deployed policy (no gradient), and the controller clips to
    [-1, 1].
    """
    raw = (samples[..., :ACTION_DIM] + 1) * 0.5 * (q99 - q01 + 1e-6) + q01
    gripper = (raw[..., 6:7] > 0.5).to(raw.dtype) * 2 - 1
    return torch.cat((raw[..., :6], gripper.detach()), -1).clamp(-1, 1)


class PhysicalWMScorer(nn.Module):
    def __init__(
        self,
        wm: nn.Module,
        q01: torch.Tensor,
        q99: torch.Tensor,
        horizon: int,
        gamma: float,
    ):
        super().__init__()
        self.wm = wm.eval().requires_grad_(False)
        self.register_buffer("q01", q01)
        self.register_buffer("q99", q99)
        self.horizon, self.gamma = int(horizon), float(gamma)

    @classmethod
    def from_config(cls, cfg: Any, device: torch.device) -> "PhysicalWMScorer":
        # ``device`` may be a bare accelerator index, which map_location rejects.
        checkpoint = torch.load(
            cfg.checkpoint_path, map_location="cpu", weights_only=False
        )
        wm = load_wm(checkpoint, device)
        with open(cfg.norm_stats_path) as f:
            stats = json.load(f)["norm_stats"]["actions"]
        q01 = torch.tensor(stats["q01"][:ACTION_DIM], dtype=torch.float32)
        q99 = torch.tensor(stats["q99"][:ACTION_DIM], dtype=torch.float32)
        return cls(wm, q01, q99, cfg.horizon, cfg.gamma).to(device)

    def env_actions(self, samples: torch.Tensor) -> torch.Tensor:
        return decode_env_actions(samples.float(), self.q01, self.q99)[
            :, : self.horizon
        ]

    def __call__(
        self, samples: torch.Tensor, forward_inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """``samples`` [B, action_horizon, model_action_dim] -> returns [B]."""
        if samples.shape[1] < self.horizon:
            raise ValueError("Action chunk is shorter than the imagination horizon")
        device = samples.device
        values = {
            name: forward_inputs[key].to(device=device, dtype=torch.float32)
            for name, key in CARRY_KEYS.items()
        }
        values["reset"] = values["reset"] > 0.5
        carry = Carry(**values)
        carry.validate(self.wm.schema)
        with torch.autocast(device_type=device.type, enabled=False):
            return imagined_return(
                self.wm, carry, self.env_actions(samples), self.gamma
            )

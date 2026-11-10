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

"""Physical-state world model: residual integration of joints and rigid bodies.

The model predicts residuals of a semi-implicit integrator over the simulator's
physical state; the TCP pose is recomputed from the predicted joints by FK. It is
differentiable with respect to the action, which is what ReFL needs.

State layout of PushCube (41-D): qpos 9, qvel 9, tcp pose 7 @18, cube pose 7 @25,
goal position 3 @32, cube linear velocity 3 @35, cube angular velocity 3 @38.
Poses are position + WXYZ quaternion in the env frame.
"""

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from rlinf.models.embodiment.physical_wm.geometry import (
    RobotFK,
    exp,
    multiply,
    normalize,
    rotation6,
    rotation_error,
)


@dataclass(frozen=True)
class Schema:
    task: str
    dim: int
    nj: int  # joints (arm + fingers)
    arm: int  # arm joints
    tcp: int  # start of the TCP pose
    bodies: tuple  # (name, pose start, linear velocity start, angular velocity start)
    quaternions: tuple  # starts of every WXYZ quaternion
    dt: float  # control period
    steps: int  # episode length

    @property
    def protocol(self) -> str:
        return self.task + "-physical-h5-v1"

    def state_features(self, state: torch.Tensor) -> torch.Tensor:
        parts, start = [], 0
        for q in self.quaternions:
            parts.extend((state[..., start:q], rotation6(state[..., q : q + 4])))
            start = q + 4
        parts.append(state[..., start:])
        return torch.cat(parts, -1)

    def difference(self, state: torch.Tensor, previous: torch.Tensor) -> torch.Tensor:
        parts, start = [], 0
        for q in self.quaternions:
            parts.extend(
                (
                    state[..., start:q] - previous[..., start:q],
                    rotation_error(state[..., q : q + 4], previous[..., q : q + 4]),
                )
            )
            start = q + 4
        parts.append(state[..., start:] - previous[..., start:])
        return torch.cat(parts, -1)

    def raw_floor(self) -> torch.Tensor:
        floor = torch.full((self.dim,), 0.05)
        floor[: self.arm] = 0.1
        floor[self.arm : self.nj] = 0.01
        floor[self.nj : self.nj + self.arm] = 0.5
        floor[self.nj + self.arm : 2 * self.nj] = 0.02
        for q in self.quaternions:
            floor[q : q + 4] = 1
        for _, _, v, w in self.bodies:
            floor[v : v + 3], floor[w : w + 3] = 0.1, 0.5
        return floor

    def feature_floor(self, difference: bool = False) -> torch.Tensor:
        floor = self.raw_floor()
        parts, start = [], 0
        for q in self.quaternions:
            parts.extend((floor[start:q], torch.ones(3 if difference else 6)))
            start = q + 4
        parts.append(floor[start:])
        return torch.cat(parts)


SCHEMAS = {
    "pushcube": Schema(
        "pushcube", 41, 9, 7, 18, (("cube", 25, 35, 38),), (21, 28), 0.05, 50
    ),
}

ACTION_DIM = 7


@dataclass
class Carry:
    """What the world model needs to take one step.

    ``previous`` and ``previous_action`` are the state and executed action of the
    step before ``state``; both are ignored where ``reset`` is true (first step of
    an episode). ``context`` is the elapsed fraction of the episode, [..., 1].
    """

    state: torch.Tensor
    previous: torch.Tensor
    previous_action: torch.Tensor
    reset: torch.Tensor
    context: torch.Tensor

    def validate(self, schema: Schema) -> None:
        lead = self.state.shape[:-1]
        if (
            self.state.shape[-1] != schema.dim
            or self.previous.shape != self.state.shape
        ):
            raise ValueError("Physical state/history shape mismatch")
        if self.previous_action.shape != (*lead, ACTION_DIM):
            raise ValueError("Missing the executed previous action")
        if self.reset.shape != lead:
            raise ValueError("Missing the reset mask")
        if self.context.shape != (*lead, 1):
            raise ValueError("Context must be the elapsed episode fraction [..., 1]")


def features(schema: Schema, carry: Carry, action: torch.Tensor) -> torch.Tensor:
    valid = (~carry.reset).to(carry.state.dtype)[..., None]
    return torch.cat(
        [
            schema.state_features(carry.state),
            schema.difference(carry.state, carry.previous) * valid,
            carry.previous_action * valid,
            action,
            carry.reset.to(carry.state.dtype)[..., None],
            carry.context,
        ],
        -1,
    )


def feature_floor(schema: Schema) -> torch.Tensor:
    return torch.cat(
        (
            schema.feature_floor(),
            schema.feature_floor(True),
            torch.ones(2 * ACTION_DIM + 1),
            torch.ones(1),
        )
    )


def _mlp(widths: list[int]) -> nn.Sequential:
    layers = []
    for i in range(len(widths) - 1):
        layers.append(nn.Linear(widths[i], widths[i + 1]))
        if i < len(widths) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class PhysicalWM(nn.Module):
    def __init__(
        self,
        task: str,
        asset: dict[str, Any],
        mean: torch.Tensor,
        scale: torch.Tensor,
        width: int = 2048,
        device: str | torch.device = "cpu",
    ):
        super().__init__()
        self.schema = SCHEMAS[task]
        self.asset, self.width = asset, width
        self.fk = RobotFK(asset, device)
        if self.fk.narm != self.schema.arm:
            raise ValueError("Wrong robot FK")
        if mean.shape != feature_floor(self.schema).shape or (scale <= 0).any():
            raise ValueError("Normalizer contract mismatch")
        self.register_buffer("mean", mean.clone().to(device))
        self.register_buffer("scale", scale.clone().to(device))
        self.trunk = nn.Sequential(_mlp([len(mean), width, width, width]), nn.SiLU())
        self.continuous = nn.Linear(
            width, self.schema.nj * 2 + 12 * len(self.schema.bodies)
        )
        nn.init.zeros_(self.continuous.weight)
        nn.init.zeros_(self.continuous.bias)
        self.to(device)

    def forward(self, carry: Carry, action: torch.Tensor) -> Carry:
        s = self.schema
        hidden = self.trunk((features(s, carry, action) - self.mean) / self.scale)
        residual = self.continuous(hidden)
        current = carry.state
        n, a = s.nj, s.arm
        qscale = current.new_tensor([0.05] * a + [0.002] * (n - a))
        vscale = current.new_tensor([0.25] * a + [0.01] * (n - a))
        v = current[..., n : 2 * n] + residual[..., n : 2 * n] * vscale
        q = (
            current[..., :n]
            + s.dt * (current[..., n : 2 * n] + v) / 2
            + residual[..., :n] * qscale
        )
        nxt = current.clone()
        nxt[..., :n], nxt[..., n : 2 * n] = q, v
        nxt[..., s.tcp : s.tcp + 7] = self.fk(q)
        offset = 2 * n
        for _, p, lv, av in s.bodies:
            r = residual[..., offset : offset + 12]
            offset += 12
            velocity = current[..., lv : lv + 3] + 0.05 * r[..., :3]
            angular = current[..., av : av + 3] + 0.5 * r[..., 3:6]
            position = (
                current[..., p : p + 3]
                + s.dt * (current[..., lv : lv + 3] + velocity) / 2
                + 0.01 * r[..., 6:9]
            )
            tangent = (
                s.dt * (current[..., av : av + 3] + angular) / 2
                + (torch.pi / 36) * r[..., 9:12]
            )
            quat = normalize(multiply(exp(tangent), current[..., p + 3 : p + 7]))
            nxt[..., p : p + 7] = torch.cat((position, quat), -1)
            nxt[..., lv : lv + 3], nxt[..., av : av + 3] = velocity, angular
        context = carry.context + 1 / s.steps
        return Carry(nxt, current, action, torch.zeros_like(carry.reset), context)

    def checkpoint(self) -> dict[str, Any]:
        return {
            "protocol": self.schema.protocol,
            "task": self.schema.task,
            "asset": self.asset,
            "width": self.width,
            "model": self.state_dict(),
        }


def load_wm(
    checkpoint: dict[str, Any], device: str | torch.device = "cpu"
) -> PhysicalWM:
    task = checkpoint["task"]
    if task not in SCHEMAS or checkpoint["protocol"] != SCHEMAS[task].protocol:
        raise ValueError("Incompatible world-model protocol")
    sd = checkpoint["model"]
    model = PhysicalWM(
        task, checkpoint["asset"], sd["mean"], sd["scale"], checkpoint["width"], device
    )
    model.load_state_dict(sd, strict=True)
    return model


def pushcube_reward(
    state: torch.Tensor, half_size: float = 0.02, goal_radius: float = 0.1
) -> tuple[torch.Tensor, torch.Tensor]:
    """ManiSkill PushCube ``normalized_dense`` reward and success of a state.

    Reaching term (TCP to the push point behind the cube) plus, once in contact,
    the cube-to-goal term; success gives 1.
    """
    cube, tcp, goal = state[..., 25:28], state[..., 18:21], state[..., 32:35]
    push = cube.clone()
    push[..., 0] -= half_size + 0.005
    distance = (push - tcp).norm(dim=-1)
    goal_distance = (cube[..., :2] - goal[..., :2]).norm(dim=-1)
    success = (goal_distance < goal_radius) & (cube[..., 2] < half_size + 0.005)
    place = 1 - torch.tanh(5 * goal_distance)
    height = 1 - torch.tanh(5 * (cube[..., 2] - half_size).abs())
    raw = 1 - torch.tanh(5 * distance) + (distance < 0.01) * place * (1 + height)
    return torch.where(success, torch.ones_like(raw), raw / 4), success.to(raw.dtype)


TASK_REWARDS = {"pushcube": pushcube_reward}


def imagined_return(
    wm: PhysicalWM,
    carry: Carry,
    actions: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Discounted analytic reward of the WM rollout of ``actions`` [B, H, 7].

    An imagined success ends the episode; that cut is not differentiated.
    """
    reward_fn = TASK_REWARDS[wm.schema.task]
    survival = actions.new_ones(actions.shape[0])
    value = torch.zeros_like(survival)
    for h in range(actions.shape[1]):
        carry = wm(carry, actions[:, h])
        reward, success = reward_fn(carry.state)
        value = value + (gamma**h) * survival * reward
        survival = survival * (1 - success).detach()
    return value

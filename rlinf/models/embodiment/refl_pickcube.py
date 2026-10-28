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

"""PickCube PPO790 actor and physical WM checkpoint-compatible layers.

Adapted from SuperMimic, MIT copyright 2026 Yuxi Liu; see
refl_pickcube.LICENSE. Simulator/FK state stays FP32 under neural autocast.
"""

from __future__ import annotations

import math

import torch
from torch import nn

CUBOILEO_ERROR_SCALES = (0.05, 0.1, 0.5, 1.0, 0.1, 2.0)
CUBOILEO_METRIC = "cuboi_nominal_rotvec_v1"
LEGACY_METRIC = "physical_threshold_quaternion_v0"
_EPS = 1.0e-8


def validate_precision(precision: str, device) -> None:
    if precision not in {"bf16", "float32"}:
        raise ValueError(f"Unknown neural precision: {precision}")
    device = torch.device(device)
    if (
        precision == "bf16"
        and device.type == "cuda"
        and not torch.cuda.is_bf16_supported()
    ):
        raise RuntimeError("This run requires native CUDA bfloat16 support")


def neural_forward(network, *inputs, precision="float32"):
    device = inputs[0].device
    with torch.autocast(
        device_type=device.type, dtype=torch.bfloat16, enabled=precision == "bf16"
    ):
        result = network(*inputs)
    # The cast remains differentiable. No reduced-precision state integration/FK.
    return result.float()


def set_precision(module, precision: str):
    validate_precision(precision, next(module.parameters()).device)
    for child in module.modules():
        if hasattr(child, "precision"):
            child.precision = precision
    return module


class MLP(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden: tuple[int, ...],
        out_dim: int,
        act=nn.SiLU,
        zero_last: bool = False,
        last_gain: float | None = None,
    ):
        super().__init__()
        layers: list[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), act()]
            d = h
        last = nn.Linear(d, out_dim)
        if zero_last:
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        elif last_gain is not None:
            nn.init.orthogonal_(last.weight, gain=last_gain)
            nn.init.zeros_(last.bias)
        layers.append(last)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DynamicsMLP(nn.Module):
    """delta = f(canonical(s), a, xi_norm) -> (…, 70). Zero-initialised last layer (identity dynamics at init)."""

    def __init__(
        self,
        feat_dim: int = 71,
        act_dim: int = 29,
        xi_dim: int = 33,
        hidden=(2048, 2048, 2048),
        out_dim: int = 70,
    ):
        super().__init__()
        self.mlp = MLP(
            feat_dim + act_dim + xi_dim, tuple(hidden), out_dim, zero_last=True
        )

    def forward(
        self, feat: torch.Tensor, action: torch.Tensor, xi_norm: torch.Tensor
    ) -> torch.Tensor:
        return self.mlp(torch.cat([feat, action, xi_norm], dim=-1))


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(_EPS)


def quat_canonical(q: torch.Tensor) -> torch.Tensor:
    """Flip sign so that w >= 0 (removes the double cover ambiguity)."""
    return torch.where(q[..., :1] < 0, -q, q)


def rotvec_from_quat(q: torch.Tensor) -> torch.Tensor:
    """Logarithmic map: unit quaternion (wxyz) -> rotation vector in (-pi, pi]."""
    q = quat_canonical(q)
    w = q[..., :1]
    xyz = q[..., 1:]
    s = xyz.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(s, w)
    small = s < 1e-6
    # angle / sin(angle/2) == angle / s ; series: 2 + angle^2/12 ... (with s ~ angle/2)
    k = torch.where(
        small,
        2.0 + (angle * angle) / 12.0,
        angle / torch.where(small, torch.ones_like(s), s),
    )
    return xyz * k


def quat_mul(a, b):
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat(
        [
            aw * bw - (av * bv).sum(-1, keepdim=True),
            aw * bv + bw * av + torch.cross(av, bv, dim=-1),
        ],
        -1,
    )


def matrix_quaternion(m):
    """WXYZ quaternion; clamped inactive branches avoid sqrt(0) backward NaNs."""
    a, b, c = m[..., 0, 0], m[..., 1, 1], m[..., 2, 2]
    roots = (
        torch.stack([1 + a + b + c, 1 + a - b - c, 1 - a + b - c, 1 - a - b + c], -1)
        .clamp_min(1e-8)
        .sqrt()
    )
    rows = torch.stack(
        [
            torch.stack(
                [
                    roots[..., 0].square(),
                    m[..., 2, 1] - m[..., 1, 2],
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] - m[..., 0, 1],
                ],
                -1,
            ),
            torch.stack(
                [
                    m[..., 2, 1] - m[..., 1, 2],
                    roots[..., 1].square(),
                    m[..., 1, 0] + m[..., 0, 1],
                    m[..., 0, 2] + m[..., 2, 0],
                ],
                -1,
            ),
            torch.stack(
                [
                    m[..., 0, 2] - m[..., 2, 0],
                    m[..., 1, 0] + m[..., 0, 1],
                    roots[..., 2].square(),
                    m[..., 2, 1] + m[..., 1, 2],
                ],
                -1,
            ),
            torch.stack(
                [
                    m[..., 1, 0] - m[..., 0, 1],
                    m[..., 2, 0] + m[..., 0, 2],
                    m[..., 2, 1] + m[..., 1, 2],
                    roots[..., 3].square(),
                ],
                -1,
            ),
        ],
        -2,
    )
    candidates = rows / (2 * roots.clamp_min(0.1)[..., None])
    idx = roots.argmax(-1)[..., None, None].expand(*roots.shape[:-1], 1, 4)
    q = candidates.gather(-2, idx).squeeze(-2)
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def cache_static_chain_indices(chain):
    """Cache only topology indices on CPU; retain the library's FK arithmetic."""
    original = chain.get_frame_indices
    cache = {}

    def cached(*names):
        key = repr(names)
        if key not in cache:
            cache[key] = original(*names).cpu()
        return cache[key]

    def cpu_indices(x):
        if isinstance(x, torch.Tensor):
            return x.cpu()
        if isinstance(x, list):
            return [cpu_indices(v) for v in x]
        if isinstance(x, tuple):
            return tuple(cpu_indices(v) for v in x)
        if isinstance(x, dict):
            return {k: cpu_indices(v) for k, v in x.items()}
        return x

    chain.get_frame_indices = cached
    for name in ["parents_indices", "joint_indices", "joint_type_indices"]:
        if hasattr(chain, name):
            setattr(chain, name, cpu_indices(getattr(chain, name)))


def cuboileo_residual(pred, true):
    """Same per-quantity scales/log-map as G1, with explicit prismatic mapping.

    Retain this task's physical quantities (including derived TCP) so this change
    does not silently change the field set. Finger positions/velocities use the
    linear scales, not the revolute scales. Goal, grasp and relative vectors are
    excluded; grasp BCE is separate. Dimensions:30(native42) or36(physical48).
    """
    pos, rot, lin, ang, jpos, jvel = CUBOILEO_ERROR_SCALES

    def rotation(q_pred, q_true):
        q1, q2 = quat_normalize(q_pred), quat_normalize(q_true)
        inverse = torch.cat([q2[..., :1], -q2[..., 1:]], -1)
        return rotvec_from_quat(quat_mul(inverse, q1)) / rot

    parts = [
        (pred[..., :7] - true[..., :7]) / jpos,
        (pred[..., 7:9] - true[..., 7:9]) / pos,
        (pred[..., 9:16] - true[..., 9:16]) / jvel,
        (pred[..., 16:18] - true[..., 16:18]) / lin,
        (pred[..., 19:22] - true[..., 19:22]) / pos,
        rotation(pred[..., 22:26], true[..., 22:26]),
        (pred[..., 29:32] - true[..., 29:32]) / pos,
        rotation(pred[..., 32:36], true[..., 32:36]),
    ]
    if pred.shape[-1] == 48:
        parts.extend(
            [
                (pred[..., 42:45] - true[..., 42:45]) / lin,
                (pred[..., 45:48] - true[..., 45:48]) / ang,
            ]
        )
    return torch.cat(parts, -1)


def state_error_loss(residual, metric):
    squared = residual.square()
    # Upstream sums dimensions for training; evaluation always averages them.
    return squared.sum(-1).mean() if metric == CUBOILEO_METRIC else squared.mean()


class SingleWorldModel(nn.Module):
    def __init__(
        self,
        asset,
        mean,
        scale,
        hidden=2048,
        device="cuda",
        error_metric=CUBOILEO_METRIC,
    ):
        super().__init__()
        self.asset = asset
        self.hidden = hidden
        self.precision = "float32"
        if error_metric not in (CUBOILEO_METRIC, LEGACY_METRIC):
            raise ValueError(error_metric)
        self.error_metric = error_metric
        self.register_buffer("mean", mean.clone())
        self.register_buffer("scale", scale.clone())
        self.register_buffer(
            "base_pose",
            torch.tensor(asset["base_pose"], device=device, dtype=torch.float32),
        )
        self.dynamics = DynamicsMLP(93, 8, 0, (hidden, hidden, hidden), 26)
        self.register_buffer(
            "q_correction", torch.tensor([0.03] * 7 + [0.002] * 2, device=device)
        )
        self.register_buffer(
            "velocity_correction", torch.tensor([1.0] * 7 + [0.1] * 2, device=device)
        )
        import pytorch_kinematics as pk

        self.chain = pk.build_serial_chain_from_urdf(
            asset["urdf"], asset["ee_link"]
        ).to(device=device, dtype=torch.float32)
        if list(self.chain.get_joint_parameter_names()) != asset["joint_names"]:
            raise ValueError("FK joint ordering mismatch")
        cache_static_chain_indices(self.chain)
        self.to(device)

    def tcp(self, qpos, reference):
        shape = qpos.shape[:-1]
        matrix = (
            self.base_pose
            @ self.chain.forward_kinematics(qpos[..., :7].reshape(-1, 7)).get_matrix()
        )
        quat = matrix_quaternion(matrix[:, :3, :3]).reshape(*shape, 4)
        quat = torch.where(
            (quat * reference[..., 22:26]).sum(-1, keepdim=True) < 0, -quat, quat
        )
        return torch.cat([matrix[:, :3, 3].reshape(*shape, 3), quat], -1)

    def step(
        self, state, action, previous_state=None, previous_action=None, reset=None
    ):
        if previous_state is None:
            previous_state = state
        if previous_action is None:
            previous_action = torch.zeros_like(action)
        if reset is None:
            reset = torch.ones_like(action[..., 0])
        feat = torch.cat(
            [
                (state - self.mean) / self.scale,
                (state - previous_state) / self.scale,
                previous_action,
                reset.float()[..., None],
            ],
            -1,
        )
        d = neural_forward(
            self.dynamics, feat, action, state[..., :0], precision=self.precision
        )
        q = state[..., :9] + 0.05 * state[..., 9:18] + d[..., :9] * self.q_correction
        v = state[..., 9:18] + d[..., 9:18] * self.velocity_correction
        cube = (
            state[..., 29:32]
            + (1 - reset.float()[..., None])
            * (state[..., 29:32] - previous_state[..., 29:32])
            + d[..., 18:21] * 0.03
        )
        quat = state[..., 32:36] + d[..., 21:25] * 0.1
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        grasp = d[..., 25:26].sigmoid()
        tcp = self.tcp(q, state)
        goal = state[..., 26:29]
        return torch.cat(
            [q, v, grasp, tcp, goal, cube, quat, cube - tcp[..., :3], goal - cube], -1
        )

    def rollout(
        self, state, actions, previous_state=None, previous_action=None, reset=None
    ):
        out = []
        for a in actions:
            nxt = self.step(state, a, previous_state, previous_action, reset)
            previous_state, previous_action = state, a
            state = nxt
            reset = torch.zeros_like(a[..., 0])
            out.append(state)
        return torch.stack(out)

    def enable_compile(self, mode):
        if mode is not None:
            raise ValueError("Eager verified FK; no whole-model compiler substitution")

    def residual(self, p, t):
        if self.error_metric == CUBOILEO_METRIC:
            return cuboileo_residual(p, t)

        def qerror(a, b):
            return torch.where((a * b).sum(-1, keepdim=True) < 0, -a, a) - b

        return torch.cat(
            [
                (p[..., :7] - t[..., :7]) / 0.03,
                (p[..., 7:9] - t[..., 7:9]) / 0.002,
                (p[..., 9:16] - t[..., 9:16]) / 0.2,
                (p[..., 16:18] - t[..., 16:18]) / 0.03,
                (p[..., 19:22] - t[..., 19:22]) / 0.01,
                qerror(p[..., 22:26], t[..., 22:26]) / 0.05,
                (p[..., 29:32] - t[..., 29:32]) / 0.01,
                qerror(p[..., 32:36], t[..., 32:36]) / 0.05,
            ],
            -1,
        )

    def replay_loss(self, w):
        ctx = {
            "previous_state": w.xi[:, :42],
            "previous_action": w.xi[:, 42:50],
            "reset": w.xi[:, 50],
        }
        pred = self.rollout(w.states[0], w.a_exec, **ctx)
        e = self.residual(pred, w.states[1:])
        state_loss = state_error_loss(e, self.error_metric)
        grasp_loss = nn.functional.binary_cross_entropy(
            pred[..., 18].clamp(1e-6, 1 - 1e-6), w.states[1:, ..., 18]
        )
        loss = state_loss + grasp_loss
        return loss, {
            "state_loss": state_loss.detach(),
            "state_normalized_mse": e.square().mean().detach(),
            "grasp_bce": grasp_loss.detach(),
        }


class SinglePhysicalWorldModel(SingleWorldModel):
    """Native observation42 plus true cube linear/angular velocity6; PPO still sees42."""

    def __init__(
        self,
        asset,
        mean,
        scale,
        hidden=2048,
        device="cuda",
        goal_free=False,
        error_metric=CUBOILEO_METRIC,
        cube_linear_loss_scale=0.5,
        cube_angular_loss_scale=1.0,
    ):
        super().__init__(asset, mean, scale, hidden, device, error_metric=error_metric)
        self.dynamics = DynamicsMLP(105, 8, 0, (hidden, hidden, hidden), 32).to(device)
        self.goal_free = goal_free
        if not all(
            math.isfinite(v) and v > 0
            for v in (cube_linear_loss_scale, cube_angular_loss_scale)
        ):
            raise ValueError("Training velocity scales must be finite and positive")
        if error_metric != CUBOILEO_METRIC and (
            cube_linear_loss_scale,
            cube_angular_loss_scale,
        ) != (0.5, 1.0):
            raise ValueError("Velocity loss reweighting requires the CuboiLeo metric")
        self.cube_linear_loss_scale = float(cube_linear_loss_scale)
        self.cube_angular_loss_scale = float(cube_angular_loss_scale)
        self.register_buffer(
            "physics_mask", torch.ones(48, device=device), persistent=False
        )
        if goal_free:
            self.physics_mask[26:29] = 0
            self.physics_mask[39:42] = 0

    def step(
        self, state, action, previous_state=None, previous_action=None, reset=None
    ):
        if previous_state is None:
            previous_state = state
        if previous_action is None:
            previous_action = torch.zeros_like(action)
        if reset is None:
            reset = torch.ones_like(action[..., 0])
        feat = torch.cat(
            [
                ((state - self.mean) / self.scale) * self.physics_mask,
                ((state - previous_state) / self.scale) * self.physics_mask,
                previous_action,
                reset.float()[..., None],
            ],
            -1,
        )
        d = neural_forward(
            self.dynamics, feat, action, state[..., :0], precision=self.precision
        )
        v = state[..., 9:18] + d[..., 9:18] * self.velocity_correction
        q = state[..., :9] + 0.05 * v + d[..., :9] * self.q_correction
        cv = state[..., 42:45] + 0.5 * d[..., 18:21]
        cw = state[..., 45:48] + 2 * d[..., 21:24]
        cube = state[..., 29:32] + 0.05 * cv + 0.01 * d[..., 24:27]
        # Same world-coordinate quaternion integration as existing state_wm.constant_velocity.
        rv = 0.05 * cw
        angle = rv.norm(dim=-1, keepdim=True)
        dq = torch.cat(
            [torch.cos(angle / 2), rv * (0.5 * torch.sinc(angle / (2 * torch.pi)))], -1
        )
        quat = quat_mul(dq, state[..., 32:36]) + 0.02 * d[..., 27:31]
        quat = quat / quat.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        tcp = self.tcp(q, state)
        goal = state[..., 26:29]
        grasp = d[..., 31:32].sigmoid()
        return torch.cat(
            [
                q,
                v,
                grasp,
                tcp,
                goal,
                cube,
                quat,
                cube - tcp[..., :3],
                goal - cube,
                cv,
                cw,
            ],
            -1,
        )

    def residual(self, p, t):
        if self.error_metric == CUBOILEO_METRIC:
            return cuboileo_residual(p, t)
        return torch.cat(
            [
                super().residual(p, t),
                (p[..., 42:45] - t[..., 42:45]) / 0.1,
                (p[..., 45:48] - t[..., 45:48]),
            ],
            -1,
        )

    def training_residual(self, e):
        if self.error_metric != CUBOILEO_METRIC:
            return e
        # Only loss weighting changes. Physical residual/evaluation scales stay fixed.
        return torch.cat(
            [
                e[..., :-6],
                e[..., -6:-3]
                * (CUBOILEO_ERROR_SCALES[2] / self.cube_linear_loss_scale),
                e[..., -3:] * (CUBOILEO_ERROR_SCALES[3] / self.cube_angular_loss_scale),
            ],
            -1,
        )

    def training_scale_config(self):
        return {
            "cube_linear_mps": self.cube_linear_loss_scale,
            "cube_angular_rps": self.cube_angular_loss_scale,
        }

    def replay_loss(self, w):
        ctx = {
            "previous_state": w.xi[:, :48],
            "previous_action": w.xi[:, 48:56],
            "reset": w.xi[:, 56],
        }
        pred = self.rollout(w.states[0], w.a_exec, **ctx)
        e = self.residual(pred, w.states[1:])
        weighted = self.training_residual(e)
        state_loss = state_error_loss(weighted, self.error_metric)
        grasp_loss = nn.functional.binary_cross_entropy(
            pred[..., 18].clamp(1e-6, 1 - 1e-6), w.states[1:, ..., 18]
        )
        metrics = {
            "state_loss": state_loss.detach(),
            "state_normalized_mse": e.square().mean().detach(),
            "grasp_bce": grasp_loss.detach(),
        }
        for name, sl in [
            ("other_state", slice(None, -6)),
            ("cube_linear_velocity", slice(-6, -3)),
            ("cube_angular_velocity", slice(-3, None)),
        ]:
            metrics[name + "_unweighted_sq_sum"] = (
                e[..., sl].square().sum(-1).mean().detach()
            )
            metrics[name + "_weighted_sq_sum"] = (
                weighted[..., sl].square().sum(-1).mean().detach()
            )
        return state_loss + grasp_loss, metrics


def ppo_net(output):
    return nn.Sequential(
        nn.Linear(42, 256),
        nn.Tanh(),
        nn.Linear(256, 256),
        nn.Tanh(),
        nn.Linear(256, 256),
        nn.Tanh(),
        nn.Linear(256, output),
    )


class SingleCritic(nn.Module):
    """PPO790 state42 value network; preserve derivatives through input states."""

    def __init__(self):
        super().__init__()
        self.net = ppo_net(1)
        self.precision = "float32"

    def forward(self, state):
        return neural_forward(
            self.net, state[..., :42], precision=self.precision
        ).squeeze(-1)


class SingleActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = ppo_net(8)
        self.precision = "float32"

    def raw(self, state):
        return neural_forward(self.net, state[..., :42], precision=self.precision)

    def forward(self, state):
        return self.raw(state).clamp(-1, 1)


def validate_warmstart(warm, gamma):
    expected = ("PickCube-v1", 42, 8, 0.8, 1.0, 20, 100, "identity")
    actual = tuple(
        warm.get(k)
        for k in [
            "env_id",
            "state_dim",
            "action_dim",
            "gamma",
            "reward_scale",
            "control_hz",
            "sim_hz",
            "input_normalization",
        ]
    )
    if (
        actual != expected
        or gamma != warm["gamma"]
        or not warm["evaluation"]["accepted"]
    ):
        raise ValueError("Actor/value warmstart contract mismatch")


def load_world_model(checkpoint, device):
    """Load the explicit physical48, goal-free WM protocol."""
    c = checkpoint
    if c.get("state_dim") != 48 or c.get("goal_free") is not True:
        raise ValueError("Expected a goal-free physical48 world model")
    if c.get("error_metric") != CUBOILEO_METRIC:
        raise ValueError("Expected cuboi_nominal_rotvec_v1 WM metric")
    scales = c["training_loss_scales"]
    model = SinglePhysicalWorldModel(
        c["asset"],
        c["model"]["mean"],
        c["model"]["scale"],
        c["hidden"],
        device=device,
        goal_free=True,
        error_metric=c["error_metric"],
        cube_linear_loss_scale=scales["cube_linear_mps"],
        cube_angular_loss_scale=scales["cube_angular_rps"],
    )
    model.load_state_dict(c["model"], strict=True)
    return model

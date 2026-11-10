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

"""WXYZ quaternions, world-frame tangent vectors and differentiable robot FK."""

from typing import Any

import torch
from torch import nn


def normalize(q: torch.Tensor) -> torch.Tensor:
    return nn.functional.normalize(q, dim=-1)


def multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, av, bw, bv = a[..., :1], a[..., 1:], b[..., :1], b[..., 1:]
    return torch.cat(
        (
            aw * bw - (av * bv).sum(-1, keepdim=True),
            aw * bv + bw * av + torch.cross(av, bv, dim=-1),
        ),
        -1,
    )


def inverse(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), -1)


def exp(v: torch.Tensor) -> torch.Tensor:
    """Quaternion of the rotation vector ``v`` (axis * angle)."""
    angle = v.norm(dim=-1, keepdim=True)
    return torch.cat(
        (torch.cos(angle / 2), 0.5 * torch.sinc(angle / (2 * torch.pi)) * v), -1
    )


def log(q: torch.Tensor) -> torch.Tensor:
    q = normalize(q)
    q = torch.where(q[..., :1] < 0, -q, q)
    v = q[..., 1:]
    n = v.norm(dim=-1, keepdim=True)
    # atan2 avoids the infinite derivative of acos at the identity.
    factor = 2 * torch.atan2(n, q[..., :1].clamp_min(0)) / n.clamp_min(1e-8)
    return v * torch.where(n > 1e-7, factor, 2 * torch.ones_like(n))


def rotation_error(pred: torch.Tensor, true: torch.Tensor) -> torch.Tensor:
    return log(multiply(normalize(pred), inverse(normalize(true))))


def rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = normalize(q)
    uv = torch.cross(q[..., 1:], v, dim=-1)
    return v + 2 * (q[..., :1] * uv + torch.cross(q[..., 1:], uv, dim=-1))


def rotation6(q: torch.Tensor) -> torch.Tensor:
    """First two columns of the rotation matrix, a continuous rotation feature."""
    x = torch.zeros_like(q[..., 1:])
    x[..., 0] = 1
    y = torch.zeros_like(x)
    y[..., 1] = 1
    return torch.cat((rotate(q, x), rotate(q, y)), -1)


def matrix_quaternion(m: torch.Tensor) -> torch.Tensor:
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


def _rpy_matrix(rpy: list[float]) -> torch.Tensor:
    """URDF fixed-axis roll-pitch-yaw: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    r, p, y = (torch.tensor(float(v), dtype=torch.float64) for v in rpy)
    cr, sr, cp, sp, cy, sy = r.cos(), r.sin(), p.cos(), p.sin(), y.cos(), y.sin()
    return torch.stack(
        [
            torch.stack([cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr]),
            torch.stack([sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr]),
            torch.stack([-sp, cp * sr, cp * cr]),
        ]
    )


def _axis_angle_matrix(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """Rodrigues rotation about a unit ``axis`` [3] by ``angle`` [N] -> [N, 3, 3]."""
    x, y, z = axis.to(angle)
    zero = torch.zeros_like(x)
    k = torch.stack(
        [
            torch.stack([zero, -z, y]),
            torch.stack([z, zero, -x]),
            torch.stack([-y, x, zero]),
        ]
    )
    s, c = angle.sin()[:, None, None], angle.cos()[:, None, None]
    eye = torch.eye(3, dtype=angle.dtype, device=angle.device)
    return eye + s * k + (1 - c) * (k @ k)


def _serial_chain(urdf: str, ee_link: str) -> list[dict[str, Any]]:
    """Joints from the root link to ``ee_link``, each with its origin and axis."""
    import xml.etree.ElementTree as ET

    joints = {}
    for joint in ET.fromstring(urdf).iter("joint"):
        origin = joint.find("origin")
        axis = joint.find("axis")
        child = joint.find("child").get("link")
        joints[child] = {
            "name": joint.get("name"),
            "type": joint.get("type"),
            "parent": joint.find("parent").get("link"),
            "xyz": [0.0, 0.0, 0.0]
            if origin is None
            else [float(v) for v in origin.get("xyz", "0 0 0").split()],
            "rpy": [0.0, 0.0, 0.0]
            if origin is None
            else [float(v) for v in origin.get("rpy", "0 0 0").split()],
            "axis": [1.0, 0.0, 0.0]
            if axis is None
            else [float(v) for v in axis.get("xyz").split()],
        }
    chain, link = [], ee_link
    while link in joints:
        chain.append(joints[link])
        link = joints[link]["parent"]
    if not chain:
        raise ValueError(f"Link {ee_link!r} is not the child of any URDF joint")
    return chain[::-1]


class RobotFK(nn.Module):
    """Differentiable TCP pose (position + WXYZ quaternion) in the env frame.

    ``asset`` holds the robot URDF text, the end-effector link, the arm joint
    names in chain order and the 4x4 robot base pose, as stored in the
    world-model checkpoint.
    """

    def __init__(self, asset: dict[str, Any], device: str | torch.device = "cpu"):
        super().__init__()
        self.asset = asset
        chain = _serial_chain(asset["urdf"], asset["ee_link"])
        moving = [j for j in chain if j["type"] != "fixed"]
        if any(j["type"] not in ("revolute", "continuous") for j in moving):
            raise ValueError("Only revolute arm joints are supported")
        names = [j["name"] for j in moving]
        if names != list(asset["arm_joint_names"]):
            raise ValueError(
                f"FK arm joints {names} do not match the asset {asset['arm_joint_names']}"
            )
        # Each joint i applies origins[i] (homogeneous) and then its rotation;
        # fixed joints after the last arm joint fold into ``tail``.
        origins, axes, pending = [], [], torch.eye(4, dtype=torch.float64)
        for joint in chain:
            origin = torch.eye(4, dtype=torch.float64)
            origin[:3, :3] = _rpy_matrix(joint["rpy"])
            origin[:3, 3] = torch.tensor(joint["xyz"], dtype=torch.float64)
            pending = pending @ origin
            if joint["type"] != "fixed":
                axis = torch.tensor(joint["axis"], dtype=torch.float64)
                origins.append(pending)
                axes.append(axis / axis.norm())
                pending = torch.eye(4, dtype=torch.float64)
        base = torch.tensor(asset["base_pose"], dtype=torch.float64)
        origins[0] = base @ origins[0]
        self.register_buffer("base", base.float().to(device))
        # Rebuilt from ``asset``; checkpoints carry only ``base``.
        chain_buffers = {
            "origins": torch.stack(origins),
            "axes": torch.stack(axes),
            "tail": pending,
        }
        for name, value in chain_buffers.items():
            self.register_buffer(name, value.float().to(device), persistent=False)
        self.narm = len(names)

    def forward(self, qpos: torch.Tensor) -> torch.Tensor:
        shape = qpos.shape[:-1]
        q = qpos[..., : self.narm].reshape(-1, self.narm)
        rotation = self.origins.new_zeros(q.shape[0], 3, 3) + torch.eye(
            3, device=q.device, dtype=self.origins.dtype
        )
        position = self.origins.new_zeros(q.shape[0], 3)
        for i in range(self.narm):
            position = position + rotation @ self.origins[i, :3, 3]
            rotation = (
                rotation
                @ self.origins[i, :3, :3]
                @ _axis_angle_matrix(self.axes[i], q[:, i].to(self.origins.dtype))
            )
        position = position + rotation @ self.tail[:3, 3]
        rotation = rotation @ self.tail[:3, :3]
        return torch.cat((position, matrix_quaternion(rotation)), -1).reshape(*shape, 7)

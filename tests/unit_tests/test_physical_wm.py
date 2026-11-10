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

"""Physical-state world model, its ReFL scorer and the env-side carry."""

import json
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.algorithms.refl_scorers import build_refl_scorer
from rlinf.models.embodiment.physical_wm.geometry import RobotFK
from rlinf.models.embodiment.physical_wm.model import (
    SCHEMAS,
    Carry,
    PhysicalWM,
    feature_floor,
    load_wm,
    pushcube_reward,
)
from rlinf.models.embodiment.physical_wm.refl_scorer import (
    CARRY_KEYS,
    CarryTracker,
    PhysicalWMScorer,
    decode_env_actions,
)


def _toy_asset():
    """Seven revolute joints about z with 0.1 m links; the TCP is link7."""
    links = "".join(f'<link name="link{i}"/>' for i in range(8))
    joints = "".join(
        f'<joint name="joint{i}" type="revolute"><parent link="link{i - 1}"/>'
        f'<child link="link{i}"/><origin xyz="0.1 0 0" rpy="0 0 0"/>'
        '<axis xyz="0 0 1"/><limit lower="-3" upper="3" effort="1" velocity="1"/>'
        "</joint>"
        for i in range(1, 8)
    )
    return {
        "urdf": f'<robot name="toy">{links}{joints}</robot>',
        "ee_link": "link7",
        "arm_joint_names": [f"joint{i}" for i in range(1, 8)],
        "base_pose": torch.eye(4).tolist(),
    }


def test_robot_fk_composes_urdf_joints():
    fk = RobotFK(_toy_asset())
    qpos = torch.zeros(2, 9)
    qpos[1, 0] = torch.pi / 2
    pose = fk(qpos)
    torch.testing.assert_close(pose[0], torch.tensor([0.7, 0, 0, 1, 0, 0, 0]))
    half = 2**-0.5
    torch.testing.assert_close(
        pose[1], torch.tensor([0.1, 0.6, 0, half, 0, 0, half]), atol=1e-6, rtol=0
    )
    assert not {"origins", "axes", "tail"} & set(fk.state_dict())


def _wm():
    floor = feature_floor(SCHEMAS["pushcube"])
    wm = PhysicalWM("pushcube", _toy_asset(), torch.zeros_like(floor), floor, width=32)
    # A trained WM has a nonzero residual head; the zero init has no action path.
    torch.manual_seed(0)
    torch.nn.init.normal_(wm.continuous.weight, std=0.05)
    return wm


def _state(batch):
    """PushCube states at rest: random arm joints, the cube on the table."""
    state = torch.zeros(batch, 41)
    generator = torch.Generator().manual_seed(1)
    state[:, :9] = 0.1 * torch.randn(batch, 9, generator=generator)
    state[:, 21] = 1  # TCP quaternion
    state[:, 25:29] = torch.tensor([0.0, 0.0, 0.02, 1.0])  # cube pose
    state[:, 32:35] = torch.tensor([0.3, 0.0, 0.0])  # goal
    return state


def test_pushcube_reward_matches_native_terms():
    state = _state(2)
    # Env 0: TCP at the push point and the cube on the goal: success.
    state[0, 25:28] = torch.tensor([0.3, 0.0, 0.02])
    state[0, 18:21] = torch.tensor([0.3 - 0.025, 0.0, 0.02])
    # Env 1: TCP 0.2 m above the push point, the cube far from the goal.
    state[1, 18:21] = torch.tensor([-0.025, 0.0, 0.22])
    reward, success = pushcube_reward(state)
    assert success.tolist() == [1.0, 0.0]
    assert reward[0] == 1
    torch.testing.assert_close(reward[1], (1 - torch.tanh(torch.tensor(1.0))) / 4)


def test_world_model_step_keeps_kinematics_and_survives_checkpoint():
    wm = _wm()
    state = _state(3)
    carry = Carry(
        state,
        state.clone(),
        torch.zeros(3, 7),
        torch.ones(3, dtype=torch.bool),
        torch.zeros(3, 1),
    )
    nxt = wm(carry, torch.zeros(3, 7))
    torch.testing.assert_close(nxt.state[:, 18:25], wm.fk(nxt.state[:, :9]))
    torch.testing.assert_close(nxt.state[:, 28:32].norm(dim=-1), torch.ones(3))
    torch.testing.assert_close(nxt.previous, carry.state)
    assert not nxt.reset.any()
    torch.testing.assert_close(nxt.context, torch.full((3, 1), 1 / 50))
    torch.testing.assert_close(
        load_wm(wm.checkpoint())(carry, torch.zeros(3, 7)).state, nxt.state
    )


def _start(batch):
    """Scorer inputs at the first step of an episode, as the env sends them."""
    tracker = CarryTracker(episode_steps=50)
    tracker.reset(_state(batch))
    return tracker.observation(torch.zeros(batch, dtype=torch.long))


def test_decode_env_actions_unnormalizes_and_keeps_the_hard_gripper():
    samples = torch.zeros(1, 10, 32)
    samples[..., 0] = 3.0  # past q99: the controller saturates
    samples[..., 6] = 1.0  # raw gripper above 0.5: open
    q01, q99 = torch.full((7,), -0.5), torch.full((7,), 0.5)
    actions = decode_env_actions(samples, q01, q99)
    assert actions.shape == (1, 10, 7)
    torch.testing.assert_close(actions[..., 1], torch.zeros(1, 10), atol=1e-5, rtol=0)
    assert (actions[..., 0] == 1).all() and (actions[..., 6] == 1).all()


def test_scorer_gradient_reaches_only_the_executed_arm_actions(tmp_path):
    torch.save(_wm().checkpoint(), tmp_path / "wm.pt")
    stats = {"norm_stats": {"actions": {"q01": [-1.0] * 32, "q99": [1.0] * 32}}}
    (tmp_path / "norm_stats.json").write_text(json.dumps(stats))
    cfg = OmegaConf.create(
        {
            "type": "physical_wm",
            "checkpoint_path": str(tmp_path / "wm.pt"),
            "norm_stats_path": str(tmp_path / "norm_stats.json"),
            "horizon": 5,
            "gamma": 0.99,
        }
    )
    scorer = build_refl_scorer(cfg, torch.device("cpu"))
    assert not any(p.requires_grad for p in scorer.parameters())
    samples = (0.1 * torch.randn(4, 10, 32)).requires_grad_()
    rewards = scorer(samples, _start(4))
    assert rewards.shape == (4,) and torch.isfinite(rewards).all()
    rewards.sum().backward()
    grad = samples.grad
    assert grad[:, :5, :6].abs().sum() > 0
    assert grad[:, :, 6].abs().sum() == 0  # hard gripper threshold
    assert grad[:, 5:].abs().sum() == 0  # past the imagination horizon
    assert grad[..., 7:].abs().sum() == 0  # padded model action dims


def test_scorer_rejects_a_malformed_carry():
    scorer = PhysicalWMScorer(_wm(), -torch.ones(7), torch.ones(7), 5, 0.99)
    obs = _start(2)
    obs["refl_context"] = torch.zeros(2)
    with pytest.raises(ValueError, match="elapsed episode fraction"):
        scorer(torch.zeros(2, 10, 32), obs)


def test_carry_tracker_follows_steps_and_partial_resets():
    tracker = CarryTracker(episode_steps=50)
    s0 = torch.zeros(2, 41)
    tracker.reset(s0)
    obs = tracker.observation(torch.zeros(2))
    assert set(obs) == set(CARRY_KEYS.values())
    assert obs["refl_reset"].tolist() == [1.0, 1.0]

    tracker.step(s0 + 1, torch.full((2, 7), 2.0))
    obs = tracker.observation(torch.ones(2))
    torch.testing.assert_close(obs["refl_previous"], s0)
    torch.testing.assert_close(obs["refl_state"], s0 + 1)
    torch.testing.assert_close(obs["refl_previous_action"], torch.ones(2, 7))
    assert obs["refl_reset"].tolist() == [0.0, 0.0]
    torch.testing.assert_close(obs["refl_context"], torch.full((2, 1), 1 / 50))

    s1 = s0 + 1
    s1[1] = -5
    tracker.reset(s1, torch.tensor([1]))
    obs = tracker.observation(torch.tensor([1, 0]))
    assert obs["refl_reset"].tolist() == [0.0, 1.0]
    torch.testing.assert_close(obs["refl_previous"][0], s0[0])
    torch.testing.assert_close(obs["refl_previous"][1], torch.full((41,), -5.0))
    torch.testing.assert_close(obs["refl_previous_action"][0], torch.ones(7))
    assert obs["refl_previous_action"][1].abs().sum() == 0


def test_pushcube_refl_config_composes(monkeypatch):
    from hydra import compose, initialize_config_dir

    repo = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(repo / "examples/embodiment"))
    with initialize_config_dir(
        config_dir=str(repo / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(config_name="maniskill_refl_openpi_pi05_pushcube")
        OmegaConf.resolve(cfg)
    assert cfg.algorithm.loss_type == "embodied_refl"
    assert cfg.algorithm.refl_margin is None
    assert cfg.actor.refl_reward_model.type == "physical_wm"
    assert cfg.actor.model.openpi.refl_late_fraction == 0.5
    assert cfg.actor.model.num_steps == 4
    assert cfg.env.train.refl_physical_state and cfg.env.eval.refl_physical_state
    assert cfg.env.train.init_params.id == "PushCubeReFL-v1"

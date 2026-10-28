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

"""ReFL gradient, window-boundary and source-provenance contracts."""

import subprocess

import pytest
import torch
from torch import nn

from rlinf.algorithms.refl import (
    TransitionWindows,
    frozen_parameters,
    imagined_objective,
    isolated_rng,
    mixed_windows,
    score_native_reward,
    validate_source_columns,
    world_model_batch,
)
from rlinf.data.refl_replay import ReplayBuffer, Windows, record_online
from rlinf.runners.refl_runner import assert_clean_source


class LinearDynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))

    def step(self, state, action, **context):
        return state + self.scale * action


class Reward:
    def __init__(self, terminal=False):
        self.terminal = terminal

    def __call__(self, state):
        return state.squeeze(-1)

    def terminated(self, state):
        return torch.full(state.shape[:-1], self.terminal, device=state.device)


def start():
    return {
        k: torch.ones(1, 4, 1)
        for k in ("state", "previous_state", "previous_action", "reset")
    }


def test_pathwise_gradient_and_frozen_weights():
    actor = nn.Linear(1, 1, bias=False)
    actor.weight.data.fill_(0.1)
    wm = LinearDynamics()
    with frozen_parameters(wm):
        objective = imagined_objective(actor, wm, Reward(), start(), 3, 0.8)
        objective.backward()
        analytic = actor.weight.grad.item()
        with torch.no_grad():
            actor.weight.add_(1e-3)
            plus = imagined_objective(actor, wm, Reward(), start(), 3, 0.8)
            actor.weight.sub_(2e-3)
            minus = imagined_objective(actor, wm, Reward(), start(), 3, 0.8)
            actor.weight.add_(1e-3)
        assert analytic > 0
        assert analytic == pytest.approx(float((plus - minus) / 2e-3), rel=1e-3)
        assert wm.scale.grad is None
        assert not wm.scale.requires_grad
    assert wm.scale.requires_grad


def test_terminal_reward_counts_once_and_no_bootstrap():
    actor = nn.Linear(1, 1, bias=False)
    actor.weight.data.fill_(0.1)
    objective = imagined_objective(
        actor, LinearDynamics(), Reward(True), start(), 3, 0.8
    )
    assert objective.item() == pytest.approx(1.2)
    objective.backward()
    assert actor.weight.grad.item() == pytest.approx(2.0)


@pytest.mark.parametrize("horizon", [3, 13])
def test_completion_cost_prefers_immediate_success_to_waiting(horizon):
    class Schedule:
        def __init__(self, delay):
            self.delay, self.step_count = delay, 0

        def step(self, state, action, **context):
            done = self.step_count >= self.delay
            self.step_count += 1
            return state.new_tensor([1.0 if done else 0.55, float(done)]).expand_as(
                state
            )

    class ScheduledReward:
        def __call__(self, state):
            return state[:, 0]

        def terminated(self, state):
            return state[:, 1].bool()

    initial = {k: torch.zeros(1, 4, 2) for k in start()}

    def actor(state):
        return torch.zeros_like(state[:, :1])

    values = [
        imagined_objective(
            actor,
            Schedule(delay),
            ScheduledReward(),
            initial,
            horizon,
            0.8,
            "completion_cost_v1",
        ).item()
        for delay in [0, 1, 2, horizon]
    ]
    assert values[0] == 0.0
    assert all(a > b for a, b in zip(values, values[1:]))
    old = [
        imagined_objective(
            actor, Schedule(delay), ScheduledReward(), initial, horizon, 0.8
        ).item()
        for delay in [0, 1]
    ]
    assert old == pytest.approx([1.0, 1.35])


def test_completion_cost_changes_gradient_not_only_displayed_score():
    class Dynamics:
        def step(self, state, action, **context):
            return state + torch.cat([action, torch.zeros_like(action)], -1)

    class SumReward:
        def __init__(self, offset=0.0):
            self.offset = offset

        def __call__(self, state):
            return state.sum(-1) + self.offset

        def terminated(self, state):
            return torch.zeros(len(state), dtype=torch.bool)

    actor = nn.Linear(2, 1, bias=False)
    actor.weight.data.fill_(0.1)
    initial = {k: torch.tensor([[[0.2, 0.0], [0.0, 0.6]]]) for k in start()}

    def objective(rule="native", offset=0.0):
        return imagined_objective(
            actor, Dynamics(), SumReward(offset), initial, 1, 0.8, rule
        )

    native_gradient = torch.autograd.grad(objective(), actor.weight)[0]
    shifted_gradient = torch.autograd.grad(objective(offset=-1), actor.weight)[0]
    cost_gradient = torch.autograd.grad(objective("completion_cost_v1"), actor.weight)[
        0
    ]
    torch.testing.assert_close(shifted_gradient, native_gradient)
    assert not torch.allclose(
        cost_gradient / cost_gradient.norm(), native_gradient / native_gradient.norm()
    )
    for i in range(2):
        with torch.no_grad():
            actor.weight[0, i] += 1e-3
            plus = objective("completion_cost_v1")
            actor.weight[0, i] -= 2e-3
            minus = objective("completion_cost_v1")
            actor.weight[0, i] += 1e-3
        assert cost_gradient[0, i].item() == pytest.approx(
            ((plus - minus) / 2e-3).item(), rel=2e-3, abs=1e-4
        )


@pytest.mark.parametrize("value", [-0.01, 1.01, float("nan")])
def test_completion_cost_rejects_invalid_native_score(value):
    with pytest.raises(ValueError, match="native rewards in"):
        score_native_reward(torch.tensor([value]), "completion_cost_v1")


def test_completion_cost_is_bounded_and_improves_with_progress():
    rewards = torch.tensor([0.0, 0.2, 0.55, 0.8, 1.0])
    scores = score_native_reward(rewards, "completion_cost_v1")
    assert ((scores >= -1) & (scores <= 0)).all()
    assert (scores[1:] > scores[:-1]).all()
    assert scores[-1] == 0
    with pytest.raises(ValueError, match="Unknown"):
        score_native_reward(rewards, "unknown")


def test_freeze_restores_flags_on_exception():
    model = nn.Linear(1, 1)
    model.bias.requires_grad_(False)
    with pytest.raises(RuntimeError), frozen_parameters(model):
        raise RuntimeError("test")
    assert model.weight.requires_grad
    assert not model.bias.requires_grad


def test_native_reward_soft_grasp_keeps_input_gradient():
    pytest.importorskip("mani_skill")
    from mani_skill.agents.robots.panda.panda import Panda
    from mani_skill.envs.tasks.tabletop.pick_cube import PickCubeEnv

    from rlinf.envs.sim.maniskill.refl_pickcube import NativeRewardOracle

    oracle = NativeRewardOracle.__new__(NativeRewardOracle)
    oracle.task, oracle.static = PickCubeEnv, Panda.is_static
    oracle.goal_thresh, oracle.robot_uids = 0.025, "panda"
    state = torch.zeros(3, 48)
    state[:, 26] = 0.2
    state[:, 29] = 0.04
    state[:, 18] = torch.tensor([0.0, 0.3, 1.0])
    state.requires_grad_()
    reward = oracle(state)
    torch.testing.assert_close(reward[1], 0.7 * reward[0] + 0.3 * reward[2])
    reward[1].backward()
    assert state.grad[1, 18] > 0
    assert torch.isfinite(state.grad).all()


def transitions():
    time, envs = 6, 2
    state = torch.arange(time * envs * 48).reshape(time, envs, 48).float()
    return {
        "state": state,
        "next_state": state + 1,
        "action": torch.ones(time, envs, 8),
        "previous_state": state - 1,
        "previous_action": torch.zeros(time, envs, 8),
        "reset": torch.zeros(time, envs, dtype=torch.bool),
        "alive": torch.ones(time, envs, dtype=torch.bool),
        "terminated": torch.zeros(time, envs, dtype=torch.bool),
        "truncated": torch.zeros(time, envs, dtype=torch.bool),
        "episode_id": torch.zeros(time, envs, dtype=torch.long),
    }


@pytest.mark.parametrize(
    "field", ["terminated", "truncated", "reset", "alive", "episode_id"]
)
def test_windows_reject_boundaries_and_preserve_split(field):
    data = transitions()
    if field in ("terminated", "truncated"):
        data[field][2, 0] = True
    elif field == "reset":
        data[field][3, 0] = True
    elif field == "episode_id":
        data[field][3:, 0] = 1
    else:
        data[field][3, 0] = False
    windows = TransitionWindows(data, torch.tensor([0]), 3)
    assert torch.all(windows.envs == 0)
    assert (
        1 not in windows.starts.tolist()
        if field in ("terminated", "truncated")
        else 2 not in windows.starts.tolist()
    )
    sample = windows.sample(16)
    batch = world_model_batch(sample)
    torch.testing.assert_close(batch.a_exec, sample["action"])
    torch.testing.assert_close(batch.states[1:], sample["next_state"])
    assert batch.xi.shape == (16, 57)


def test_dead_windows_fail_and_mixture_keeps_batch_axes():
    data = transitions()
    first = TransitionWindows(data, torch.tensor([0]), 3)
    second = TransitionWindows(data, torch.tensor([1]), 3)
    assert mixed_windows(first, second, 10)["state"].shape == (3, 10, 48)
    data["alive"].zero_()
    with pytest.raises(ValueError, match="No live"):
        TransitionWindows(data, torch.tensor([0]), 3)


def test_source_identity_rejects_column_permutation():
    columns = torch.arange(4).expand(6, 4).clone()
    validate_source_columns(columns, 4)
    with pytest.raises(ValueError, match="declared split"):
        validate_source_columns(columns[:, [0, 2, 1, 3]], 4)


def test_diagnostics_and_evaluation_do_not_advance_training_rng():
    import random

    import numpy as np

    data = transitions()
    windows = TransitionWindows(data, torch.tensor([0]), 3)
    generator = torch.Generator().manual_seed(31000)
    torch_state = torch.get_rng_state().clone()
    windows.sample(512, generator)
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    with pytest.raises(RuntimeError), isolated_rng("cpu"):
        torch.rand(100)
        random.random()
        np.random.random(100)
        raise RuntimeError("evaluation interrupted")
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
    assert random.getstate() == python_state
    np.testing.assert_equal(np.random.get_state(), numpy_state)


def test_original_ring_preserves_executed_actions_and_terminal_states():
    data = transitions()
    for t in range(6):
        data["state"][t, :, 0] = torch.tensor([t, 100 + t])
        data["next_state"][t] = data["state"][t] + 1
        data["action"][t] = data["state"][t, :, :1]
    data["clean_action"] = -data["action"]
    data["reward"] = torch.zeros(6, 2)
    data["terminated"][2, 0] = True
    data["alive"][3:, 0] = False
    data["truncated"][5, 1] = True
    data["reset"][0] = True
    replay = ReplayBuffer(2, 6, 48, 8, 57, "cpu")
    record_online(replay, data, torch.arange(2))
    assert replay.window_mask(3)[0, 0]
    batch = replay.sample_windows(512, 3)
    torch.testing.assert_close(batch.states[1:, :, 0], batch.states[:-1, :, 0] + 1)
    torch.testing.assert_close(batch.a_exec[:, :, 0], batch.states[:-1, :, 0])
    torch.testing.assert_close(batch.a_clean, -batch.a_exec)
    torch.testing.assert_close(batch.done.bool(), batch.states[1:, :, 0] == 3)
    starts = replay.sample_starts(256, 1)
    assert not torch.isin(starts.state[:, 0], torch.tensor([3.0, 106.0])).any()
    assert (batch.states[-1, :, 0] == 3).any()
    assert (batch.states[-1, :, 0] == 106).any()


def test_original_fixed_windows_include_terminal_target_not_reset_crossing():
    data = transitions()
    data["terminated"][2, 0] = True
    windows = Windows(data, torch.tensor([0]), 3)
    assert windows.starts.tolist() == [0, 3]


def test_bin_sampler_adapts_training_columns_only():
    pytest.importorskip("mani_skill")
    from rlinf.envs.sim.maniskill.refl_pickcube import BinSampler

    valid = torch.ones(3, 4, dtype=torch.bool)
    valid[2, 0] = False
    sampler = BinSampler(valid, torch.tensor([0, 20, 40]), gamma=0.99)
    weights, _ = sampler.weights()
    torch.testing.assert_close(weights.sum(0), torch.ones(4))
    assert weights[2, 0] == 0
    sampler.update(
        torch.ones(4, dtype=torch.long),
        torch.ones(4) * 50,
        torch.tensor([True, False, True, False]),
    )
    updated, _ = sampler.weights()
    torch.testing.assert_close(updated[:, [1, 3]], weights[:, [1, 3]])
    assert (updated[1, [0, 2]] < weights[1, [0, 2]]).all()


def test_provenance_rejects_dirty_ignored_and_wrong_commit(tmp_path):
    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(tmp_path), *args], text=True
        ).strip()

    git("init")
    (tmp_path / "code.py").write_text("x=1\n")
    (tmp_path / ".gitignore").write_text("cache/\n")
    git("add", ".")
    git(
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "test",
    )
    commit = git("rev-parse", "HEAD")
    assert assert_clean_source(tmp_path, commit)["head"] == commit
    with pytest.raises(ValueError, match="Expected commit"):
        assert_clean_source(tmp_path, "0" * 40)
    (tmp_path / "cache").mkdir()
    (tmp_path / "cache" / "runtime").write_text("cache")
    with pytest.raises(ValueError, match="ignored"):
        assert_clean_source(tmp_path, commit)
    (tmp_path / "cache" / "runtime").unlink()
    (tmp_path / "cache").rmdir()
    (tmp_path / "code.py").write_text("x=2\n")
    with pytest.raises(ValueError, match="modified"):
        assert_clean_source(tmp_path, commit)

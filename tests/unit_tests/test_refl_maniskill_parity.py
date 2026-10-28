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

"""Numerical parity against frozen56814f6 source, not a re-derived target formula."""

import ast
import copy
import hashlib
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from torch import nn

from rlinf.algorithms import refl_maniskill as port
from rlinf.data.refl_replay import ReplayBuffer, StartBatch, WindowBatch

REFERENCE_SHA256 = {
    "types.py.txt": "b5a7322b84a4dec1af7ea836d14a954d44b294442063be758a150b7bc604844a",
    "objective.py.txt": "fef8bab264efe858bfb9d8a70a1d7f7bebad318cab38e008c3a0e33c95a857de",
    "policy_trainer.py.txt": "dc8528cbd7434510d1f7d85d2d7c92831a2dd9e1fdee2b01a0c31bd8f39fcc21",
    "wm_trainer.py.txt": "dce7970e168247a8536ffddb429993715f354bbf903b350abd4659c9e9610a36",
    "distributed.py.txt": "ebf2d6d37ce6102f55941c0ab328c9deb4021964c46f7bd35a8bc1bb767f5a21",
    "single_supermimic.py.txt": "d5ac93e59c3ed8579e7e3ec0384c1470a25773e04270d52af7809691130e2d4a",
}


def reference():
    module = types.ModuleType("pinned_maniskill_reference")
    sys.modules[module.__name__] = module
    module.__dict__.update(
        torch=torch,
        nn=nn,
        dist=torch.distributed,
        dataclass=dataclass,
        ReplayBuffer=ReplayBuffer,
        WindowBatch=WindowBatch,
    )
    for filename, names in [
        ("distributed.py.txt", {"world_size", "allreduce_grads"}),
        ("types.py.txt", {"ImagRollout"}),
        ("objective.py.txt", {"policy_objective", "gae_targets"}),
        ("single_supermimic.py.txt", {"context_from_xi", "SingleImagination"}),
        ("policy_trainer.py.txt", {"PolicyTrainerCfg", "PolicyTrainer"}),
        ("wm_trainer.py.txt", {"WMTrainerCfg", "WorldModelTrainer"}),
    ]:
        path = Path(__file__).parent / "refl_reference" / filename
        assert (
            hashlib.sha256(path.read_bytes()).hexdigest() == REFERENCE_SHA256[filename]
        )
        tree = ast.parse(path.read_text())
        selected = [n for n in tree.body if getattr(n, "name", None) in names]
        future = ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        )
        tree = ast.fix_missing_locations(
            ast.Module(body=[future, *selected], type_ignores=[])
        )
        exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(42, 8)

    def raw(self, state):
        return self.net(state[..., :42])

    def forward(self, state):
        return self.raw(state).clamp(-1, 1)


class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(42, 1)

    def forward(self, state, goal=None, xi=None, e_root=None):
        return self.net(state[..., :42]).squeeze(-1)


class Dynamics(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Linear(8, 48)

    def step(self, state, action, **context):
        return state + 0.1 * self.net(action)

    def enable_compile(self, mode):
        assert mode is None

    def replay_loss(self, window):
        predicted = self.step(window.states[0], window.a_exec[0])
        loss = (predicted - window.states[1]).square().mean()
        return loss, {"state_loss": loss.detach()}


class Reward(nn.Module):
    def forward(self, state):
        return (state[..., 0].sigmoid(),)


def make_start():
    state = torch.randn(8, 48) * 0.1
    state[:, 0] = torch.linspace(-1, 1, 8)
    action = torch.zeros(8, 8)
    xi = torch.cat([state, action, torch.ones(8, 1)], -1)
    return StartBatch(
        state, xi, torch.zeros(8, dtype=torch.long), action, state[None], action[None]
    )


class Starts:
    def __init__(self, start):
        self.start = start

    def sample_starts(self, n, hist):
        assert (n, hist) == (8, 1)
        return self.start


def assert_parameters_same(a, b):
    for (ka, pa), (kb, pb) in zip(a.named_parameters(), b.named_parameters()):
        assert ka == kb
        torch.testing.assert_close(pa, pb, rtol=0, atol=0)
        if pa.grad is not None:
            torch.testing.assert_close(pa.grad, pb.grad, rtol=0, atol=0)


def assert_optimizer_same(a, b):
    a, b = a.state_dict(), b.state_dict()
    assert a["param_groups"] == b["param_groups"]
    for index, state in a["state"].items():
        for key, value in state.items():
            torch.testing.assert_close(value, b["state"][index][key], rtol=0, atol=0)


@pytest.mark.parametrize("use_critic", [False, True])
def test_full_actor_critic_update_matches_original(use_critic):
    ref = reference()
    torch.manual_seed(19)
    start = make_start()
    actor = Actor()
    critic = Critic() if use_critic else None
    model = port.SingleModel(Dynamics(), Reward())
    original_actor = copy.deepcopy(actor)
    original_model = copy.deepcopy(model)
    original_critic = copy.deepcopy(critic)

    def terminated(state):
        return state[..., 0] > 0.5

    new_rollout = port.SingleImagination(model, actor, terminated)
    old_rollout = ref.SingleImagination(original_model, original_actor, terminated)
    kw = {
        "horizon": 3,
        "batch_size": 8,
        "history": 1,
        "lr": 1e-4,
        "critic_lr": 1e-4,
        "gamma": 0.8,
        "gae_lambda": 0.95,
        "use_critic": use_critic,
        "obs_noise": False,
        "critic_epochs": 2,
        "critic_minibatches": 2,
        "compile": None,
    }
    new = port.PolicyTrainer(
        model,
        actor,
        critic,
        None,
        port.PolicyTrainerCfg(**kw),
        rollout_fn=new_rollout,
        critic_inputs_fn=new_rollout.critic_inputs,
    )
    old = ref.PolicyTrainer(
        original_model,
        original_actor,
        original_critic,
        None,
        ref.PolicyTrainerCfg(**kw),
        rollout_fn=old_rollout,
        critic_inputs_fn=old_rollout.critic_inputs,
    )
    for step in range(3):
        torch.manual_seed(100 + step)
        expected = old.update(Starts(start))
        torch.manual_seed(100 + step)
        actual = new.update(Starts(start))
        assert actual == pytest.approx(expected, rel=0, abs=0)
        assert_parameters_same(actor, original_actor)
        assert_parameters_same(model, original_model)
        assert_optimizer_same(new.opt, old.opt)
        if use_critic:
            assert_parameters_same(critic, original_critic)
            assert_optimizer_same(new.critic_opt, old.critic_opt)


def test_world_model_optimizer_matches_original():
    ref = reference()
    torch.manual_seed(9)
    model = Dynamics()
    original = copy.deepcopy(model)
    states = torch.randn(4, 8, 48)
    actions = torch.randn(3, 8, 8)
    batch = WindowBatch(
        states,
        actions,
        actions,
        torch.zeros(8, 57),
        torch.zeros(3, 8),
        torch.zeros(3, 8),
        torch.zeros(3, 8, dtype=torch.long),
    )

    class Buffer:
        def sample_windows(self, n, k):
            assert (n, k) == (8, 3)
            return batch

    kw = {
        "window": 3,
        "batches_per_update": 2,
        "batch_size": 8,
        "lr": 1e-4,
        "compile": None,
    }
    old = ref.WorldModelTrainer(
        original, None, ref.WMTrainerCfg(**kw), loss_fn=original.replay_loss
    )
    new = port.WorldModelTrainer(
        model, None, port.WMTrainerCfg(**kw), loss_fn=model.replay_loss
    )
    for _ in range(3):
        expected = old.update(Buffer())
        actual = new.update(Buffer())
        assert actual == pytest.approx(expected, rel=0, abs=0)
        assert_parameters_same(model, original)
        assert_optimizer_same(new.opt, old.opt)


def test_image_formula_and_lambda_targets_match_original():
    ref = reference()
    torch.manual_seed(2)
    rewards = torch.rand(13, 8, requires_grad=True)
    done = torch.zeros(13, 8)
    done[2:, 0] = 1
    done[7:, 1] = 1
    done[-1, 2] = 1
    survival = torch.cat([torch.ones(1, 8), torch.cumprod(1 - done, 0)])
    ro = types.SimpleNamespace(rewards=rewards, survival=survival)
    terminal = torch.randn(8, requires_grad=True)
    a = port.policy_objective(ro, terminal, 0.8)
    b = ref.policy_objective(ro, terminal, 0.8)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    ga = torch.autograd.grad(a, (rewards, terminal), retain_graph=True)
    gb = torch.autograd.grad(b, (rewards, terminal))
    for x, y in zip(ga, gb):
        torch.testing.assert_close(x, y, rtol=0, atol=0)
    assert ga[1][:3].eq(0).all()
    values = torch.randn(14, 8)
    torch.testing.assert_close(
        port.gae_targets(rewards, done, values, 0.8, 0.95),
        ref.gae_targets(rewards, done, values, 0.8, 0.95),
        rtol=0,
        atol=0,
    )


def test_real_checkpoint_gpu_parity():
    """Run only with an external committed-run config on an allocated GPU."""
    import json
    import os

    from rlinf.data.refl_replay import WarmupReplay, Windows
    from rlinf.envs.sim.maniskill.refl_pickcube import NativeRewardOracle
    from rlinf.models.embodiment.refl_pickcube import (
        SingleActor,
        SingleCritic,
        load_world_model,
        set_precision,
    )
    from rlinf.runners.refl_runner import prepare_run, sha256, write_json

    config_path = os.environ.get("REFL_PARITY_CONFIG")
    if not config_path:
        pytest.skip("Set REFL_PARITY_CONFIG for the real-checkpoint GPU parity gate")
    root = Path(__file__).resolve().parents[2]
    cfg = prepare_run(json.loads(Path(config_path).read_text()), root)
    output = Path(cfg["output_dir"])
    passed = False
    try:
        ref = reference()
        torch.manual_seed(cfg["seed"])
        torch.set_float32_matmul_precision("highest")
        device = "cuda"
        loaded = {
            k: torch.load(v["path"], map_location=device, weights_only=False)
            for k, v in cfg["inputs"].items()
        }
        oracle = NativeRewardOracle()
        model = port.SingleModel(
            load_world_model(loaded["world_model"], device),
            port.NativeRewardModule(oracle),
        )
        actor = SingleActor().to(device)
        actor.net.load_state_dict(loaded["actor"]["actor_mean"])
        critic = SingleCritic().to(device)
        critic.net.load_state_dict(loaded["actor"]["critic"])
        for module in (model, actor, critic):
            set_precision(module, cfg["precision"])
        windows = Windows(
            loaded["dataset"]["data"], torch.arange(3584, device=device), 3
        )
        batch = windows.sample(8, torch.Generator(device=device).manual_seed(31415))
        state = batch["state"][0]
        action = batch["previous_action"][0]
        xi = torch.cat(
            [batch["previous_state"][0], action, batch["reset"][0, :, None].float()], -1
        )
        start = StartBatch(
            state,
            xi,
            torch.zeros(8, dtype=torch.long, device=device),
            action,
            state[None],
            action[None],
        )
        old_model = copy.deepcopy(model)
        old_actor = copy.deepcopy(actor)
        old_critic = copy.deepcopy(critic)
        new_rollout = port.SingleImagination(model, actor, oracle.terminated)
        old_rollout = ref.SingleImagination(old_model, old_actor, oracle.terminated)
        kw = {
            "horizon": 13,
            "batch_size": 8,
            "history": 1,
            "gamma": 0.8,
            "gae_lambda": 0.95,
            "lr": 1e-4,
            "critic_lr": 1e-4,
            "use_critic": True,
            "critic_epochs": 5,
            "critic_minibatches": 4,
            "obs_noise": False,
            "compile": None,
        }
        new = port.PolicyTrainer(
            model,
            actor,
            critic,
            None,
            port.PolicyTrainerCfg(**kw),
            rollout_fn=new_rollout,
            critic_inputs_fn=new_rollout.critic_inputs,
        )
        old = ref.PolicyTrainer(
            old_model,
            old_actor,
            old_critic,
            None,
            ref.PolicyTrainerCfg(**kw),
            rollout_fn=old_rollout,
            critic_inputs_fn=old_rollout.critic_inputs,
        )
        reports = []
        for update in range(3):
            torch.manual_seed(123 + update)
            expected = old.update(Starts(start))
            torch.manual_seed(123 + update)
            actual = new.update(Starts(start))
            assert actual == pytest.approx(expected, rel=0, abs=0)
            assert_parameters_same(actor, old_actor)
            assert_parameters_same(critic, old_critic)
            assert_parameters_same(model, old_model)
            assert_optimizer_same(new.opt, old.opt)
            assert_optimizer_same(new.critic_opt, old.critic_opt)
            reports.append(actual)
        kw = {"window": 3, "batches_per_update": 2, "batch_size": 8, "lr": 1e-4, "compile": None}
        new_wm = port.WorldModelTrainer(
            model.dynamics,
            None,
            port.WMTrainerCfg(**kw),
            loss_fn=model.dynamics.replay_loss,
        )
        old_wm = ref.WorldModelTrainer(
            old_model.dynamics,
            None,
            ref.WMTrainerCfg(**kw),
            loss_fn=old_model.dynamics.replay_loss,
        )
        for update in range(3):
            torch.manual_seed(321 + update)
            a = old_wm.update(WarmupReplay(windows))
            torch.manual_seed(321 + update)
            b = new_wm.update(WarmupReplay(windows))
            assert a == pytest.approx(b, rel=0, abs=0)
            assert_parameters_same(model, old_model)
            assert_optimizer_same(new_wm.opt, old_wm.opt)
        write_json(
            output / "parity.json",
            {
                "passed": True,
                "reference_commit": "56814f65a9916d5fcbba46018858fce63cc99fa1",
                "neural_precision": cfg["precision"],
                "actor_critic_updates": 3,
                "critic_updates": 60,
                "wm_updates": 6,
                "exact_parameter_gradient_optimizer_equality": True,
                "metrics": reports,
            },
        )
        passed = True
    finally:
        write_json(
            output / "completion.json",
            {
                "completed": passed,
                "exit_code": 0 if passed else 1,
                "evidence": "real-checkpoint GPU parity",
                "start_sha256": sha256(output / "start.json"),
            },
        )

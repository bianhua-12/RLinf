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

import asyncio
import os
import queue
import sys

from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import rlinf.runners.async_embodied_runner as async_runner_module
import rlinf.runners.embodied_runner as embodied_runner_module
from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner


class _FakeHandle:
    def __init__(self, result=None, durations=None, per_rank_durations=None):
        self._result = result
        self._durations = durations or {}
        self._per_rank_durations = per_rank_durations or []
        self.wait_calls = 0

    def wait(self):
        self.wait_calls += 1
        return self._result

    def consume_durations(self, return_per_rank=False):
        if return_per_rank:
            return self._durations, self._per_rank_durations
        return self._durations

    def done(self):
        return True


class _FakeChannel:
    def __init__(self, name: str, maxsize: int = 0):
        self.name = name
        self.maxsize = maxsize

    def get_nowait(self):
        raise asyncio.QueueEmpty

    def qsize(self):
        return 0


class _FakeMetricLogger:
    def __init__(self, cfg):
        del cfg
        self.logged = []

    def log(self, data, step, **kwargs):
        self.logged.append((data, step, kwargs))

    def finish(self):
        return None


class _FakeLogger:
    def info(self, *args, **kwargs):
        del args, kwargs

    def warning(self, *args, **kwargs):
        del args, kwargs


class _FakeActor:
    worker_group_name = "ActorGroup"

    def __init__(self):
        self.sync_model_to_rollout_calls = 0
        self.recv_rollout_trajectories_calls = []
        self.recv_rollout_handle = _FakeHandle()
        self.run_training_handle = _FakeHandle(
            result=[{"sac/critic_loss": 0.1}],
            durations={"run_training": 0.0},
            per_rank_durations=[],
        )

    def sync_model_to_rollout(self):
        self.sync_model_to_rollout_calls += 1
        return _FakeHandle()

    def recv_rollout_trajectories(self, input_channel):
        self.recv_rollout_trajectories_calls.append(input_channel)
        return self.recv_rollout_handle

    def run_training(self):
        return self.run_training_handle

    def stop(self):
        return _FakeHandle()


class _FakeRollout:
    worker_group_name = "RolloutGroup"

    def __init__(self):
        self.sync_model_from_actor_calls = 0
        self.generate_calls = []

    def sync_model_from_actor(self):
        self.sync_model_from_actor_calls += 1
        return _FakeHandle()

    def generate(self, input_channel, output_channel, metric_channel):
        self.generate_calls.append(
            {
                "input_channel": input_channel,
                "output_channel": output_channel,
                "metric_channel": metric_channel,
            }
        )
        return _FakeHandle()

    def stop(self):
        return _FakeHandle()


class _FakeEnv:
    worker_group_name = "EnvGroup"

    def __init__(self):
        self.interact_calls = []

    def interact(
        self,
        input_channel,
        rollout_channel,
        reward_channel,
        actor_channel,
        metric_channel,
    ):
        self.interact_calls.append(
            {
                "input_channel": input_channel,
                "rollout_channel": rollout_channel,
                "reward_channel": reward_channel,
                "actor_channel": actor_channel,
                "metric_channel": metric_channel,
            }
        )
        return _FakeHandle()

    def stop(self):
        return _FakeHandle()


class _FakeReward:
    def __init__(self):
        self.compute_calls = []
        self.handle = _FakeHandle()

    def compute_rewards_async(self, input_channel, output_channel):
        self.compute_calls.append(
            {
                "input_channel": input_channel,
                "output_channel": output_channel,
            }
        )
        return self.handle

    def stop(self):
        return _FakeHandle()


def _build_cfg():
    return OmegaConf.create(
        {
            "runner": {
                "task_type": "embodied",
                "max_epochs": 1,
                "max_steps": 1,
                "val_check_interval": -1,
                "save_interval": 100,
                "weight_sync_interval": 1,
                "logger": {
                    "log_path": "/tmp/rlinf-test",
                    "project_name": "rlinf",
                    "experiment_name": "async-sac-test",
                    "logger_backends": [],
                },
            },
            "actor": {
                "sync_weight_no_wait": False,
            },
            "algorithm": {
                "loss_type": "embodied_sac",
            },
        }
    )


def test_async_embodied_runner_starts_reward_worker_with_reward_channel(monkeypatch):
    created_channels = []

    def fake_channel_create(name: str, maxsize: int = 0, **kwargs):
        del kwargs
        channel = _FakeChannel(name, maxsize=maxsize)
        created_channels.append(channel)
        return channel

    monkeypatch.setattr(
        embodied_runner_module.Channel, "create", staticmethod(fake_channel_create)
    )
    monkeypatch.setattr(embodied_runner_module, "MetricLogger", _FakeMetricLogger)
    monkeypatch.setattr(embodied_runner_module, "get_logger", lambda: _FakeLogger())
    monkeypatch.setattr(
        async_runner_module,
        "check_progress",
        lambda *args, **kwargs: (False, False, False),
    )

    runner = AsyncEmbodiedRunner(
        cfg=_build_cfg(),
        actor=_FakeActor(),
        rollout=_FakeRollout(),
        env=_FakeEnv(),
        reward=_FakeReward(),
    )
    runner.print_metrics_table_async = lambda *args, **kwargs: None

    runner.run()

    assert runner.reward_channel is not None
    assert runner.reward_channel.name == "Reward"
    assert runner.reward.compute_calls[0]["input_channel"] is runner.reward_channel
    assert runner.env.interact_calls[0]["reward_channel"] is runner.reward_channel
    assert runner.actor.recv_rollout_handle.wait_calls >= 1

# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

import sys
from concurrent.futures import Future

import gymnasium as gym
import numpy as np

from examples.embodiment import franka_fold_pi05_client as client


def test_episode_timeout_defaults_to_120_seconds(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["franka_fold_pi05_client.py"])

    args = client.parse_args()

    assert args.episode_timeout_s == 120.0


def test_episode_timeout_terminates_as_failure(monkeypatch):
    class _NeverDoneEnv(gym.Env):
        def reset(self, *, seed=None, options=None):
            return _observation(), {}

        def step(self, action):
            return _observation(), 1.0, False, False, {}

    times = iter([10.0, 130.0])
    monkeypatch.setattr(client.time, "monotonic", lambda: next(times))
    env = client._EpisodeTimeoutWrapper(_NeverDoneEnv(), timeout_s=120.0)

    env.reset()
    _, reward, terminated, truncated, info = env.step(None)

    assert reward == 0.0
    assert terminated
    assert not truncated
    assert info["eval_result"] == "failure"
    assert info["success"] is False


def _observation():
    image = np.zeros((8, 9, 3), dtype=np.uint8)
    return {
        "state": {"proprio": np.zeros(client.ACTION_DIM, dtype=np.float32)},
        "frames": {
            "base_0_rgb": image,
            "left_wrist_0_rgb": image,
            "right_wrist_0_rgb": image,
        },
    }


def _response(delay=None):
    response = {
        "actions": np.zeros(
            (client.ACTION_HORIZON, client.ACTION_DIM), dtype=np.float32
        )
    }
    if delay is not None:
        response["training_rtc_delay_steps"] = delay
    return response


class _DelayedFuture:
    def __init__(self, env, ready_at, response):
        self.env = env
        self.ready_at = ready_at
        self.response = response

    def done(self):
        return self.env.steps >= self.ready_at

    def result(self):
        return self.response


class _Env:
    def __init__(self):
        self.steps = 0

    def reset(self):
        return _observation(), {}

    def step(self, action):
        self.steps += 1
        terminated = self.steps >= 15
        return _observation(), 0.0, terminated, False, {"eval_result": "success"}


class _Policy:
    def __init__(self, env):
        self.env = env
        self.observations = []

    def submit(self, observation):
        self.observations.append(observation)
        delay = observation.get("training_rtc_delay_steps")
        if len(self.observations) == 2:
            return _DelayedFuture(self.env, 13, _response(delay))
        future = Future()
        future.set_result(_response(delay))
        return future


def test_run_policy_discards_late_rtc_response_and_retries():
    env = _Env()
    policy = _Policy(env)

    result = client.run_policy(env, policy, client.DEFAULT_TASK)

    assert result == "success"
    assert len(policy.observations) >= 3
    assert policy.observations[1]["training_rtc_delay_steps"] == 10
    assert policy.observations[2]["training_rtc_delay_steps"] == 10


class _TakeoverEnv(_Env):
    def __init__(self):
        super().__init__()
        self.release_count = 0

    def get_wrapper_attr(self, name):
        return getattr(self, name)

    def get_hold_action(self):
        return np.zeros(client.ACTION_DIM, dtype=np.float32)

    def release_to_policy(self):
        self.release_count += 1

    def step(self, action):
        del action
        self.steps += 1
        active = self.steps in (3, 4)
        takeover = 3 <= self.steps <= 6
        terminated = self.steps >= 8
        return (
            _observation(),
            0.0,
            terminated,
            False,
            {
                "eval_result": "success" if terminated else None,
                "pico_active": active,
                "pico_ready": True,
                "pico_takeover": takeover,
            },
        )


class _TakeoverPolicy:
    def __init__(self, env):
        self.env = env
        self.observations = []

    def submit(self, observation):
        self.observations.append(observation)
        delay = observation.get("training_rtc_delay_steps")
        response = _response(delay)
        if len(self.observations) == 2:
            return _DelayedFuture(self.env, 4, response)
        future = Future()
        future.set_result(response)
        return future


def test_run_policy_reinfers_from_hold_after_pico_takeover():
    env = _TakeoverEnv()
    policy = _TakeoverPolicy(env)

    result = client.run_policy(env, policy, client.DEFAULT_TASK)

    assert result == "success"
    assert env.release_count == 1
    resume_observation = policy.observations[-1]
    assert resume_observation["training_rtc_delay_steps"] == 10
    np.testing.assert_array_equal(
        resume_observation["training_rtc_action_prefix"][:10],
        np.zeros((10, client.ACTION_DIM)),
    )

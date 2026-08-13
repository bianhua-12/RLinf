# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

from concurrent.futures import Future

import numpy as np

from examples.embodiment import franka_fold_pi05_client as client


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

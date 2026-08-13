# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

from concurrent.futures import Future
from io import StringIO

import numpy as np

from examples.embodiment import franka_fold_gr00t_client as client


def _observation():
    image = np.zeros((8, 9, 3), dtype=np.uint8)
    return {
        "state": {"proprio": np.zeros(16, dtype=np.float32)},
        "frames": {
            "base_0_rgb": image,
            "left_wrist_0_rgb": image,
            "right_wrist_0_rgb": image,
        },
    }


def _response(delay=None):
    actions = {
        key: np.zeros((1, client.ACTION_HORIZON, width), dtype=np.float32)
        for key, width in client.ACTION_LAYOUT
    }
    info = {} if delay is None else {"training_rtc_delay_steps": delay}
    return actions, info


class _DelayedFuture:
    def __init__(self, env, response):
        self.env = env
        self.response = response

    def done(self):
        return self.env.steps >= 4

    def result(self):
        return self.response


class _Env:
    def __init__(self):
        self.steps = 0
        self.release_count = 0
        self.hold_action = np.arange(16, dtype=np.float32) / 100.0

    def reset(self):
        return _observation(), {}

    def get_wrapper_attr(self, name):
        return getattr(self, name)

    def get_hold_action(self):
        return self.hold_action.copy()

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


class _Policy:
    def __init__(self, env):
        self.env = env
        self.requests = []

    def submit(self, observation, options=None):
        self.requests.append((observation, options))
        delay = None if options is None else options["training_rtc_delay_steps"]
        response = _response(delay)
        if len(self.requests) == 2:
            return _DelayedFuture(self.env, response)
        future = Future()
        future.set_result(response)
        return future


def test_run_policy_reinfers_from_hold_after_pico_takeover():
    env = _Env()
    policy = _Policy(env)

    result = client.run_policy(env, policy, client.DEFAULT_TASK, StringIO())

    assert result == "success"
    assert env.release_count == 1
    resume_options = policy.requests[2][1]
    assert resume_options["training_rtc_delay_steps"] == 10
    np.testing.assert_array_equal(
        client.flatten_actions(resume_options["training_rtc_action_prefix"]),
        np.repeat(env.hold_action[None, :], 10, axis=0),
    )

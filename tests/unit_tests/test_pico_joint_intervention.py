# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from rlinf.envs.realworld.common.wrappers.pico_joint_intervention import (
    DualFrankaJointPicoIntervention,
)


class _Env(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(-10.0, 10.0, shape=(16,), dtype=np.float32)
        self.joints = np.zeros((2, 7), dtype=np.float32)
        self.last_action = None

    def get_joint_positions(self):
        return self.joints.copy()

    def reset(self, *, seed=None, options=None):
        del seed, options
        self.joints.fill(0.0)
        return {}, {}

    def step(self, action):
        self.last_action = np.asarray(action, dtype=np.float32).copy()
        self.joints[0] = self.last_action[:7]
        self.joints[1] = self.last_action[8:15]
        return {}, 0.0, False, False, {}


class _Expert:
    def __init__(self, responses):
        self.responses = list(responses)
        self.index = 0
        self.current_gripper_actions = []

    def get_action(
        self,
        tcp_pose,
        action_scale,
        *,
        gripper_enabled,
        direct,
        current_gripper_action,
    ):
        del tcp_pose, action_scale, gripper_enabled
        assert direct
        self.current_gripper_actions.append(current_gripper_action)
        response = self.responses[min(self.index, len(self.responses) - 1)]
        self.index += 1
        return response

    def stop(self):
        pass


class _Kinematics:
    def compute(self, joints):
        del joints
        return np.array([0.0, 0.0, 0.174, 0.0, 0.0, 0.0, 1.0]), np.eye(6, 7)


def _response(*, active=False, ready=True, calibrated=True, valid=True, action=None):
    if action is None:
        action = np.zeros(7, dtype=np.float32)
    return (
        np.asarray(action, dtype=np.float32),
        active,
        {
            "pico_active": active,
            "pico_ready": ready,
            "pico_calibrated": calibrated,
            "pico_invalid_pose": not valid,
        },
    )


def _wrapper(left_responses, right_responses, **kwargs):
    wrapper = DualFrankaJointPicoIntervention(
        _Env(),
        experts={
            "left": _Expert(left_responses),
            "right": _Expert(right_responses),
        },
        **kwargs,
    )
    wrapper._kinematics = _Kinematics()
    return wrapper


def test_reset_waits_for_ready_released_controllers_without_stepping_robot():
    wrapper = _wrapper([_response()], [_response()])

    wrapper.reset()

    assert wrapper.env.last_action is None
    assert wrapper._ready == {"left": True, "right": True}


def test_reset_rejects_stale_pico_startup():
    wrapper = _wrapper(
        [_response(ready=False)],
        [_response(ready=False)],
        ready_timeout_s=0.01,
    )

    with pytest.raises(TimeoutError, match="startup timed out"):
        wrapper.reset()


def test_takeover_holds_both_arms_until_fresh_policy_release():
    tcp_error = np.array([0.01] * 6 + [0.0], dtype=np.float32)
    wrapper = _wrapper(
        [_response(active=True, action=tcp_error), _response(), _response()],
        [_response(), _response(), _response()],
    )
    policy_action = np.full(16, 0.5, dtype=np.float32)

    _, _, _, _, active_info = wrapper.step(policy_action)
    assert active_info["pico_active"]
    assert active_info["pico_takeover"]
    expected_left = np.array([0.01 / (1.0 + 0.05**2)] * 6 + [0.0])
    np.testing.assert_allclose(wrapper.env.last_action[:7], expected_left)
    np.testing.assert_allclose(wrapper.env.last_action[8:15], 0.0)

    _, _, _, _, released_info = wrapper.step(policy_action)
    assert not released_info["pico_active"]
    assert released_info["pico_takeover"]
    np.testing.assert_allclose(wrapper.env.last_action[:7], expected_left)
    np.testing.assert_allclose(wrapper.env.last_action[8:15], 0.0)

    resume_action = wrapper.env.last_action.copy()
    resume_action[:7] += 0.02
    resume_action[8:15] += 0.02
    wrapper.release_to_policy()
    wrapper.step(resume_action)
    np.testing.assert_allclose(wrapper.env.last_action[:7], expected_left + 0.02)
    np.testing.assert_allclose(wrapper.env.last_action[8:15], 0.02)


def test_stale_pico_data_exits():
    wrapper = _wrapper(
        [_response(active=True), _response(ready=False)],
        [_response(), _response(ready=False)],
    )
    wrapper.step(np.zeros(16, dtype=np.float32))

    with pytest.raises(RuntimeError, match="PICO left controller data timed out"):
        wrapper.step(np.zeros(16, dtype=np.float32))


def test_takeover_uses_model_kinematics():
    wrapper = _wrapper(
        [_response(active=True, action=np.array([0.01] + [0.0] * 6))],
        [_response()],
    )
    wrapper.step(np.zeros(16, dtype=np.float32))

    assert wrapper.env.last_action[0] > 0.0


def test_takeover_passes_previous_gripper_targets_as_relative_baselines():
    wrapper = _wrapper(
        [_response(), _response(active=True, action=np.zeros(7))],
        [_response(), _response(active=True, action=np.zeros(7))],
    )
    policy_action = np.zeros(16, dtype=np.float32)
    policy_action[[7, 15]] = [0.4, -0.3]
    wrapper.step(policy_action)

    next_policy_action = policy_action.copy()
    next_policy_action[[7, 15]] = [-0.9, 0.8]
    wrapper.step(next_policy_action)

    assert wrapper.experts["left"].current_gripper_actions == pytest.approx([0.0, 0.4])
    assert wrapper.experts["right"].current_gripper_actions == pytest.approx(
        [0.0, -0.3]
    )


def test_invalid_pico_pose_keeps_takeover_latched():
    wrapper = _wrapper(
        [_response(active=True), _response(valid=False)],
        [_response(), _response()],
    )
    wrapper.step(np.zeros(16, dtype=np.float32))
    _, _, _, _, info = wrapper.step(np.zeros(16, dtype=np.float32))

    assert info["pico_takeover"]
    assert not info["pico_ready"]
    with pytest.raises(ValueError, match="stale"):
        wrapper.release_to_policy()

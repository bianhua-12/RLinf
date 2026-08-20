# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

import sys
from argparse import Namespace
from concurrent.futures import Future
from types import ModuleType

import gymnasium as gym
import numpy as np

from examples.embodiment import franka_fold_pi05_cfgrl_server as cfgrl_server
from examples.embodiment import franka_fold_pi05_client as client


def test_episode_timeout_defaults_to_120_seconds(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["franka_fold_pi05_client.py"])

    args = client.parse_args()

    assert args.episode_timeout_s == 120.0
    assert args.base_camera_type == "realsense"
    assert args.base_camera_serial == "327122078534"


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


def test_validate_server_metadata_accepts_baseline_and_cfgrl():
    for config_name in client.SUPPORTED_SERVER_CONFIGS:
        client.validate_server_metadata(
            {
                "config_name": config_name,
                "action_horizon": client.ACTION_HORIZON,
                "action_dim": client.ACTION_DIM,
            }
        )


def test_validate_server_metadata_rejects_wrong_action_contract():
    with np.testing.assert_raises_regex(RuntimeError, "action contract mismatch"):
        client.validate_server_metadata(
            {
                "config_name": "pi05_franka_fold_recap_cfgrl",
                "action_horizon": 20,
                "action_dim": client.ACTION_DIM,
            }
        )


def test_cfgrl_server_rewrites_prompt_to_one_positive_condition():
    class _Delegate:
        metadata = {"config_name": "delegate"}

        def __init__(self):
            self.observation = None

        def infer(self, observation):
            self.observation = observation
            return {"actions": np.zeros((client.ACTION_HORIZON, client.ACTION_DIM))}

    delegate = _Delegate()
    policy = cfgrl_server.PositiveConditionPolicy(delegate)
    observation = {"prompt": client.DEFAULT_TASK, "state": np.zeros(client.ACTION_DIM)}

    policy.infer(observation)

    assert observation["prompt"] == client.DEFAULT_TASK
    assert delegate.observation["prompt"] == (
        "fold the clothes\nAdvantage: positive"
    )
    assert cfgrl_server.positive_condition_prompt(
        delegate.observation["prompt"]
    ) == delegate.observation["prompt"]


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
        self.action_space = gym.spaces.Box(
            -np.ones(client.ACTION_DIM, dtype=np.float32),
            np.ones(client.ACTION_DIM, dtype=np.float32),
        )

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


def test_clip_action_chunk_matches_recorded_action_space():
    env = _Env()
    actions = np.zeros((client.ACTION_HORIZON, client.ACTION_DIM), dtype=np.float32)
    actions[:, 7] = 1.027
    actions[:, 15] = -1.026

    clipped = client._clip_action_chunk(env, actions)

    assert clipped.dtype == np.float32
    assert clipped.flags.c_contiguous
    assert np.all(clipped[:, 7] == 1.0)
    assert np.all(clipped[:, 15] == -1.0)


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


def test_create_env_defers_video_encoding_for_pi05_and_pico(monkeypatch, tmp_path):
    class _HardwareEnv(gym.Env):
        action_space = gym.spaces.Box(-1.0, 1.0, (client.ACTION_DIM,))
        observation_space = gym.spaces.Dict({})

    captured = []

    class _CollectEpisode:
        def __init__(self, env, **kwargs):
            captured.append(kwargs)

    monkeypatch.setattr(client.gym, "make", lambda *args, **kwargs: _HardwareEnv())
    monkeypatch.setattr("rlinf.envs.wrappers.CollectEpisode", _CollectEpisode)
    pico_module = ModuleType(
        "rlinf.envs.realworld.common.wrappers.pico_joint_intervention"
    )
    pico_module.DualFrankaJointPicoIntervention = lambda env, **kwargs: env
    monkeypatch.setitem(sys.modules, pico_module.__name__, pico_module)
    args = Namespace(
        left_robot_ip="left",
        right_robot_ip="right",
        base_camera_serial="base",
        left_camera_serial="left-camera",
        right_camera_serial="right-camera",
        base_camera_type="realsense",
        left_gripper_connection="left-gripper",
        right_gripper_connection="right-gripper",
        joint_reset_qpos=client.DEFAULT_JOINT_RESET_QPOS,
        task=client.DEFAULT_TASK,
        enable_pico=False,
        pico_zmq_addr="ipc:///tmp/test-pico.ipc",
        pico_control_threshold=0.85,
        pico_ready_timeout_s=1.0,
        episode_timeout_s=120.0,
        rollout_dir=str(tmp_path),
    )

    client.create_env(args)
    args.enable_pico = True
    client.create_env(args)

    assert len(captured) == 2
    for kwargs in captured:
        assert kwargs["fps"] == 30
        assert kwargs["use_videos"] is True
        assert kwargs["finalize_interval"] == 0
        assert kwargs["defer_video_encoding_until_finalize"] is True
        assert kwargs["isolate_episode_stats"] is True
        assert kwargs["image_writer_threads"] == 12
        assert kwargs["image_writer_processes"] == 0
        assert kwargs["copy_observations"] is False


def test_close_rollout_resources_stops_hardware_before_encoding():
    events = []

    class _HardwareEnv:
        def close(self):
            events.append("hardware")

    class _CollectionEnv:
        env = _HardwareEnv()

        def close(self):
            events.append("writer")

    class _Policy:
        def close(self):
            events.append("policy")
            raise RuntimeError("policy close failed")

    with np.testing.assert_raises_regex(RuntimeError, "policy close failed"):
        client._close_rollout_resources(_CollectionEnv(), _Policy())

    assert events == ["hardware", "policy", "writer"]

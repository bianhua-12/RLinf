# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

import gymnasium as gym
import numpy as np
import pytest

from rlinf.envs.realworld.common.wrappers.keyboard_eval_control_wrapper import (
    KeyboardEvalControlWrapper,
)
from rlinf.envs.wrappers import CollectEpisode


class _Listener:
    def __init__(self):
        self.keys = []
        self.closed = False

    def pop_pressed_keys(self):
        keys, self.keys = self.keys, []
        return keys

    def close(self):
        self.closed = True


class _Env(gym.Env):
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    observation_space = gym.spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        return np.zeros(1, dtype=np.float32), {}

    def _get_observation(self):
        return np.ones(1, dtype=np.float32)

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}


class _StartListener(_Listener):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def pop_pressed_keys(self):
        self.calls += 1
        return ["a"] if self.calls == 2 else []


def test_reset_refreshes_observation_after_start_pedal(monkeypatch):
    listener = _StartListener()
    monkeypatch.setattr(
        "rlinf.envs.realworld.common.wrappers.keyboard_eval_control_wrapper.KeyboardListener",
        lambda: listener,
    )
    monkeypatch.setattr(
        "rlinf.envs.realworld.common.wrappers.keyboard_eval_control_wrapper.time.sleep",
        lambda _: None,
    )
    env = KeyboardEvalControlWrapper(_Env())

    observation, _ = env.reset()

    np.testing.assert_array_equal(observation, np.ones(1, dtype=np.float32))
    np.testing.assert_array_equal(env._last_obs, observation)


@pytest.mark.parametrize(
    ("key", "expected_result", "expected_reward", "expected_success"),
    [("c", "success", 1.0, True), ("b", "failure", 0.0, False)],
)
def test_terminal_pedal_result_is_exposed_for_collection(
    monkeypatch, key, expected_result, expected_reward, expected_success
):
    listener = _Listener()
    monkeypatch.setattr(
        "rlinf.envs.realworld.common.wrappers.keyboard_eval_control_wrapper.KeyboardListener",
        lambda: listener,
    )
    env = KeyboardEvalControlWrapper(_Env())
    env._running = True
    listener.keys = [key]

    _, reward, terminated, truncated, info = env.step(np.zeros(1))

    assert reward == expected_reward
    assert terminated is True
    assert truncated is False
    assert info["eval_result"] == expected_result
    assert info["success"] is expected_success


def test_q_aborts_rollout(monkeypatch):
    listener = _Listener()
    monkeypatch.setattr(
        "rlinf.envs.realworld.common.wrappers.keyboard_eval_control_wrapper.KeyboardListener",
        lambda: listener,
    )
    env = KeyboardEvalControlWrapper(_Env())
    env._running = True
    listener.keys = ["q"]

    with pytest.raises(KeyboardInterrupt):
        env.step(np.zeros(1))


@pytest.mark.parametrize(("key", "expected_success"), [("c", True), ("b", False)])
def test_collect_episode_receives_terminal_result(
    monkeypatch, tmp_path, key, expected_success
):
    listener = _Listener()
    monkeypatch.setattr(
        "rlinf.envs.realworld.common.wrappers.keyboard_eval_control_wrapper.KeyboardListener",
        lambda: listener,
    )
    controlled = KeyboardEvalControlWrapper(_Env())
    controlled._running = True
    collector = CollectEpisode(
        controlled,
        save_dir=str(tmp_path),
        export_format="pickle",
        only_success=False,
    )
    saved = []
    monkeypatch.setattr(
        collector,
        "_flush_episode",
        lambda env_idx, is_success: saved.append((env_idx, is_success)),
    )
    listener.keys = [key]

    collector.step(np.zeros(1))

    assert saved == [(0, expected_success)]


def test_collect_episode_close_skips_periodically_finalized_writer(tmp_path):
    collector = CollectEpisode(
        _Env(),
        save_dir=str(tmp_path),
        export_format="lerobot",
    )

    class _FinalizedWriter:
        dataset = None

        def finalize(self):
            raise AssertionError("already-finalized writer must not be finalized again")

    collector._lerobot_writer = _FinalizedWriter()
    collector.close()

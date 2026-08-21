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

"""Dual-arm GELLO intervention wrapper for joint-space control.

Step-gated (forwarded via ``env.step``) or direct-stream (a daemon pushes
targets at the configured period, bypassing ``env.step``'s rate gate).
"""

from __future__ import annotations

import threading
import time

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.gello.gello_joint_expert import GelloJointExpert
from rlinf.utils.logging import get_logger


class DualGelloJointIntervention(gym.ActionWrapper):
    def __init__(
        self,
        env: gym.Env,
        left_port: str,
        right_port: str,
        gripper_enabled: bool = True,
        use_delta: bool = False,
        action_scale: float = 0.1,
        direct_stream: bool = False,
        stream_period: float = 0.001,
        stream_watchdog_timeout: float = 0.25,
        ready_timeout: float = 10.0,
        left_expert: GelloJointExpert | None = None,
        right_expert: GelloJointExpert | None = None,
    ):
        super().__init__(env)

        self.gripper_enabled = gripper_enabled
        self.use_delta = use_delta
        self.action_scale = action_scale
        self.left_expert = left_expert or GelloJointExpert(port=left_port)
        self.right_expert = right_expert or GelloJointExpert(port=right_port)
        self.last_intervene = 0.0

        self._direct_stream = direct_stream
        self._stream_period = stream_period
        self._stream_watchdog_timeout = stream_watchdog_timeout
        self._ready_timeout = ready_timeout
        self._logger = get_logger()
        self._stream_thread: threading.Thread | None = None
        self._stream_running = False
        self._stream_error: Exception | None = None
        self._stream_last_success_time: float | None = None
        self._closed = False
        self._stream_gate = threading.Event()
        self._stream_gate.set()  # gate open = stream tick allowed
        self._aligned = False
        try:
            self._wait_for_experts()
        except Exception:
            self.left_expert.close()
            self.right_expert.close()
            raise

    def _wait_for_experts(self) -> None:
        deadline = time.monotonic() + self._ready_timeout
        while not (self.left_expert.ready and self.right_expert.ready):
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    "timed out waiting for both GELLO devices to become ready"
                )
            time.sleep(0.01)

    def _resolve_controllers(self):
        inner = self.unwrapped
        return (getattr(inner, "_left_ctrl", None), getattr(inner, "_right_ctrl", None))

    def _start_stream_thread(self) -> None:
        if self._resolve_controllers() == (None, None):
            return
        if self._stream_thread is not None and self._stream_thread.is_alive():
            return
        self._stream_running = True
        self._stream_thread = threading.Thread(
            target=self._stream_loop,
            name="DualGelloJointStream",
            daemon=True,
        )
        self._stream_thread.start()

    def _stream_loop(self) -> None:
        # The environment step owns gripper commands at the collection rate;
        # sending serial commands from this loop would starve the bus.
        period = self._stream_period
        left_target = None
        right_target = None
        try:
            left_ctrl, right_ctrl = self._resolve_controllers()
            if left_ctrl is None or right_ctrl is None:
                raise RuntimeError("direct GELLO stream requires both arm controllers")
            while self._stream_running:
                self._stream_gate.wait()
                if not self._stream_running:
                    break

                loop_start = time.monotonic()

                if not (self.left_expert.ready and self.right_expert.ready):
                    raise RuntimeError(
                        "GELLO input was lost; controlled realignment is required"
                    )

                left_q, _ = self.left_expert.get_action()
                right_q, _ = self.right_expert.get_action()

                left_target = np.asarray(left_q, dtype=np.float64)
                right_target = np.asarray(right_q, dtype=np.float64)
                lf = left_ctrl.move_joints(left_target)
                rf = right_ctrl.move_joints(right_target)
                lf.wait()
                rf.wait()
                self._stream_last_success_time = time.monotonic()

                elapsed = time.monotonic() - loop_start
                sleep_for = period - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
        except Exception as exc:
            self._stream_error = exc
            self._aligned = False
            self._stream_running = False
            self._stream_gate.clear()
            now = time.monotonic()
            last_success_age = (
                None
                if self._stream_last_success_time is None
                else now - self._stream_last_success_time
            )
            self._logger.exception(
                "Dual GELLO stream stopped: left_target=%s right_target=%s "
                "last_success_age_s=%s",
                None if left_target is None else left_target.tolist(),
                None if right_target is None else right_target.tolist(),
                ("never" if last_success_age is None else f"{last_success_age:.3f}"),
            )

    def _raise_if_stream_unhealthy(self) -> None:
        if self._stream_error is not None:
            raise RuntimeError("dual GELLO stream failed") from self._stream_error
        if not (self._direct_stream and self._aligned):
            return
        if not self._stream_running:
            raise RuntimeError("dual GELLO stream stopped unexpectedly")
        thread = self._stream_thread
        if thread is None or not thread.is_alive():
            raise RuntimeError("dual GELLO stream thread exited unexpectedly")
        if self._stream_last_success_time is None:
            raise RuntimeError("dual GELLO stream has no successful command heartbeat")
        heartbeat_age = time.monotonic() - self._stream_last_success_time
        if heartbeat_age > self._stream_watchdog_timeout:
            raise RuntimeError(
                "dual GELLO stream command heartbeat is stale "
                f"({heartbeat_age:.3f}s > {self._stream_watchdog_timeout:.3f}s)"
            )

    def _get_current_joint_positions(self) -> np.ndarray:
        return self.get_wrapper_attr("get_joint_positions")()

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        if not (self.left_expert.ready and self.right_expert.ready):
            return action, False

        left_q, left_g = self.left_expert.get_action()
        right_q, right_g = self.right_expert.get_action()
        current = self._get_current_joint_positions()  # (2, 7)

        per_arm = []
        for target_q, current_q in zip((left_q, right_q), (current[0], current[1])):
            if self.use_delta:
                delta_q = (target_q - current_q) / self.action_scale
                arm_a = np.clip(delta_q, -1.0, 1.0)
            else:
                arm_a = target_q.copy()
            per_arm.append(arm_a)

        if self.gripper_enabled:
            grippers = []
            for grip in (left_g, right_g):
                g = -(2 * grip - 1.0)
                g = np.clip(g, -1.0, 1.0)
                grippers.append(g)
            expert_a = np.concatenate(
                [per_arm[0], grippers[0], per_arm[1], grippers[1]], axis=0
            )
        else:
            expert_a = np.concatenate(per_arm, axis=0)

        if self._direct_stream and self._aligned:
            return expert_a, True

        movement = np.linalg.norm(
            np.concatenate([left_q, right_q]) - np.concatenate([current[0], current[1]])
        )
        if movement > 0.01 or self.gripper_enabled:
            self.last_intervene = time.time()

        if time.time() - self.last_intervene < 0.5:
            return expert_a, True
        return action, False

    def _align_to_gello(self) -> bool:
        # Without this, direct-stream's first loop would push a far target
        # straight into the impedance tracker, causing a reference jump.
        self._aligned = False
        if not (self.left_expert.ready and self.right_expert.ready):
            return False
        left_ctrl, right_ctrl = self._resolve_controllers()
        if left_ctrl is None or right_ctrl is None:
            return False
        left_q, _ = self.left_expert.get_action()
        right_q, _ = self.right_expert.get_action()
        lf = left_ctrl.reset_joint(np.asarray(left_q, dtype=np.float64).tolist())
        rf = right_ctrl.reset_joint(np.asarray(right_q, dtype=np.float64).tolist())
        lf.wait()
        rf.wait()
        setattr(self.unwrapped, "_left_state", left_ctrl.get_state().wait()[0])
        setattr(self.unwrapped, "_right_state", right_ctrl.get_state().wait()[0])
        self._stream_last_success_time = time.monotonic()
        self._aligned = True
        return True

    def reset(self, **kwargs):
        # Skip the inner env's home slew: aligning to GELLO directly avoids
        # a "home → GELLO" double-slew that breaks tracking continuity.
        options = dict(kwargs.get("options") or {})
        options.setdefault("skip_reset_to_home", True)
        kwargs["options"] = options

        self._stream_gate.clear()
        try:
            self._wait_for_experts()
            result = self.env.reset(**kwargs)
            if self._direct_stream:
                if not self._align_to_gello():
                    raise RuntimeError("failed to align both Frankas to GELLO")
                _, info = result
                result = (self.env.get_wrapper_attr("_get_observation")(), info)
            self._stream_error = None
            self._stream_gate.set()
            if self._direct_stream and self._aligned:
                self._start_stream_thread()
        except Exception:
            self._aligned = False
            self._stream_gate.clear()
            raise
        return result

    def step(self, action):
        self._raise_if_stream_unhealthy()
        new_action, replaced = self.action(action)
        if self._direct_stream and self._aligned:
            replaced = True
            self._start_stream_thread()
        obs, rew, done, truncated, info = self.env.step(new_action)
        self._raise_if_stream_unhealthy()
        if replaced:
            info["intervene_action"] = new_action
            info["intervene_flag"] = np.ones(1)
        return obs, rew, done, truncated, info

    def close(self):
        if self._closed:
            return None
        self._closed = True
        self._stream_running = False
        self._stream_gate.set()
        t = self._stream_thread
        if t is not None and t.is_alive():
            t.join(timeout=2.0)
        try:
            self.left_expert.close()
        finally:
            try:
                self.right_expert.close()
            finally:
                super().close()
        return None

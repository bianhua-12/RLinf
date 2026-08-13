# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the dual-Franka clothes-folding pi0.5 policy through RLinf."""

from __future__ import annotations

import argparse
import functools
import json
import os
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import msgpack
import numpy as np
import websockets.sync.client

ACTION_HORIZON = 30
ACTION_DIM = 16
CONTROL_HZ = 30.0
TRAINING_RTC_MAX_DELAY = 10
DEFAULT_PEDAL = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"
DEFAULT_TASK = "fold the clothes"
DEFAULT_JOINT_RESET_QPOS = [
    [
        -0.02556161,
        0.30659358,
        -0.07418893,
        -1.70056259,
        -0.00549571,
        2.11270901,
        1.16654093,
    ],
    [
        -0.09129598,
        0.39645433,
        0.24130687,
        -1.49827293,
        -0.11264549,
        1.97891465,
        1.86833598,
    ],
]


def _pack_array(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"Unsupported NumPy dtype: {value.dtype}")
        value = np.ascontiguousarray(value)
        return {
            b"__ndarray__": True,
            b"data": value.tobytes(),
            b"dtype": value.dtype.str,
            b"shape": value.shape,
        }
    if isinstance(value, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": value.item(),
            b"dtype": value.dtype.str,
        }
    return value


def _unpack_array(value: Any) -> Any:
    if b"__ndarray__" in value:
        return np.ndarray(
            buffer=value[b"data"],
            dtype=np.dtype(value[b"dtype"]),
            shape=value[b"shape"],
        )
    if b"__npgeneric__" in value:
        return np.dtype(value[b"dtype"]).type(value[b"data"])
    return value


_Packer = functools.partial(msgpack.Packer, default=_pack_array)
_unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array)


class Pi05Client:
    """Minimal client for OpenPI's WebSocket policy server."""

    def __init__(self, host: str, port: int, timeout_s: float):
        self.uri = f"ws://{host}:{port}"
        self.timeout_s = timeout_s
        self.packer = _Packer()
        self.websocket = websockets.sync.client.connect(
            self.uri,
            compression=None,
            max_size=None,
            open_timeout=timeout_s,
        )
        self.metadata = _unpackb(self.websocket.recv(timeout=timeout_s))

    def infer(self, observation: dict[str, Any]) -> dict[str, Any]:
        self.websocket.send(self.packer.pack(observation))
        response = self.websocket.recv(timeout=self.timeout_s)
        if isinstance(response, str):
            raise RuntimeError(f"OpenPI server error:\n{response}")
        return _unpackb(response)

    def close(self) -> None:
        self.websocket.close()


class AsyncPi05Client:
    """Keep all WebSocket operations on one dedicated thread."""

    def __init__(self, host: str, port: int, timeout_s: float):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi05")
        self.host = host
        self.port = port
        self.timeout_s = timeout_s
        self.client: Pi05Client | None = None

    def _connect(self) -> dict[str, Any]:
        self.client = Pi05Client(self.host, self.port, self.timeout_s)
        return self.client.metadata

    def connect(self) -> dict[str, Any]:
        return self.executor.submit(self._connect).result()

    def submit(self, observation: dict[str, Any]) -> Future:
        if self.client is None:
            raise RuntimeError("Pi0.5 client is not connected")
        return self.executor.submit(self.client.infer, observation)

    def close(self) -> None:
        if self.client is not None:
            self.executor.submit(self.client.close).result()
        self.executor.shutdown(wait=True, cancel_futures=True)


def build_observation(raw_obs: dict[str, Any], task: str) -> dict[str, Any]:
    """Map the current RLinf dual-Franka observation to the training schema."""
    frames = raw_obs["frames"]
    state_data = raw_obs["state"]
    if "proprio" in state_data:
        state = np.asarray(state_data["proprio"], dtype=np.float32)
    else:
        joints = np.asarray(state_data["joint_position"], dtype=np.float32)
        grippers = np.asarray(state_data["gripper_position"], dtype=np.float32)
        if joints.shape != (14,) or grippers.shape != (2,):
            raise ValueError(
                "Franka state must contain proprio (16,), or joint_position (14,) "
                f"and gripper_position (2,), got {joints.shape} and {grippers.shape}"
            )
        state = np.concatenate(
            [joints[:7], grippers[:1], joints[7:], grippers[1:]], dtype=np.float32
        )
    if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
        raise ValueError(f"Franka proprio must be finite shape ({ACTION_DIM},), got {state.shape}")

    def image(key: str) -> np.ndarray:
        value = np.asarray(frames[key], dtype=np.uint8)
        if value.ndim != 3 or value.shape[-1] != 3:
            raise ValueError(f"Camera {key} must be HWC RGB, got {value.shape}")
        return np.ascontiguousarray(value)

    return {
        "observation.image": image("base_0_rgb"),
        "observation.extra_view_image-0": image("left_wrist_0_rgb"),
        "observation.extra_view_image-1": image("right_wrist_0_rgb"),
        "observation.state": state,
        "prompt": task,
    }


def validate_actions(response: dict[str, Any]) -> np.ndarray:
    actions = np.ascontiguousarray(response["actions"], dtype=np.float32)
    if actions.shape != (ACTION_HORIZON, ACTION_DIM):
        raise ValueError(
            f"Pi0.5 actions must have shape ({ACTION_HORIZON},{ACTION_DIM}), got {actions.shape}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("Pi0.5 actions contain NaN or infinity")
    return actions


def add_rtc_prefix(
    observation: dict[str, Any], previous_actions: np.ndarray, executed: int, delay: int
) -> dict[str, Any]:
    if not 1 <= delay <= TRAINING_RTC_MAX_DELAY:
        raise ValueError(f"RTC delay must be in [1,{TRAINING_RTC_MAX_DELAY}], got {delay}")
    if executed < 0 or executed + delay > ACTION_HORIZON:
        raise ValueError(
            f"RTC prefix [{executed}:{executed + delay}] exceeds horizon {ACTION_HORIZON}"
        )
    prefix = np.zeros_like(previous_actions)
    prefix[:delay] = previous_actions[executed : executed + delay]
    return {
        **observation,
        "training_rtc_action_prefix": prefix,
        "training_rtc_delay_steps": delay,
    }


def _fresh_observation(env) -> dict[str, Any]:
    return env.unwrapped._get_observation()


def _wait_for_start(listener) -> None:
    listener.pop_pressed_keys()
    print("Arms are homed. Arrange the clothes, then press pedal A to start.")
    while True:
        time.sleep(0.05)
        for key in listener.pop_pressed_keys():
            if key == "a":
                return
            if key == "q":
                raise KeyboardInterrupt


def _pedal_result(listener) -> str | None:
    for key in listener.pop_pressed_keys():
        if key == "b":
            return "failure"
        if key == "c":
            return "success"
        if key == "q":
            raise KeyboardInterrupt
    return None


def run_policy(env, listener, policy: AsyncPi05Client, task: str, max_steps: int) -> str:
    """Execute one rolling Training RTC episode at the environment's 30 Hz rate."""
    env.reset()
    _wait_for_start(listener)
    latest_obs = _fresh_observation(env)
    action_chunk = validate_actions(policy.submit(build_observation(latest_obs, task)).result())
    action_index = 0
    episode_step = 0
    delay_history = deque([TRAINING_RTC_MAX_DELAY], maxlen=8)
    pending: Future | None = None
    request_start_step = 0
    requested_delay = 0

    print("Policy running. Pedal B=failure, C=success, Q=abort.")
    while episode_step < max_steps:
        result = _pedal_result(listener)
        if result is not None:
            return result

        if pending is not None and pending.done():
            response = pending.result()
            observed_delay = episode_step - request_start_step
            if observed_delay > requested_delay:
                raise RuntimeError(
                    "Pi0.5 response exceeded its RTC prefix: "
                    f"observed={observed_delay}, conditioned={requested_delay}"
                )
            if response.get("training_rtc_delay_steps") != requested_delay:
                raise RuntimeError("Pi0.5 server did not confirm the requested RTC delay")
            action_chunk = validate_actions(response)
            action_index = observed_delay
            delay_history.append(max(observed_delay, 1))
            pending = None

        if pending is None and action_index >= 2:
            requested_delay = min(max(delay_history), TRAINING_RTC_MAX_DELAY)
            observation = build_observation(latest_obs, task)
            observation = add_rtc_prefix(
                observation, action_chunk, action_index, requested_delay
            )
            pending = policy.submit(observation)
            request_start_step = episode_step

        if pending is not None and episode_step - request_start_step >= requested_delay:
            raise RuntimeError(
                f"Pi0.5 inference exceeded the {requested_delay}-step RTC prefix"
            )
        if action_index >= ACTION_HORIZON:
            raise RuntimeError("Action chunk exhausted before Pi0.5 returned the next chunk")
        latest_obs, *_ = env.step(action_chunk[action_index])
        episode_step += 1
        action_index += 1

    return "timeout"


def create_env(args: argparse.Namespace):
    import gymnasium as gym

    import rlinf.envs.realworld.franka.tasks  # noqa: F401

    override_cfg = {
        "left_robot_ip": args.left_robot_ip,
        "right_robot_ip": args.right_robot_ip,
        "base_camera_serials": [args.base_camera_serial],
        "left_camera_serials": [args.left_camera_serial],
        "right_camera_serials": [args.right_camera_serial],
        "base_camera_type": "realsense",
        "left_camera_type": "realsense",
        "right_camera_type": "realsense",
        "left_gripper_type": "robotiq",
        "right_gripper_type": "robotiq",
        "left_gripper_connection": args.left_gripper_connection,
        "right_gripper_connection": args.right_gripper_connection,
        "joint_reset_qpos": args.joint_reset_qpos,
        "joint_action_mode": "absolute",
        "step_frequency": CONTROL_HZ,
        "max_num_steps": args.max_steps + 100,
        "task_description": args.task,
        "controlled_motion_tolerance": 0.05,
    }
    env_cfg = {
        "no_gripper": False,
        "use_gello_joint": False,
        "use_gello": False,
        "use_pico": False,
        "use_spacemouse": False,
        "keyboard_reward_wrapper": None,
    }
    return gym.make(
        "Ros2DualFrankaJointEnv-v1",
        override_cfg=override_cfg,
        worker_info=None,
        hardware_info=None,
        env_idx=0,
        env_cfg=env_cfg,
    )


def _joint_reset(value: str) -> list[list[float]]:
    try:
        parsed = json.loads(value)
        array = np.asarray(parsed, dtype=np.float64)
    except (json.JSONDecodeError, TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("joint reset must be a JSON 2x7 array") from exc
    if array.shape != (2, 7) or not np.isfinite(array).all():
        raise argparse.ArgumentTypeError("joint reset must be a finite JSON 2x7 array")
    return array.tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", default=os.environ.get("OPENPI_SERVER_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("OPENPI_SERVER_PORT", "8000"))
    )
    parser.add_argument("--timeout-s", type=float, default=15.0)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--joint-reset-qpos", type=_joint_reset, default=DEFAULT_JOINT_RESET_QPOS)
    parser.add_argument("--left-robot-ip", default="172.16.0.1")
    parser.add_argument("--right-robot-ip", default="172.16.0.2")
    parser.add_argument("--base-camera-serial", default="327122078534")
    parser.add_argument("--left-camera-serial", default="261922076829")
    parser.add_argument("--right-camera-serial", default="262322073199")
    parser.add_argument(
        "--left-gripper-connection",
        default="/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_AM6YM2MI-if00-port0",
    )
    parser.add_argument(
        "--right-gripper-connection",
        default="/dev/serial/by-id/usb-FTDI_FT232R_USB_UART_BG046F7F-if00-port0",
    )
    parser.add_argument("--pedal-device", default=DEFAULT_PEDAL)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--enable-policy", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    return parser.parse_args()


def self_test() -> None:
    raw_obs = {
        "frames": {
            "base_0_rgb": np.zeros((8, 9, 3), dtype=np.uint8),
            "left_wrist_0_rgb": np.ones((8, 9, 3), dtype=np.uint8),
            "right_wrist_0_rgb": np.full((8, 9, 3), 2, dtype=np.uint8),
        },
        "state": {
            "joint_position": np.arange(14, dtype=np.float32),
            "gripper_position": np.array([0.04, 0.08], dtype=np.float32),
        },
    }
    observation = build_observation(raw_obs, DEFAULT_TASK)
    expected_state = np.array(
        [0, 1, 2, 3, 4, 5, 6, 0.04, 7, 8, 9, 10, 11, 12, 13, 0.08],
        dtype=np.float32,
    )
    np.testing.assert_allclose(observation["observation.state"], expected_state)
    ros2_observation = build_observation(
        {**raw_obs, "state": {"proprio": expected_state.copy()}}, DEFAULT_TASK
    )
    np.testing.assert_array_equal(ros2_observation["observation.state"], expected_state)
    actions = np.arange(ACTION_HORIZON * ACTION_DIM, dtype=np.float32).reshape(
        ACTION_HORIZON, ACTION_DIM
    )
    rtc_observation = add_rtc_prefix(observation, actions, executed=3, delay=10)
    np.testing.assert_array_equal(
        rtc_observation["training_rtc_action_prefix"][:10], actions[3:13]
    )
    decoded = _unpackb(_Packer().pack(rtc_observation))
    np.testing.assert_array_equal(decoded["observation.state"], expected_state)
    assert validate_actions({"actions": actions}).shape == (30, 16)
    print("Self-test passed: state layout, images, 30x16 actions, RTC prefix, and wire format.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.enable_policy:
        raise SystemExit("Refusing to open hardware without --enable-policy")
    if not 1 <= args.port <= 65535 or args.timeout_s <= 0 or args.max_steps <= 0:
        raise SystemExit("port, timeout-s, and max-steps must be positive and valid")
    input("Press Enter to open hardware (Ctrl+C to cancel): ")

    os.environ["RLINF_KEYBOARD_DEVICE"] = args.pedal_device
    from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener

    policy = AsyncPi05Client(args.host, args.port, args.timeout_s)
    env = None
    listener = None
    try:
        metadata = policy.connect()
        if metadata.get("config_name") != "pi05_franka_fold_full_rtc":
            raise RuntimeError(f"Unexpected OpenPI server metadata: {metadata}")
        print(f"Connected to pi0.5 server at {args.host}:{args.port}")
        listener = KeyboardListener()
        env = create_env(args)
        result = run_policy(env, listener, policy, args.task, args.max_steps)
        print(f"Episode result: {result}")
    finally:
        policy.close()
        if listener is not None:
            listener.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

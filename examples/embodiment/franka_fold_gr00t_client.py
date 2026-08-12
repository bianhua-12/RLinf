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

"""Run the Franka clothes-folding GR00T policy through the RLinf ROS 2 env."""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any

import msgpack
import numpy as np
import zmq

ACTION_LAYOUT = (
    ("left_arm", 7),
    ("left_gripper", 1),
    ("right_arm", 7),
    ("right_gripper", 1),
)
ACTION_HORIZON = 30
CONTROL_HZ = 30.0
TRAINING_RTC_MAX_DELAY = 10
DEFAULT_PEDAL = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"


def _encode_numpy(value: Any) -> Any:
    """Encode numeric arrays using GR00T's msgpack-numpy wire format."""
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("O", "V"):
            raise TypeError(f"Cannot serialize ndarray dtype {value.dtype}")
        array = np.ascontiguousarray(value)
        return {
            b"nd": True,
            b"type": array.dtype.str,
            b"kind": b"",
            b"shape": array.shape,
            b"data": array.tobytes(),
        }
    if isinstance(value, (np.bool_, np.number)):
        scalar = np.asarray(value)
        return {b"nd": False, b"type": scalar.dtype.str, b"data": scalar.tobytes()}
    raise TypeError(f"Cannot serialize type {type(value).__name__}")


def _decode_numpy(value: Any) -> Any:
    """Decode numeric arrays using GR00T's msgpack-numpy wire format."""
    if not isinstance(value, dict):
        return value
    marker = value.get(b"nd", value.get("nd"))
    if marker is None:
        return value
    dtype_value = value.get(b"type", value.get("type"))
    data = value.get(b"data", value.get("data"))
    if dtype_value is None or data is None:
        raise ValueError("Malformed NumPy payload")
    dtype = np.dtype(dtype_value)
    if dtype.kind in ("O", "V"):
        raise ValueError(f"Refusing to decode ndarray dtype {dtype}")
    if marker is True:
        shape = value.get(b"shape", value.get("shape"))
        if shape is None:
            raise ValueError("Malformed ndarray payload: shape is missing")
        return np.frombuffer(data, dtype=dtype).reshape(tuple(shape))
    return np.frombuffer(data, dtype=dtype)[0]


def _pack(value: Any) -> bytes:
    return msgpack.packb(value, default=_encode_numpy)


def _unpack(value: bytes) -> Any:
    return msgpack.unpackb(value, object_hook=_decode_numpy, raw=False)


class Gr00tClient:
    """Minimal client for the existing Isaac-GR00T ZeroMQ policy server."""

    def __init__(self, host: str, port: int, timeout_ms: int):
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms
        self.context = zmq.Context()
        self.socket: zmq.Socket | None = None
        self._new_socket()

    def _new_socket(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self.socket.connect(f"tcp://{self.host}:{self.port}")

    def call(self, endpoint: str, data: dict[str, Any] | None = None) -> Any:
        request = {"endpoint": endpoint}
        if data is not None:
            request["data"] = data
        assert self.socket is not None
        try:
            self.socket.send(_pack(request))
            response = _unpack(self.socket.recv())
        except zmq.error.Again:
            self._new_socket()
            raise TimeoutError(
                f"GR00T server timed out at {self.host}:{self.port}"
            ) from None
        if isinstance(response, dict) and "error" in response:
            raise RuntimeError(f"GR00T server error: {response['error']}")
        return response

    def ping_and_reset(self) -> None:
        response = self.call("ping")
        if not isinstance(response, dict) or response.get("status") != "ok":
            raise RuntimeError(f"Unexpected GR00T ping response: {response!r}")
        self.call("reset", {"options": None})

    def get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        response = self.call(
            "get_action", {"observation": observation, "options": options}
        )
        if not isinstance(response, list) or len(response) != 2:
            raise ValueError("GR00T get_action response must be [actions, info]")
        return response[0], response[1]

    def close(self) -> None:
        if self.socket is not None:
            self.socket.close(linger=0)
            self.socket = None
        self.context.term()


class AsyncGr00tClient:
    """Keep every ZeroMQ operation on one dedicated thread."""

    def __init__(self, host: str, port: int, timeout_ms: int):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gr00t")
        self.client: Gr00tClient | None = None
        self.host = host
        self.port = port
        self.timeout_ms = timeout_ms

    def _get_client(self) -> Gr00tClient:
        if self.client is None:
            self.client = Gr00tClient(self.host, self.port, self.timeout_ms)
        return self.client

    def _connect(self) -> None:
        self._get_client().ping_and_reset()

    def _get_action(
        self, observation: dict[str, Any], options: dict[str, Any] | None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        return self._get_client().get_action(observation, options)

    def connect(self) -> None:
        self.executor.submit(self._connect).result()

    def submit(
        self, observation: dict[str, Any], options: dict[str, Any] | None = None
    ) -> Future:
        return self.executor.submit(self._get_action, observation, options)

    def close(self) -> None:
        if self.client is not None:
            self.executor.submit(self.client.close).result()
        self.executor.shutdown(wait=True, cancel_futures=True)


def build_observation(raw_obs: dict[str, Any], task: str) -> dict[str, Any]:
    """Map the ROS 2 Franka env observation to the checkpoint modalities."""
    frames = raw_obs["frames"]
    state = np.asarray(raw_obs["state"]["proprio"], dtype=np.float32)
    required_frames = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
    missing = [key for key in required_frames if key not in frames]
    if missing:
        raise ValueError(f"Missing camera frames: {missing}")
    if state.shape != (16,) or not np.isfinite(state).all():
        raise ValueError(
            f"Franka proprio must be finite shape (16,), got {state.shape}"
        )

    def video(key: str) -> np.ndarray:
        image = np.asarray(frames[key], dtype=np.uint8)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise ValueError(f"Camera {key} must be HWC RGB, got {image.shape}")
        return np.ascontiguousarray(image[None, None])

    return {
        "video": {
            "front": video("base_0_rgb"),
            "left_wrist": video("left_wrist_0_rgb"),
            "right_wrist": video("right_wrist_0_rgb"),
        },
        "state": {
            "left_arm": state[None, None, 0:7],
            "left_gripper": state[None, None, 7:8],
            "right_arm": state[None, None, 8:15],
            "right_gripper": state[None, None, 15:16],
        },
        "language": {"annotation.human.task_description": [[task]]},
    }


def validate_actions(actions: dict[str, Any]) -> dict[str, np.ndarray]:
    """Validate one physical-space 30x16 dual-Franka action chunk."""
    result = {}
    for key, width in ACTION_LAYOUT:
        value = actions.get(key, actions.get(f"action.{key}"))
        if value is None:
            raise ValueError(f"GR00T response is missing action key {key!r}")
        array = np.ascontiguousarray(value, dtype=np.float32)
        if array.shape != (1, ACTION_HORIZON, width):
            raise ValueError(
                f"Action {key!r} must have shape (1,{ACTION_HORIZON},{width}), "
                f"got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"Action {key!r} contains NaN or infinity")
        result[key] = array
    return result


def flatten_actions(actions: dict[str, np.ndarray]) -> np.ndarray:
    return np.concatenate([actions[key] for key, _ in ACTION_LAYOUT], axis=-1)[0]


def rtc_options(
    previous: dict[str, np.ndarray], executed: int, delay: int
) -> dict[str, Any]:
    """Build the physical action prefix consumed by Training RTC inference."""
    if not 1 <= delay <= TRAINING_RTC_MAX_DELAY:
        raise ValueError(
            f"RTC delay must be in [1,{TRAINING_RTC_MAX_DELAY}], got {delay}"
        )
    if executed < 0 or executed + delay > ACTION_HORIZON:
        raise ValueError(
            f"RTC prefix [{executed}:{executed + delay}] exceeds horizon {ACTION_HORIZON}"
        )
    return {
        "training_rtc_action_prefix": {
            key: value[:, executed : executed + delay].copy()
            for key, value in previous.items()
        },
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


def run_policy(
    env, listener, policy: AsyncGr00tClient, task: str, max_steps: int
) -> str:
    """Execute one rolling Training RTC episode."""
    env.reset()
    _wait_for_start(listener)
    latest_obs = _fresh_observation(env)
    observation = build_observation(latest_obs, task)
    actions, _ = policy.submit(observation).result()
    action_dict = validate_actions(actions)
    action_chunk = flatten_actions(action_dict)
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
            actions, info = pending.result()
            observed_delay = episode_step - request_start_step
            if observed_delay > requested_delay:
                raise RuntimeError(
                    "GR00T response exceeded its Training RTC prefix: "
                    f"observed={observed_delay}, conditioned={requested_delay}"
                )
            if info.get("training_rtc_delay_steps") != requested_delay:
                raise RuntimeError(
                    "GR00T server did not confirm the requested RTC delay"
                )
            action_dict = validate_actions(actions)
            action_chunk = flatten_actions(action_dict)
            action_index = observed_delay
            delay_history.append(max(observed_delay, 1))
            pending = None

        if pending is None and action_index >= 2:
            requested_delay = min(max(delay_history), TRAINING_RTC_MAX_DELAY)
            options = rtc_options(action_dict, action_index, requested_delay)
            observation = build_observation(latest_obs, task)
            pending = policy.submit(observation, options)
            request_start_step = episode_step

        if action_index >= ACTION_HORIZON:
            raise RuntimeError(
                "Action chunk exhausted before GR00T returned the next chunk"
            )
        latest_obs, *_ = env.step(action_chunk[action_index])
        episode_step += 1
        action_index += 1

    return "timeout"


def create_env(args: argparse.Namespace):
    """Construct the existing RLinf ROS 2 dual-Franka environment directly."""
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
        "teleop_direct_stream": False,
        "camera_fps": 30,
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
        raise argparse.ArgumentTypeError(
            "joint reset must be a JSON 2x7 array"
        ) from exc
    if array.shape != (2, 7) or not np.isfinite(array).all():
        raise argparse.ArgumentTypeError("joint reset must be a finite JSON 2x7 array")
    return array.tolist()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--host", default=os.environ.get("GR00T_SERVER_HOST", "127.0.0.1")
    )
    parser.add_argument(
        "--port", type=int, default=int(os.environ.get("GR00T_SERVER_PORT", "5555"))
    )
    parser.add_argument("--timeout-ms", type=int, default=15000)
    parser.add_argument("--task", default=os.environ.get("RLINF_TASK_DESCRIPTION"))
    parser.add_argument(
        "--joint-reset-qpos",
        type=_joint_reset,
        default=(
            _joint_reset(os.environ["RLINF_JOINT_RESET_QPOS"])
            if "RLINF_JOINT_RESET_QPOS" in os.environ
            else None
        ),
    )
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
    parser.add_argument(
        "--pedal-device", default=os.environ.get("RLINF_KEYBOARD_DEVICE", DEFAULT_PEDAL)
    )
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
        "state": {"proprio": np.arange(16, dtype=np.float32)},
    }
    observation = build_observation(raw_obs, "fold the clothes")
    assert observation["video"]["front"].shape == (1, 1, 8, 9, 3)
    assert observation["state"]["right_arm"].shape == (1, 1, 7)
    actions = {
        key: np.arange(ACTION_HORIZON * width, dtype=np.float32).reshape(
            1, ACTION_HORIZON, width
        )
        for key, width in ACTION_LAYOUT
    }
    validated = validate_actions(actions)
    assert flatten_actions(validated).shape == (ACTION_HORIZON, 16)
    options = rtc_options(validated, executed=3, delay=TRAINING_RTC_MAX_DELAY)
    assert options["training_rtc_action_prefix"]["left_arm"].shape == (1, 10, 7)
    decoded = _unpack(_pack({"observation": observation, "options": options}))
    np.testing.assert_array_equal(
        decoded["observation"]["state"]["left_arm"],
        observation["state"]["left_arm"],
    )
    print("Self-test passed: observation, 30x16 action, RTC prefix, and wire format.")


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.enable_policy:
        raise SystemExit("Refusing to open hardware without --enable-policy")
    if not args.task:
        raise SystemExit("Set RLINF_TASK_DESCRIPTION or pass --task")
    if args.joint_reset_qpos is None:
        raise SystemExit("Set RLINF_JOINT_RESET_QPOS or pass --joint-reset-qpos")
    if not 1 <= args.port <= 65535 or args.timeout_ms <= 0 or args.max_steps <= 0:
        raise SystemExit("port, timeout-ms, and max-steps must be positive and valid")
    confirmation = input("Type RUN FRANKA POLICY to open hardware: ").strip()
    if confirmation != "RUN FRANKA POLICY":
        raise SystemExit("Cancelled")

    os.environ["RLINF_KEYBOARD_DEVICE"] = args.pedal_device
    from rlinf.envs.realworld.common.keyboard.keyboard_listener import KeyboardListener

    policy = AsyncGr00tClient(args.host, args.port, args.timeout_ms)
    env = None
    listener = None
    try:
        policy.connect()
        print(f"Connected to GR00T server at {args.host}:{args.port}")
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

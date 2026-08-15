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

import gymnasium as gym
import msgpack
import numpy as np
import websockets.sync.client

ACTION_HORIZON = 30
ACTION_DIM = 16
CONTROL_HZ = 30.0
TRAINING_RTC_MAX_DELAY = 10
DEFAULT_PEDAL = "/dev/input/by-id/usb-PCsensor_FootSwitch-event-kbd"
DEFAULT_TASK = "fold the clothes"
DEFAULT_PICO_ZMQ_ADDR = "ipc:///tmp/vr_data.ipc"
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


class _EpisodeTimeoutWrapper(gym.Wrapper):
    """End an overlong policy episode as a failure."""

    def __init__(self, env: gym.Env, timeout_s: float):
        super().__init__(env)
        self.timeout_s = timeout_s
        self.deadline = 0.0

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        self.deadline = time.monotonic() + self.timeout_s
        return observation, info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        if time.monotonic() >= self.deadline:
            reward = 0.0
            terminated = True
            truncated = False
            info = {**info, "eval_result": "failure", "success": False}
        return observation, reward, terminated, truncated, info


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


def collection_observation(raw_obs: dict[str, Any], task: str) -> dict[str, Any]:
    """Add the standard LeRobot fields while preserving the pi0.5 observation."""
    frames = raw_obs["frames"]
    adapted = dict(raw_obs)
    adapted["states"] = build_observation(raw_obs, task)["observation.state"]
    adapted["main_images"] = frames["base_0_rgb"]
    adapted["extra_view_images"] = np.stack(
        [frames["left_wrist_0_rgb"], frames["right_wrist_0_rgb"]]
    )
    adapted["task_descriptions"] = task
    return adapted


def run_policy(env, policy: AsyncPi05Client, task: str) -> str:
    """Execute one rolling Training RTC episode at the environment's 30 Hz rate."""
    latest_obs, _ = env.reset()
    action_chunk = validate_actions(policy.submit(build_observation(latest_obs, task)).result())
    action_index = 0
    episode_step = 0
    delay_history = deque([TRAINING_RTC_MAX_DELAY], maxlen=8)
    pending: Future | None = None
    pending_kind: str | None = None
    pending_generation = 0
    request_start_step = 0
    requested_delay = 0
    generation = 0
    pico_active = False
    pico_ready = False
    pico_resume_required = False

    print("Policy running. Pedal B=failure, C=success, Q=abort.")
    while True:
        if pending is not None and pending.done():
            response = pending.result()
            completed_kind = pending_kind
            completed_generation = pending_generation
            pending = None
            pending_kind = None
            if completed_generation != generation:
                print("Discarding a pi0.5 response issued before PICO takeover.")
            elif completed_kind == "resume":
                candidate = validate_actions(response)
                observed_delay = episode_step - request_start_step
                if observed_delay > requested_delay:
                    print(
                        "Pi0.5 resume response exceeded its RTC prefix; "
                        "holding and re-inferring."
                    )
                elif response.get("training_rtc_delay_steps") != requested_delay:
                    raise RuntimeError(
                        "Pi0.5 server did not confirm the resume RTC delay"
                    )
                else:
                    release = env.get_wrapper_attr("release_to_policy")
                    try:
                        release()
                    except (RuntimeError, ValueError) as exc:
                        print(f"PICO release rejected; holding and re-inferring: {exc}")
                    else:
                        action_chunk = candidate
                        action_index = observed_delay
                        pico_resume_required = False
                        delay_history.clear()
                        delay_history.append(TRAINING_RTC_MAX_DELAY)
                        print("Fresh pi0.5 action accepted; PICO takeover released.")
            else:
                observed_delay = episode_step - request_start_step
                if observed_delay > requested_delay:
                    print(
                        "Pi0.5 response exceeded its RTC prefix: "
                        f"observed={observed_delay}, conditioned={requested_delay}; "
                        "discarding the stale response."
                    )
                    delay_history.append(
                        min(observed_delay + 1, TRAINING_RTC_MAX_DELAY)
                    )
                else:
                    if response.get("training_rtc_delay_steps") != requested_delay:
                        raise RuntimeError(
                            "Pi0.5 server did not confirm the requested RTC delay"
                        )
                    action_chunk = validate_actions(response)
                    action_index = observed_delay
                    delay_history.append(
                        min(max(observed_delay + 1, 1), TRAINING_RTC_MAX_DELAY)
                    )

        if pico_resume_required and not pico_active and pico_ready and pending is None:
            hold_action = np.asarray(
                env.get_wrapper_attr("get_hold_action")(), dtype=np.float32
            )
            hold_chunk = np.repeat(hold_action[None, :], ACTION_HORIZON, axis=0)
            requested_delay = TRAINING_RTC_MAX_DELAY
            observation = add_rtc_prefix(
                build_observation(latest_obs, task),
                hold_chunk,
                executed=0,
                delay=requested_delay,
            )
            pending = policy.submit(observation)
            pending_kind = "resume"
            pending_generation = generation
            request_start_step = episode_step
        elif (
            not pico_resume_required
            and pending is None
            and 2 <= action_index < ACTION_HORIZON
        ):
            requested_delay = min(
                max(delay_history), ACTION_HORIZON - action_index
            )
            observation = build_observation(latest_obs, task)
            observation = add_rtc_prefix(
                observation, action_chunk, action_index, requested_delay
            )
            pending = policy.submit(observation)
            pending_kind = "rtc"
            pending_generation = generation
            request_start_step = episode_step

        if not pico_resume_required and action_index >= ACTION_HORIZON:
            print("Pi0.5 inference is still pending; holding the last joint target.")
            action_chunk = np.repeat(action_chunk[-1:], ACTION_HORIZON, axis=0)
            action_index = 0

        command_index = min(action_index, ACTION_HORIZON - 1)
        latest_obs, _, terminated, truncated, info = env.step(
            action_chunk[command_index]
        )
        episode_step += 1
        if not pico_resume_required:
            action_index += 1

        now_pico_active = bool(info.get("pico_active", False))
        pico_ready = bool(info.get("pico_ready", False))
        now_pico_takeover = bool(info.get("pico_takeover", False))
        if now_pico_active and not pico_active:
            generation += 1
            pico_resume_required = True
            print("PICO takeover active; invalidating pending pi0.5 actions.")
        elif now_pico_takeover:
            pico_resume_required = True
        pico_active = now_pico_active

        if terminated or truncated:
            return info.get("eval_result") or "timeout"


def create_env(args: argparse.Namespace):
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
        "camera_fps": int(CONTROL_HZ),
        "step_frequency": CONTROL_HZ,
        "task_description": args.task,
        "controlled_motion_tolerance": 0.05,
    }
    env_cfg = {
        "no_gripper": False,
        "use_gello_joint": False,
        "use_gello": False,
        "use_pico": False,
        "use_spacemouse": False,
        "keyboard_reward_wrapper": "eval_control",
    }
    env = gym.make(
        "Ros2DualFrankaJointEnv-v1",
        override_cfg=override_cfg,
        worker_info=None,
        hardware_info=None,
        env_idx=0,
        env_cfg=env_cfg,
    )
    if args.enable_pico:
        from rlinf.envs.realworld.common.wrappers.pico_joint_intervention import (
            DualFrankaJointPicoIntervention,
        )

        try:
            env = DualFrankaJointPicoIntervention(
                env,
                zmq_addr=args.pico_zmq_addr,
                control_trigger="grip",
                control_threshold=args.pico_control_threshold,
                max_stale_s=0.2,
                ready_timeout_s=args.pico_ready_timeout_s,
                trajectory_filter={
                    "min_cutoff": 1.0,
                    "beta": 0.1,
                    "d_cutoff": 1.0,
                },
                calibration={
                    "enabled": True,
                    "required": True,
                    "auto_calibrate_on_start": True,
                    "button": "trigger",
                    "threshold": 0.5,
                    "head_forward_axis": "-z",
                    "base_position": [0.0, 0.0, 0.0],
                },
                left={"gripper_close_button": "X", "gripper_open_button": "Y"},
                right={"gripper_close_button": "A", "gripper_open_button": "B"},
            )
        except Exception:
            env.close()
            raise

    env = _EpisodeTimeoutWrapper(env, args.episode_timeout_s)

    class CollectionObservationWrapper(gym.ObservationWrapper):
        def observation(self, observation):
            return collection_observation(observation, args.task)

    from rlinf.envs.wrappers import CollectEpisode

    env = CollectionObservationWrapper(env)
    return CollectEpisode(
        env,
        save_dir=args.rollout_dir,
        export_format="lerobot",
        robot_type="dual_FR3",
        fps=int(CONTROL_HZ),
        use_videos=True,
        only_success=False,
        finalize_interval=0,
        resume=True,
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
    parser.add_argument("--episode-timeout-s", type=float, default=120.0)
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
    parser.add_argument("--enable-pico", action="store_true")
    parser.add_argument("--pico-zmq-addr", default=DEFAULT_PICO_ZMQ_ADDR)
    parser.add_argument("--pico-control-threshold", type=float, default=0.85)
    parser.add_argument("--pico-ready-timeout-s", type=float, default=10.0)
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=int(os.environ.get("RLINF_PI05_NUM_EPISODES", "50")),
    )
    parser.add_argument(
        "--rollout-dir",
        default=os.environ.get(
            "RLINF_PI05_ROLLOUT_DIR",
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "logs",
                    "franka_fold_pi05_rollouts",
                )
            ),
        ),
    )
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
    collected = collection_observation(raw_obs, DEFAULT_TASK)
    np.testing.assert_array_equal(collected["states"], expected_state)
    assert collected["main_images"] is raw_obs["frames"]["base_0_rgb"]
    assert collected["extra_view_images"].shape == (2, 8, 9, 3)
    print(
        "Self-test passed: state layout, images, 30x16 actions, RTC prefix, "
        "wire format, and rollout observation mapping."
    )


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if not args.enable_policy:
        raise SystemExit("Refusing to open hardware without --enable-policy")
    if (
        not 1 <= args.port <= 65535
        or args.timeout_s <= 0
        or args.episode_timeout_s <= 0
        or args.num_episodes <= 0
    ):
        raise SystemExit(
            "port, timeout-s, episode-timeout-s, and num-episodes must be positive and valid"
        )
    if args.enable_pico and (
        not 0.0 < args.pico_control_threshold <= 1.0
        or args.pico_ready_timeout_s <= 0.0
    ):
        raise SystemExit("PICO threshold and timeout must be positive and valid")
    input("Press Enter to open hardware (Ctrl+C to cancel): ")

    os.environ["RLINF_KEYBOARD_DEVICE"] = args.pedal_device
    policy = AsyncPi05Client(args.host, args.port, args.timeout_s)
    env = None
    try:
        metadata = policy.connect()
        if metadata.get("config_name") != "pi05_franka_fold_full_rtc":
            raise RuntimeError(f"Unexpected OpenPI server metadata: {metadata}")
        print(f"Connected to pi0.5 server at {args.host}:{args.port}")
        env = create_env(args)
        print(f"Saving policy rollouts to {args.rollout_dir}")
        for episode_index in range(args.num_episodes):
            result = run_policy(env, policy, args.task)
            print(
                f"Episode {episode_index + 1}/{args.num_episodes} result: {result}"
            )
            if episode_index + 1 < args.num_episodes:
                print("Resetting arms for the next episode.")
    finally:
        policy.close()
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()

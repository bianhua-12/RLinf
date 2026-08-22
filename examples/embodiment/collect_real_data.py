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

import errno
import os
import select
import signal
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import hydra
import numpy as np
import torch
from tqdm import tqdm

from rlinf.data.schema.embodied_trajectory_builder import EmbodiedTrajectoryBuilder
from rlinf.data.schema.embodied_types import (
    ChunkStepResult,
)
from rlinf.data.storage.replay import TrajectoryReplayBuffer
from rlinf.envs.realworld.realworld_env import RealWorldEnv
from rlinf.scheduler import Cluster, ComponentPlacement, Worker
from rlinf.utils.logging import get_logger


class _GracefulStopSignalHandler:
    """Request a clean actor shutdown on first SIGINT, force on the second."""

    def __init__(self, stop_request_path: Path, logger):
        self._stop_request_path = stop_request_path
        self._logger = logger
        self._signal_count = 0

    def __call__(self, _signum, _frame) -> None:
        self._signal_count += 1
        if self._signal_count == 1:
            self._logger.warning(
                "Graceful stop requested; waiting for hardware shutdown and "
                "final video encoding. Press Ctrl-C again to force exit."
            )
            self._stop_request_path.touch()
            return
        raise KeyboardInterrupt


def _relay_terminal_keys(fifo_path: str, stop_event: threading.Event) -> None:
    writer_fd = None
    terminal_fd = sys.stdin.fileno()
    terminal_attrs = None
    try:
        while not stop_event.is_set():
            try:
                writer_fd = os.open(fifo_path, os.O_WRONLY | os.O_NONBLOCK)
                break
            except OSError as exc:
                if exc.errno != errno.ENXIO:
                    raise
                stop_event.wait(0.1)
        if writer_fd is None:
            return

        terminal_attrs = termios.tcgetattr(terminal_fd)
        tty.setcbreak(terminal_fd)
        while not stop_event.is_set():
            readable, _, _ = select.select([terminal_fd], [], [], 0.1)
            if readable:
                key = os.read(terminal_fd, 1).lower()
                if key in (b"a", b"b", b"c"):
                    os.write(writer_fd, key)
    except BrokenPipeError:
        return
    finally:
        if terminal_attrs is not None:
            termios.tcsetattr(terminal_fd, termios.TCSADRAIN, terminal_attrs)
        if writer_fd is not None:
            os.close(writer_fd)


def _configure_keyboard_device(cfg) -> None:
    device = cfg.env.eval.get("keyboard_device")
    if not device:
        return

    device_path = Path(str(device))
    if device_path.parent != Path("/dev/input/by-id") or not device_path.name.endswith(
        "-event-kbd"
    ):
        raise ValueError(
            "env.eval.keyboard_device must use a stable "
            "/dev/input/by-id/...-event-kbd path"
        )
    if not device_path.exists():
        raise FileNotFoundError(f"Keyboard input device not found: {device_path}")
    if not os.access(device_path, os.R_OK):
        raise PermissionError(f"Keyboard input device is not readable: {device_path}")
    os.environ["RLINF_KEYBOARD_DEVICE"] = str(device_path)


class DataCollector(Worker):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.num_data_episodes = cfg.runner.num_data_episodes
        self._stop_request_path = Path(cfg.runner.logger.log_path) / ".stop_collection"
        self.total_cnt = 0
        override_cfg = cfg.env.eval.get("override_cfg", {})
        self.manual_episode_control_only = bool(
            override_cfg.get("manual_episode_control_only", False)
        )
        self.env = RealWorldEnv(
            cfg.env.eval,
            num_envs=1,
            seed_offset=0,
            total_num_processes=1,
            worker_info=self.worker_info,
            return_numpy=True,
        )
        self._hardware_env = self.env
        try:
            self._initialize_collection_storage()
        except Exception:
            try:
                self._hardware_env.close()
            finally:
                if self.env is not self._hardware_env:
                    self.env.close()
            raise

    def _initialize_collection_storage(self):
        self.save_demos = bool(self.cfg.runner.get("save_demos", True))
        dc_cfg = self.cfg.env.eval.get("data_collection")
        if dc_cfg and getattr(dc_cfg, "enabled", False):
            from rlinf.envs.wrappers import CollectEpisode

            self.env = CollectEpisode(
                self.env,
                save_dir=dc_cfg.save_dir,
                export_format=dc_cfg.get("export_format", "pickle"),
                robot_type=dc_cfg.get("robot_type", "panda"),
                fps=dc_cfg.get("fps", 10),
                use_videos=bool(dc_cfg.get("use_videos", False)),
                only_success=dc_cfg.get("only_success", False),
                finalize_interval=dc_cfg.get("finalize_interval", 100),
                defer_video_encoding_until_finalize=bool(
                    dc_cfg.get("defer_video_encoding_until_finalize", False)
                ),
                isolate_episode_stats=bool(dc_cfg.get("isolate_episode_stats", False)),
                image_writer_threads=int(dc_cfg.get("image_writer_threads", 10)),
                image_writer_processes=int(dc_cfg.get("image_writer_processes", 0)),
                resume=bool(dc_cfg.get("resume", False)),
                video_write_mode=dc_cfg.get("video_write_mode", "lerobot_png"),
                stream_video_queue_size=int(dc_cfg.get("stream_video_queue_size", 60)),
                stream_max_pending_commits=int(
                    dc_cfg.get("stream_max_pending_commits", 2)
                ),
                stream_commit_watchdog_timeout=float(
                    dc_cfg.get("stream_commit_watchdog_timeout", 30.0)
                ),
                # RealWorldEnv allocates fresh observation arrays on every step.
                copy_observations=False,
            )
            self._preexisting_success = int(
                getattr(self.env, "preexisting_episode_count", 0)
            )
            if self._preexisting_success:
                self.log_info(
                    f"[resume] {self._preexisting_success} pre-existing episodes; "
                    f"continuing toward {self.num_data_episodes}"
                )
        else:
            self._preexisting_success = 0

        # Read from the wrapped action space so GripperCloseEnv / dual-arm all just work.
        self.action_dim = int(self.env.action_space.shape[-1])

        self.buffer = None
        if self.save_demos:
            buffer_path = os.path.join(self.cfg.runner.logger.log_path, "demos")
            self.log_info(f"Initializing ReplayBuffer at: {buffer_path}")
            self.buffer = TrajectoryReplayBuffer(
                seed=self.cfg.seed if hasattr(self.cfg, "seed") else 1234,
                enable_cache=False,
                auto_save=True,
                auto_save_path=buffer_path,
                trajectory_format="pt",
            )
        else:
            self.log_info("ReplayBuffer demo saving is disabled.")

        # Outer rate limiter for envs that don't self-pace (e.g. direct-stream).
        fps = dc_cfg.get("fps") if dc_cfg else None
        self._target_step_period = 1.0 / float(fps) if fps else None

    def _process_obs(self, obs):
        """Reshape env obs into the dict EmbodiedTrajectoryBuilder expects."""
        if not self.cfg.runner.record_task_description:
            obs.pop("task_descriptions", None)

        ret_obs = {}
        for key, val in obs.items():
            if isinstance(val, np.ndarray):
                val = torch.from_numpy(val)
            val = val.cpu()
            if key == "images":
                ret_obs["main_images"] = val.clone()
            else:
                ret_obs[key] = val.clone()
        return ret_obs

    def _wait_for_step_deadline(self, next_deadline: float) -> float:
        """Wait for an absolute deadline so scheduler oversleep does not drift."""
        if self._target_step_period is None:
            return next_deadline

        next_deadline += self._target_step_period
        sleep_for = next_deadline - time.perf_counter()
        if sleep_for > 0:
            time.sleep(sleep_for)
        return next_deadline

    def _stop_requested(self) -> bool:
        """Return whether the driver requested a graceful collection stop."""
        path = getattr(self, "_stop_request_path", None)
        return path is not None and path.exists()

    def run(self):
        try:
            self._run_collection()
        finally:
            try:
                self._hardware_env.close()
            finally:
                try:
                    if self.buffer is not None:
                        self.buffer.close()
                finally:
                    if self.env is not self._hardware_env:
                        self.log_info(
                            "Hardware collection stopped; finalizing collected data."
                        )
                        self.env.close()
                        self.log_info("Collected data finalized.")

    def _run_collection(self):
        obs, _ = self.env.reset()
        # Seed from preexisting episodes so resume bar + stop target line up.
        success_cnt = self._preexisting_success
        if success_cnt >= self.num_data_episodes:
            self.log_info(f"[resume] target {self.num_data_episodes} already met.")
            return
        progress_bar = tqdm(
            total=self.num_data_episodes,
            initial=success_cnt,
            desc="Collecting Data Episodes:",
        )

        current_rollout = None
        current_obs_processed = None
        if self.save_demos:
            current_rollout = EmbodiedTrajectoryBuilder(
                max_episode_length=self.cfg.env.eval.max_episode_steps,
            )
            current_obs_processed = self._process_obs(obs)

        next_step_deadline = time.perf_counter()
        while success_cnt < self.num_data_episodes and not self._stop_requested():
            # Teleop wrapper overrides this via info["intervene_action"].
            action = np.zeros((1, self.action_dim))
            next_obs, reward, terminated, truncated, info = self.env.step(action)

            # ``kb_phase is None`` ⇒ no keyboard wrapper attached → upstream "record every step".
            kb_event = info["keyboard_event"][0] if "keyboard_event" in info else None
            kb_phase = info["keyboard_phase"][0] if "keyboard_phase" in info else None
            if kb_event:
                self.log_info(f"[keyboard] {kb_event}")

            if "intervene_action" in info:
                action = info["intervene_action"]

            done = bool((terminated | truncated).any().item())

            if self.save_demos:
                next_obs_processed = self._process_obs(next_obs)
                terminated_tensor = torch.as_tensor(
                    terminated, dtype=torch.bool
                ).unsqueeze(1)
                truncated_tensor = torch.as_tensor(
                    truncated, dtype=torch.bool
                ).unsqueeze(1)
                done_tensor = terminated_tensor | truncated_tensor
                action_tensor = torch.as_tensor(action, dtype=torch.float32)
                reward_tensor = torch.as_tensor(reward, dtype=torch.float32).unsqueeze(
                    1
                )
                step_result = ChunkStepResult(
                    actions=action_tensor,
                    rewards=reward_tensor,
                    dones=done_tensor,
                    terminations=terminated_tensor,
                    truncations=truncated_tensor,
                    forward_inputs={"action": action_tensor},
                )

                # Rebuild rollout on rec-start or abort; ``restart`` supports
                # older keyboard wrappers.
                if kb_event in ("start", "restart", "abort"):
                    current_rollout = EmbodiedTrajectoryBuilder(
                        max_episode_length=self.cfg.env.eval.max_episode_steps,
                    )
                if kb_event != "start" and kb_phase in (None, "rec"):
                    current_rollout.append_step_result(step_result)
                    current_rollout.append_transitions(
                        curr_obs=current_obs_processed,
                        next_obs=next_obs_processed,
                    )
                current_obs_processed = next_obs_processed

            obs = next_obs

            if done:
                r_val = (
                    reward[0]
                    if hasattr(reward, "__getitem__") and len(reward) > 0
                    else reward
                )
                if hasattr(r_val, "item"):
                    r_val = r_val.item()

                manual_done = False
                if "manual_done" in info:
                    md = info["manual_done"]
                    if hasattr(md, "__getitem__") and len(md) > 0:
                        manual_done = bool(md[0])
                    else:
                        manual_done = bool(md)

                self.total_cnt += 1
                if self.manual_episode_control_only:
                    save_episode = bool(manual_done)
                else:
                    save_episode = bool(r_val >= 0.5 or manual_done)

                if save_episode:
                    success_cnt += 1

                    self.log_info(
                        f"Success (reward={r_val}, manual_done={manual_done}). "
                        f"Total: {success_cnt}/{self.num_data_episodes}"
                    )

                    if self.save_demos:
                        trajectory = current_rollout.to_trajectory()
                        trajectory.intervene_flags = torch.ones_like(
                            trajectory.intervene_flags
                        )
                        self.buffer.add_trajectories([trajectory])

                    progress_bar.update(1)
                else:
                    self.log_info(
                        f"Episode ended (reward={r_val:.2f}). "
                        f"Discarded. Total success: {success_cnt}/{self.num_data_episodes}"
                    )

                if success_cnt < self.num_data_episodes:
                    obs, _ = self.env.reset()
                    next_step_deadline = time.perf_counter()
                    if self.save_demos:
                        current_obs_processed = self._process_obs(obs)
                        current_rollout = EmbodiedTrajectoryBuilder(
                            max_episode_length=self.cfg.env.eval.max_episode_steps,
                        )

            if success_cnt < self.num_data_episodes:
                next_step_deadline = self._wait_for_step_deadline(next_step_deadline)

        if self._stop_requested():
            self.log_info(
                f"Graceful stop requested after {success_cnt} completed episodes."
            )
        elif self.save_demos:
            self.log_info(
                "Finished. Demos saved in: "
                f"{os.path.join(self.cfg.runner.logger.log_path, 'demos')}"
            )
        else:
            self.log_info("Finished. ReplayBuffer demo saving was disabled.")


@hydra.main(
    version_base="1.1", config_path="config", config_name="realworld_collect_data"
)
def main(cfg):
    _configure_keyboard_device(cfg)
    fifo_path = None
    stop_event = threading.Event()
    relay_thread = None
    stop_request_path = Path(cfg.runner.logger.log_path) / ".stop_collection"
    logger = get_logger()
    stop_request_path.unlink(missing_ok=True)
    if cfg.env.eval.get("keyboard_fifo_path") == "terminal":
        if not sys.stdin.isatty():
            raise RuntimeError("Terminal keyboard relay requires an interactive stdin")
        fifo_path = f"/tmp/rlinf_keyboard_{os.getpid()}.fifo"
        os.mkfifo(fifo_path, mode=0o600)
        cfg.env.eval.keyboard_fifo_path = fifo_path
        relay_thread = threading.Thread(
            target=_relay_terminal_keys,
            args=(fifo_path, stop_event),
            name="rlinf-terminal-key-relay",
            daemon=True,
        )
        relay_thread.start()

    try:
        cluster = Cluster(cluster_cfg=cfg.cluster)
        component_placement = ComponentPlacement(cfg, cluster)
        env_placement = component_placement.get_strategy("env")
        collector = DataCollector.create_group(cfg).launch(
            cluster, name=cfg.env.group_name, placement_strategy=env_placement
        )
        collection_work = collector.run()
        previous_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(
            signal.SIGINT,
            _GracefulStopSignalHandler(stop_request_path, logger),
        )
        try:
            collection_work.wait()
        finally:
            signal.signal(signal.SIGINT, previous_sigint_handler)
    finally:
        stop_request_path.unlink(missing_ok=True)
        stop_event.set()
        if relay_thread is not None:
            relay_thread.join(timeout=1.0)
        if fifo_path is not None:
            try:
                os.unlink(fifo_path)
            except FileNotFoundError:
                pass


if __name__ == "__main__":
    main()

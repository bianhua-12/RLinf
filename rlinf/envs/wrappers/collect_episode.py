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

from __future__ import annotations

import atexit
import copy
import json
import os
import pickle
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from threading import Lock
from typing import Any, Optional

import gymnasium as gym
import numpy as np
import torch

from rlinf.utils.logging import get_logger

_VALID_FORMATS = ("pickle", "lerobot")
_VALID_VIDEO_WRITE_MODES = ("lerobot_png", "stream_mp4")


_ID_DIR_RE = re.compile(r"^id_(\d+)$")


def _scan_existing_lerobot_shards(save_dir: str, rank: int) -> tuple[int, int]:
    """Return ``(total_episodes, next_shard_id)`` for resume.

    ``next_shard_id`` is ``max(existing_id_numbers) + 1`` over every
    ``id_<int>/`` directory (regardless of whether ``meta/info.json`` is
    finalized) so a crashed session's partial shard is never overwritten.
    Shards with unparseable ``info.json`` contribute 0 to ``total_episodes``.
    """
    rank_dir = os.path.join(save_dir, f"rank_{rank}")
    if not os.path.isdir(rank_dir):
        return 0, 0

    total = 0
    max_id = -1
    for entry in os.listdir(rank_dir):
        m = _ID_DIR_RE.match(entry)
        if m is None:
            continue
        if not os.path.isdir(os.path.join(rank_dir, entry)):
            continue
        max_id = max(max_id, int(m.group(1)))

        info_path = os.path.join(rank_dir, entry, "meta", "info.json")
        try:
            with open(info_path) as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            continue
        count = meta.get("total_episodes")
        if isinstance(count, int) and count > 0:
            total += count
    next_shard_id = max_id + 1 if max_id >= 0 else 0
    return total, next_shard_id


class CollectEpisode(gym.Wrapper):
    """Wrapper for collecting rollout data episode by episode.

    Records observations, actions, rewards, termination flags, and info dicts
    at each step. Completed episodes are asynchronously saved to disk in either
    pickle or LeRobot format.

    Supports both single and vectorized environments. When used with
    auto-resetting environments (those that embed ``final_observation`` in
    ``info``), the pre-reset observation is correctly attributed to the finished
    episode, and the post-reset observation is carried over to the next episode.

    Args:
        env: The gymnasium environment to wrap.
        save_dir: Directory for saving collected episode data.
        rank: Worker rank for file naming in distributed settings. Defaults to 0.
        num_envs: Number of parallel environments. Defaults to 1.
        show_goal_site: Whether to show goal visualization in renders.
            Defaults to True.
        export_format: Episode export format, ``"pickle"`` or ``"lerobot"``.
            Defaults to ``"pickle"``.
        robot_type: Robot type for LeRobot metadata. Defaults to ``"panda"``.
        fps: FPS for LeRobot metadata. Defaults to 10.
        use_videos: Whether to encode LeRobot camera features as MP4 videos.
            Defaults to False.
        only_success: Whether to save only successful episodes. Defaults to False.
        finalize_interval: Call ``writer.finalize()`` every this many completed
            episodes to flush ``info.json`` and ``stats.json`` as a checkpoint.
            ``0`` disables periodic flushing (lerobot only). Defaults to 100.
        defer_video_encoding_until_finalize: Keep video frames as PNGs while
            collecting and encode them only after the hardware loop stops.
            Requires ``use_videos=True`` and ``finalize_interval=0``.
        isolate_episode_stats: Compute LeRobot image statistics in a spawned
            process so their Python memory cleanup cannot pause collection.
        image_writer_threads: Number of LeRobot image-writer threads.
        image_writer_processes: Number of LeRobot image-writer processes. Use
            ``0`` inside Ray actors to avoid forking from the actor process.
        resume: If True and ``export_format == "lerobot"``, reuse ``save_dir``
            across sessions — new episodes land in a fresh ``id_{N}`` shard
            (N = sum of episodes across pre-existing shards) so the in-progress
            write never touches previously-finalized data. Ignored for pickle.
            Defaults to False.
        copy_observations: Whether to deep-copy observations before buffering.
            Set to False only when the wrapped environment returns newly
            allocated tensors or arrays on every reset and step. Defaults to
            True.
        video_write_mode: ``"lerobot_png"`` keeps the LeRobot compatibility
            writer. ``"stream_mp4"`` continuously encodes one MP4 per camera
            and episode, then publishes LeRobot v3 metadata transactionally.
        stream_video_queue_size: Maximum synchronized camera frame groups
            waiting for the spawned stream encoder. Queue overflow fails closed.
        stream_max_pending_commits: Maximum ready episodes awaiting the isolated
            metadata commit process.
        stream_commit_watchdog_timeout: Maximum age in seconds for the oldest
            pending metadata commit before collection fails closed.
    """

    def __init__(
        self,
        env: gym.Env,
        save_dir: str,
        rank: int = 0,
        num_envs: int = 1,
        show_goal_site: bool = True,
        export_format: str = "pickle",
        robot_type: str = "panda",
        fps: int = 10,
        use_videos: bool = False,
        only_success: bool = False,
        finalize_interval: int = 100,
        defer_video_encoding_until_finalize: bool = False,
        isolate_episode_stats: bool = False,
        image_writer_threads: int = 10,
        image_writer_processes: int = 0,
        resume: bool = False,
        copy_observations: bool = True,
        video_write_mode: str = "lerobot_png",
        stream_video_queue_size: int = 60,
        stream_max_pending_commits: int = 2,
        stream_commit_watchdog_timeout: float = 30.0,
    ):
        if isinstance(env, gym.Env):
            super().__init__(env)
        else:
            self.env = env

        if export_format not in _VALID_FORMATS:
            raise ValueError(
                f"Unsupported export_format={export_format!r}, "
                f"expected one of {_VALID_FORMATS}"
            )
        if defer_video_encoding_until_finalize and not use_videos:
            raise ValueError(
                "defer_video_encoding_until_finalize requires use_videos=True"
            )
        if defer_video_encoding_until_finalize and finalize_interval != 0:
            raise ValueError(
                "defer_video_encoding_until_finalize requires finalize_interval=0"
            )
        if video_write_mode not in _VALID_VIDEO_WRITE_MODES:
            raise ValueError(
                f"Unsupported video_write_mode={video_write_mode!r}, "
                f"expected one of {_VALID_VIDEO_WRITE_MODES}"
            )
        if video_write_mode == "stream_mp4":
            if export_format != "lerobot" or not use_videos:
                raise ValueError(
                    "video_write_mode='stream_mp4' requires "
                    "export_format='lerobot' and use_videos=True"
                )
            if defer_video_encoding_until_finalize:
                raise ValueError(
                    "stream_mp4 writes encoded video continuously and conflicts with "
                    "defer_video_encoding_until_finalize=True"
                )
            if num_envs != 1:
                raise ValueError("stream_mp4 currently requires num_envs=1")
            if int(stream_video_queue_size) <= 0:
                raise ValueError("stream_video_queue_size must be positive")
            if int(stream_max_pending_commits) <= 0:
                raise ValueError("stream_max_pending_commits must be positive")
            if float(stream_commit_watchdog_timeout) <= 0:
                raise ValueError("stream_commit_watchdog_timeout must be positive")
        self.save_dir = save_dir
        self.rank = rank
        self.num_envs = num_envs
        self.show_goal_site = show_goal_site
        self.export_format = export_format
        self.robot_type = robot_type
        self.fps = fps
        self.use_videos = use_videos
        self.only_success = only_success
        self.finalize_interval = finalize_interval
        self.defer_video_encoding_until_finalize = defer_video_encoding_until_finalize
        self.isolate_episode_stats = isolate_episode_stats
        self.image_writer_threads = image_writer_threads
        self.image_writer_processes = image_writer_processes
        self.copy_observations = copy_observations
        self.video_write_mode = video_write_mode
        self.stream_video_queue_size = int(stream_video_queue_size)
        self.stream_max_pending_commits = int(stream_max_pending_commits)
        self.stream_commit_watchdog_timeout = float(stream_commit_watchdog_timeout)

        self._preexisting_episode_count = 0
        self._next_shard_id = 0
        if export_format == "lerobot":
            self._lerobot_writer: Optional[Any] = None
            self._lerobot_lock = Lock()
            if resume and video_write_mode == "stream_mp4":
                from rlinf.data.storage.lerobot.streaming_writer import (
                    StreamingLeRobotDatasetWriter,
                )

                StreamingLeRobotDatasetWriter.recover_rank(save_dir, rank)
            if resume:
                (
                    self._preexisting_episode_count,
                    self._next_shard_id,
                ) = _scan_existing_lerobot_shards(save_dir, rank)
            self._episodes_written = (
                self._preexisting_episode_count
            )  # guarded by _lerobot_lock

        # Single-worker executor keeps write ordering deterministic.
        self._executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix=f"collect_episode_export_rank_{self.rank}",
        )
        self._futures: list[Future] = []

        # Per-environment episode state.
        self._episode_ids = [0] * num_envs
        self._episode_success = [False] * num_envs
        self._segment_ids: list[int] = [0] * num_envs
        self._global_step = 0
        # Holds the post-reset obs for auto-reset envs to prepend to next episode.
        self._pending_obs: list[Any] = [None] * num_envs
        self._pending_info: list[Any] = [None] * num_envs
        self._buffers: list[dict[str, list]] = [
            self._new_buffer() for _ in range(num_envs)
        ]

        self._closed = False
        self.logger = get_logger()

        os.makedirs(self.save_dir, exist_ok=True)
        atexit.register(self._finalize_on_exit)

    @property
    def preexisting_episode_count(self) -> int:
        """Number of episodes on disk at construction time (resume mode only)."""
        return self._preexisting_episode_count

    @property
    def is_start(self):
        return getattr(self.env, "is_start")

    @is_start.setter
    def is_start(self, value):
        setattr(self.env, "is_start", value)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ):
        """Reset the environment and initialise episode buffers.

        Args:
            seed: Optional random seed for environment reset.
            options: Optional dictionary of reset options.

        Returns:
            Tuple of (observation, info) from the underlying environment.
        """
        if self.video_write_mode == "stream_mp4":
            self._abort_stream_episode()
        self._buffers = [self._new_buffer() for _ in range(self.num_envs)]
        self._episode_success = [False] * self.num_envs
        self._pending_obs = [None] * self.num_envs
        self._pending_info = [None] * self.num_envs

        try:
            obs, info = self.env.reset(seed=seed, options=options)
        except TypeError:
            obs, info = self.env.reset()

        self._show_goal_site_visual()
        self._record_reset_obs(obs, info)
        return obs, info

    def step(self, action, **kwargs):
        """Execute a step and record the transition.

        Args:
            action: Action to execute in the environment.
            **kwargs: Additional arguments forwarded to the underlying step.

        Returns:
            Tuple of (obs, reward, terminated, truncated, info).
        """
        obs, reward, terminated, truncated, info = self.env.step(action, **kwargs)
        self._record_step(action, obs, reward, terminated, truncated, info)
        self._maybe_flush(terminated, truncated)
        return obs, reward, terminated, truncated, info

    def chunk_step(self, chunk_actions):
        """Execute a chunk of actions, recording each sub-step individually.

        Both pickle and lerobot formats receive step-level records for maximum
        data fidelity.

        Args:
            chunk_actions: Action chunk, typically a tensor of shape
                ``[num_envs, chunk_size, action_dim]``.

        Returns:
            Tuple of (obs_list, rewards, terminations, truncations, infos_list).
        """
        obs_list, rewards, terminations, truncations, infos_list = self.env.chunk_step(
            chunk_actions
        )

        chunk_size = len(obs_list) if isinstance(obs_list, (list, tuple)) else 1
        for step_idx in range(chunk_size):
            step_action = (
                chunk_actions[:, step_idx]
                if isinstance(chunk_actions, (torch.Tensor, np.ndarray))
                and chunk_actions.ndim > 1
                else chunk_actions
            )
            step_obs = (
                obs_list[step_idx] if isinstance(obs_list, (list, tuple)) else obs_list
            )
            step_reward = (
                rewards[:, step_idx] if getattr(rewards, "ndim", 1) > 1 else rewards
            )
            step_term = (
                terminations[:, step_idx]
                if getattr(terminations, "ndim", 1) > 1
                else terminations
            )
            step_trunc = (
                truncations[:, step_idx]
                if getattr(truncations, "ndim", 1) > 1
                else truncations
            )
            step_info = (
                copy.deepcopy(infos_list[step_idx])
                if isinstance(infos_list, (list, tuple))
                else infos_list
            )
            self._record_step(
                step_action, step_obs, step_reward, step_term, step_trunc, step_info
            )
            self._maybe_flush(step_term, step_trunc)

        return obs_list, rewards, terminations, truncations, infos_list

    def close(self):
        if self._closed:
            # Resource shutdown is one-shot, but a transient finalization error
            # may leave the writer available for an explicit best-effort retry.
            self._finalize_lerobot()
            return None
        self._closed = True
        first_error: Optional[BaseException] = None
        result = None
        try:
            self._finalize_lerobot()
            self._wait_futures()
        except BaseException as exc:
            first_error = exc
        finally:
            if self._executor is not None:
                try:
                    self._executor.shutdown(wait=True)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                self._executor = None
            if hasattr(self.env, "close"):
                try:
                    result = self.env.close()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
        if first_error is not None:
            raise first_error
        return result

    def _new_buffer(self) -> dict[str, list]:
        return {
            "observations": [],
            "actions": [],
            "rewards": [],
            "terminated": [],
            "truncated": [],
            "infos": [],
            "segment_ids": [],
            "observation_timestamps_ns": [],
        }

    def _seed_reset_frame(
        self,
        env_idx: int,
        env_obs: Any,
        observation_timestamp_ns: int,
        env_info: Any = None,
    ) -> None:
        """Seed a fresh buffer with the post-reset state-aligned entry.

        State-aligned fields (observations / rewards / terminated / truncated /
        infos / observation timestamps) get a leading reset entry;
        action-aligned fields (actions, segment_ids) stay empty and fill on the
        first regular step.
        """
        buf = self._buffers[env_idx]
        buf["observations"].append(env_obs)
        buf["rewards"].append(0.0)
        buf["terminated"].append(False)
        buf["truncated"].append(False)
        buf["infos"].append(env_info if env_info is not None else {})
        buf["observation_timestamps_ns"].append(observation_timestamp_ns)

    def _record_reset_obs(self, obs, info) -> None:
        """Record the initial observation from reset into every env's buffer."""
        for env_idx in range(self.num_envs):
            env_info = self._slice_copy(info, env_idx)
            self._seed_reset_frame(
                env_idx,
                self._slice_observation(obs, env_idx),
                self._observation_timestamp_ns(env_info),
                env_info,
            )

    @staticmethod
    def _bool_from_env_info(env_info: Any, key: str) -> bool:
        """Read a per-env bool flag (scalar / 0-d / size-1 array all OK)."""
        if not isinstance(env_info, dict) or env_info.get(key) is None:
            return False
        return bool(np.asarray(env_info[key]).any())

    def _record_step(self, action, obs, reward, terminated, truncated, info) -> None:
        """Record one transition into every env's buffer."""
        self._global_step += 1

        has_final_obs = isinstance(info, dict) and "final_observation" in info
        if has_final_obs:
            final_observation = info["final_observation"]
            final_info_batch = info["final_info"]
            info_no_reset = {
                key: value
                for key, value in info.items()
                if key not in ("final_observation", "final_info")
            }

        for env_idx in range(self.num_envs):
            # Auto-reset envs store the pre-reset obs in info["final_observation"];
            # the current `obs` is the post-reset obs for the *next* episode.
            # Only use final_observation for envs that are actually done this step.
            env_done = self._scalar_flag(terminated, env_idx) or self._scalar_flag(
                truncated, env_idx
            )
            if has_final_obs and env_done:
                obs_source = final_observation
                info_source = final_info_batch
            else:
                obs_source = obs
                info_source = info_no_reset if has_final_obs else info

            # Inspect the small info structure before taking ownership of image
            # observations. Pre-record frames must not enter the data buffer.
            env_info_view = self._slice_data(info_source, env_idx)
            record_reset = self._bool_from_env_info(env_info_view, "record_reset")
            pre_record = self._bool_from_env_info(env_info_view, "pre_record")

            if pre_record:
                continue

            env_info = self._slice_copy(info_source, env_idx)
            env_obs = self._slice_observation(obs_source, env_idx)

            if has_final_obs and env_done:
                self._pending_obs[env_idx] = self._slice_observation(obs, env_idx)
                self._pending_info[env_idx] = self._slice_copy(info_no_reset, env_idx)
                if "intervene_action" in env_info:
                    env_info["intervene_action"] = env_info["intervene_action"][-1]
                    env_info["intervene_flag"] = env_info["intervene_flag"][-1]

            if record_reset:
                if self.video_write_mode == "stream_mp4":
                    self._abort_stream_episode()
                self._buffers[env_idx] = self._new_buffer()
                self._episode_success[env_idx] = False
                self._segment_ids[env_idx] = 0
                self._seed_reset_frame(
                    env_idx,
                    env_obs,
                    self._observation_timestamp_ns(env_info),
                    env_info,
                )
                continue

            if self._bool_from_env_info(env_info, "segment_advance"):
                self._segment_ids[env_idx] += 1

            buf = self._buffers[env_idx]
            if self.video_write_mode == "stream_mp4":
                frame = self._transition_to_lerobot_frame(
                    buf, env_idx, action, env_info
                )
                if frame is None:
                    raise RuntimeError(
                        "stream_mp4 cannot skip a frame with missing state/action; "
                        "failing closed to preserve frame alignment"
                    )
                self._ensure_streaming_writer(frame).append_frame(frame)
                # The encoder owns the prior images now. Retain only the task
                # description needed at successful episode commit.
                prior = buf["observations"][-1]
                if isinstance(prior, dict):
                    buf["observations"][-1] = {
                        "task_descriptions": prior.get(
                            "task_descriptions", "unknown task"
                        )
                    }
            buf["observations"].append(env_obs)
            buf["actions"].append(self._slice_copy(action, env_idx))
            buf["rewards"].append(self._slice_copy(reward, env_idx))
            buf["terminated"].append(self._slice_copy(terminated, env_idx))
            buf["truncated"].append(self._slice_copy(truncated, env_idx))
            buf["infos"].append(env_info)
            buf["segment_ids"].append(int(self._segment_ids[env_idx]))
            buf["observation_timestamps_ns"].append(
                self._observation_timestamp_ns(env_info)
            )

            self._update_success(env_idx, env_info)

    def _reset_env_buffer(self, env_idx: int) -> None:
        """Advance episode counter, clear the buffer, and carry over pending obs."""
        if self.video_write_mode == "stream_mp4":
            self._abort_stream_episode()
        self._episode_ids[env_idx] += 1
        self._buffers[env_idx] = self._new_buffer()
        self._episode_success[env_idx] = False
        self._segment_ids[env_idx] = 0

        if self._pending_obs[env_idx] is not None:
            pending_info = self._pending_info[env_idx]
            self._seed_reset_frame(
                env_idx,
                self._pending_obs[env_idx],
                self._observation_timestamp_ns(pending_info),
                pending_info,
            )
            self._pending_obs[env_idx] = None
            self._pending_info[env_idx] = None

    def _maybe_flush(self, terminated, truncated) -> None:
        """Save finished episodes and reset their buffers."""
        for env_idx in range(self.num_envs):
            done_by_term = self._scalar_flag(terminated, env_idx)
            done_by_trunc = self._scalar_flag(truncated, env_idx)
            if not (done_by_term or done_by_trunc):
                continue

            is_success = self._get_episode_success(self._buffers[env_idx], env_idx)
            if self.only_success:
                if is_success and done_by_term:
                    self._flush_episode(env_idx, is_success)
                    self._reset_env_buffer(env_idx)
                else:
                    if done_by_trunc:
                        self._reset_env_buffer(env_idx)
            else:
                if done_by_term or done_by_trunc:
                    self._flush_episode(env_idx, is_success)
                    self._reset_env_buffer(env_idx)

    def _flush_episode(self, env_idx: int, is_success: bool) -> None:
        """Dispatch a completed episode to the appropriate format writer."""
        self.logger.info(f"Flush env {env_idx}")
        buf = self._buffers[env_idx]
        if not buf["actions"]:
            return

        if self.export_format == "lerobot":
            if self.video_write_mode == "stream_mp4":
                if self._lerobot_writer is None or not self._lerobot_writer.active:
                    raise RuntimeError(
                        "stream_mp4 episode has actions but no active encoder"
                    )
                self._lerobot_writer.finish_episode(
                    task=self._extract_task_description(buf, env_idx),
                    is_success=is_success,
                )
                self._episodes_written += 1
                return
            ep_data = self._buffer_to_lerobot_ep(buf, env_idx, is_success)
            if ep_data is not None:
                self._submit(self._write_lerobot_episode, ep_data)
        else:
            episode_data = self._copy(
                {
                    "rank": self.rank,
                    "env_idx": env_idx,
                    "episode_id": self._episode_ids[env_idx],
                    "step": self._global_step,
                    "success": is_success,
                    "observations": buf["observations"],
                    "actions": buf["actions"],
                    "rewards": buf["rewards"],
                    "terminated": buf["terminated"],
                    "truncated": buf["truncated"],
                    "infos": buf["infos"],
                    "observation_timestamps_ns": buf["observation_timestamps_ns"],
                }
            )
            label = "success" if is_success else "fail"
            filename = (
                f"rank_{self.rank}_env_{env_idx}_"
                f"episode_{self._episode_ids[env_idx]}_"
                f"step_{self._global_step}_"
                f"{label}.pkl"
            )
            self._submit(
                self._write_pickle, os.path.join(self.save_dir, filename), episode_data
            )

    def _buffer_to_lerobot_ep(
        self, buf: dict, env_idx: int, is_success: bool
    ) -> Optional[list[dict[str, Any]]]:
        """Convert a raw episode buffer into a list of per-step frame dicts.
        Produces the format expected by ``LeRobotDatasetWriter.add_episode``:
        a ``list[dict]`` where every dict represents one step and carries the
        fields ``image``, ``state``, ``actions``, ``task``, ``is_success``,
        ``done``, ``intervene_flag``, ``observation_timestamp_ns``, and
        optionally ``wrist_image`` / ``extra_view_image``.
        The observations list contains one extra entry prepended at reset time,
        so it is aligned to the actions list by taking the leading N entries.
        Steps where any required field (image, state, action) is missing are
        silently skipped.
        Args:
            buf: Raw episode buffer produced by ``_new_buffer``.
            env_idx: Index of the parallel environment this buffer belongs to.
            is_success: Whether the episode was successful.
        Returns:
            A list of per-step frame dicts, or ``None`` if no valid frames
            could be extracted.
        """
        actions = buf["actions"]
        terminated = buf["terminated"]
        obs_steps = buf["observations"]
        observation_timestamps_ns = buf.get("observation_timestamps_ns", [])
        seg_ids = buf.get("segment_ids", [])
        if not actions:
            return None
        if len(obs_steps) > len(actions):
            obs_steps = obs_steps[: len(actions)]
        task_desc = self._extract_task_description(buf, env_idx)
        steps: list[dict[str, Any]] = []
        first_term_step: Optional[int] = None
        for i, action in enumerate(actions):
            obs = obs_steps[i] if i < len(obs_steps) else None
            image, wrist_image, extra_view_image, state = self._extract_obs_image_state(
                obs
            )
            # Overwrite action with intervene action if present.
            np_action = self._to_numpy(action)
            raw_info = buf["infos"][i + 1]
            if isinstance(raw_info, dict) and "final_info" in raw_info:
                # _record_step normally pops final_info before storing; if it
                # leaked through (e.g. nested wrapper that also auto-resets)
                # drop the frame instead of crashing the writer.
                self.logger.warning(
                    "collect_episode: dropping frame %d because info still "
                    "carries final_info; check upstream auto-reset wrappers",
                    i,
                )
                continue
            info_with_intervene = copy.deepcopy(raw_info)

            if (
                "intervene_flag" in info_with_intervene
                and "intervene_action" in info_with_intervene
            ):
                if info_with_intervene["intervene_flag"].all():
                    np_action = self._to_numpy(info_with_intervene["intervene_action"])
            if state is None or np_action is None:
                continue
            intervene_flag = self._intervene_flag_from_info(info_with_intervene)
            seg_id = int(seg_ids[i]) if i < len(seg_ids) else 0
            frame: dict[str, Any] = {
                "state": np.asarray(state).astype(np.float32),
                "actions": np.asarray(np_action).astype(np.float32).flatten(),
                "task": task_desc,
                "is_success": np.array([is_success], dtype=bool),
                "done": np.array([False], dtype=bool),
                "intervene_flag": np.array([intervene_flag], dtype=bool),
                "segment_id": np.array([seg_id], dtype=np.uint8),
            }
            if i < len(observation_timestamps_ns):
                frame["observation_timestamp_ns"] = np.array(
                    [observation_timestamps_ns[i]], dtype=np.int64
                )
            if image is not None:
                frame["image"] = self._to_uint8(np.asarray(image))
            for key, img in self._expand_multi_view_images(
                "wrist_image", wrist_image
            ).items():
                frame[key] = self._to_uint8(np.asarray(img))
            for key, img in self._expand_multi_view_images(
                "extra_view_image", extra_view_image
            ).items():
                frame[key] = self._to_uint8(np.asarray(img))
            steps.append(frame)
            if bool(terminated[i]) and first_term_step is None:
                first_term_step = len(steps)
        if not steps:
            return None
        end = first_term_step if first_term_step is not None else len(steps)
        steps = steps[:end]
        steps[-1]["done"] = np.array([True], dtype=bool)
        return steps

    def _ensure_lerobot_writer(self, ep_data: dict):
        """Get-or-create the LeRobot writer. Must be called under ``_lerobot_lock``."""
        from rlinf.data.storage.lerobot import LeRobotDatasetWriter

        if self._lerobot_writer is None:
            self._lerobot_writer = LeRobotDatasetWriter()
        shard_id = self._next_shard_id

        if self._lerobot_writer.dataset is None:
            first = ep_data[0]
            wrist_image_keys = self._collect_image_keys(first, "wrist_image")
            extra_view_image_keys = self._collect_image_keys(first, "extra_view_image")
            self._lerobot_writer.create(
                repo_id=os.path.join(
                    self.save_dir, f"rank_{self.rank}", f"id_{shard_id}"
                ),
                robot_type=self.robot_type,
                fps=self.fps,
                image_shape=first["image"].shape if "image" in first else None,
                state_dim=int(first["state"].shape[-1]),
                action_dim=int(first["actions"].shape[-1]),
                has_image="image" in first,
                wrist_image_keys=wrist_image_keys,
                extra_view_image_keys=extra_view_image_keys,
                has_intervene_flag="intervene_flag" in first,
                has_segment_id="segment_id" in first,
                has_observation_timestamp="observation_timestamp_ns" in first,
                use_videos=self.use_videos,
                defer_video_encoding_until_finalize=(
                    self.defer_video_encoding_until_finalize
                ),
                isolate_episode_stats=self.isolate_episode_stats,
                image_writer_threads=self.image_writer_threads,
                image_writer_processes=self.image_writer_processes,
            )
            self._next_shard_id = shard_id + 1
        return self._lerobot_writer

    def _transition_to_lerobot_frame(
        self, buf: dict, env_idx: int, action: Any, env_info: Any
    ) -> Optional[dict[str, Any]]:
        """Build the current action-aligned frame from the preceding observation."""
        if not buf["observations"]:
            return None
        observation = buf["observations"][-1]
        image, wrist_image, extra_view_image, state = self._extract_obs_image_state(
            observation
        )
        np_action = self._to_numpy(self._slice_copy(action, env_idx))
        if isinstance(env_info, dict):
            intervene = env_info.get("intervene_flag")
            intervene_action = env_info.get("intervene_action")
            if intervene is not None and intervene_action is not None:
                intervene_array = np.asarray(self._to_numpy(intervene), dtype=bool)
                if intervene_array.all():
                    np_action = self._to_numpy(intervene_action)
        if state is None or np_action is None:
            return None
        frame: dict[str, Any] = {
            "state": np.asarray(state, dtype=np.float32),
            "actions": np.asarray(np_action, dtype=np.float32).flatten(),
            "is_success": np.asarray([False], dtype=bool),
            "done": np.asarray([False], dtype=bool),
            "intervene_flag": np.asarray(
                [self._intervene_flag_from_info(env_info)], dtype=bool
            ),
            "segment_id": np.asarray([self._segment_ids[env_idx]], dtype=np.uint8),
        }
        timestamps = buf.get("observation_timestamps_ns", [])
        if timestamps:
            frame["observation_timestamp_ns"] = np.asarray(
                [timestamps[-1]], dtype=np.int64
            )
        if image is not None:
            frame["image"] = self._to_uint8(np.asarray(image))
        for key, value in self._expand_multi_view_images(
            "wrist_image", wrist_image
        ).items():
            frame[key] = self._to_uint8(np.asarray(value))
        for key, value in self._expand_multi_view_images(
            "extra_view_image", extra_view_image
        ).items():
            frame[key] = self._to_uint8(np.asarray(value))
        return frame

    def _ensure_streaming_writer(self, frame: dict[str, Any]):
        from rlinf.data.storage.lerobot.streaming_writer import (
            StreamingLeRobotDatasetWriter,
        )

        if self._lerobot_writer is None:
            writer = StreamingLeRobotDatasetWriter(
                queue_size=self.stream_video_queue_size,
                max_pending_commits=self.stream_max_pending_commits,
                commit_watchdog_timeout=self.stream_commit_watchdog_timeout,
            )
            try:
                wrist_image_keys = self._collect_image_keys(frame, "wrist_image")
                extra_view_image_keys = self._collect_image_keys(
                    frame, "extra_view_image"
                )
                shard_id = self._next_shard_id
                writer.create(
                    repo_id=os.path.join(
                        self.save_dir, f"rank_{self.rank}", f"id_{shard_id}"
                    ),
                    robot_type=self.robot_type,
                    fps=self.fps,
                    image_shape=frame["image"].shape if "image" in frame else None,
                    state_dim=int(frame["state"].shape[-1]),
                    action_dim=int(frame["actions"].shape[-1]),
                    has_image="image" in frame,
                    wrist_image_keys=wrist_image_keys,
                    extra_view_image_keys=extra_view_image_keys,
                    has_intervene_flag="intervene_flag" in frame,
                    has_segment_id="segment_id" in frame,
                    has_observation_timestamp="observation_timestamp_ns" in frame,
                )
            except BaseException:
                try:
                    writer.finalize()
                except BaseException:
                    pass
                raise
            self._next_shard_id = shard_id + 1
            self._lerobot_writer = writer
        return self._lerobot_writer

    def _abort_stream_episode(self) -> None:
        if self._lerobot_writer is not None and self._lerobot_writer.active:
            self._lerobot_writer.abort_episode()

    @staticmethod
    def _collect_image_keys(
        frame: dict[str, Any],
        prefix: str,
    ) -> dict[str, tuple[int, ...]]:
        """Return ``{key: (H, W, C)}`` for all frame keys matching *prefix*.

        Matches both the bare ``prefix`` (e.g. ``wrist_image``) and indexed
        variants (``wrist_image-0``, ``wrist_image-1``, …). The separator is a
        hyphen, not ``/``: lerobot >= 0.3 rejects feature names containing
        ``/`` in ``LeRobotDatasetMetadata.create``.
        """
        return {
            k: tuple(frame[k].shape)
            for k in frame
            if (k == prefix or k.startswith(f"{prefix}-"))
            and isinstance(frame[k], np.ndarray)
            and frame[k].ndim == 3
        }

    def _write_lerobot_episode(self, ep_data: dict) -> None:
        with self._lerobot_lock:
            writer = self._ensure_lerobot_writer(ep_data)
            writer.add_episode(ep_data, consume=True)
            self._episodes_written += 1
            count = self._episodes_written
            if self.finalize_interval > 0 and count % self.finalize_interval == 0:
                writer.finalize()

    def _finalize_lerobot(self) -> None:
        """Drain pending futures then write the LeRobot dataset metadata."""
        if self.export_format != "lerobot":
            return
        first_error: Optional[BaseException] = None
        try:
            self._wait_futures()
        except BaseException as exc:
            first_error = exc

        finalize_succeeded = False
        try:
            with self._lerobot_lock:
                if self._lerobot_writer is not None and (
                    self.video_write_mode == "stream_mp4"
                    or self._lerobot_writer.dataset is not None
                ):
                    if first_error is None:
                        self._lerobot_writer.finalize()
                    else:
                        self._lerobot_writer.finalize(prior_error=first_error)
                finalize_succeeded = True
        except BaseException as exc:
            if first_error is None:
                first_error = exc
        finally:
            if finalize_succeeded:
                self._lerobot_writer = None

        if first_error is not None:
            raise first_error

    def _write_pickle(self, save_path: str, episode_data: dict) -> None:
        with open(save_path, "wb") as f:
            pickle.dump(episode_data, f)

    def _submit(self, fn, *args) -> None:
        if self._executor is None:
            return
        self._futures.append(self._executor.submit(fn, *args))
        self.logger.info(f"Futures queue length: {len(self._futures)}")
        self._drain_futures()

    def _drain_futures(self) -> None:
        remaining = []
        for f in self._futures:
            if f.done():
                f.result()
            else:
                remaining.append(f)
        self._futures = remaining

    def _wait_futures(self) -> None:
        first_error: Optional[BaseException] = None
        for f in self._futures:
            try:
                f.result()
            except BaseException as exc:
                if first_error is None:
                    first_error = exc
        self._futures = []
        if first_error is not None:
            raise first_error

    def _finalize_on_exit(self) -> None:
        self.close()

    def _update_success(self, env_idx: int, env_info) -> None:
        """Update the per-env success flag from a single-env info dict."""
        if not isinstance(env_info, dict):
            return

        success = self._extract_success_from_info(env_info)
        if success is not None:
            # Keep success sticky during an episode.
            self._episode_success[env_idx] = self._episode_success[env_idx] or success

    def _get_episode_success(self, buf: dict, env_idx: int) -> bool:
        """Determine final episode success by scanning recorded info dicts.

        Checks (in priority order): ``final_info``, ``episode``, and the root
        info dict, looking for ``success_once``, ``success_at_end``, and
        ``success`` keys. Falls back to the incrementally-updated
        ``_episode_success`` flag.
        """
        if self._episode_success[env_idx]:
            return True

        found_any = False
        is_success = False

        for info in buf["infos"]:
            if not isinstance(info, dict):
                continue
            success = self._extract_success_from_info(info)
            if success is not None:
                found_any = True
                is_success = is_success or success

        if found_any:
            return is_success
        return self._episode_success[env_idx]

    @staticmethod
    def _intervene_flag_from_info(info: Any) -> bool:
        """Whether this timestep used human / expert intervention (per-env info)."""
        if not isinstance(info, dict):
            return False
        val = info.get("intervene_flag")
        if val is None:
            return False
        arr = CollectEpisode._to_numpy(val)
        if arr is None:
            return False
        return bool(np.asarray(arr, dtype=bool).reshape(-1).any())

    @staticmethod
    def _to_bool_scalar(val) -> Optional[bool]:
        if val is None:
            return None
        if isinstance(val, torch.Tensor):
            if val.numel() != 1:
                return None
            return bool(val.item())
        if isinstance(val, np.ndarray):
            if val.size != 1:
                return None
            return bool(val.reshape(-1)[0])
        return bool(val)

    def _extract_success_from_source(self, src) -> Optional[bool]:
        if not isinstance(src, dict):
            return None
        for key in ("success_once", "success_at_end", "success"):
            val = self._to_bool_scalar(src.get(key))
            if val is not None:
                return val
        return None

    def _extract_success_from_info(self, info: dict) -> Optional[bool]:
        """Extract success with episode-level fields taking priority."""
        episode_values: list[bool] = []

        final_info = info.get("final_info", None)
        if isinstance(final_info, dict):
            final_info_success = self._extract_success_from_source(final_info)
            if final_info_success is not None:
                episode_values.append(final_info_success)
            final_episode_success = self._extract_success_from_source(
                final_info.get("episode")
            )
            if final_episode_success is not None:
                episode_values.append(final_episode_success)

        current_episode_success = self._extract_success_from_source(info.get("episode"))
        if current_episode_success is not None:
            episode_values.append(current_episode_success)

        if episode_values:
            return any(episode_values)

        return self._extract_success_from_source(info)

    def _extract_task_description(self, buf: dict, env_idx: int) -> str:
        for obs in reversed(buf["observations"]):
            if not isinstance(obs, dict) or "task_descriptions" not in obs:
                continue
            desc = obs["task_descriptions"]
            if isinstance(desc, (list, tuple)):
                return str(desc[env_idx] if len(desc) == self.num_envs else desc[0])
            return str(desc)
        return "unknown task"

    def _extract_obs_image_state(self, obs):
        """Return ``(image, wrist_image, extra_view_image, state)`` from an obs dict.

        ``wrist_image`` and ``extra_view_image`` are returned as raw numpy
        arrays and may have shape ``[H, W, C]`` *or* ``[N, H, W, C]``.
        Use :meth:`_expand_multi_view_images` to fan them out into
        individually-keyed views before writing.
        """
        if not isinstance(obs, dict):
            return None, None, None, None
        image = obs.get("main_images", obs.get("image", obs.get("full_image")))
        wrist_image = obs.get("wrist_images", obs.get("wrist_image"))
        extra_view_image = obs.get("extra_view_images", obs.get("extra_view_image"))
        state = obs.get("states", obs.get("state"))
        return (
            self._to_numpy(image),
            self._to_numpy(wrist_image),
            self._to_numpy(extra_view_image),
            self._to_numpy(state),
        )

    @staticmethod
    def _expand_multi_view_images(
        base_key: str,
        arr: Optional[np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Expand a potentially batched image array into per-view entries.

        * ``[H, W, C]``           → ``{base_key: img}``
        * ``[1, H, W, C]``        → ``{base_key: img[0]}``
        * ``[N, H, W, C]`` (N>1)  → ``{base_key-0: img[0], …, base_key-N-1: img[N-1]}``
        """
        if arr is None:
            return {}
        if arr.ndim == 3:
            return {base_key: arr}
        if arr.ndim == 4:
            if arr.shape[0] == 1:
                return {base_key: arr[0]}
            return {f"{base_key}-{i}": arr[i] for i in range(arr.shape[0])}
        return {base_key: arr}

    def _slice_data(self, data, env_idx: int):
        """Slice batched data for a single env without copying."""
        if isinstance(data, torch.Tensor):
            return (
                data[env_idx]
                if data.dim() > 0 and data.shape[0] == self.num_envs
                else data
            )
        if isinstance(data, np.ndarray):
            return (
                data[env_idx]
                if data.ndim > 0 and data.shape[0] == self.num_envs
                else data
            )
        if isinstance(data, dict):
            return {k: self._slice_data(v, env_idx) for k, v in data.items()}
        if isinstance(data, list):
            return data[env_idx] if len(data) == self.num_envs else data
        return data

    def _slice_copy(self, data, env_idx: int):
        """Slice batched data for a single env and deep-copy the result."""
        return self._copy(self._slice_data(data, env_idx))

    def _slice_observation(self, data, env_idx: int):
        """Slice an observation and either copy it or take shared ownership."""
        sliced = self._slice_data(data, env_idx)
        if self.copy_observations:
            return self._copy(sliced)
        return self._own_observation(sliced)

    def _own_observation(self, data):
        """Copy containers while retaining fresh CPU tensor and array storage."""
        if isinstance(data, torch.Tensor):
            detached = data.detach()
            return detached.cpu() if detached.device.type != "cpu" else detached
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, dict):
            return {key: self._own_observation(value) for key, value in data.items()}
        if isinstance(data, list):
            return [self._own_observation(value) for value in data]
        if isinstance(data, tuple):
            return tuple(self._own_observation(value) for value in data)
        return data

    @staticmethod
    def _observation_timestamp_ns(env_info: Any) -> int:
        """Extract a monotonic observation timestamp, with a local fallback."""
        if isinstance(env_info, dict):
            value = env_info.get("observation_timestamp_ns")
            if value is not None:
                array = CollectEpisode._to_numpy(value)
                if array is not None and array.size:
                    return int(array.reshape(-1)[0])
        return time.monotonic_ns()

    def _scalar_flag(self, flags, env_idx: int) -> bool:
        """Extract a boolean flag for ``env_idx`` from a batched flag."""
        if isinstance(flags, torch.Tensor):
            if flags.dim() > 1:
                return bool(flags[env_idx, -1].item())
            if flags.dim() == 1 and flags.shape[0] == self.num_envs:
                return bool(flags[env_idx].item())
            return bool(flags.item())
        if isinstance(flags, np.ndarray):
            if flags.ndim > 0 and flags.shape[0] == self.num_envs:
                return bool(flags[env_idx])
            return bool(flags)
        return bool(flags)

    @staticmethod
    def _to_numpy(data) -> Optional[np.ndarray]:
        if data is None:
            return None
        if isinstance(data, np.ndarray):
            return data
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().numpy()
        return np.asarray(data)

    @staticmethod
    def _to_uint8(arr: np.ndarray) -> np.ndarray:
        if arr.dtype == np.uint8:
            return arr
        return (
            (arr * 255).astype(np.uint8) if arr.max() <= 1.0 else arr.astype(np.uint8)
        )

    def _copy(self, data):
        if isinstance(data, torch.Tensor):
            return data.detach().cpu().clone()
        if isinstance(data, np.ndarray):
            return data.copy()
        if isinstance(data, dict):
            return {k: self._copy(v) for k, v in data.items()}
        if isinstance(data, list):
            return [self._copy(item) for item in data]
        if isinstance(data, tuple):
            return tuple(self._copy(item) for item in data)
        return data

    def _show_goal_site_visual(self) -> None:
        """Unhide the goal site in environments that support it."""
        if not self.show_goal_site:
            return

        unwrapped = self.env
        while hasattr(unwrapped, "env"):
            unwrapped = unwrapped.env
        if hasattr(unwrapped, "unwrapped"):
            unwrapped = unwrapped.unwrapped

        if not hasattr(unwrapped, "goal_site"):
            return

        goal_site = unwrapped.goal_site
        if hasattr(unwrapped, "_hidden_objects"):
            while goal_site in unwrapped._hidden_objects:
                unwrapped._hidden_objects.remove(goal_site)
        if hasattr(goal_site, "show_visual"):
            goal_site.show_visual()

    def update_reset_state_ids(self):
        if hasattr(self.env, "update_reset_state_ids"):
            self.env.update_reset_state_ids()

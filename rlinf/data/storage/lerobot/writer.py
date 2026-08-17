# Copyright 2025 The RLinf Authors.
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

"""LeRobot dataset writer for saving rollout data."""

import gc
import importlib
import multiprocessing
import sys
from concurrent.futures import ProcessPoolExecutor
from threading import Lock
from typing import Any

from rlinf.data.storage.lerobot.compat import add_frame_to_dataset
from rlinf.utils.logging import get_logger

_COMPUTE_STATS_PATCH_LOCK = Lock()


def _silence_hf_datasets_progress_bars() -> None:
    # Disable HF ``datasets`` Map / parquet tqdm bars; called from
    # ``create()`` so importing this module doesn't affect other consumers.
    try:
        import datasets as _hf_datasets

        _hf_datasets.disable_progress_bar()
    except ImportError:
        pass


def _compute_episode_stats_in_process(
    episode_data: dict[str, Any], features: dict[str, Any]
) -> dict[str, Any]:
    """Compute LeRobot episode statistics outside the collector process."""
    from lerobot.datasets.compute_stats import compute_episode_stats

    return compute_episode_stats(episode_data, features)


class LeRobotDatasetWriter:
    """
    Wrapper for LeRobotDataset that provides a simplified interface for writing episodes.

    Usage:
        writer = LeRobotDatasetWriter()
        writer.create(
            repo_id="my-dataset",
            robot_type="franka_panda",
            fps=5,
            features={...}
        )

        for episode_data in episodes:
            writer.add_episode(episode_data)

        writer.finalize(push_to_hub=False)
    """

    def __init__(self):
        """Initialize the writer."""
        self.dataset = None
        self.use_videos = False
        self.defer_video_encoding_until_finalize = False
        self._episode_stats_executor: ProcessPoolExecutor | None = None
        self.logger = get_logger()

    def create(
        self,
        repo_id: str,
        robot_type: str = "franka_panda",
        fps: int = 5,
        features: dict[str, dict[str, Any]] | None = None,
        image_writer_threads: int = 10,
        image_writer_processes: int = 0,
        image_shape: tuple[int, int, int] = (256, 256, 3),
        state_dim: int = 8,
        action_dim: int = 7,
        has_image: bool = True,
        wrist_image_keys: dict[str, tuple[int, ...]] | None = None,
        extra_view_image_keys: dict[str, tuple[int, ...]] | None = None,
        has_intervene_flag: bool = True,
        has_segment_id: bool = False,
        has_observation_timestamp: bool = False,
        use_videos: bool = False,
        defer_video_encoding_until_finalize: bool = False,
        isolate_episode_stats: bool = False,
    ) -> None:
        """
        Create a new LeRobot dataset.

        Args:
            repo_id: The identifier for the new LeRobot dataset
            robot_type: Robot type (default "franka_panda")
            fps: Frame rate (default 5)
            features: Feature schema dictionary defining the dataset structure.
                If None, auto-generated from dimensions.
            image_writer_threads: Number of threads for image writing
            image_writer_processes: Number of processes for image writing
            image_shape: Image shape (H, W, C) for the main ``image`` feature.
            state_dim: State dimension for auto-generated features
            action_dim: Action dimension for auto-generated features
            has_image: Whether to include the main ``image`` feature.
            wrist_image_keys: Mapping of wrist-camera image key names to their
                ``(H, W, C)`` shapes.  A single view produces
                ``{"wrist_image": (H, W, C)}``; multiple views produce
                ``{"wrist_image-0": …, "wrist_image-1": …, …}``.
            extra_view_image_keys: Same as *wrist_image_keys* but for the
                extra-view camera(s).
            has_intervene_flag: Whether to include per-frame human-intervention
                flag (bool, shape ``(1,)``) in auto-generated features.
            has_segment_id: Whether to include per-frame ``segment_id``
                (uint8, shape ``(1,)``) in auto-generated features. Used for
                in-episode sub-task boundaries set by KeyboardStartEndWrapper.
            has_observation_timestamp: Whether to include the monotonic
                ``observation_timestamp_ns`` (int64, shape ``(1,)``) captured
                when the observation was received.
            use_videos: Whether to encode camera features as MP4 videos.
            defer_video_encoding_until_finalize: Keep completed-episode PNGs
                during collection and encode all videos in :meth:`finalize`.
            isolate_episode_stats: Compute per-episode image statistics in a
                persistent spawned process so image loading and array cleanup
                cannot pause the collection process's Python threads.

        """

        try:  # lerobot >= 0.2 layout
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ModuleNotFoundError:  # lerobot < 0.2
            from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

        _silence_hf_datasets_progress_bars()

        if features is None:
            camera_dtype = "video" if use_videos else "image"
            features = {
                "state": {
                    "dtype": "float32",
                    "shape": (state_dim,),
                    "names": ["state"],
                },
                "actions": {
                    "dtype": "float32",
                    "shape": (action_dim,),
                    "names": ["actions"],
                },
                "done": {
                    "dtype": "bool",
                    "shape": (1,),
                    "names": ["done"],
                },
                "is_success": {
                    "dtype": "bool",
                    "shape": (1,),
                    "names": ["is_success"],
                },
            }
            if has_intervene_flag:
                features["intervene_flag"] = {
                    "dtype": "bool",
                    "shape": (1,),
                    "names": ["intervene_flag"],
                }
            if has_segment_id:
                features["segment_id"] = {
                    "dtype": "uint8",
                    "shape": (1,),
                    "names": ["segment_id"],
                }
            if has_observation_timestamp:
                features["observation_timestamp_ns"] = {
                    "dtype": "int64",
                    "shape": (1,),
                    "names": ["observation_timestamp_ns"],
                }
            if has_image:
                features["image"] = {
                    "dtype": camera_dtype,
                    "shape": list(image_shape),
                    "names": ["height", "width", "channel"],
                }
            for keys in (wrist_image_keys, extra_view_image_keys):
                if keys:
                    for key, shape in keys.items():
                        features[key] = {
                            "dtype": camera_dtype,
                            "shape": list(shape),
                            "names": ["height", "width", "channel"],
                        }

        self.logger.info(
            f"Creating LeRobot dataset: repo_id={repo_id}, robot_type={robot_type}, fps={fps}"
        )
        create_kwargs = {}
        if defer_video_encoding_until_finalize:
            if not use_videos:
                raise ValueError(
                    "defer_video_encoding_until_finalize requires use_videos=True"
                )
            # LeRobot performs synchronous video encoding when this threshold is
            # reached. Defer every practical run to finalize().
            create_kwargs["batch_encoding_size"] = sys.maxsize

        self.dataset = LeRobotDataset.create(
            repo_id=repo_id,
            robot_type=robot_type,
            fps=fps,
            features=features,
            use_videos=use_videos,
            image_writer_threads=image_writer_threads,
            image_writer_processes=image_writer_processes,
            **create_kwargs,
        )
        self.use_videos = use_videos
        self.defer_video_encoding_until_finalize = defer_video_encoding_until_finalize
        if isolate_episode_stats:
            self._episode_stats_executor = ProcessPoolExecutor(
                max_workers=1,
                mp_context=multiprocessing.get_context("spawn"),
            )

    def add_episode(
        self,
        episode_data: list[dict[str, Any]],
        *,
        consume: bool = False,
    ) -> None:
        """
        Add an episode to the dataset.

        Args:
            episode_data: List of frame dictionaries, where each frame contains:
                - image: np.ndarray [H, W, C]
                - wrist_image: np.ndarray [H, W, C] (optional)
                - state: np.ndarray [state_dim]
                - actions: np.ndarray [action_dim]
                - task: str (task instruction)
                - intervene_flag: np.ndarray [1] of bool (optional; matches schema)
                - observation_timestamp_ns: np.ndarray [1] of int64 (optional)
                - Any other fields defined in the features schema
            consume: Clear frames after LeRobot accepts them. The caller must
                transfer ownership of ``episode_data`` when enabling this.

        The frames will be automatically processed to include both the original
        image format and the observation.images format (transposed to [C, H, W]).
        """
        if self.dataset is None:
            raise RuntimeError("Dataset not created. Call create() first.")

        if not episode_data:
            self.logger.warning("Empty episode_data provided, skipping.")
            return
        frame_count = len(episode_data)
        task = episode_data[0].get("task", "N/A")
        accepted_frames = 0
        try:
            for frame_data in episode_data:
                add_frame_to_dataset(self.dataset, frame_data)
                accepted_frames += 1
                if consume:
                    frame_data.clear()
        except BaseException:
            if consume:
                del episode_data[:accepted_frames]
            raise

        self._save_episode()
        if consume:
            episode_data.clear()
        self.logger.info(f"Saved episode with {frame_count} frames, task: '{task}'")

    def _save_episode(self) -> None:
        """Save metadata while optionally isolating image-stat computation."""
        # LeRobot imports compute_episode_stats into its dataset module. Guard
        # every save across all writer instances so a non-isolated writer cannot
        # observe another writer's temporary replacement.
        with _COMPUTE_STATS_PATCH_LOCK:
            if self._episode_stats_executor is None:
                self.dataset.save_episode()
                return

            dataset_module = importlib.import_module(type(self.dataset).__module__)
            original_compute_stats = dataset_module.compute_episode_stats

            def isolated_compute_stats(episode_buffer, features):
                future = self._episode_stats_executor.submit(
                    _compute_episode_stats_in_process,
                    episode_buffer,
                    features,
                )
                return future.result()

            dataset_module.compute_episode_stats = isolated_compute_stats
            try:
                self.dataset.save_episode()
            finally:
                dataset_module.compute_episode_stats = original_compute_stats

    def _encode_pending_videos(self, prior_error: BaseException | None = None) -> None:
        """Encode episodes deferred while hardware collection was active."""
        pending = int(getattr(self.dataset, "episodes_since_last_encoding", 0))
        if pending <= 0 and prior_error is None:
            return

        try:
            from lerobot.datasets.video_utils import VideoEncodingManager
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "The installed LeRobot version does not support deferred video encoding"
            ) from exc

        if pending > 0:
            self.logger.info(
                "Encoding %d deferred episodes after hardware collection stopped.",
                pending,
            )
        manager = VideoEncodingManager(self.dataset)
        manager.__enter__()
        manager.__exit__(
            type(prior_error) if prior_error is not None else None,
            prior_error,
            prior_error.__traceback__ if prior_error is not None else None,
        )
        self.dataset.episodes_since_last_encoding = 0

    def finalize(self, prior_error: BaseException | None = None) -> None:
        """Finalize the dataset and properly clean up all resources."""
        if self.dataset is None:
            raise RuntimeError("Dataset not created. Call create() first.")

        first_error = prior_error
        try:
            if (
                hasattr(self.dataset, "image_writer")
                and self.dataset.image_writer is not None
            ):
                self.dataset.image_writer.wait_until_done()
            if self.defer_video_encoding_until_finalize or (
                prior_error is not None and self.use_videos
            ):
                self._encode_pending_videos(prior_error)
        except BaseException as exc:
            if first_error is None:
                first_error = exc
            else:
                self.logger.error("Dataset finalization also failed: %s", exc)
        finally:
            if self._episode_stats_executor is not None:
                try:
                    self._episode_stats_executor.shutdown(wait=True)
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                    else:
                        self.logger.error(
                            "Episode statistics process shutdown also failed: %s", exc
                        )
                self._episode_stats_executor = None
            if (
                hasattr(self.dataset, "image_writer")
                and self.dataset.image_writer is not None
            ):
                try:
                    self.dataset.image_writer.stop()
                except BaseException as exc:
                    if first_error is None:
                        first_error = exc
                    else:
                        self.logger.error("Image writer shutdown also failed: %s", exc)
                self.dataset.image_writer = None

        if first_error is not None:
            raise first_error

        if hasattr(self.dataset, "episode_buffer"):
            self.dataset.episode_buffer = None

        if hasattr(self.dataset, "hf_dataset"):
            self.dataset.hf_dataset = None

        del self.dataset
        self.dataset = None
        gc.collect()
        self.logger.info("Dataset finalized.")

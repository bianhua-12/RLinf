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

"""Runner for CFG SFT with optional environment-based evaluation."""

from __future__ import annotations

import os
from typing import Optional

from omegaconf.dictconfig import DictConfig
from tqdm import tqdm

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.runner_utils import check_progress


class CfgSFTRunner:
    """Run offline CFG training with optional LIBERO environment evaluation."""

    def __init__(
        self,
        cfg: DictConfig,
        actor,
        rollout=None,
        env=None,
        run_timer: Optional[ScopedTimer] = None,
    ) -> None:
        self.cfg = cfg
        self.actor = actor
        self.rollout = rollout
        self.env = env
        self.run_timer = run_timer

        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")

        self.consumed_samples = 0
        self.global_step = 0

        self.set_max_steps()
        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

    def init_workers(self) -> None:
        """Initialize worker groups and optionally restore from checkpoint."""
        actor_future = self.actor.init_worker()
        rollout_future = (
            self.rollout.init_worker() if self.rollout is not None else None
        )
        env_future = self.env.init_worker() if self.env is not None else None

        actor_future.wait()
        if rollout_future is not None:
            rollout_future.wait()
        if env_future is not None:
            env_future.wait()

        steps_per_epoch = self.actor.get_training_steps_per_epoch().wait()[0]
        self.set_max_steps(steps_per_epoch=steps_per_epoch)

        resume_dir = self.cfg.runner.get("resume_dir", None)
        if resume_dir is None:
            return

        actor_checkpoint_path = os.path.join(resume_dir, "actor")
        assert os.path.exists(actor_checkpoint_path), (
            f"resume_dir {actor_checkpoint_path} does not exist."
        )
        self.actor.load_checkpoint(actor_checkpoint_path).wait()
        self.global_step = int(resume_dir.split("global_step_")[-1])

    def run(self) -> None:
        """Execute the CFG training loop."""
        start_step = self.global_step
        global_pbar = tqdm(
            initial=start_step,
            total=self.max_steps,
            desc="Global Step",
            ncols=800,
        )

        self.actor.set_global_step(self.global_step)
        if self.rollout is not None:
            self.rollout.set_global_step(self.global_step)

        if (
            start_step == 0
            and self._should_run_env_eval()
            and self.cfg.runner.val_check_interval > 0
        ):
            with self.timer("eval"):
                eval_metrics = self._run_env_eval()
                eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                self.metric_logger.log(data=eval_metrics, step=start_step)

        for _step in range(start_step, self.max_steps):
            eval_metrics = {}

            self.actor.set_global_step(self.global_step)
            if self.rollout is not None:
                self.rollout.set_global_step(self.global_step)

            with self.timer("step"):
                if (
                    self._should_run_env_eval()
                    and self.cfg.runner.val_check_interval > 0
                    and self.global_step > 0
                    and self.global_step % self.cfg.runner.val_check_interval == 0
                ):
                    with self.timer("eval"):
                        eval_metrics = self._run_env_eval()
                        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
                        self.metric_logger.log(data=eval_metrics, step=_step)

                actor_handle: Handle = self.actor.run_training()
                actor_metrics = actor_handle.wait()

                self.global_step += 1

                _, save_model, _ = check_progress(
                    self.global_step,
                    self.max_steps,
                    self.cfg.runner.val_check_interval,
                    self.cfg.runner.save_interval,
                    1.0,
                    run_time_exceeded=False,
                )

                if save_model:
                    self._save_checkpoint()

            time_metrics = self.timer.consume_durations()
            time_metrics["training"] = actor_handle.consume_duration()
            time_metrics = {f"time/{k}": v for k, v in time_metrics.items()}

            training_metrics = {f"train/{k}": v for k, v in actor_metrics[0].items()}
            self.metric_logger.log(time_metrics, _step)
            self.metric_logger.log(training_metrics, _step)

            logging_metrics = time_metrics
            logging_metrics.update(training_metrics)

            if eval_metrics:
                logging_metrics.update(eval_metrics)

            global_pbar.set_postfix(logging_metrics, refresh=False)
            global_pbar.update(1)

        self.metric_logger.finish()

    def _should_run_env_eval(self) -> bool:
        """Return whether this run should use rollout+env evaluation."""
        return self.rollout is not None and self.env is not None

    def _run_eval(self) -> dict:
        """Run the configured evaluation path."""
        if self._should_run_env_eval():
            return self._run_env_eval()

        eval_handle: Handle = self.actor.run_eval()
        eval_metrics = eval_handle.wait()
        return (
            eval_metrics[0] if isinstance(eval_metrics, (list, tuple)) else eval_metrics
        )

    def _run_env_eval(self) -> dict:
        """Run environment-based evaluation using rollout and env workers."""
        rollout_handle: Handle = self.rollout.sync_model_from_actor()
        actor_handle: Handle = self.actor.sync_model_to_rollout()
        actor_handle.wait()
        rollout_handle.wait()

        env_handle: Handle = self.env.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        rollout_handle = self.rollout.evaluate(
            input_channel=self.env_channel,
            output_channel=self.rollout_channel,
        )
        env_results = env_handle.wait()
        rollout_handle.wait()
        eval_metrics_list = [result for result in env_results if result is not None]
        return compute_evaluate_metrics(eval_metrics_list)

    def _save_checkpoint(self) -> None:
        """Persist an actor checkpoint."""
        base_output_dir = os.path.join(
            self.cfg.runner.logger.log_path,
            self.cfg.runner.logger.experiment_name,
            f"checkpoints/global_step_{self.global_step}",
        )
        actor_save_path = os.path.join(base_output_dir, "actor")
        os.makedirs(actor_save_path, exist_ok=True)
        self.actor.save_checkpoint(actor_save_path, self.global_step).wait()

    def set_max_steps(self, steps_per_epoch: int | None = None) -> None:
        """Compute max training steps from runner config.

        Args:
            steps_per_epoch: Optional optimizer steps per epoch. When omitted,
                the runner falls back to ``1`` until workers are initialized.
        """
        self.num_steps_per_epoch = max(1, steps_per_epoch or 1)
        self.max_steps = self.num_steps_per_epoch * self.cfg.runner.max_epochs

        if (max_steps := self.cfg.runner.get("max_steps", -1)) >= 0:
            self.max_steps = min(self.max_steps, max_steps)

    @property
    def epoch(self) -> int:
        """Return the current epoch index."""
        return self.global_step // self.num_steps_per_epoch

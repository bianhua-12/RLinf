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

"""Finite ReFL experiments with native held-out evaluation and sealed provenance."""

import hashlib
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from rlinf.utils.logging import get_logger

logger = get_logger()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value, exclusive=False):
    with Path(path).open("x" if exclusive else "w") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def assert_clean_source(repository, expected_commit):
    """Reject dirty, untracked, ignored, or mismatched execution sources."""

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(repository), *args], text=True
        ).strip()

    head = git("rev-parse", "HEAD")
    if len(expected_commit) != 40 or head != expected_commit:
        raise ValueError(f"Expected commit {expected_commit}; found {head}")
    status = git("status", "--porcelain=v2", "--untracked-files=all", "--ignored")
    if status:
        raise ValueError(
            "Execution checkout contains modified, untracked or ignored files"
        )
    submodules = git("submodule", "status", "--recursive")
    # This entrypoint has no submodule dependency. Refuse unrecorded source trees.
    if submodules:
        raise ValueError(
            "ReFL launch requires an explicit submodule provenance extension"
        )
    return {
        "head": head,
        "ref": git("rev-parse", "--abbrev-ref", "HEAD"),
        "status": status,
        "submodules": submodules,
    }


def prepare_run(config, repository):
    """Seal code, inputs, configuration and installed environment before workers run."""
    import torch

    config = dict(config)
    repository = Path(repository).resolve()
    code = assert_clean_source(repository, config["commit"])
    output = Path(config["output_dir"]).resolve()
    if output == repository or repository in output.parents:
        raise ValueError("Run output must be outside the source checkout")
    if not config.get("container_image"):
        raise ValueError("An explicit execution container image identity is required")
    if set(config["inputs"]) not in (
        {"actor", "world_model", "dataset", "bank"},
        {"actor", "world_model", "dataset", "bank", "bank_mask"},
    ):
        raise ValueError(
            "Expected actor, world_model, dataset, bank and optional bank_mask inputs"
        )
    for name, item in config["inputs"].items():
        if not all(item.get(k) for k in ("path", "sha256", "version", "protocol")):
            raise ValueError(f"Missing input identity fields for {name}")
        if sha256(item["path"]) != item["sha256"]:
            raise ValueError(f"Input SHA256 mismatch: {name}")
        receipt = json.loads(Path(item["receipt_path"]).read_text())
        if (
            receipt.get("verified") is not True
            or receipt.get("version") != item["version"]
            or receipt.get("sha256") != item["sha256"]
            or receipt.get("bytes") != Path(item["path"]).stat().st_size
        ):
            raise ValueError(f"Input receipt/version mismatch: {name}")
    output.mkdir(parents=True, exist_ok=False)
    config["repository"] = str(repository)
    config["output_dir"] = str(output)
    write_json(output / "config.json", config, exclusive=True)
    packages = sorted(
        [
            {"name": d.metadata["Name"], "version": d.version}
            for d in importlib.metadata.distributions()
        ],
        key=lambda d: d["name"].lower(),
    )
    write_json(output / "packages.json", packages, exclusive=True)
    sources = {}
    for name in ("torch", "mani-skill", "sapien", "ray", "pytorch-kinematics"):
        distribution = importlib.metadata.distribution(name)
        direct = json.loads(distribution.read_text("direct_url.json") or "{}")
        if direct.get("dir_info", {}).get("editable"):
            raise ValueError(
                f"Editable dependency needs separate source provenance: {name}"
            )
        sources[name] = {
            "version": distribution.version,
            "vcs": direct.get("vcs_info"),
            "archive": direct.get("archive_info"),
        }
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,uuid,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    record = {
        "schema": "rlinf-refl-start-v1",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "code": code,
        "repository": str(repository),
        "command": sys.argv,
        "cwd": str(Path.cwd().resolve()),
        "config_sha256": sha256(output / "config.json"),
        "packages_sha256": sha256(output / "packages.json"),
        "environment": {
            "python": sys.version,
            "os": platform.platform(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": gpu,
            "container_image": config["container_image"],
            "dependency_sources": sources,
        },
        "inputs": config["inputs"],
        "input_receipt_sha256": {
            name: sha256(item["receipt_path"])
            for name, item in config["inputs"].items()
        },
        "seed": config["seed"],
    }
    write_json(output / "start.json", record, exclusive=True)
    (output / "start.sha256").write_text(sha256(output / "start.json") + "\n")
    logger.info("Sealed ReFL start manifest: %s", json.dumps(record))
    return config


class ReFLRunner:
    def __init__(self, config, worker):
        self.cfg, self.worker = config, worker
        self.root = Path(config["output_dir"])
        self.wandb = None
        self.actor_step = 0

    def log(self, metrics):
        self.actor_step = metrics.get("updates/actor", self.actor_step)
        metrics = {**metrics, "updates/actor": self.actor_step}
        scalars = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
        if not all(math.isfinite(v) for v in scalars.values()):
            raise FloatingPointError("Non-finite ReFL metrics")
        with (self.root / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(metrics, allow_nan=False) + "\n")
        logger.info("ReFL metrics: %s", json.dumps(metrics))
        if self.wandb is not None:
            self.wandb.log(scalars)

    def run(self):
        import wandb

        completed = False
        try:
            self.wandb = wandb.init(
                project=self.cfg["wandb_project"],
                id=self.cfg["run_id"],
                name=self.cfg["run_id"],
                mode=self.cfg["wandb_mode"],
                dir=str(self.root),
                config=self.cfg,
                resume="never",
            )
            self.wandb.define_metric("updates/actor")
            self.wandb.define_metric("*", step_metric="updates/actor", step_sync=True)
            write_json(
                self.root / "wandb.json",
                {
                    "id": self.wandb.id,
                    "mode": self.cfg["wandb_mode"],
                    "url": None if self.wandb.offline else self.wandb.url,
                },
            )
            self.log(self.worker.initialize().wait()[0])
            baseline = self.worker.evaluate(checkpoint="initial.pt", seed=40000).wait()[
                0
            ]
            write_json(self.root / "initial_eval.json", baseline)
            self.log({"initial/" + k: v for k, v in baseline.items()})
            lo, hi = self.cfg["initial_success_range"]
            if not lo <= baseline["success_rate"] <= hi:
                raise ValueError(
                    f"Initial success {baseline['success_rate']} outside [{lo}, {hi}]"
                )
            selected = self.worker.evaluate().wait()[0]
            best = (selected["success_rate"], selected["mean_return"])
            self.worker.save_checkpoint("best.pt").wait()
            self.log({"eval/" + k: v for k, v in selected.items()})
            for iteration in range(1, self.cfg["iterations"] + 1):
                self.log(self.worker.train_iteration().wait()[0])
                if (
                    iteration % self.cfg["eval_every"] == 0
                    or iteration == self.cfg["iterations"]
                ):
                    metrics = self.worker.evaluate().wait()[0]
                    self.log({"eval/" + k: v for k, v in metrics.items()})
                    score = (metrics["success_rate"], metrics["mean_return"])
                    if score > best:
                        best = score
                        self.worker.save_checkpoint("best.pt").wait()
            self.worker.save_checkpoint("final.pt").wait()
            independent = {"initial": baseline}
            for label, checkpoint in (("selected", "best.pt"), ("final", "final.pt")):
                independent[label] = self.worker.evaluate(
                    checkpoint=checkpoint, seed=40000
                ).wait()[0]
            self.log(
                {
                    f"independent/{label}/{key}": value
                    for label, metrics in independent.items()
                    for key, value in metrics.items()
                }
            )
            write_json(self.root / "independent_eval.json", independent)
            completed = True
            return independent
        finally:
            cleanup_errors = []
            try:
                self.worker.close().wait()
            except Exception as exc:
                cleanup_errors.append("worker_close:" + type(exc).__name__)
            if self.wandb is not None:
                try:
                    self.wandb.finish(exit_code=0 if completed else 1)
                except Exception as exc:
                    cleanup_errors.append("wandb_finish:" + type(exc).__name__)
            expected = (self.root / "start.sha256").read_text().strip()
            if sha256(self.root / "start.json") != expected:
                raise ValueError("Start provenance was modified")
            outputs = {
                p.name: sha256(p)
                for p in self.root.iterdir()
                if p.is_file()
                and (p.suffix == ".pt" or p.name == "independent_eval.json")
            }
            write_json(
                self.root / "completion.json",
                {
                    "schema": "rlinf-refl-completion-v1",
                    "start_sha256": expected,
                    "completed": completed and not cleanup_errors,
                    "training_and_evaluation_completed": completed,
                    "cleanup_errors": cleanup_errors,
                    "exit_code": 0 if completed and not cleanup_errors else 1,
                    "finished_at": datetime.now(timezone.utc).isoformat(),
                    "evidence": "native_closed_loop_evaluation"
                    if completed
                    else "incomplete",
                    "outputs": outputs,
                },
                exclusive=True,
            )
            if cleanup_errors:
                raise RuntimeError("ReFL cleanup failed: " + ", ".join(cleanup_errors))

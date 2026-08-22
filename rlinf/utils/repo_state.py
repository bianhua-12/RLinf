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

"""Repository-state checks for reproducible RLinf launches."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

_STATUS_PREVIEW_LIMIT = 50
_PATH_ENV_MERGE_MODE_NAME = "RLINF_PATH_ENV_MERGE_MODE"


def is_python_source_environment_variable(name: str) -> bool:
    """Return whether an environment variable can change Python code lookup."""
    return name == "PATH" or name == "VIRTUAL_ENV" or name.startswith("PYTHON")


def _run_git(repo_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    git_env = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    try:
        return subprocess.run(
            ["git", "--no-optional-locks", "-C", str(repo_path), *args],
            capture_output=True,
            text=True,
            check=False,
            env=git_env,
        )
    except OSError as exc:
        raise RuntimeError(
            f"Cannot verify RLinf repository state: failed to run git: {exc}"
        ) from exc


def _git_failure_message(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stderr or result.stdout).strip() or "git command failed"


def require_clean_worktree(repo_path: str | Path | None = None) -> str:
    """Require a named commit and a clean non-ignored Git worktree.

    Args:
        repo_path: A path inside the repository. Defaults to the checkout that
            contains the imported ``rlinf`` package.

    Returns:
        The full commit SHA verified for the launch.

    Raises:
        RuntimeError: If Git is unavailable, no commit can be resolved, or the
            index/worktree contains staged, unstaged, or untracked changes.
    """
    candidate = (
        Path(repo_path).resolve()
        if repo_path is not None
        else Path(__file__).resolve().parents[2]
    )
    root_result = _run_git(candidate, "rev-parse", "--show-toplevel")
    if root_result.returncode != 0:
        raise RuntimeError(
            "Refusing to start RLinf without a Git worktree: "
            f"{_git_failure_message(root_result)}"
        )
    repo_root = Path(root_result.stdout.strip()).resolve()

    commit_result = _run_git(repo_root, "rev-parse", "--verify", "HEAD^{commit}")
    if commit_result.returncode != 0:
        raise RuntimeError(
            "Refusing to start RLinf without a named commit: "
            f"{_git_failure_message(commit_result)}"
        )
    commit_sha = commit_result.stdout.strip()

    status_result = _run_git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if status_result.returncode != 0:
        raise RuntimeError(
            "Cannot verify whether the RLinf worktree is clean: "
            f"{_git_failure_message(status_result)}"
        )

    dirty_entries = status_result.stdout.splitlines()
    if dirty_entries:
        preview = dirty_entries[:_STATUS_PREVIEW_LIMIT]
        if len(dirty_entries) > _STATUS_PREVIEW_LIMIT:
            preview.append(
                f"... and {len(dirty_entries) - _STATUS_PREVIEW_LIMIT} more entries"
            )
        details = "\n".join(f"  {entry}" for entry in preview)
        raise RuntimeError(
            "Refusing to start RLinf from a dirty Git worktree at "
            f"{repo_root}. Commit or remove all staged, unstaged, and "
            f"non-ignored untracked files before launch:\n{details}"
        )

    return commit_sha


def check_worktree_commit(
    expected_commit: str,
    node_id: str,
    node_ip: str,
    repo_path: str | Path | None = None,
    num_nodes: int = 1,
    runtime_label: str = "default Ray runtime",
) -> dict[str, object]:
    """Report whether one execution node is clean at the expected commit.

    The structured result lets the driver report every failing node instead of
    losing node identity in a remote-task exception.

    Args:
        expected_commit: Full commit SHA verified by the launch driver.
        node_id: Ray node identifier used for diagnostics.
        node_ip: Ray node address used for diagnostics.
        repo_path: Optional checkout path, primarily for tests. By default the
            imported ``rlinf`` package determines the execution checkout.
        num_nodes: Number of nodes expected by the launch.
        runtime_label: Human-readable runtime identity for diagnostics.

    Returns:
        A structured success or failure result for the node.
    """
    result: dict[str, object] = {
        "node_id": node_id,
        "node_ip": node_ip,
        "ok": False,
        "runtime": runtime_label,
        "python_executable": sys.executable,
        "source_env": {
            key: value
            for key, value in os.environ.items()
            if is_python_source_environment_variable(key)
            or key == _PATH_ENV_MERGE_MODE_NAME
        },
    }
    node_rank_raw = os.environ.get("RLINF_NODE_RANK")
    if num_nodes == 1:
        node_rank = 0
    else:
        try:
            node_rank = int(node_rank_raw) if node_rank_raw is not None else -1
        except ValueError:
            node_rank = -1
        if node_rank < 0 or node_rank >= num_nodes:
            result["error"] = (
                "invalid RLINF_NODE_RANK: "
                f"expected an integer in [0, {num_nodes - 1}], "
                f"found {node_rank_raw!r}"
            )
            return result
    result["node_rank"] = node_rank

    try:
        actual_commit = require_clean_worktree(repo_path)
    except RuntimeError as exc:
        result["error"] = str(exc)
        return result

    result["commit"] = actual_commit
    if actual_commit != expected_commit:
        result["error"] = (
            f"commit mismatch: expected {expected_commit}, found {actual_commit}"
        )
        return result

    result["ok"] = True
    return result

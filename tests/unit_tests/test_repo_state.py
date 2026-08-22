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

import subprocess
from pathlib import Path

import pytest

from rlinf.utils.repo_state import check_worktree_commit, require_clean_worktree


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def _initialized_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")
    _git(repo, "config", "user.email", "tests@rlinf.invalid")
    _git(repo, "config", "user.name", "RLinf Tests")
    (repo / ".gitignore").write_text("ignored/\n", encoding="ascii")
    (repo / "tracked.txt").write_text("committed\n", encoding="ascii")
    _git(repo, "add", ".gitignore", "tracked.txt")
    _git(repo, "commit", "--quiet", "-m", "test: initialize repository")
    return repo


def test_clean_worktree_returns_full_commit_sha(tmp_path):
    repo = _initialized_repo(tmp_path)

    assert require_clean_worktree(repo) == _git(repo, "rev-parse", "HEAD")


@pytest.mark.parametrize("dirty_kind", ["unstaged", "staged", "untracked"])
def test_dirty_worktree_is_rejected(tmp_path, dirty_kind):
    repo = _initialized_repo(tmp_path)
    if dirty_kind == "untracked":
        (repo / "untracked.txt").write_text("new\n", encoding="ascii")
        expected_filename = "untracked.txt"
    else:
        (repo / "tracked.txt").write_text("modified\n", encoding="ascii")
        expected_filename = "tracked.txt"
        if dirty_kind == "staged":
            _git(repo, "add", "tracked.txt")

    with pytest.raises(RuntimeError, match="dirty Git worktree") as exc_info:
        require_clean_worktree(repo)

    assert expected_filename in str(exc_info.value)


def test_git_environment_cannot_redirect_worktree_check(tmp_path, monkeypatch):
    checked_parent = tmp_path / "checked"
    checked_parent.mkdir()
    checked_repo = _initialized_repo(checked_parent)
    (checked_repo / "tracked.txt").write_text("modified\n", encoding="ascii")

    redirect_parent = tmp_path / "redirect"
    redirect_parent.mkdir()
    redirect_repo = _initialized_repo(redirect_parent)
    monkeypatch.setenv("GIT_DIR", str(redirect_repo / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(redirect_repo))

    with pytest.raises(RuntimeError, match="dirty Git worktree") as exc_info:
        require_clean_worktree(checked_repo)

    assert "tracked.txt" in str(exc_info.value)


def test_ignored_runtime_artifacts_do_not_block_launch(tmp_path):
    repo = _initialized_repo(tmp_path)
    ignored_dir = repo / "ignored"
    ignored_dir.mkdir()
    (ignored_dir / "run.log").write_text("runtime output\n", encoding="ascii")

    assert require_clean_worktree(repo) == _git(repo, "rev-parse", "HEAD")


def test_path_outside_git_worktree_is_rejected(tmp_path):
    with pytest.raises(RuntimeError, match="without a Git worktree"):
        require_clean_worktree(tmp_path)


def test_repository_without_commit_is_rejected(tmp_path):
    repo = tmp_path / "empty-repo"
    repo.mkdir()
    _git(repo, "init", "--quiet")

    with pytest.raises(RuntimeError, match="without a named commit"):
        require_clean_worktree(repo)


def test_dirty_submodule_is_rejected(tmp_path):
    child = tmp_path / "child"
    child.mkdir()
    _git(child, "init", "--quiet")
    _git(child, "config", "user.email", "tests@rlinf.invalid")
    _git(child, "config", "user.name", "RLinf Tests")
    (child / "source.py").write_text("VALUE = 1\n", encoding="ascii")
    _git(child, "add", "source.py")
    _git(child, "commit", "--quiet", "-m", "test: initialize child")

    repo = _initialized_repo(tmp_path)
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        "--quiet",
        str(child),
        "dependency",
    )
    _git(repo, "commit", "--quiet", "-am", "test: add submodule")
    (repo / "dependency" / "source.py").write_text("VALUE = 2\n", encoding="ascii")

    with pytest.raises(RuntimeError, match="dirty Git worktree") as exc_info:
        require_clean_worktree(repo)

    assert "dependency" in str(exc_info.value)


def test_node_worktree_commit_mismatch_is_reported(tmp_path):
    repo = _initialized_repo(tmp_path)

    result = check_worktree_commit("f" * 40, "node-id", "10.0.0.2", repo)

    assert result["node_id"] == "node-id"
    assert result["node_ip"] == "10.0.0.2"
    assert result["node_rank"] == 0
    assert result["ok"] is False
    assert result["commit"] == _git(repo, "rev-parse", "HEAD")
    assert result["error"] == (
        f"commit mismatch: expected {'f' * 40}, found {_git(repo, 'rev-parse', 'HEAD')}"
    )


def test_invalid_multi_node_rank_is_reported(tmp_path, monkeypatch):
    repo = _initialized_repo(tmp_path)
    monkeypatch.delenv("RLINF_NODE_RANK", raising=False)

    result = check_worktree_commit(
        _git(repo, "rev-parse", "HEAD"),
        "node-id",
        "10.0.0.2",
        repo,
        num_nodes=2,
    )

    assert result["ok"] is False
    assert "invalid RLINF_NODE_RANK" in str(result["error"])

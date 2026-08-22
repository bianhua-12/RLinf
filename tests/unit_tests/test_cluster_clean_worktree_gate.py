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

import os
import sys
from types import SimpleNamespace

import pytest
from omegaconf import OmegaConf

import rlinf.scheduler.cluster.cluster as cluster_module
from rlinf.scheduler.cluster.cluster import Cluster
from rlinf.scheduler.cluster.config import ClusterConfig


def _uninitialized_cluster(monkeypatch, calls):
    cluster = object.__new__(Cluster)
    cluster._has_initialized = False
    monkeypatch.setattr(cluster, "_setup_logger", lambda: calls.append("logger"))
    monkeypatch.setattr(
        cluster,
        "_init_and_launch_managers",
        lambda *args: calls.append("launch"),
    )
    cluster._logger = SimpleNamespace(info=lambda *args: calls.append("log_commit"))
    return cluster


def test_cluster_launch_checks_worktree_before_initialization(monkeypatch):
    calls = []
    cluster = _uninitialized_cluster(monkeypatch, calls)
    monkeypatch.setattr(
        cluster_module,
        "require_clean_worktree",
        lambda: calls.append("gate") or "a" * 40,
    )

    Cluster.__init__(cluster, num_nodes=1)

    assert calls == ["gate", "logger", "log_commit", "launch"]
    assert cluster._source_commit == "a" * 40


def test_dirty_worktree_stops_cluster_before_logger_or_ray(monkeypatch):
    calls = []
    cluster = _uninitialized_cluster(monkeypatch, calls)

    def reject_dirty_worktree():
        calls.append("gate")
        raise RuntimeError("dirty Git worktree")

    monkeypatch.setattr(cluster_module, "require_clean_worktree", reject_dirty_worktree)

    with pytest.raises(RuntimeError, match="dirty Git worktree"):
        Cluster.__init__(cluster, cluster_cfg={"num_nodes": 1})

    assert calls == ["gate"]


def test_cluster_without_launch_arguments_is_attach_only(monkeypatch):
    calls = []
    cluster = _uninitialized_cluster(monkeypatch, calls)

    def fail_attach():
        calls.append("attach")
        raise ConnectionError

    monkeypatch.setattr(cluster, "_init_from_existing_managers", fail_attach)

    with pytest.raises(RuntimeError, match="attach-only"):
        Cluster.__init__(cluster)

    assert calls == ["logger", "attach"]


def test_execution_nodes_are_pinned_and_checked_at_driver_commit(monkeypatch):
    node_a = "a" * 56
    node_b = "b" * 56
    nodes = [
        {"NodeID": node_a, "NodeManagerAddress": "10.0.0.1"},
        {"NodeID": node_b, "NodeManagerAddress": "10.0.0.2"},
    ]
    remote_calls = []

    class FakeRemoteCheck:
        def options(self, **options):
            strategy = options["scheduling_strategy"]
            remote_calls.append(("options", strategy.node_id, strategy.soft))
            return self

        def remote(self, *args):
            remote_calls.append(("remote", *args))
            return f"ref-{args[1]}"

    monkeypatch.setattr(
        cluster_module.ray,
        "remote",
        lambda **options: (
            lambda function: (
                FakeRemoteCheck()
                if options == {"num_cpus": 0}
                and function is cluster_module.check_worktree_commit
                else None
            )
        ),
    )

    def fake_get(refs, timeout):
        assert refs == [f"ref-{node_a}", f"ref-{node_b}"]
        assert timeout == Cluster.WORKTREE_CHECK_TIMEOUT
        source_env = Cluster._source_environment(dict(os.environ))
        return [
            {
                "node_id": node_a,
                "node_ip": "10.0.0.1",
                "node_rank": 0,
                "python_executable": sys.executable,
                "source_env": source_env,
                "ok": True,
            },
            {
                "node_id": node_b,
                "node_ip": "10.0.0.2",
                "node_rank": 1,
                "python_executable": "/remote/python",
                "source_env": source_env,
                "ok": True,
            },
        ]

    monkeypatch.setattr(cluster_module.ray, "get", fake_get)
    monkeypatch.setattr(
        cluster_module.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_node_id=lambda: node_a),
    )
    cluster = object.__new__(Cluster)
    cluster._source_commit = "c" * 40
    cluster._ray_code_sync_fragment = None
    cluster._cluster_cfg = None
    cluster._logger = SimpleNamespace(info=lambda *args: None)

    cluster._verify_execution_node_worktrees(nodes)

    assert remote_calls == [
        ("options", node_a, False),
        (
            "remote",
            "c" * 40,
            node_a,
            "10.0.0.1",
            None,
            2,
            "default Ray runtime",
        ),
        ("options", node_b, False),
        (
            "remote",
            "c" * 40,
            node_b,
            "10.0.0.2",
            None,
            2,
            "default Ray runtime",
        ),
    ]


def test_configured_python_runtime_uses_node_override_merge_mode(monkeypatch):
    node_id = "a" * 56
    monkeypatch.setattr(
        cluster_module.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_node_id=lambda: node_id),
    )
    cluster = object.__new__(Cluster)
    cluster._cluster_cfg = ClusterConfig.from_dict_cfg(
        OmegaConf.create(
            {
                "num_nodes": 1,
                "component_placement": {},
                "node_groups": [
                    {
                        "label": "custom",
                        "node_ranks": "0",
                        "env_configs": [
                            {
                                "node_ranks": "0",
                                "python_interpreter_path": "/opt/custom/bin/python",
                                "env_vars": [{"PYTHONPATH": "/opt/custom/src"}],
                            }
                        ],
                    }
                ],
            }
        )
    )
    source_env = Cluster._source_environment(dict(os.environ))
    merge_mode_key = Cluster.get_full_env_var_name(
        cluster_module.ClusterEnvVar.PATH_ENV_MERGE_MODE
    )
    source_env[merge_mode_key] = "override"
    source_env["PYTHONPATH"] = "/clean/default"

    runtimes = cluster._configured_worktree_runtimes(
        [
            {
                "node_id": node_id,
                "node_ip": "10.0.0.1",
                "node_rank": 0,
                "python_executable": sys.executable,
                "source_env": source_env,
                "ok": True,
            }
        ]
    )

    assert len(runtimes) == 2
    configured_runtime = next(
        runtime for runtime in runtimes if runtime["runtime"] == "node group 'custom'"
    )
    assert configured_runtime["node_id"] == node_id
    assert configured_runtime["node_ip"] == "10.0.0.1"
    assert configured_runtime["node_rank"] == 0
    assert configured_runtime["python_executable"] == "/opt/custom/bin/python"
    assert configured_runtime["source_env"]["PYTHONPATH"] == "/opt/custom/src"
    assert configured_runtime["source_env"][merge_mode_key] == "override"


def test_code_sync_source_must_match_driver_commit(monkeypatch):
    cluster = object.__new__(Cluster)
    cluster._source_commit = "a" * 40
    cluster._ray_code_sync_fragment = {"py_modules": ["/other/rlinf"]}
    monkeypatch.setattr(
        cluster_module,
        "require_clean_worktree",
        lambda path: "b" * 40,
    )

    with pytest.raises(RuntimeError, match="code-sync source.*imported driver"):
        cluster._verify_code_sync_source()


def test_execution_node_failure_aborts_launch(monkeypatch):
    node_id = "a" * 56

    class FakeRemoteCheck:
        def options(self, **options):
            return self

        def remote(self, *args):
            return "result-ref"

    monkeypatch.setattr(
        cluster_module.ray,
        "remote",
        lambda **options: lambda function: FakeRemoteCheck(),
    )
    monkeypatch.setattr(
        cluster_module.ray,
        "get",
        lambda refs, timeout: [
            {
                "node_id": node_id,
                "node_ip": "10.0.0.1",
                "ok": False,
                "error": "dirty Git worktree",
            }
        ],
    )
    cluster = object.__new__(Cluster)
    cluster._source_commit = "c" * 40
    cluster._ray_code_sync_fragment = None
    cluster._logger = SimpleNamespace(info=lambda *args: None)

    with pytest.raises(RuntimeError, match="10.0.0.1.*dirty Git worktree"):
        cluster._verify_execution_node_worktrees(
            [{"NodeID": node_id, "NodeManagerAddress": "10.0.0.1"}]
        )

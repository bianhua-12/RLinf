# Clean-worktree gate for Cluster-based runs

Status: Implemented

## Problem

RLinf could start training, evaluation, or real-hardware collection from a
checkout containing staged, unstaged, or untracked source and configuration.
The resulting run could not be bound reliably to a named commit, and a worker
could launch hardware before the repository state was noticed.

## Decision

Every explicit `Cluster` launch must resolve a Git `HEAD` commit and require an
empty porcelain status on the driver before logger, Ray, manager, worker, or
hardware initialization. Reject staged changes, unstaged changes, non-ignored
untracked files, and dirty submodules with the exact status entries in the
error.

Do not provide an environment-variable bypass. Ignore files already covered by
Git ignore rules so local environments, caches, and configured runtime output
do not make the existing checkout permanently unusable. Git subprocesses remove
inherited `GIT_*` variables so callers cannot redirect the gate to different
metadata, worktree, or index paths.

When Ray code sync is enabled, workers receive the clean package snapshot from
the verified driver; an explicit sync path must itself be clean at the same
commit. Otherwise, after Ray connects but before node probing, manager launch,
or hardware enumeration, hard-affinity tasks on every alive Ray node verify the
default Ray runtime plus every configured Python interpreter and
source-affecting environment (`PATH`, `VIRTUAL_ENV`, and `PYTHON*`). A timeout,
remote error, dirty checkout, invalid/duplicate node rank, or SHA mismatch
disconnects the driver and aborts the launch. Internal `Cluster()` calls are
attach-only and never fall back to creating an unvalidated run. Worker-specific
source-path changes that were not present at launch are rejected unless code
sync owns the package source.

## Research contract

- A Cluster-based run begins from a full `HEAD^{commit}` SHA.
- The driver checks the imported execution checkout, not a submission-host
  assertion.
- The check covers index, tracked worktree, non-ignored untracked, and submodule
  status as reported by `git status --porcelain=v1 --untracked-files=all
  --ignore-submodules=none`.
- Driver gate failure occurs before RLinf initializes logging or Ray. Remote
  validation necessarily runs after Ray connection but before `NodeProbe`,
  managers, environments, cameras, or robots.
- Without code sync, every alive Ray node must be clean at the driver SHA. With
  code sync, the clean driver package snapshot is the worker code source.
- Custom interpreters and Python source paths from `node_groups.env_configs`
  are validated in their actual Ray runtime before `NodeProbe`. Later
  worker-specific source path injection is fail-closed.
- There is no automatic stash, cleanup, commit, or bypass.
- Ignored files are outside this gate. They must not be treated as evidence or
  silently imported as source/configuration by a run.
- Standalone utilities that do not create a `Cluster` are outside this gate and
  must call `require_clean_worktree` before becoming supported experiment
  launchers.

## Alternatives considered

- Check only `git diff`: rejected because it misses staged and untracked files.
- Check inside a shell launcher: rejected because direct Python entrypoints
  could bypass it.
- Check every `Cluster()` attachment in workers: rejected because attachments
  do not create a run and Ray code-sync packages intentionally lack repository
  metadata. Launch-time validation owns the source contract.
- Reject ignored files: not selected because this checkout already keeps local
  environments and runtime logs in ignored paths. Those artifacts remain
  outside the reproducibility claim.
- Auto-commit or stash changes: rejected because it mutates user work and hides
  the exact code decision behind a launch side effect.

## Verification

- Focused unit tests create real temporary Git repositories and cover clean,
  staged, unstaged, untracked, ignored, and missing-repository states.
- Constructor-ordering unit tests verify that the driver gate precedes logging
  and Ray launch, `Cluster()` cannot fall back to launch, remote checks are
  pinned to each node, configured Python runtimes are included, and a node
  failure aborts initialization. This is evidence level 1.
- Ruff, format, and diff checks cover the changed Python and documentation
  surface. No training, Ray cluster, or real-hardware run is claimed.

## Consequences

Cluster-based tests and runs that create a real cluster must be executed from a
clean commit. Developers can still run pure unit tests while editing, but any
test that launches a Cluster must first commit the exact implementation under
test. Existing ignored logs and environments do not block a launch.

## Artifacts

- `rlinf/utils/repo_state.py`
- `rlinf/scheduler/cluster/cluster.py`
- `tests/unit_tests/test_repo_state.py`
- `tests/unit_tests/test_cluster_clean_worktree_gate.py`
- `tests/unit_tests/bench_broadcast.py`
- `AGENTS.md`

# Streaming LeRobot v3 video writer

Status: Implemented

## Problem

The Franka collector used LeRobot's deferred video mode. Every 30 Hz camera
frame was first written as PNG, and controller recovery then waited while all
pending PNGs were converted to MP4. With roughly 100 trajectories of about
130 seconds and three cameras, recovery work scaled with the full historical
pixel set instead of the unfinished episode.

## Decision

Add an opt-in `video_write_mode: stream_mp4` path while retaining
`lerobot_png` as the compatibility default. Stream mode sends synchronized
camera frame groups through a bounded 60-group queue to a spawned PyAV worker.
The worker directly encodes independent per-episode MP4 files with
`libsvtav1`, CRF 30, `yuv420p`, GOP 2, and preset 8.

Each episode stages its three MP4 files, numeric parquet, raw-image statistics,
task, checksums, and schema/encoding contract under `.streaming/partial`.
`manifest.ready.json` makes the staged episode replayable. Immutable artifacts
are then moved to LeRobot v3 one-file-per-episode paths; metadata is rebuilt
from committed manifests. Recovery removes an incomplete partial episode and
idempotently republishes ready episodes. Conflicting checksums fail closed.
Normal success commits update only the new episode and aggregate statistics;
historical media hashing and full metadata rebuild occur only during recovery.

The implementation is pinned to `lerobot==0.3.4` and protocol
`rlinf_lerobot_stream_mp4_v1` because it relies on that release's v3 schema.
The existing controller recovery boundary is unchanged: controller polling
raises when a Franka controller becomes inactive, collection closes in
`finally`, and the launcher rejects duplicate collectors before reinitializing
the control stack on rerun. Stream mode makes that close independent of all
previously collected video.

## Research contract

- Preserve the action-to-previous-observation alignment used by the old writer.
- Preserve task, `done`, `is_success`, intervention, `segment_id`, and monotonic
  observation timestamp fields.
- Require an exact and shape-stable camera set for every frame group.
- Never silently drop a frame: queue overflow, worker death, malformed state,
  camera mismatch, or artifact conflict raises immediately.
- Emit no PNG files in stream mode.
- Keep the legacy writer behavior unchanged unless stream mode is selected.
- Do not convert or delete datasets collected by the old path.
- Do not use a code checkout for physical collection until it has a named clean
  commit and separate robot-cell smoke authorization.

## Alternatives

- Deferred PNG encoding: rejected because restart time grows with all buffered
  images and was the observed operational bottleneck.
- A single ever-growing MP4 per camera: rejected because a crash can damage the
  entire session and episode-local recovery becomes ambiguous.
- Hardware NVENC: not selected for this change; the existing AV1 quality and
  decoding contract is kept constant for a controlled storage-only change.
- Best-effort queue drops: rejected because it produces plausible but
  misaligned robot demonstrations.

## Verification

- Focused unit tests cover prior-observation alignment, option conflicts,
  raw-image statistics, native LeRobot v3 load/decode, no-PNG output, idempotent
  ready-transaction replay, multi-episode independent files, version pinning
  before recovery writes, queue overflow, encoder death, camera mismatch,
  record-reset abort, success commit, and legacy writer regression. The final
  focused run passed 71 tests.
- The CPU smoke contract is three 224x224 cameras at 30 Hz, including one
  130-second episode and a manifest-only 100-episode recovery benchmark. It
  recorded 3900 frames, flushed in 0.989 seconds, decoded through native
  LeRobot, produced no PNGs, and rebuilt 100 episode manifests in 1.178 seconds
  without changing MP4 mtimes. Appending after 100 historical manifests flushed
  in 0.335 seconds.
- A real Franka smoke is intentionally excluded until separately authorized.

## Consequences

Successful episode flush cost is bounded by draining the active encoder queue
and publishing a few small metadata files. Restart recovery scans compact JSON
manifests and checksums; it never transcodes historical video. The v3 layout
contains more small parquet/MP4 files than LeRobot's default consolidation, but
the failure boundary is an episode and recovery remains local and deterministic.

## Artifacts

- `rlinf/data/storage/lerobot/streaming_writer.py`
- `rlinf/envs/wrappers/collect_episode.py`
- `examples/embodiment/config/realworld_collect_data_ros2_gello_dual_franka_pnp.yaml`
- `examples/embodiment/franka_fold_pi05_client.py`
- `tests/unit_tests/test_lerobot_streaming_writer.py`

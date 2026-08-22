# Dual-GELLO episode-reset stream continuity

Status: Implemented

## Problem

The direct dual-GELLO stream was gated off on every data-collection episode
reset. The inner dual-Franka reset intentionally skipped home motion, but the
operator could move either GELLO while observation and camera reset work ran.
Reopening the stream gate then sent the latest GELLO joint targets, producing a
delayed catch-up motion near personnel resetting the workspace.

## Decision

Keep an established, healthy direct GELLO stream active across episode resets.
The inner environment still receives `skip_reset_to_home=True`, resets episode
counters and observations, and checks controller health, but it does not pause
joint target publication or re-run GELLO alignment.

Retain controlled `reset_joint` alignment for the first reset, before the
direct-stream thread starts. Check stream health immediately before and after a
continuous reset. Before explicit home/alignment motion, close the stream gate
and wait for any in-flight command tick to finish. If reset or either health
check fails, perform the same acknowledged pause and mark alignment invalid so
the process fails closed instead of resuming tracking silently.

## Research contract

- This decision applies only when `direct_stream=True` and controlled initial
  alignment has completed.
- Joint targets remain the two seven-joint GELLO readings published at the
  configured stream period; gripper commands remain owned by `env.step`.
- Normal episode reset does not command home or realignment; joint target
  ownership remains with the continuous GELLO stream.
- An explicit `skip_reset_to_home=False` request still pauses direct streaming
  before home motion and controlled GELLO realignment.
- A healthy episode reset must not clear the stream gate, call `reset_joint`,
  or start a second stream thread.
- Initial startup must keep the stream gate closed until both arms complete
  controlled alignment.
- Explicit home, reset-failure, and stream-health failure paths must wait for
  any in-flight direct-stream command to finish before reset motion proceeds or
  the pause is considered complete.
- Reset and stream-health failures invalidate alignment and leave the stream
  gate closed.
- Dataset fields, action shapes, camera ordering, episode labels, and sampling
  rate are unchanged.

## Alternatives considered

- Reuse an upstream fix: rejected because RLinf `main` and the related closed
  PR #1463 retain the same per-reset gate-and-realign behavior.
- Add target slew limiting only when the gate reopens: rejected because it
  preserves the unexpected delayed motion while merely stretching it in time.
- Require a new clutch or operator confirmation after every episode: not
  selected because it changes the collection protocol and hardware interface;
  it remains a possible independent safety layer.
- Stop collecting before the final reset: retained for the terminal episode,
  but it does not address resets between successful episodes.

## Verification

- Focused unit tests cover continuous healthy reset, controlled initial
  alignment, and fail-closed reset errors. This is evidence level 1.
- Existing direct-stream tests cover float64 targets, controller failure
  propagation, foreground error reporting, and heartbeat expiry.
- No real-robot validation is claimed. A guarded hardware check with the cell
  clear and emergency stop available is required before collection resumes.

## Consequences

Operators no longer experience a software-induced GELLO freeze and catch-up at
normal episode boundaries. The arms continue following GELLO while episode and
camera state resets, so the operator must retain control of both leaders during
workspace reset. A stream failure or reset exception requires a controlled
restart/re-alignment rather than transparent recovery.

## Artifacts

- `rlinf/envs/realworld/common/wrappers/dual_gello_joint_intervention.py`
- `rlinf/envs/realworld/franka/dual_franka_env.py`
- `tests/unit_tests/test_dual_gello_joint_intervention.py`
- `examples/embodiment/config/env/realworld_ros2_dual_franka_joint.yaml`
- RLinf upstream `main` at `a3816b596478dcd8a5c69a6ec1468c9519f77b5b`
- RLinf pull request #1463 (closed without merge)

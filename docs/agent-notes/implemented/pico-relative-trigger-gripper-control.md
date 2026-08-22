# PICO relative-trigger gripper control

Status: Implemented

## Problem

The deployed dual-Franka PICO takeover used discrete open and close buttons for
the grippers. A continuous trigger is more ergonomic, but interpreting its
absolute value as the gripper command would make takeover jump whenever the
policy command and trigger position differed at the activation boundary.

## Decision

Add an explicit `relative_trigger` gripper mode while retaining `buttons` as
the compatibility default. When the arm's `grip` value first crosses
`control_threshold`, capture both the current executed gripper action and the
current value of `gripper_trigger`. During that takeover, compute

`reference_action - gripper_trigger_scale * (trigger - reference_trigger)`

for the default direction and clip the result to `[-1, 1]`. A larger trigger
value therefore closes the gripper. `gripper_invert: true` reverses the sign.
Releasing takeover clears both references, so the next takeover anchors again
to the then-current command without a discontinuity.

The Franka-fold pi0.5 client opts into this mode with the controller trigger
and a scale of `2.0`. It disables trigger-based calibration for that client so
one analog control cannot simultaneously change the gripper and recalibrate
the controller frame.

## Research contract

- PICO arm takeover remains gated by `grip`; the analog `trigger` controls only
  the gripper after takeover is active.
- The takeover boundary is continuous: the first relative-trigger output is
  the incoming gripper command, independent of the trigger's initial value.
- TCP and joint intervention wrappers pass the command currently governing each
  arm as the takeover reference. Subsequent policy gripper outputs do not move
  that reference until takeover is released and activated again.
- Trigger values are clipped to `[0, 1]`; output actions are clipped to
  `[-1, 1]`. The scale must be positive and finite, and a non-finite incoming
  gripper reference is rejected.
- `buttons` remains the default mode and preserves the existing A/B-style
  discrete command behavior for other consumers.
- This changes only human-intervention actions. It does not change the policy
  observation contract, model image inputs, or four-camera recording order.

## Alternatives considered

- Map the absolute trigger value directly to `[-1, 1]`: rejected because it can
  command a discontinuous gripper jump at takeover.
- Continue using two face buttons: retained as the compatibility default, but
  not selected for the Franka-fold client because it cannot express continuous
  motion.
- Re-anchor on every policy step: rejected because policy outputs during active
  takeover would move the human operator's reference and make the trigger
  response unpredictable.
- Use the trigger for both calibration and the gripper: rejected because one
  gesture would alter two independent control states.

## Verification

- Focused unit tests cover activation continuity, trigger scaling and clipping,
  release/re-activation anchoring, per-arm reference propagation in joint
  takeover, and the Franka-fold client configuration. This is evidence level 1.
- The client self-test covers construction and serialization boundaries but is
  not a real-robot validation of trigger ergonomics or task performance.

## Consequences

The Franka-fold PICO path now uses continuous trigger-relative gripper commands.
Other PICO users remain on button control unless they opt in. A real-robot
operator check is still required before making any claim about ergonomics or
closed-loop task performance.

## Artifacts

- `examples/embodiment/franka_fold_pi05_client.py`
- `rlinf/envs/realworld/common/pico/pico_expert.py`
- `rlinf/envs/realworld/common/wrappers/pico_intervention.py`
- `rlinf/envs/realworld/common/wrappers/pico_joint_intervention.py`
- `tests/unit_tests/test_franka_fold_pi05_client.py`
- `tests/unit_tests/test_pico_expert_threading.py`
- `tests/unit_tests/test_pico_joint_intervention.py`
- `docs/source-en/rst_source/examples/embodied/franka_vr.rst`
- `docs/source-zh/rst_source/examples/embodied/franka_vr.rst`

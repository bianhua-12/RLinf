# Dual-Franka four-camera observation contract

Status: Implemented

## Problem

The dual-Franka cell gained a second external camera, a RealSense D435, while
the deployed Franka-fold pi0.5 checkpoint retained its three-image training
contract. Camera discovery order and legacy field names were not sufficient to
distinguish the four-image recording schema from the three-image policy
payload. Reinterpreting either order silently would change dataset semantics or
feed an untrained image role to the policy.

## Decision

Use one explicit hardware and recording mapping across GELLO collection,
pi0.5 rollout, and PICO intervention:

| Physical role | Serial | Raw frame key | LeRobot field |
| --- | --- | --- | --- |
| Main view 1 | `DA6135161` | `base_0_rgb` | `image` |
| Left wrist | `261922076829` | `left_wrist_0_rgb` | `extra_view_image-0` |
| Right wrist | `262322073199` | `right_wrist_0_rgb` | `extra_view_image-1` |
| D435 main view 2 | `327122078534` | `base_1_rgb` | `extra_view_image-2` |

Configure the two base cameras with aligned `base_camera_serials` and
`base_camera_types` lists so the Hikrobot and RealSense backends can coexist.
Require `extra_view_image_keys` to list every non-main camera exactly once;
missing, duplicate, or unknown entries fail before recording.

Bind each RealSense pipeline directly to its requested serial instead of
enumerating and querying every connected device during startup. Make the
vector-environment close path idempotent, and signal the ROS2 launch process
group even when its leader has already exited, so a completed stage cannot
leave camera or robot-controller ownership behind for the next stage.

Keep the deployed pi0.5 policy payload at three images:

| OpenPI field | Raw frame key | Physical role |
| --- | --- | --- |
| `observation.image` | `base_0_rgb` | Main view 1 |
| `observation.extra_view_image-0` | `base_1_rgb` | D435 main view 2 |
| `observation.extra_view_image-1` | `right_wrist_0_rgb` | Right wrist |

The left wrist image remains in the four-camera recording but is omitted from
this policy payload. PICO takeover changes actions only; it does not change the
observation or recording mapping.

## Research contract

- Four-camera recordings use the fixed LeRobot order `image`,
  `extra_view_image-0`, `extra_view_image-1`, `extra_view_image-2` shown above.
- The deployed checkpoint receives exactly three HWC RGB `uint8` image arrays
  and no `observation.extra_view_image-2` field.
- The legacy OpenPI field name `extra_view_image-0` denotes D435 main view 2
  for this checkpoint. It must not be inferred from the LeRobot field with the
  same suffix; policy and recording adapters are separate contracts.
- GELLO collection and the pi0.5/PICO rollout client configure camera capture,
  control, and video writing at 30 Hz. Observation timestamps retain measured
  wall-clock timing; a 30 FPS container does not claim every interval is
  exactly 33.3 ms.
- All cameras in one recorded frame group must have stable, stack-compatible
  shapes. Missing or reordered cameras are errors, not best-effort fallbacks.
- Closing the outer real-world environment releases owned hardware exactly
  once. ROS2 controller children must not outlive their dedicated process
  group between collection stages.
- Existing three-camera datasets and the deployed three-image checkpoint are
  not silently reinterpreted as four-image artifacts.

## Alternatives considered

- Send all four images to the deployed checkpoint: rejected because its model
  and training data define only three image slots.
- Continue sending the left wrist in the second policy slot: rejected because
  the confirmed training-time role for that slot is the D435 main view.
- Derive recording order from sorted frame keys: rejected because names and
  discovery order do not constitute a durable semantic contract.
- Use one base-camera backend for both devices: rejected because the installed
  main cameras require Hikrobot and RealSense drivers respectively.

## Verification

- Focused unit tests cover mixed base-camera backends, misaligned backend
  lists, explicit four-camera ordering, missing-camera rejection, LeRobot field
  fan-out, direct RealSense binding, idempotent environment close, ROS2 process
  group cleanup, the three-image policy adapter, and four-image recording
  adapter. This is evidence level 1.
- The client self-test covers state layout, MessagePack serialization, action
  shape, and the distinct policy/recording image mappings.
- A diagnostic real-hardware run produced four synchronized 224x224 MP4
  streams for GELLO, pi0.5 rollout, and PICO intervention, and a serialized
  three-image request received a `(30, 16)` policy response. These checks were
  run from a dirty development checkout, so they are operational diagnostics,
  not formal experiment evidence or a task-performance claim.

## Consequences

New four-camera datasets have one additional LeRobot video field and require
consumers that understand this explicit order. The deployed checkpoint remains
shape-compatible because its payload stays at three images, but its second
field must be populated from `base_1_rgb`. A future four-image policy requires
an explicit OpenPI data/model contract and retraining rather than an adapter
that guesses from tensor shape.

## Artifacts

- `examples/embodiment/config/realworld_collect_data_ros2_gello_dual_franka_pnp.yaml`
- `examples/embodiment/config/realworld_dual_franka_collect_data_pico.yaml`
- `examples/embodiment/franka_fold_pi05_client.py`
- `rlinf/envs/realworld/common/camera/realsense_camera.py`
- `rlinf/envs/realworld/franka/dual_franka_env.py`
- `rlinf/envs/realworld/franka/ros2_controller.py`
- `rlinf/envs/realworld/realworld_env.py`
- `rlinf/scheduler/hardware/robots/dual_franka.py`
- `tests/unit_tests/test_collect_episode.py`
- `tests/unit_tests/test_franka_fold_pi05_client.py`
- `tests/unit_tests/test_realsense_camera.py`
- `tests/unit_tests/test_ros2_dual_franka_joint_env.py`

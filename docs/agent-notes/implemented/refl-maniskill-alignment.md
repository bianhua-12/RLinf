# Align with the original ManiSkill trainer / 对齐原 ManiSkill 训练器

Status: implementation restored; CPU and real-checkpoint GPU parity passed.
Native learning validation failed (53.91% initial,12.50% final); the port is not
accepted as successful training. See the
[experiment report](../../experiments/refl-maniskill-alignment-20261028.md).

## Correction / 更正

The55158db9 variant did not implement the user's supplied SuperMimic equation.
It added success-boundary value terms and changed critic targets to preserve
V(success), then trained only live critic rows. Its82.42% selected result is
not evidence against the original critic implementation.

上一版加了图中没有的“成功时刻补值”，还改了 critic 目标和训练样本。这不是旧版
critic 的等价恢复；82.42%的结果不能用来否定旧实现。本次以旧源码为准。

## Pinned source / 固定来源

Use SuperMimic commit `56814f65a9916d5fcbba46018858fce63cc99fa1`, its native reward
branch with the critic enabled. The user-requested no-RM setting remains explicit;
this is not the historical RM-enabled824 experiment. The execution image remains
the requested RLinf0.4 ManiSkill image, rather than silently downgrading packages.

`refl_maniskill.py` directly ports the original PolicyTrainer, WorldModelTrainer,
SingleImagination, policy_objective, gae_targets and single-Panda adapters. Only
unused generic MuJoCo entry paths are replaced with explicit adapter errors.
The active rollout/update arithmetic is retained. MIT attribution is preserved.
Verbatim original source files and SHA256 checks live in
[the reference fixtures](../../../tests/unit_tests/refl_reference/SOURCE.md).

## Restored contract / 恢复的契约

- Actor objective exactly follows the supplied equation:
  `E[sum(t=0..H-1) gamma**t*c_t*R(s_(t+1)) + gamma**H*c_H*V(o_H)]`.
  The only value contribution is at the surviving H-step horizon. Early native
  termination makes c_H zero; there is no intermediate success bonus/value.
- Optimize `-J/H`, then original norm1 and value0.1 actor gradient clips.
- Critic targets use the original masked lambda return:
  `R_t=r_t+gamma*(1-d_next)*((1-lambda)*V_next+lambda*R_next)`.
  True termination zeros continuation. R_H is V_H.
- Critic trains all H*N flattened rows, including original absorbing padded
  states whose post-terminal rewards/targets are zero. Five epochs, four
  minibatches, original shared/global torch.randperm, Adam1e-4 and norm clip1.
- Critic and WM parameters freeze during actor BPTT; their input derivatives
  remain enabled. Critic fitting uses detached rollout/targets.
- Restore original source bank valid mask44317, without the108-row runtime overlay.
  The prior overlay was a separate migration choice. Source bank lineage remains
  immutable; this choice restores sampling identity, not cross-image physics
  equivalence. No claim is made that every old frame passes new-image tolerances.
- Restore Oracle -> WM -> actor -> critic construction order and original shared
  critic permutation RNG. The PPO790 actor/critic weights still load exactly.
- Original ring replay, fixed Windows, WarmupReplay, MixedReplay and BinSampler
  remain in use; WM gets three executed actions, half fixed/half online data,
  ten WM optimizer steps before each actor update.
- H13, batch256, gamma0.8, lambda0.95, seed0, BF16,200 actor/2000 WM/4000 critic
  updates and the native512 evaluation populations remain explicit.

这次同时恢复：图中的唯一末端价值项、终止处归零的 critic 目标、全部想象行训练、
原随机打乱、原 bank 有效帧、模型构造顺序和原更新函数。只保留用户指定的无 RM
原生奖励分支及新镜像；不声称跨 Torch/SAPIEN 版本逐位复现旧 A800 学习曲线。

## Parity gates / 一致性验证

The tests load independently frozen original source, not a re-derived formula.
They compare objective values, reward/value gradients, lambda targets, actor and
critic parameter/gradient updates and Adam states across three consecutive
updates, and three WM updates. CPU tests require exact equality. A separate
clean-commit GPU parity run uses the actual PPO790, WM822 and dataset793 at BF16,
H13 and the original critic update settings, again requiring exact equality.

The source snapshots are verified against their pinned Git commit and SHA256.
This task's external launch controller requires the GPU parity result for its
executed commit. The portable training entrypoint itself checks source/input
provenance, not this external parity gate. The GPU parity profile uses batch8,
three actor updates,60 critic updates and6 WM updates (two per call), versus
production batch256 and ten WM steps per actor update. It checks the same
H13/BF16 equations at reduced batch/update scale, not production throughput.
After parity passes, native initial/selected/final evaluation determines learning;
matching derivatives is not a task-success claim.

## Boundaries and compatibility / 边界与兼容

RLinf still owns process placement, logging, input checks and durable records.
Validation uses isolated RNG and additional read-only input/probe checks; these
are not alternative losses. Native initial evaluation precedes training as a
runtime acceptance check. Simulator/compiler versions remain those in the chosen
image, so bitwise equivalence to a historical A800 run is not promised.

Checkpointv3 records `implementation: maniskill_56814f6` and
`critic_bootstrap: horizon_only`. Historicalv2 retains its success-and-horizon
meaning for inference reload; it is never silently reinterpreted asv3. Exact
training resume remains unsupported. The rejected variant's artifacts are kept.

## Read-only gradient diagnosis / 只读梯度诊断

`examples/embodiment/diagnose_refl_critic.py` measures reward and horizon-value
actor gradients on the same 256 held-out real starts, crossing initial/final
actor-WM and critic weights. No parameters or losses are changed. Raw-gradient
cosines precede clipping/Adam and cannot establish a causal training attribution.
It also measures whether native512 evaluation consumes global Torch RNG. Input
checkpoint hashes, clean diagnostic commit and environment are recorded.

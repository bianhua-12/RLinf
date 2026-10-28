# Restore replay before comparing collection samplers / 恢复 replay 后比较采集器

Status: implemented; 16 focused CPU tests and two controlled GPU runs completed.
[Results and provenance](../../experiments/refl-replay-sampler-20261028.md).

## Problem / 问题

The first RLinf port combined several changes: WoVR-style KIR collection,
trajectory-window replay, new runtime bank filtering, different software image,
and diagnostic draws from the training RNG. Its87.70% selected success cannot
be treated as an equivalent reproduction of the old no-RM run872 (93.75%).

首次迁移混合了采集器、replay、有效帧和运行环境等变化；87.70%不能视为旧872的
等价复现。本次先恢复旧 replay 并隔离诊断随机数，再只切换采集器。

## Decision / 决策

Restore the original SuperMimic `ReplayBuffer`, `record_online`, `MixedReplay`,
`Windows`, `WarmupReplay` and `BinSampler` from commit
`56814f65a9916d5fcbba46018858fce63cc99fa1`. Formatting and a dict-literal lint
rewrite are the only changes inside those copied components. Preserve the MIT
notice. AST comparison against the source confirms structural equivalence after
normalizing the dict spelling.

Select both mechanisms explicitly in configuration:

| Setting | Values | Default |
| --- | --- | --- |
| `replay_backend` | `ring`, `trajectory` | `ring` |
| `collection_sampler` | `bins`, `kir` | `bins` |

Missing selectors are errors, not inferred legacy compatibility. The original
trajectory implementation remains selectable for a future controlled comparison;
this request schedules two runs with the ring backend.

使用显式配置选择 ring/trajectory 和 bins/kir，默认恢复 bins+ring。保留旧迁移的
trajectory 路径用于后续受控对照，但本次两组都固定为 ring。

## Matched experiment / 配对实验

1. A: original BinSampler + original ring replay.
2. B: KIR + the same original ring replay.

Both run serially on the same Zenith A100/image from the same code commit and
seed0. Each independently reloads PPO790 and WM822 and receives200 actor and2000
WM updates. No RM, critic or bootstrap; H13 actor imagination, three executed
WM actions, BF16 neural compute, batch256, gamma0.8 and unchanged optimizer
settings. Native512 checkpoint selection and independent seed40000 evaluations
follow the existing contract. Run identity/output paths necessarily differ;
`collection_sampler` is the only algorithm configuration difference.

Use the same already-verified runtime bank mask from the first Zenith run:
SHA256 `fe79725a7690498fc132b1afda72fef32125536f032942f77c6ff8f4c3c11827`,
TOS version `ec11bf697c74dbc2ff80f2c99e9a8ed89ef015c0ea316af389e75ee28925def1`.
Bind it as the optional `bank_mask` input with its version/hash/verification
receipt. The loader checks bool dtype, shape, subset-of-source validity and at
least one valid frame per environment. This reuses the same-runtime restoration
proof and prevents a different mask from becoming another experimental variable.
It is not a mask-validity guarantee for another image or robot contract.

两组从同一底座独立开始，唯一算法配置差异是采集器。固定使用第一次 Zenith
验证并已存入 TOS 的有效帧 mask；其他镜像不能仅凭这个 mask 省略恢复验证。

## Replay and randomness / Replay 与随机数

- Ring dimensions: 3584 training environments,1000 rows per environment,
  state48, action8, context57. Terminal target rows and original episode IDs,
  pointer/wrap masks and `multinomial` sampling are preserved.
- WM batches mix equal fixed and ring populations using the original adapters.
  The fixed population uses the original `Windows` filter and `randint` draws.
- Actor starts use the original `sample_starts(batch256, history1)` and a
  shape-only adapter into the existing differentiable objective.
- Fixed/current diagnostics use a private accelerator generator seeded31000.
  Native evaluation and model reconstruction preserve global Python, NumPy,
  CPU Torch and accelerator RNG states, including exceptional exits.
- Initial training RNG fingerprints are logged to check A/B starting alignment.
- The earlier87.70% run used both trajectory replay and shared diagnostic RNG.
  Comparing it with the new KIR+ring run cannot isolate replay alone. A third
  KIR+trajectory run with the same isolated RNG would be needed for that claim.
- A/B is a same-image recipe comparison, not a bitwise reproduction of the
  A800/Torch2.9/SAPIEN3.0.3 run872. Current image and fixed bank mask stay intact.

诊断和评估不再消耗训练共用 RNG。旧87.70%同时使用 trajectory replay 和共用
诊断 RNG，因此不能把它与新 KIR+ring 的全部差值都归因于 replay；单独量化 replay
需要额外的同 RNG 对照。本次也不宣称与旧 A800 环境逐位一致。

## Verification / 验证

Sixteen tests passed in the requested image, including ring wraparound, terminal
target retention, executed/clean-action alignment, fixed-window boundaries,
BinSampler held-out isolation, and RNG restoration after an exception. The
copied eight classes/functions/dataclasses pass normalized AST equality checks.
Independent review passed. The two GPU runs reached91.99% (BinSampler) and90.43%
(KIR) selected native success; this is partial recovery, not full reproduction
of the historical93.75% no-RM result. Exact training resume remains unsupported.

Implementation: [replay](../../../rlinf/data/refl_replay.py),
[samplers](../../../rlinf/envs/sim/maniskill/refl_pickcube.py),
[worker](../../../rlinf/workers/actor/refl_actor_worker.py),
[tests](../../../tests/unit_tests/test_refl.py).

# Zenith PickCube ReFL validation / Zenith PickCube ReFL 验证

A single-seed run improved independent native success from **276/512 (53.91%)**
to **449/512 (87.70%)** for the validation-selected checkpoint. The final
checkpoint reached **445/512 (86.91%)**. This validates the implemented path for
this run; it does not establish convergence, robustness across seeds, or
superiority to PPO or a learned-RM control.

一次固定种子实验中，独立原生成功率从 **53.91% 提高到 87.70%**；最终 checkpoint
为 **86.91%**。这是本次管线和训练结果的验证，不代表已经收敛、多种子稳健性或优于
PPO / 使用 RM 的对照。

## Implementation and execution / 实现与执行

- Official RLinf baseline: `e1a82ebf0c1af69da6ab9b1519f5f3692fe33566`.
- Executed clean commit: `85c61ed594dc587f814d824f728a4546831a9507`.
- Branch: `bianhua-12/RLinf`, `features/refl`.
- Algorithm and data contract: [Agent Note](../agent-notes/implemented/refl-pickcube.md).
- Zenith development instance `di-20261028103155-sgw7j`, one A100-SXM4-80GB.
- Image: `docker.1ms.run/rlinf/rlinf:agentic-rlinf0.4-maniskill_libero`.
- Environment: Torch2.11.0+cu128, ManiSkill3.0.0b22, SAPIEN3.0.1,
  PyTorch Kinematics0.7.6, Ray2.58.0. Full package snapshot is in the archive.
- Seed0; 20 collection iterations, 200 actor / 2000 WM updates; BF16 neural
  compute and FP32 state/FK. RM updates0, critic updates0, value bootstrap0.
- Execution record: 2026-10-28 11:25:13–11:30:04 Asia/Shanghai (about291 seconds,
  including runtime initialization, bank validation and evaluation; not an
  optimizer-only throughput measurement).
- Start-record SHA256: `12bcb513fc3c5bcc05589ade96d0a6cad0c55a7e973699c4e5de52b380e57c9f`.
- Completion exit0; no cleanup errors. The execution checkout was clean at launch.

## Evaluation / 评估

Every row below uses512 native-reset episodes, 128 environments per batch,
50 control steps, seeds40000..40003, with freshly loaded checkpoint weights.
Success means native `success_once`; it is not the KIR collection rate.

| Checkpoint | Actor updates | Successes | Success rate | Mean undiscounted return |
| --- | ---: | ---: | ---: | ---: |
| Initial PPO790 | 0 | 276/512 | 53.90625% | 25.57417 |
| Selected | 150 | 449/512 | 87.69531% | 33.20851 |
| Final | 200 | 445/512 | 86.91406% | 32.69639 |

Selected improvement: **+33.78906 percentage points**. Selection used separate
seeds30000..30003 and retained the initial checkpoint as a candidate:

| Actor updates | Selection successes /512 |
| ---: | ---: |
| 0 | 265 |
| 50 | 124 |
| 100 | 328 |
| 150 | 453 |
| 200 | 425 |

The early decline and later final-checkpoint decline remain part of the result.
The selected checkpoint is iteration15, with150 actor and1500 WM updates.

## Verification / 验证

- Twelve focused CPU tests and Ruff checks passed on the requested image.
- Independent static review completed; source-column and completion-state
  findings were fixed before launch.
- Native bank validation retained44209 of44317 previously valid frames. The
  run-local mask was saved without modifying the original bank.
- Native state reward parity maximum error was0 in every logged iteration.
- Actor gradients were finite and positive. Frozen WM parameters received no
  actor-update gradients. Actual WM and actor probe outputs passed checkpoint
  reload comparison with absolute tolerance1e-6.
- W&B cloud history independently confirmed20 iteration rows, the200/2000 update
  totals, zero RM/critic/bootstrap values, zero reward-parity error, all three
  independent evaluations, and run state `finished`.
- The repository heuristic audit reports two existing relative-link errors in
  the upstream test-install skill. That file was not changed.

[W&B run and curves](https://wandb.ai/rleai/rlinf-refl-pickcube/runs/refl-pickcube-20261028-85c61ed5).

## Archive / 归档

TOS asset version:
`ec11bf697c74dbc2ff80f2c99e9a8ed89ef015c0ea316af389e75ee28925def1`.

Physical manifest:
`tos://qianzhangshucai-01/post-training/_assets/v1/versions/ec11bf697c74dbc2ff80f2c99e9a8ed89ef015c0ea316af389e75ee28925def1/manifest.json`.

The existing qz asset protocol published20 content objects, 213747843 bytes,
with actual TOS HEAD byte/CRC64 verification. Manifest, event and READY were
published in order. No remote full-SHA256 readback was performed; cloud-catalog
acceptance remains pending and is separate from verified object storage.

The archive contains selected/final actor and WM bundles, exact source patch,
configuration, environment snapshot, sealed start/completion records, per-episode
outcomes, metrics and W&B verification. Identical original training inputs are
referenced by immutable version/hash rather than copied into this package.

- Selected checkpoint SHA256: `765b5befb85665d4d3e7fb90df363c2fda1d4e81600847829a308fbca23293ad`.
- Final checkpoint SHA256: `91b7aa0113d19b241ea04e67eeb34f90af8e3942bec1a1417e0506867b3188e0`.
- Payload moved directly from Zenith to TOS. The existing A800 authorization
  controller issued temporary object URLs; neither A800 nor the Mac relayed
  checkpoint, dataset or log bytes.

选中与最终权重、逐回合结果和复现记录已完成 TOS 大小/CRC 校验；目录索引接受状态
仍是 pending。源码提交和实验结果绑定到执行时的85c61ed5，而非之后新增文档的提交。

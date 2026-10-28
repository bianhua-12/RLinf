# PickCube replay/sampler comparison / PickCube replay 与采集器对照

Two matched, single-seed runs completed on the same Zenith A100/image. Restoring
the original ring replay with BinSampler reached **471/512 (91.99%)** independent
selected success. Changing only the sampler to KIR reached **463/512 (90.43%)**.
The observed difference is8 episodes, **1.5625 percentage points**; it does not
establish a general ranking across seeds.

同一 Zenith 镜像的两组配对实验已完成：原 ring + BinSampler 为 **91.99%**，仅切换
为 KIR 后为 **90.43%**。差8个回合、1.5625个百分点，不据此宣称多种子普遍优劣。

## Protocol and checks / 协议与核对

- Executed clean commit: `375a55c1883e6deccd1450c19848de34380dcf39`.
- [Research decision](../agent-notes/implemented/refl-replay-sampler-ablation.md).
- Same PPO790, WM822, dataset793, reference bank and cached runtime validity mask.
- Same seed0, BF16, H13, batch256, Adam settings, 200 actor / 2000 WM updates.
- RM0, critic0, bootstrap0. Real-state native reward parity maximum error0.
- Both initial native evaluations exactly276/512; per-episode outcome files match.
- Actual resolved configs differ only in `collection_sampler`, `run_id` and
  `output_dir`. Same code commit and installed-package snapshot:
  `e4e76acf575bd81ec3a39caeac10e373094a8a148ada3dd07f91d189a14b0075`.
- Initial CPU+accelerator Torch RNG fingerprint matches:
  `c78cfd3ff4c9b54e556e86b7ed13d1dd2d55fefc334c0c4630b98f7a939f41e9`.
- Runtime bank mask matches:
  `fe79725a7690498fc132b1afda72fef32125536f032942f77c6ff8f4c3c11827`.
- Sixteen focused CPU tests and Ruff checks pass. Eight restored components
  match the original source under normalized AST comparison. Independent review
  found no launch blocker. No unrelated upstream behavior changed.
- Both completion records report exit0 and no cleanup errors. W&B cloud history
  verifies20 iteration rows, update totals, zero RM/critic/bootstrap, reward parity
  and independent evaluations; both runs are `finished`.

两组的实际配置、初始回合结果、Torch RNG 指纹、依赖快照及 mask 均已核对；唯一
算法配置差异是采集器。验证抽样使用独立 generator，评估保存并恢复全局 RNG。

## Native evaluation / 原生评估

Independent evaluation uses512 native-reset episodes, seeds40000..40003,
128 environments per batch and50 control steps. Selection uses different seeds.

| Run | Initial | Selected | Final |
| --- | ---: | ---: | ---: |
| A: BinSampler + original ring | 276/512 (53.90625%) | 471/512 (91.99219%) | 471/512 (91.99219%) |
| B: KIR + original ring | 276/512 (53.90625%) | 463/512 (90.42969%) | 461/512 (90.03906%) |

Both selected checkpoints come from update200. Their selected/final actor and WM
parameter tensors are equal within each run. Repeated simulator evaluation can
vary: B returned463 versus461 successes for those equal weights. This is not an
additional optimizer step or a different initialization.

两个 selected 都来自第200次更新。各组 best/final 的 actor 与 WM 参数张量一致；
B 的同权重重复评估得到463和461个成功，存在模拟器重复评估波动。

Training selection curve, seeds30000..30003 (successes out of512):

| Actor updates | A: BinSampler | B: KIR |
| ---: | ---: | ---: |
| 0 | 265 | 265 |
| 50 | 136 | 226 |
| 100 | 376 | 323 |
| 150 | 447 | 234 |
| 200 | 468 | 457 |

## Interpretation / 解释边界

The [initial RLinf port](refl-pickcube-20261028.md) used KIR, trajectory-window
replay and shared diagnostic RNG, and reached87.69531% selected success. B is
higher by **2.73438 points**, but replay, diagnostic/evaluation RNG handling and
startup validation procedure changed together. This does **not** isolate replay's
individual contribution. A versus B isolates the selected sampler configuration
within the current paired protocol, subject to the single-seed limitation.

The historical no-RM run872 reached93.75% selected and92.96875% final. A remains
1.75781 points below its selected result, or0.97656 points below its final result.
The old image/runtime and original bank validity differ, so this is not a bitwise
historical reproduction. The95.70%/95.51% run873 also trained an RM and is a
separate algorithmic condition.

首次迁移87.70%到本次 KIR+ring 的90.43%，改善2.73个百分点，但不能全归给 replay，
因为 RNG 和启动验证流程也变化了。A/B 的采集器差为1.56个百分点；恢复后仍未完全
达到旧无 RM 的872。原95%左右的873带 RM，不能把它当作完全相同的算法条件。

Final WM diagnostics (not task-success evidence):

| Metric | A | B |
| --- | ---: | ---: |
| Fixed validation H3 normalized MSE | 0.76656 | 0.74914 |
| Fixed validation reward MAE | 0.019710 | 0.018786 |
| Current-policy validation H3 normalized MSE | 0.63062 | 0.80427 |
| Current-policy validation reward MAE | 0.023825 | 0.024226 |
| Valid collected training transitions | 1484737 | 1816448 |

B collected more valid transitions and has slightly lower fixed-data WM error,
yet lower native task success. Neither transition count nor aggregate WM fitting
error is a substitute for the independent task evaluation. Current-policy
validation populations differ, so their errors are descriptive, not a matched
state-distribution comparison.

## Provenance and archives / 血缘与归档

- A run: `refl-bins-ring-20261028-375a55c1`.
  [W&B](https://wandb.ai/rleai/rlinf-refl-pickcube/runs/refl-bins-ring-20261028-375a55c1).
  Start SHA256 `1849fdec33753ba974e59e98c58cade0fbb81a0a86a413aae7c02e495313edb6`.
  Best SHA256 `2c7ee677eb6f4c4bbb244657d65ab99371ef71cc4d3bff55d8bc2d4f9fe1282d`.
  TOS version `1b7096675e5cb53a4a33005e6458c0834d9f7a44ed3bf59c418c315529a891e6`.
  Verified20 content objects /213784634 bytes;16 uploaded,4 reused.
- B run: `refl-kir-ring-20261028-375a55c1`.
  [W&B](https://wandb.ai/rleai/rlinf-refl-pickcube/runs/refl-kir-ring-20261028-375a55c1).
  Start SHA256 `48aa59e8d56bdbce19bf6b24be816317025643da09b2bdc961a6420b1cab21e7`.
  Best SHA256 `c140fd1081b16a2210c6c8c9cbe2a2ef5e15061979b8eb29e12942ff9fd1bf6c`.
  TOS version `17626d4f3fbf54d3a9001e9b0a9338888459790b0d71971cc897cd8762286715`.
  Verified20 content objects /213784671 bytes;15 uploaded,5 reused.

Each physical manifest is at
`tos://qianzhangshucai-01/post-training/_assets/v1/versions/<version>/manifest.json`.
The archive includes selected/final bundles, episode outcomes, configuration,
source patch, environment, sealed start/completion, metrics and W&B verification.
Payloads went directly Zenith -> TOS using existing qz asset manifests/events and
READY ordering. Actual byte/CRC64 checks passed; no full remote SHA readback was
performed. Catalog acceptance remains pending and is distinct from TOS storage.

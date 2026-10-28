# Completion-cost scoring trial / 完成成本计分对照

The score-ordering change passed its mathematical/gradient checks, but the
single-seed training run did **not** establish a stable performance improvement.
Selected native success was473/512 (92.38%) versus471/512 (91.99%) for the
matched native-score control; final success fell to421/512 (82.23%).

计分顺序已修正，但训练未稳定变好：最佳92.38%仅比对照91.99%多2个成功回合，
最终降至82.23%。不能把中途峰值当成解决了性能差距。

## Rule and scope / 规则与范围

[Decision and proof](../agent-notes/implemented/refl-completion-cost.md).
Actor imagined score is `r_native**2 - 1`, bounded in[-1,0]. An unfinished step
costs negative points, success costs0, absorbing successors cost0. Immediate
success therefore has higher forward score than any delayed completion under
this task's normalized native reward bounds. Process progress still supplies
input gradients; the square changes their relative weighting.

The constant subtraction alone would not alter computed pathwise gradients
because success masks are detached. This trial therefore changes dense-gradient
weighting as well as forward score ordering. It does not supply an exact gradient
of the hard arrival time, and cannot establish that a delay incentive caused the
previous performance gap. Successful score ordering is distinct from task learning.

规则是“未完成每步扣成本，过程分越高扣得越少；成功扣0”。仅减常数不会改变当前
硬终止掩码下的反传梯度，平方才改变过程梯度的权重。因此本实验不能单独证明旧的
掉点由拖延激励造成，也不保证计分正确后策略一定学得更好。

- Executed clean commit: `e8de5ad782830894e81d12ee9053166a6626f44b`.
- Run: `refl-completion-cost-20261028-e8de5ad7`.
- Control: `refl-bins-ring-20261028-375a55c1`.
- Same BinSampler, original ring, seed0, PPO790/WM822, dataset/bank/valid mask,
  BF16, H13, batch256, gamma0.8, optimizer and200 actor/2000 WM updates.
- No learned RM, critic or value bootstrap. Native environment rewards, WM
  labels, collection reward weights and native success criteria remain unchanged.
- Resolved configs differ only in reward_rule and required commit/run identities.
  The old control's implicit native rule is explicitly identified for this audit;
  its stored config is not rewritten.
- Same package snapshot and initial Torch RNG fingerprint
  `c78cfd3ff4c9b54e556e86b7ed13d1dd2d55fefc334c0c4630b98f7a939f41e9`.

## Native evaluation / 原生评估

512 native-reset episodes, seed40000..40003, horizon50, reloaded checkpoints:

| Policy | Actor updates | Successes | Success rate |
| --- | ---: | ---: | ---: |
| Initial PPO790 | 0 | 276/512 | 53.90625% |
| Native-score control selected/final | 200 | 471/512 | 91.99219% |
| Completion-cost selected | 150 | 473/512 | 92.38281% |
| Completion-cost final | 200 | 421/512 | 82.22656% |

The selected gain is only0.390625 points; the final drop versus control is
9.765625 points. This is one seed, with no confidence or superiority claim.
The experiment does not measure a change in time-to-success.

Selection curve, separate seeds30000..30003 (successes out of512):

| Actor updates | Native control | Completion cost |
| ---: | ---: | ---: |
| 0 | 265 | 265 |
| 50 | 136 | 229 |
| 100 | 376 | 407 |
| 150 | 447 | 472 |
| 200 | 468 | 425 |

Current-policy WM H3 normalized MSE increased from0.67834 at update150 to1.02967
at200; corresponding reward MAE increased from0.02433 to0.03059. These are
policy-dependent populations, not a fixed-state gradient-fidelity test. They do
not prove why the late decline occurred. The imagined objective also worsened
from-3.19947 to-3.26719; it would be incorrect to claim it kept improving while
native success fell. Thirteen-step physical gradient fidelity remains unmeasured.

末段真实成功率和想象目标都变差；当前策略状态上的模型误差也变大。但样本分布随
策略变化，不能据此直接判定根因。13步梯度在真实模拟器里是否可靠，尚未验证。

## Validation and provenance / 验证与血缘

- 23 focused tests and Ruff passed; independent review found no launch blocker.
- Tests cover H3/H13 immediate/delayed/never-success ordering, score bounds,
  invalid values, non-collinear policy-gradient changes and finite differences.
- Native reward parity remained exactly0. Actor gradients were finite/nonzero;
  actor and WM inference probes passed independent weight reload checks.
- Start SHA256: `431a046b36c0d9c42ba64356791197b05b5db7b7ee78b9ce25fd45057ceebcff`.
- Selected SHA256: `eaf9c9a6efa87a4d6240550759730bce1389426aac9874a0331ba405ff98cbb3`.
- Final SHA256: `66af2fb9cc28eca8ef8e9126727c939f00316535482c2ed90626c4cdcbead1e6`.
- Start:2026-10-28 15:37:49 Asia/Shanghai; native evaluations logged by15:42:39;
  completion record15:55:13. Completion includes slow W&B shutdown, not only compute.
- Completion reports exit0 and no cleanup exception. W&B initially had HTTP408
  upload failures; replaying its existing local record under the same run ID with
  console replay skipped succeeded. No retraining occurred. A fresh cloud-history
  check verified20 iteration rows,200/2000 updates, zero RM/critic/bootstrap,
  all independent evaluations and run state `finished`.

[W&B run](https://wandb.ai/rleai/rlinf-refl-pickcube/runs/refl-completion-cost-20261028-e8de5ad7).

## Archive / 归档

TOS version `00e99a9d932cb2a78d90e8c63846780f6cb4c2f82d3f2a77bff5c217d9011299`.
Physical manifest:
`tos://qianzhangshucai-01/post-training/_assets/v1/versions/00e99a9d932cb2a78d90e8c63846780f6cb4c2f82d3f2a77bff5c217d9011299/manifest.json`.

20 content objects /213803143 bytes verified by TOS HEAD size/CRC64;16 uploaded,
4 reused. Source patch, selected/final bundles, episode outcomes, environment,
configuration, metrics and provenance are included. Full remote SHA readback was
not performed; catalog acceptance remains pending. Business bytes moved directly
Zenith -> TOS using the existing qz protocol, without a Mac/A800 payload relay.

# Completion cost for direct-gradient PickCube / 尽早完成的计分规则

Status: implemented and tested; the controlled run did not establish stable
improvement. [Results and provenance](../../experiments/refl-completion-cost-20261028.md).

## Problem / 问题

The native episodic objective gives success reward1, then masks all future
rewards. With gamma0.8, immediate success scores1 while one step at reward0.55
followed by success scores1.35. This rewards some delayed completions in the
forward objective. It does not prove the policy has learned to exploit the rule.

成功后清零后续奖励时，立即成功得1分，先拿0.55再成功却得1.35分。这个算分规则
存在拖延激励，但此前的任务失败不能直接归因于它。

## Decision / 决策

Keep the native environment reward `r`, success predicate, collection rewards,
WM supervision, replay and independent evaluation unchanged. Transform only the
actor's imagined training score:

```
completion_cost_v1: score(r) = r**2 - 1
native:             score(r) = r
```

The current PickCube normalized dense reward is in[0,1]. Unfinished states score
strictly below1 (their native formula is bounded by4/5); native success is1.
Reject nonfinite/out-of-range values in the completion-cost path. Therefore:

- every unfinished transition has negative score;
- native success has score0, followed by absorbing zero score;
- immediate success has total0;
- any delayed success contains at least one strictly negative discounted term,
  hence its total is lower than immediate success;
- the same time limit and H13 imagination horizon remain in force.

There is no critic, learned RM, terminal value estimate or bootstrap. The actor
still differentiates through the WM. Global reward scale, learning rate, clipping,
optimizer and rollout budget are held fixed for the comparison.

新规则是“未完成每一步扣成本，过程分越高扣得越少；成功扣0，之后也扣0”。因此
立刻成功的0分高于任何拖延产生的负分。仅改变 actor 的想象计分；环境原分、原生
成功率、WM 标签、replay、采集器以及无 RM/无 critic 设置保持不变。

## Why square / 为什么用平方

With detached hard success masks, merely replacing `r` by `r-1` changes the
forward score but not the computed pathwise gradient on a fixed rollout. Adding
a detached success bonus or constant absorbing tail has the same problem.

The square gives derivative `2*r*dr/dtheta`, changing the relative weight of
pre-success states and steps. It provides a differentiable monotone progress
signal while the offset makes unfinished transitions costly. Compared with the
old score, low-progress states receive less gradient weight relative to states
closer to success. This is an explicit shaping choice, not a claim of policy
invariance or an exact gradient of a discontinuous success-time objective.

The hard native success mask remains detached. The ordering guarantee concerns
forward scores, not optimizer convergence, real WM derivative accuracy, or task
success. A rule can pass these mathematical checks yet hurt policy learning.

只减常数可能只改分数、不改反传梯度。平方会改变不同状态的梯度权重，因此是一次
真实的计分 shaping。成功硬判定仍不求导；算分顺序正确不保证策略一定学得更好。

## Compatibility and experiment / 兼容与实验

`reward_rule` must explicitly be `native` or `completion_cost_v1` at launch;
unknown/missing values fail. The template defaults to `completion_cost_v1`.
Existing checkpoint tensors remain loadable for native evaluation. Historical
configs/records are not edited and exact resume remains unsupported.

Run one fresh PPO790 continuation against the completed native-score control
`refl-bins-ring-20261028-375a55c1`: BinSampler + original ring, same initial WM822,
dataset793, bank and verified mask, seed0, BF16, H13, batch256, gamma0.8,
200 actor/2000 WM updates and native512 evaluation/selection seeds. Only
`reward_rule` and required code/run identity fields differ. Record initial RNG
and environment fingerprints for the comparison. The historical control achieved
91.99219% selected success; this is not a target used to select hyperparameters.

从同一53.91%底座重新开始，与已完成的 BinSampler+ring 原计分实验对照。不从
高成功率权重续训，也不同时更换采样、数据或预算。

## Checks / 检查

Tests exercise immediate versus delayed/never completion at H3/H13, negative
score bounds and monotonicity, invalid-score rejection, and finite-difference
policy gradients. They explicitly demonstrate that a constant shift alone leaves
the gradient unchanged while the squared score changes its direction on a
multi-state batch. Existing native objective tests remain intact.

Native reward parity diagnostics continue checking the unmodified environment
oracle. Training objective curves now have a different scale and must not be
compared numerically across rules; native task success is the acceptance measure.

The completed trial reached92.38% selected but82.23% final versus91.99% for the
native control. The score-order property is verified; a robust learning benefit
is not. This rule remains experimental.

计分顺序验证通过；实际训练最佳92.38%、最终82.23%，未证明稳健收益。

The user subsequently stopped this shaping trial; the template now restores
native rewards with a critic. See [the critic decision](refl-critic-native.md).

# Restore native reward and add a value critic / 恢复原奖励并加回 critic

Status: historical variant, superseded for training. Its success-boundary
bootstrap differs from the original ManiSkill equation. See the
[alignment decision](refl-maniskill-alignment.md).
The user explicitly superseded the previous no-critic preference and stopped
completion-cost shaping. No reward model is introduced.

## Decision / 决策

Default to `reward_rule: native`, `use_critic: true`. Initialize the state42,
3x256 Tanh value network from the same PPO790 warmstart's `critic` tensors.
Preserve the actor/WM initialization, BinSampler, original ring, bank mask,
BF16 compute, optimizer settings and200 actor/2000 WM updates. Critic uses Adam
LR1e-4, five epochs with four minibatches after each actor update, norm clip1,
and lambda0.95:4000 critic updates in the default run. A private generator seeded
41000+seed controls critic permutations, and construction preserves global RNG.

默认恢复原生奖励、启用 critic，并从 PPO790 自带的价值网络初始化。只恢复未来价值
估计，不加 RM，也不继续使用扣分规则。Actor 和 WM 的预算保持不变，新增4000次
critic 更新；这不是同算力预算对照。

## Bootstrap contract / 补价值的位置

The actor maximizes native discounted imagined reward plus a frozen critic's
value at the first native-success cut, or at H13 if it survives. Each trajectory
gets exactly one boundary value, discounted by gamma raised to the number of
executed imagined actions. Success at the last imagined step must not get both
success and horizon bootstraps. Critic parameters are frozen during this actor
update while derivatives through V(state), WM and actor remain enabled.

This implements the successful-final-observation treatment discussed for PPO830.
It is deliberately not identical to old SuperMimic824, which used a learned RM
and only bootstrapped the surviving H13 horizon. This remains a pathwise WM
actor update, not a PPO clipped/log-prob update.

成功时补一次价值；13步结束还未成功，也补一次价值。不会两边重复加。更新 actor
时冻结 critic 权重，但保留其输入梯度。它仍是 WM 梯度直传，不是改成 PPO 更新。

## Critic supervision / Critic 怎么学

Use detached states/rewards from the actor's imagined rollout. Calculate fixed
lambda-return targets with current critic values before the critic fit:

- at a success cut: `r + gamma * V(next_state)`;
- elsewhere: `r + gamma * ((1-lambda)*V(next_state) + lambda*next_return)`;
- at the final horizon, next_return is V(last_state).

Only pre-terminal live states train the critic. Success cuts stop the lambda
trace so absorbing rows cannot leak into earlier targets, but do not zero the
final-state value. Tensor-split minibatches retain all valid samples. Targets
and inputs are detached, so critic fitting cannot update the actor or WM.

The critic is trained on imagined data, not new ground-truth value labels.
Successful states receive no invented fixed value of5; their value comes from
the warmstart and fitted function. This can still suffer model/value bias, and
adding a critic does not guarantee that earlier success will receive a better
estimated value or that task success will improve.

critic 用想象轨迹的多步回报目标训练；成功处切断后续串联，但保留成功状态的
价值估计。不人为指定“成功价值必须等于5”。模型和 critic 仍可能估错，要靠原生
任务评估判断是否真的改善。

## Compatibility / 兼容

Launch requires explicit boolean use_critic; critic mode requires native reward
and `critic_bootstrap: success_and_horizon`. A disabled critic constructs no
value network and logs zero critic updates/bootstrap.

New checkpoints use `rlinf-refl-pickcube-v2`, recording actor, WM, critic,
separate optimizer states, use_critic and a value probe. Payload/config/probe
contradictions fail before loading. Historicalv1 checkpoints remain explicitly
critic-free and load for evaluation; they are not silently reinterpreted as
critic-enabled. Exact resume is still unsupported because replay and simulator
states are not included. No historical artifacts are rewritten.

## Validation / 验证

Tests cover success-versus-horizon discounting, no double counting, gradient
flow through frozen critic inputs, detached lambda targets, success trace cuts,
and checkpoint contradictions. Existing no-critic/native and completion-cost
regression tests remain. Native512 initial/selected/final evaluation uses the
same seeds as the prior experiments. Checkpoint reload also checks the critic's
value probe at tolerance1e-6.

The experiment begins from PPO790 again, not from the degraded completion-cost
final policy. Its reference control is the native-score no-critic BinSampler+ring
run at375a55c1 (91.99% selected/final). We report actual critic optimizer counts,
parameter changes, finite value loss, native reward parity, cloud history and
independent task success separately.

The55158db9 run completed with82.42% selected/82.62% final success. These results
belong to the altered objective and must not be labeled original-ManiSkill parity.

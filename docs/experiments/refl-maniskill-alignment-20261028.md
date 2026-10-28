# ManiSkill alignment and failed native-reward critic run / 对齐与失败结果

The implementation correction is verified; learning is not fixed. Restoring the
pinned ManiSkill update rules with native reward and a critic reduced independent
success from276/512 (53.90625%) to64/512 (12.50%). The selected checkpoint remains
the untrained initialization. Do not present numerical parity as task success.

更新规则已按旧源码恢复，但训练没有修好。初始成功率53.91%，最终12.50%；
最佳仍是初始权重。此前自行添加成功补值、修改 critic 样本和目标的版本不是等价
迁移，不能拿它的82.42%结果来判断旧 critic 的有效性。

## Run and verification / 执行与验证

- Run: `refl-maniskill-20261028-2e718692`.
- Executed commit: `2e7186924e17ec8af29b6992b58475a5f4edbbd2`.
- Pinned original: `56814f65a9916d5fcbba46018858fce63cc99fa1`.
- Image: `docker.1ms.run/rlinf/rlinf:agentic-rlinf0.4-maniskill_libero`;
  A100-SXM4-80GB, Torch2.11.0+cu128, ManiSkill3.0.0b22, SAPIEN3.0.1.
- PPO790 actor/critic, WM822, dataset793 and original44317-frame source mask;
  source hashes and receipts are bound in the sealed run configuration.
- Native reward, no RM; BinSampler, original ring replay; H13, gamma0.8,
  lambda0.95, BF16, seed0, batch256.
- Completed200 actor,2000 WM and4000 critic updates, exit0. Independent reload
  evaluation uses512 native-reset episodes, seeds40000..40003,50 steps.
-31 CPU tests passed. Separate real-checkpoint GPU comparison used batch8,
  three actor/60 critic/six WM updates; parameters, gradients and Adam states
  exactly matched independently frozen original functions.
- Independent source review found collect, replay, observation/FK, reward proxy,
  model math and precision adapters equivalent. Remaining differences include
  image/package versions, old visual-build suppression and evaluation RNG
  isolation. Inspection alone does not measure their effect.

| Checkpoint | Successes | Independent success |
|---|---:|---:|
| Initial |276/512|53.90625%|
| Selected (iteration0) |276/512|53.90625%|
| Final (iteration20) |64/512|12.50000%|

Selection-population success at actor updates0/50/100/150/200 was
51.76/50.39/30.86/31.64/12.70%. Final grasp rate remained99.41%; grasping alone
therefore does not explain task completion. This does not identify whether
transport, final pose or velocity is the cause.

## Historical comparisons / 历史对照

Live A800 records were rechecked. The prior96.289% critic run824 also trained an
RM for2000 updates. The prior native-reward93.75% run872 had no critic. Thus
neither establishes that this native-reward-plus-critic combination previously
reached96%. RM changes the differentiated reward function as well as fitting
error; removing it is not merely removing an unused network.

旧96.29%是“有 RM、有 critic”；旧93.75%是“无 RM、无 critic”。本轮是“无 RM、
有 critic”，不能当成同一配置的复现，也不能据此认定必须加回 RM。当前保留用户
指定的无 RM，不再擅自改变奖励或价值公式。

## Records / 记录

[W&B run](https://wandb.ai/rleai/rlinf-refl-pickcube/runs/refl-maniskill-20261028-2e718692)
was independently checked: finished,20 training rows, all update counters and
independent evaluation history match local records.

TOS version `e93a5124a14aca0dd34f097abf55fc561c2ff24a19a770f41ff9b08b4a470323`:
`tos://qianzhangshucai-01/post-training/_assets/v1/versions/e93a5124a14aca0dd34f097abf55fc561c2ff24a19a770f41ff9b08b4a470323/manifest.json`.
All22 objects/145496912 bytes verified by remote size/CRC64;19 newly uploaded,
three reused. Full remote SHA reread not performed; catalog acceptance pending.
The package retains failed final weights, selected initialization, per-episode
outcomes, source patch, parity evidence and sealed run records.

## Gradient diagnosis / 梯度诊断

Diagnostic run `refl-gradient-diagnostic-20261028-2e9c83f8`, executed commit
`2e9c83f85b7e144d907f03203d48c28086d5ba56`, completed without model updates.
On the same256 sampled validation starts (with replacement), holding the final
actor+WM fixed and replacing only the value network:

| Critic | Value/reward gradient norm ratio | Gradient cosine |
|---|---:|---:|
| Initial PPO790 |0.1823|+0.5712|
| Final trained critic |0.5772|-0.4277|

The trained critic opposes the reward gradient on this fixed batch. However,
the combined gradient still has a positive dot product with the reward gradient
(2.8225); this is not proof that the actor update reverses the reward direction.
Gradients precede clipping and Adam. This probe cannot attribute the entire
success decline to the critic or separate actor changes from WM changes.

A seed30000 native512 evaluation left global Torch CPU/current-CUDA RNG states
unchanged. This narrows the RNG concern for this call; it does not cover Python,
NumPy, simulator internal state or environment construction. The old headless
wrapper only suppresses visual construction by inspection; its physical effect
has not been measured across images. No further reward or critic training rule
was changed in response to this failed run.

固定最终 actor和WM后，换回初始 critic，梯度与奖励方向一致；换成训后的 critic，
两者冲突。但合成方向并没有整体反转，且这里不是实际 Adam 更新，因此只能说
critic 是具体可疑点，不能声称已锁定全部失败原因。当前训练目标尚未达成。

Diagnostic records are stored separately under TOS version
`da9ebd12bbb5dd6edbdc24f77b8a7caf7a0accbdd0c5d4447cf9098d20d688cb`;
eight objects/272016 bytes checked by remote size/CRC64, catalog pending. They
reference the existing training checkpoints without uploading them again.
Independent review found no gradient decomposition error; a subsequent guard
rejects legacy/non-horizon checkpoints and missing initial/final labels.

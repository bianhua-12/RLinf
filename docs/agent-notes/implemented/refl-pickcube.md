# PickCube ReFL through a physical world model / PickCube 世界模型梯度直传

Status: implemented and validated in one native GPU run. See the
[execution evidence](../../experiments/refl-pickcube-20261028.md) for the clean
commit, independent success rates, W&B verification and TOS archive.

## Decision / 决策

Add a separate RLinf entrypoint, runner and colocated actor worker. Existing PPO,
model registries and environment workers keep their behavior. Policy and learned
dynamics stay in one process so Ray transport cannot detach the imagined graph.

The policy maximizes discounted native state rewards through a physical world
model. There is no learned reward network, critic, advantage estimator or value
bootstrap. During an actor update, WM parameters are frozen while derivatives
with respect to actions and states remain enabled. WM weights update separately
against real executed-action transitions.

新增独立入口、runner 和同进程 worker；已有 PPO 路径保持不变。Actor 通过 WM
预测状态上的原生任务奖励反传。WM 在 actor 更新时冻结参数但保留输入梯度，
在监督更新时使用实际执行动作和真实后继状态。不构建 RM、critic 或 bootstrap。

## Research contract / 研究契约

- Native `PickCube-v1`, Panda `pd_joint_delta_pos`, 8 actions, state observation42;
  WM additionally receives raw cube linear/angular velocity for physical48.
- 20 Hz control, 100 Hz simulation, 50 control-step episodes, normalized dense
  reward, unchanged 3x256 Tanh PPO790 actor and goal-free 3x2048 physical WM.
- Initial inputs: PPO790, WM822, dataset793 and its restorable reference bank.
  Exact SHA256 and TOS version identities are in
  [the configuration](../../../examples/embodiment/config/maniskill_refl_mlp.yaml).
  This starts from WM822 and collects new WoVR-style online data; it does not
  silently substitute the WM877 prefit checkpoint.
- Actor: H13 imagination, gamma 0.8, Adam LR 1e-4, betas 0.9/0.95, norm clip1
  and value clip0.1. Optimize negative discounted return divided by horizon.
  Terminal reward counts once, followed by absorbing states and detached masks.
- WM: 10 Adam steps per actor update, LR1e-4, norm clip1, exactly 3 executed
  actions per training window. Batch256 is half immutable dataset793 training
  windows and half rolling current-policy training windows.
- Default budget: 20 collection iterations, 10 actor updates each. The latest
  20 iterations form the online buffer. Actor starts include live short episodes;
  only WM samples require complete three-step windows.
- Neural MLP compute uses BF16; parameters, optimizer, state geometry/FK and
  reductions remain FP32. No compiler or architecture conversion is introduced.
- Source environment columns 0..3583 train, 3584..3839 validate, 3840..4095 test.
  Actual `source_env` columns are checked. Windows reject dead transitions,
  internal termination/truncation/reset and episode-ID changes. Training never
  uses held-out columns.
- Collection reuses the prior WoVR-style KIR implementation: equal phase requests
  for approach/ungrasped, grasped transport, near goal, near object/ungrasped;
  per-environment fallback to valid frames when a phase is unavailable. Every
  fifth collection uses native reset. KIR branches use32 steps, native branches50;
  current actor actions receive Gaussian noise std0.1 then clamp to[-1,1].
  This is the existing project's WoVR-style sampler, not a claim to reproduce
  every component of the WoVR paper.
- Restored bank frames are rechecked on the execution runtime against cached
  poses and executed-action next states. The original collector tolerances are
  used: pose1e-4, cube3mm, joint0.01rad, reward0.05, exact binary grasp agreement.
  A run-local valid-frame mask is saved; immutable source data are untouched.
  An environment with no reproducible frame is an error.
- Native reward code is called directly. Binary real-state rewards and done flags
  must match actual environment outputs. Soft grasp probabilities use the native
  linear grasp terms; CPU tests check interpolation and input gradients.
- Selection uses512 native-reset episodes at seeds30000..30003, success then
  undiscounted return, retaining the initial policy as a candidate. Independent
  initial/selected/final evaluation uses the same512 episodes at seeds40000..40003.
  KIR success rates and imagined rewards are never task-success evidence.
- Initial independent success must be45%-60% before updates. This guards the
  requested approximately50% starting point without claiming the historical
  53.91% result must be identical under a different image/runtime.
- Checkpoint format `rlinf-refl-pickcube-v1` stores actor, WM constructor/weights,
  optimizer state, RNG and an inference probe. Reload checks actor and WM outputs.
  Replay and simulator states are not stored; exact resume is explicitly unsupported.

固定 PPO790 起点、WM822 初值和 dataset793；线上采集采用现有 WoVR/KIR 四类状态
初始化，每五轮原生 reset。WM 用三步真实动作窗口，actor 用 H13 想象回报。选模与
独立评估使用不同种子，初始、selected、final 独立评估采用相同的512个原生回合。
起始成功率必须在45%-60%内；不同镜像下的历史53.91%需要重新实测。

## Running / 运行

Use the requested ManiSkill-compatible image and an allocated NVIDIA GPU. Keep
configuration, inputs, caches and outputs outside the checkout. Copy the YAML
above to an external config directory, set `commit` to the exact clean checkout
HEAD, bind input paths and their TOS receipt paths, output path, run ID and the observed container identity.
Use an existing W&B credential for online logging; explicitly select offline only
when authentication is unavailable and retain its files for later sync.

```bash
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$PWD"
python examples/embodiment/train_refl.py --config "$REFL_CONFIG"
```

The entrypoint verifies input hashes and execution source cleanliness before
launching the RLinf worker. `start.json`, `start.sha256`, `config.json`, package
snapshot, metrics, checkpoints and `completion.json` stay in the external run
directory. Worker startup independently checks the same source commit. A dirty,
untracked or ignored file blocks execution. Completion records distinguish
finished native evaluation from cleanup failures. Publication uses the existing
source -> TOS -> destination workflow separately from training.

配置中的路径、镜像和运行 ID 必须绑定实际执行环境。入口和 worker 启动都验证代码
提交；日志和权重放在源码目录外。TOS 发布独立于训练，不把文件存在或作业启动当成
训练完成，也不把完成训练当成成功率提高。

## Evidence and limitations / 证据与边界

The12 focused CPU tests passed in the requested image: finite-difference
policy gradients, frozen WM weights, terminal masking, native soft-grasp reward,
replay boundaries, executed-action alignment and dirty-source rejection.
Ruff checks passed for the new Python files. The repository-wide heuristic audit
reports two pre-existing relative-link errors in the upstream test-install skill;
that file is unchanged. Independent static review found missing source-column validation and incorrect
cleanup completion handling; both were addressed before GPU launch.

The recorded GPU run subsequently passed physics/input loading, reference
restoration, actor/WM checkpoint probes and initial/selected/final native evaluation.
Its selected success increased from53.91% to87.70%. This is single-seed evidence;
it does not establish convergence or a comparison against other algorithms.

## Source attribution / 来源

Checkpoint-compatible MLP/FK layers, state reward proxy and task collection logic
were adapted from SuperMimic commit `56814f65a9916d5fcbba46018858fce63cc99fa1`.
The WoVR-style KIR sampler comes from the existing project's data-refinement
branch. The MIT copyright notice is preserved in
[refl_pickcube.LICENSE](../../../rlinf/models/embodiment/refl_pickcube.LICENSE).

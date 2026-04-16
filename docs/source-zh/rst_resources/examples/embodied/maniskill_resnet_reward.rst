ManiSkill PPO（基于 ResNet Reward Model）
=========================================

本文档介绍如何在 ManiSkill PPO 中使用 ResNet reward model。
主要参考配置为 ``examples/embodiment/config/maniskill_ppo_mlp_resnet_reward.yaml``。

在开始前，建议先阅读：

1. :doc:`/rst_source/examples/embodied/maniskill` 以熟悉基础 ManiSkill PPO 训练流程。
2. :doc:`/rst_source/tutorials/extend/reward_model` 以了解离线 reward 数据预处理与 ResNet reward model 训练流程。

配置概览
--------

该配置保持策略网络使用状态观测，同时让 reward model 使用 RGB 观测：

.. code-block:: yaml

   env:
     train:
       wrap_obs_mode: simple
       use_full_state: True
       init_params:
         obs_mode: rgb

   actor:
     model:
       model_type: mlp_policy

   reward:
     use_reward_model: True
     reward_mode: terminal
     reward_weight: 1.0
     env_reward_weight: 0.0
     model:
       model_type: resnet
       arch: resnet18
       model_path: /path/to/reward_model_checkpoint

这一配置下：

- ManiSkill 运行在 ``rgb`` 模式，因此环境会提供 ``main_images``。
- ``wrap_obs_mode: simple`` 配合 ``use_full_state: True`` 会继续提供 ``states`` 供 MLP 策略使用。
- actor 仍然只基于 ``states`` 预测动作，而 reward worker 只消费 ``main_images``。

在线调用链
----------

运行时的 reward 路径如下：

.. code-block:: text

   examples/embodiment/train_embodied_agent.py
      -> EmbodiedRunner
      -> EnvWorker + MultiStepRolloutWorker + EmbodiedRewardWorker
      -> ResNetRewardModel

更具体地说：

1. ``train_embodied_agent.py`` 会在 ``reward.use_reward_model=True`` 时创建 ``EmbodiedRewardWorker``。
2. ``EnvWorker`` 将观测发给 rollout worker 生成动作，同时将 reward 输入发给 reward worker。
3. ``EmbodiedRewardWorker`` 通过 ``rlinf.models.embodiment.reward.get_reward_model_class`` 解析 ``reward.model.model_type="resnet"``，并实例化 ``ResNetRewardModel``。
4. ``ResNetRewardModel.compute_reward`` 读取 ``main_images``，完成图像预处理、ResNet 前向计算，并输出 sigmoid 概率。
5. ``EnvWorker.compute_bootstrap_rewards`` 会按如下方式将环境 reward 与 reward model 输出合并：

   .. code-block:: python

      reward = env_reward_weight * env_reward + reward_weight * reward_model_output

6. 当 ``reward_mode="terminal"`` 时，``EnvWorker`` 会通过 ``_scatter_terminal_reward_output`` 仅在 done 步写入 reward model 输出；当 ``reward_mode="per_step"`` 时，则每一步都会直接使用 reward model 输出。

关键配置项
----------

- ``cluster.component_placement.reward``：用于放置 reward worker 组；没有这一项就无法启动在线 reward 推理。
- ``reward.reward_mode``：``terminal`` 更适合稀疏成功任务，``per_step`` 则提供更密集的奖励，但更容易被策略利用。
- ``reward.reward_weight`` 与 ``reward.env_reward_weight``：控制 learned reward 与 env reward 的混合方式；示例中设置为 ``env_reward_weight: 0.0``。
- ``reward.model.model_path``：指向在线推理使用的 ResNet checkpoint。
- ``reward.model.pretrained``：控制训练 reward model 时，ResNet backbone 是否从 ImageNet 权重初始化。
- ``reward_threshold``：示例将这一字段放在顶层 ``reward`` 段，但当前 embodied reward worker 只会把 ``reward.model`` 传给 ``ResNetRewardModel``。也就是说，按当前在线路径，这个顶层阈值并不会被实际消费，除非将其移入 ``reward.model`` 或修改代码。

实践说明
--------

相比基础 ManiSkill PPO 配置，这个示例主要变化有三点：

1. 环境切换到 ``obs_mode: rgb``，为 reward 推理提供图像输入。
2. actor 仍然使用基于 ``states`` 的 ``mlp_policy``，因此动作生成开销较低。
3. 新增专门的 reward worker，并由 ``EnvWorker`` 统一完成最终 reward 计算。

如果还需要先训练 ResNet reward model，请先按 :doc:`/rst_source/tutorials/extend/reward_model` 完成训练，再将得到的 checkpoint 接入该 ManiSkill 配置。

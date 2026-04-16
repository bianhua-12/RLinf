ManiSkill PPO（基于 VLM Reward Model）
=====================================

本文档介绍如何在 ManiSkill PPO 中使用 VLM reward model。
主要参考配置为 ``examples/embodiment/config/maniskill_ppo_mlp_qwen3vl4b_robochallenge_reward.yaml``。

同一组配置中的相关示例还有：

- ``examples/embodiment/config/maniskill_ppo_mlp_qwen3vl4b_dualview_robochallenge_reward.yaml``
- ``examples/embodiment/config/maniskill_ppo_mlp_qwen3vl8b_confidance_robochallenge_reward.yaml``

在开始前，建议先阅读：

1. :doc:`/rst_source/examples/embodied/maniskill` 以熟悉基础 ManiSkill PPO 训练流程。
2. :doc:`/rst_source/examples/embodied/sft_vlm` 如果你需要准备或微调 reward worker 所使用的 Qwen3-VL checkpoint 或 LoRA。

配置概览
--------

主配置保持 PPO 策略网络基于状态学习，同时使用基于历史片段的 VLM reward worker 做单视角视频判断：

.. code-block:: yaml

   env:
     train:
       wrap_obs_mode: simple_prompt
       use_full_state: True
       init_params:
         obs_mode: rgb

   actor:
     model:
       model_type: mlp_policy

   reward:
     use_reward_model: True
     reward_mode: history_buffer
     history_reward_assign: True
     use_output_step: 0
     model:
       model_type: history_vlm
       model_path: /path/to/Qwen3-VL-4B-Instruct
       lora_path: /path/to/Qwen3-VL-4B-Instruct_lora
       input_builder_name: simple_robochallenge_input_builder
       reward_parser_name: robochallenge_reward_parser
       history_buffers:
         history_window:
           history_size: 5
           input_interval: 5
           history_keys: ["main_images"]

这一配置下：

- ``wrap_obs_mode: simple_prompt`` 会让 ManiSkill 同时输出 ``states``、``main_images``、``extra_view_images`` 和 ``task_descriptions``。
- actor 仍然只使用 ``states``。
- reward worker 则只使用任务文本和由 ``main_images`` 构成的短单视角历史片段。

在线调用链
----------

VLM reward 的主路径如下：

.. code-block:: text

   train_embodied_agent.py
      -> EmbodiedRunner
      -> EnvWorker(history_buffer)
      -> HistoryManager
      -> EmbodiedRewardWorker
      -> HistoryVLMRewardModel
      -> InputBuilder + Qwen3-VL generate() + RewardParser

具体流程为：

1. ``train_embodied_agent.py`` 会在 ``reward.use_reward_model=True`` 时创建 ``EmbodiedRewardWorker``。
2. ``EmbodiedRunner.run`` 会在 ``global_step >= reward.use_output_step`` 后激活 reward channel。
3. 在 ``EnvWorker.get_reward_model_output`` 中，``reward_mode="history_buffer"`` 会让当前观测先进入 ``HistoryManager``。
4. ``HistoryManager.build_history_input`` 会按照配置提取历史窗口，例如带有 ``history_size`` 和 ``input_interval`` 的 ``history_window``。
5. ``EmbodiedRewardWorker`` 解析 ``reward.model.model_type="history_vlm"``，并实例化 ``HistoryVLMRewardModel``。
6. ``HistoryVLMRewardModel.compute_reward`` 使用配置中的 ``input_builder_name`` 构造多模态输入，调用 ``AutoModelForVision2Seq.generate()``，再通过 ``reward_parser_name`` 将生成文本解析成标量 reward。
7. ``EnvWorker.compute_bootstrap_rewards`` 会先把 reward model 输出写到当前 step；如果 ``history_reward_assign=True``，``EnvWorker.assign_history_reward`` 还会把同一个 reward 回填到当前历史窗口覆盖到的更早几个 step。

主示例
------

``maniskill_ppo_mlp_qwen3vl4b_robochallenge_reward.yaml`` 是本文档的主示例。
它的主要特点是：

- ``input_builder_name: simple_robochallenge_input_builder`` 只使用 ``history_window``，并从 ``main_images`` 构造单视角视频输入。
- prompt 设计得比较直接，只判断这段视频片段是让任务变得更好还是更坏，并要求严格输出 ``positive`` 或 ``negative``。
- ``reward_parser_name: robochallenge_reward_parser`` 会把这一文本判断转换为标量 reward。
- 配置启用了 ``lora_path``，因此在线 reward worker 会先加载基础 Qwen3-VL checkpoint，再叠加 LoRA 权重。
- ``enable_debug_video_output`` 及相关字段可用于调试时导出 reward 侧的视频片段。
- ``infer_micro_batch_size`` 与 ``enable_history_offload`` 可用于控制 reward 推理阶段的显存压力。

其他变体
--------

另外两个配置主要在输入构造器和解析器上有区别：

- ``maniskill_ppo_mlp_qwen3vl4b_dualview_robochallenge_reward.yaml`` 切换到 ``dualview_robochallenge_input_builder``，用于双视角片段判断，并支持 ``unchanged`` 标签。
- ``maniskill_ppo_mlp_qwen3vl8b_confidance_robochallenge_reward.yaml`` 使用 ``confidence_robochallenge_reward_parser``，并同时配置 ``full_history`` 和 ``history_window`` 两个历史缓冲，更接近一种进度评分方案，而不是简单的二分类判别。

当前实现说明
------------

当前代码路径里有几个值得注意的实现细节：

- 这些 YAML 在顶层 ``reward`` 段配置了 ``reward_threshold``，但当前 ``history_vlm`` 实现并不会在 reward 推理阶段应用这个阈值。
- 在主单视角配置中，reward worker 只消费 ``main_images`` 的历史输入。虽然环境观测里仍然有 ``extra_view_images``，但 ``simple_robochallenge_input_builder`` 并不会使用它。
- ``robochallenge_reward_parser`` 会把最终 reward 截断到 ``[0, 1]``。实际效果上，``positive`` 会得到正分，而 ``negative`` 会被裁成 ``0``，而不是变成带符号惩罚。
- 对 dual-view 变体而言，``dualview_robochallenge_input_builder`` 会从 ``history_input`` 中同时读取 ``main_images`` 和 ``extra_view_images``。如果想真正使用双视角历史输入，需要在历史缓冲配置里同时记录这两个键。
- ``confidence_robochallenge_reward_parser`` 同样输出 ``[0, 1]`` 区间的值；对 ``negative`` 判断，它会返回 ``1 - confidence``，因此它更像是一个有界分数，而不是带符号的惩罚值。

实践说明
--------

与 ResNet reward 路径相比，VLM reward 路径更重，但表达能力也更强：

1. reward worker 消费的是任务文本和短视频历史，而不是单帧图像。
2. ``history_buffer`` 模式允许 reward 作用到一小段轨迹，而不只是当前帧。
3. 策略网络依旧可以保持为基于 ``states`` 的轻量 ``mlp_policy``，只有 reward 分支使用 Qwen3-VL。

ManiSkill PPO with VLM Reward Model
===================================

This document describes how to use a VLM reward model in ManiSkill PPO.
The main reference config is ``examples/embodiment/config/maniskill_ppo_mlp_qwen3vl4b_robochallenge_reward.yaml``.

Related configs in the same family are:

- ``examples/embodiment/config/maniskill_ppo_mlp_qwen3vl4b_dualview_robochallenge_reward.yaml``
- ``examples/embodiment/config/maniskill_ppo_mlp_qwen3vl8b_confidance_robochallenge_reward.yaml``

Before getting started, it is recommended to read:

1. :doc:`maniskill` to understand the base ManiSkill PPO pipeline.
2. :doc:`sft_vlm` if you need to prepare or fine-tune the Qwen3-VL checkpoint or LoRA used by the reward worker.

Configuration Overview
----------------------

The main config keeps PPO policy learning on state observations, and uses a history-based VLM reward worker for single-view video judgement:

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

With this setup:

- ``wrap_obs_mode: simple_prompt`` makes ManiSkill expose ``states``, ``main_images``, ``extra_view_images``, and ``task_descriptions`` together.
- The actor still uses ``states`` only.
- The reward worker uses task text plus a short single-view history built from ``main_images`` only.

Online Call Chain
-----------------

The VLM reward path is:

.. code-block:: text

   train_embodied_agent.py
      -> EmbodiedRunner
      -> EnvWorker(history_buffer)
      -> HistoryManager
      -> EmbodiedRewardWorker
      -> HistoryVLMRewardModel
      -> InputBuilder + Qwen3-VL generate() + RewardParser

The concrete flow is:

1. ``train_embodied_agent.py`` creates ``EmbodiedRewardWorker`` when ``reward.use_reward_model=True``.
2. ``EmbodiedRunner.run`` activates the reward channel once ``global_step >= reward.use_output_step``.
3. In ``EnvWorker.get_reward_model_output``, ``reward_mode="history_buffer"`` causes the current observations to be appended into ``HistoryManager``.
4. ``HistoryManager.build_history_input`` extracts the configured history windows, such as ``history_window`` with ``history_size`` and ``input_interval``.
5. ``EmbodiedRewardWorker`` resolves ``reward.model.model_type="history_vlm"`` and instantiates ``HistoryVLMRewardModel``.
6. ``HistoryVLMRewardModel.compute_reward`` builds multimodal inputs with the configured ``input_builder_name``, runs ``AutoModelForVision2Seq.generate()``, then parses generated text with ``reward_parser_name``.
7. ``EnvWorker.compute_bootstrap_rewards`` writes the reward-model output to the current step. If ``history_reward_assign=True``, ``EnvWorker.assign_history_reward`` also back-fills the same reward to earlier steps covered by the current history window.

Main Example
------------

``maniskill_ppo_mlp_qwen3vl4b_robochallenge_reward.yaml`` is the primary example in this document.
Its main characteristics are:

- ``input_builder_name: simple_robochallenge_input_builder`` uses only ``history_window`` and one video view from ``main_images``.
- The prompt is intentionally simple: it asks whether the video segment makes the task better or worse, and expects exactly ``positive`` or ``negative``.
- ``reward_parser_name: robochallenge_reward_parser`` converts that text judgement into a scalar reward.
- ``lora_path`` is enabled, so the online reward worker loads a base Qwen3-VL checkpoint and then applies LoRA weights.
- ``enable_debug_video_output`` and related debug fields are available for saving reward-side video snippets during debugging.
- ``infer_micro_batch_size`` and ``enable_history_offload`` are available to control GPU memory pressure during reward inference.

Other Variants
--------------

The other two configs mainly change the input builder and parser:

- ``maniskill_ppo_mlp_qwen3vl4b_dualview_robochallenge_reward.yaml`` switches to ``dualview_robochallenge_input_builder``. It is designed for two-view clips and also supports the ``unchanged`` label.
- ``maniskill_ppo_mlp_qwen3vl8b_confidance_robochallenge_reward.yaml`` uses ``confidence_robochallenge_reward_parser`` and two history buffers, ``full_history`` plus ``history_window``. It is closer to a progress-scoring setup than a plain binary classifier.

Current Implementation Notes
----------------------------

The current code path has a few details worth knowing:

- ``reward_threshold`` is configured at the top-level ``reward`` section in these YAML files, but the ``history_vlm`` implementation does not currently apply that threshold during reward inference.
- In the main single-view config, the reward worker only consumes ``main_images`` history. ``extra_view_images`` are still present in env observations, but they are not used by ``simple_robochallenge_input_builder``.
- ``robochallenge_reward_parser`` clamps final rewards into ``[0, 1]``. In practice, this means ``positive`` maps to a positive score while ``negative`` is clipped to ``0`` rather than becoming a signed penalty.
- For the dual-view variant, ``dualview_robochallenge_input_builder`` reads both ``main_images`` and ``extra_view_images`` from ``history_input``. To get true two-view history inputs, both keys need to be recorded in the configured history buffer.
- ``confidence_robochallenge_reward_parser`` also outputs values in ``[0, 1]``. For ``negative`` judgements it returns ``1 - confidence``, so it behaves as a bounded score rather than a signed penalty.

Practical Notes
---------------

Compared with the ResNet reward path, the VLM reward path is heavier but more expressive:

1. The reward worker consumes task text and short video history instead of a single image.
2. ``history_buffer`` mode lets reward be assigned to a short trajectory segment, not just the current frame.
3. The policy can still remain an inexpensive ``mlp_policy`` over ``states``, while only the reward branch uses Qwen3-VL.

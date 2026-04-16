ManiSkill PPO with ResNet Reward Model
======================================

This document describes how to use a ResNet reward model in ManiSkill PPO.
The main reference config is ``examples/embodiment/config/maniskill_ppo_mlp_resnet_reward.yaml``.

Before getting started, it is recommended to read:

1. :doc:`/rst_source/examples/embodied/maniskill` to understand the base ManiSkill PPO pipeline.
2. :doc:`/rst_source/tutorials/extend/reward_model` to understand offline reward-data preprocessing and ResNet reward-model training.

Configuration Overview
----------------------

This config keeps the policy on state observations while moving the reward model to RGB observations:

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

With this setup:

- ManiSkill runs in ``rgb`` mode, so the env exposes ``main_images``.
- ``wrap_obs_mode: simple`` and ``use_full_state: True`` also keep ``states`` for the MLP policy.
- The actor still predicts actions from ``states`` only, while the reward worker consumes ``main_images`` only.

Online Call Chain
-----------------

At runtime, the reward path is:

.. code-block:: text

   examples/embodiment/train_embodied_agent.py
      -> EmbodiedRunner
      -> EnvWorker + MultiStepRolloutWorker + EmbodiedRewardWorker
      -> ResNetRewardModel

More concretely:

1. ``train_embodied_agent.py`` creates ``EmbodiedRewardWorker`` when ``reward.use_reward_model=True``.
2. ``EnvWorker`` sends observations to the rollout worker for action generation, and sends reward inputs to the reward worker.
3. ``EmbodiedRewardWorker`` resolves ``reward.model.model_type="resnet"`` through ``rlinf.models.embodiment.reward.get_reward_model_class`` and instantiates ``ResNetRewardModel``.
4. ``ResNetRewardModel.compute_reward`` reads ``main_images``, applies image preprocessing, runs the ResNet head, and returns sigmoid probabilities.
5. ``EnvWorker.compute_bootstrap_rewards`` combines env reward and reward-model output as:

   .. code-block:: python

      reward = env_reward_weight * env_reward + reward_weight * reward_model_output

6. When ``reward_mode="terminal"``, ``EnvWorker`` uses ``_scatter_terminal_reward_output`` so only done steps receive reward-model output. When ``reward_mode="per_step"``, every step receives reward-model output directly.

Key Config Fields
-----------------

- ``cluster.component_placement.reward``: places the reward worker group. Without it, online reward inference cannot be launched.
- ``reward.reward_mode``: ``terminal`` is usually safer for sparse-success tasks; ``per_step`` gives denser reward but is easier to exploit.
- ``reward.reward_weight`` and ``reward.env_reward_weight``: control whether training uses only learned reward or a mixture with env reward. The example sets ``env_reward_weight: 0.0``.
- ``reward.model.model_path``: points to the trained ResNet checkpoint used for online inference.
- ``reward.model.pretrained``: controls whether the ResNet backbone starts from ImageNet weights when training the reward model.
- ``reward_threshold``: the example keeps this field at the top-level ``reward`` section, but the current embodied reward worker only passes ``reward.model`` into ``ResNetRewardModel``. In other words, this top-level threshold is not consumed by the current online path unless it is moved into ``reward.model`` or the code is updated.

Practical Notes
---------------

Compared with the base ManiSkill PPO config, this example mainly changes three things:

1. The env is switched to ``obs_mode: rgb`` so reward inference has image input.
2. The actor remains an ``mlp_policy`` over ``states``, so action generation cost stays low.
3. A dedicated reward worker is added, and final rewards are computed centrally in ``EnvWorker``.

If you still need to train the ResNet reward model itself, follow :doc:`/rst_source/tutorials/extend/reward_model` first, then plug the resulting checkpoint into this ManiSkill config.

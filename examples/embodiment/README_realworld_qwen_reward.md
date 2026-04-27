# Real-World OpenPI Actor with Qwen3-VL Reward

This setup uses two separate Python environments. Do not install Qwen3-VL reward dependencies into the OpenPI/Pi0.5 actor environment.

## Environments

Use an OpenPI/Pi0.5 environment for actor, rollout, and real-world env workers. Create it with the normal embodied installer, for example:

```bash
bash requirements/install.sh embodied --model openpi --env realworld
```

Keep the OpenPI dependency set as installed by the script. In particular, do not upgrade this environment to the Qwen3-VL `transformers` version.

Use a separate OpenVLA/Qwen reward environment for the reward worker. It must be able to import Qwen3-VL video utilities and a recent enough Transformers release:

```bash
uv pip install qwen-vl-utils "transformers>=4.57.1,<=4.57.6"
python -c "import transformers, qwen_vl_utils; print(transformers.__version__)"
```

The OpenPI environment can be checked separately:

```bash
python -c "import transformers; print(transformers.__version__)"
```

## Placement

Point actor, rollout, and env node groups at the OpenPI interpreter. Point the reward node group at the Qwen interpreter.

```yaml
cluster:
  num_nodes: 2
  component_placement:
    actor:
      node_group: openpi
      placement: 0
    rollout:
      node_group: openpi
      placement: 0
    env:
      node_group: franka
      placement: 0
    reward:
      node_group: qwen
      placement: 0
  node_groups:
    - label: openpi
      node_ranks: [0]
      env_configs:
        python_interpreter_path: /path/to/openpi-env/bin/python
    - label: qwen
      node_ranks: [0]
      env_configs:
        python_interpreter_path: /path/to/qwen-reward-env/bin/python
    - label: franka
      node_ranks: [1]
      env_configs:
        python_interpreter_path: /path/to/openpi-env/bin/python
      hardware:
        type: Franka
        configs:
          - robot_ip: ROBOT_IP
            node_rank: 1
```

For real-world standalone reward, keep `reward.standalone_realworld: true`; the env worker will launch a one-rank reward worker at the configured reward placement and pass `reward_worker_cfg` into the Franka env.

## Qwen Reward Config

The target Qwen reward path is `history_vlm` with `simple_dualview_ternary_input_builder`. It expects a 5-frame history window with both the main camera and one extra camera view:

```yaml
reward:
  use_reward_model: true
  standalone_realworld: true
  reward_mode: per_step
  model:
    model_type: history_vlm
    model_path: ${oc.env:QWEN_VL_MODEL_PATH}
    lora_path: ${oc.env:QWEN_VL_LORA_PATH,""}
    precision: bf16
    input_builder_name: simple_dualview_ternary_input_builder
    input_builder_params:
      strict_min_frames: 5
    history_buffers:
      history_window:
        history_size: 5
        input_interval: 1
        history_keys: ["main_images", "extra_view_images"]
        input_on_done: false
    infer_micro_batch_size: 1
    subprocessor_kwargs:
      video_processor:
        do_sample_frames: true
```

Set `env.train.main_image_key` to the primary Franka camera. The Franka reward path uses `reward_image_key` as the main view and picks the first other camera as `extra_view_images`; set `reward_extra_view_key` in the env override if you need a specific second view.

## Ray Startup

On every machine, set `RLINF_NODE_RANK` before `ray start`; Ray captures the environment at startup.

```bash
export RLINF_NODE_RANK=0
ray start --head --port=6379 --node-ip-address=<head_ip>
```

Worker nodes:

```bash
export RLINF_NODE_RANK=1
ray start --address=<head_ip>:6379
```

The reward node's Python environment must import `qwen_vl_utils` and the newer `transformers`. Actor and rollout nodes should stay on the OpenPI/Pi0.5 environment and should not mix in Qwen3-VL dependencies.

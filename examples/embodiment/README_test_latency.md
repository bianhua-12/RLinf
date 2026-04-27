# OpenPI + Qwen Reward Latency 测试交接

这份文档说明如何用两个 Python 环境运行 Pi0.5/OpenPI actor、rollout、env 与 Qwen3-VL reward worker，并汇总 async PPO profile/latency 结果。

## 为什么必须拆环境

OpenPI/Pi0.5 环境只运行 actor、rollout、env 和主入口脚本。这个环境需要保留 `requirements/install.sh` 为 OpenPI 安装的原始依赖，以及 OpenPI 自己的 `transformers_replace` 替换逻辑。

OpenVLA/Qwen 环境只运行 Qwen3-VL reward worker。Qwen3-VL reward 需要 `qwen-vl-utils` 和 `transformers>=4.57.1`。

不要在 OpenPI/Pi0.5 环境里安装或升级 Qwen3-VL 所需的 `transformers>=4.57.1`。这样会覆盖 OpenPI 依赖的 Transformers 行为，导致 actor 或 rollout 侧加载 Pi0.5 失败。

## 环境准备


保留这个环境的 OpenPI 原始依赖，不要额外安装 Qwen3-VL 依赖。可以用下面命令做 sanity check：

```bash
/path/to/openpi/bin/python -c "import transformers, torch, pytest; print(transformers.__version__); print(torch.__version__)"
```

OpenVLA/Qwen 环境用于 reward worker。先准备 OpenVLA 基础环境，再安装 Qwen3-VL reward 需要的包：

```bash
source /path/to/openvla/bin/activate
bash requirements/install.sh embodied --model openvla --env frankasim
uv pip install qwen-vl-utils "transformers>=4.57.1,<=4.57.6"
```

OpenVLA/Qwen 环境的 sanity check：

```bash
/path/to/openvla/bin/python -c "import transformers, qwen_vl_utils, torch, pytest; print(transformers.__version__); print(torch.__version__)"
```

如果只在真实 Franka 机器上单独放置 reward worker，也可以把 OpenVLA/Qwen 环境建在 reward 节点上；关键是该 interpreter 必须能 import `qwen_vl_utils`，且 Transformers 版本满足 Qwen3-VL。

## 两环境统筹配置

YAML 里的 `cluster.node_groups[*].env_configs[*].python_interpreter_path` 必须写绝对路径。不要依赖当前 shell 的 `python`，因为 Ray worker 会按 node group 配置选择解释器。

`actor`、`env`、`rollout` 指向 OpenPI node group；`reward` 指向 OpenVLA/Qwen node group：

```yaml
cluster:
  num_nodes: 1
  component_placement:
    actor,env,rollout:
      node_group: openpi
      placement: 0
    reward:
      node_group: qwen
      placement: 0
  node_groups:
    - label: openpi
      node_ranks: [0]
      env_configs:
        - node_ranks: [0]
          python_interpreter_path: /path/to/openpi/bin/python
    - label: qwen
      node_ranks: [0]
      env_configs:
        - node_ranks: [0]
          python_interpreter_path: /path/to/openvla/bin/python
```

单机 latency/profile smoke 可以从 `examples/embodiment/config/frankasim_async_ppo_pi05_profile.yaml` 开始改。这个配置已经开启 `profile.enabled: true`，默认 warmup 2 steps、measure 10 steps，并使用 `simple_dualview_ternary_input_builder` 的 Qwen3-VL reward。

真实 Franka + standalone Qwen reward 的主说明仍参考 [README_realworld_qwen_reward.md](README_realworld_qwen_reward.md)。本文件只补充环境拆分、Ray placement 和 latency/profile 汇总流程。

## 启动 Ray

启动 Ray 前必须设置 `RLINF_NODE_RANK`。Ray 会在启动时捕获环境变量，后面再 export 不会影响已经启动的 Ray 进程。

```bash
export RLINF_NODE_RANK=0
ray start --head --port=6379
```


## 运行 latency/profile

先设置模型路径。OpenPI 模型路径供 actor/rollout 使用，Qwen 路径供 reward worker 使用：

```bash
export PI05_MODEL_PATH=/path/to/pi05
export QWEN_VL_MODEL_PATH=/path/to/Qwen3-VL-4B-Instruct
export QWEN_VL_LORA_PATH=/path/to/reward_lora_or_checkpoint
```

用 OpenPI/Pi0.5 环境作为主入口运行。`RLINF_PYTHON` 控制 `run_embodiment.sh` 使用哪个 Python 启动主进程：

```bash
RLINF_PYTHON=/path/to/openpi/bin/python \
bash examples/embodiment/run_embodiment.sh frankasim_async_ppo_pi05_profile
```

运行时重点检查 worker log 中的 interpreter 信息。actor、rollout、env 应该显示 `/path/to/openpi/bin/python`，reward 应该显示 `/path/to/openvla/bin/python`。

profile 配置会把日志写到 `logs/<run_dir>`，并在 TensorBoard scalar 中记录 `time/`、`reward/`、`rollout/`、`train/` 前缀的 latency 指标。用 OpenPI 环境汇总：

```bash
/path/to/openpi/bin/python \
  examples/embodiment/scripts/summarize_async_ppo_profile.py \
  logs/<run_dir> --warmup-steps 2 --measure-steps 10
```

脚本默认输出：

```text
logs/<run_dir>/async_ppo_profile_summary.json
logs/<run_dir>/async_ppo_profile_summary.csv
```

如果没有找到 TensorBoard events，先确认运行命令是否完整启动到了训练 loop，以及 `runner.logger.log_path` 是否就是传给汇总脚本的目录。



## 排错 checklist

- `No module named qwen_vl_utils`：reward worker 没有使用 OpenVLA/Qwen interpreter，或该环境没有安装 `qwen-vl-utils`。
- Qwen3-VL import 或模型加载失败：检查 OpenVLA/Qwen 环境的 `transformers` 是否为 `>=4.57.1,<=4.57.6`。
- OpenPI actor 崩溃：确认没有在 OpenPI/Pi0.5 环境安装 Qwen3-VL 需要的新版 Transformers。
- Ray worker Python 不对：查看 worker log 中的 `python interpreter ...`，再检查 `cluster.node_groups[*].env_configs[*].python_interpreter_path` 是否为绝对路径。
- 多机 worker 不按预期放置：确认每台机器都在 `ray start` 前设置了正确的 `RLINF_NODE_RANK`，并且和 YAML 的 `node_ranks` 一致。
- 汇总脚本找不到 event file：确认传入的是 `logs/<run_dir>` 或包含 `tensorboard/` 子目录的 run log 目录。

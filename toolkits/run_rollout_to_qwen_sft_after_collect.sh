#!/usr/bin/env bash
set -euo pipefail

REPO_PATH="/mnt/project_rlinf/ztx/RLinf"
RUN_DIR="${REPO_PATH}/logs/20260417-21:34:32-maniskill_ppo_mlp_full_rollout_collect_200"
ROLLOUT_DIR="${RUN_DIR}/full_rollout_batches"
DATASET_DIR="${REPO_PATH}/logs/qwen3_vl_4b_5frame_dualview_rollout_gae_thr0p1_balanced"
CONFIG_NAME="qwen3_vl_4b_5frame_dualview_rollout_gae_thr0p1_sft_10000"
PIPELINE_LOG="${REPO_PATH}/logs/rollout_to_qwen_sft_$(date +'%Y%m%d-%H%M%S').log"
TARGET_STEP=299

cd "${REPO_PATH}"

log() {
  echo "[$(date +'%Y-%m-%d %H:%M:%S')] $*" | tee -a "${PIPELINE_LOG}"
}

count_step_files() {
  find "${ROLLOUT_DIR}" -maxdepth 1 -type f \
    -name "$(printf 'global_step_%06d_rank_*.pkl' "${TARGET_STEP}")" | wc -l
}

log "pipeline started"
log "run_dir=${RUN_DIR}"
log "rollout_dir=${ROLLOUT_DIR}"
log "dataset_dir=${DATASET_DIR}"
log "config=${CONFIG_NAME}"

while true; do
  count="$(count_step_files)"
  latest="$(find "${ROLLOUT_DIR}" -maxdepth 1 -type f -name 'global_step_*_rank_*.pkl' | sort | tail -1 || true)"
  log "waiting for target step ${TARGET_STEP}: count=${count}/8 latest=${latest}"
  if [[ "${count}" -ge 8 ]]; then
    break
  fi
  if ! pgrep -f "train_embodied_agent.py.*20260417-21:34:32-maniskill_ppo_mlp_full_rollout_collect_200" >/dev/null; then
    log "collector process is not running; continuing only if target files exist"
    if [[ "$(count_step_files)" -lt 8 ]]; then
      log "target step files are incomplete; aborting"
      exit 1
    fi
    break
  fi
  sleep 300
done

log "target rollout files are ready; exporting Qwen dataset"
rm -rf "${DATASET_DIR}"
"${REPO_PATH}/.venv/bin/python" toolkits/export_rollout_pkl_qwen_dataset.py \
  --rollout-dir "${ROLLOUT_DIR}" \
  --output-dir "${DATASET_DIR}" \
  --window-size 5 \
  --stride 1 \
  --score-source returns \
  --delta-threshold 0.1 \
  --balance downsample \
  --max-per-label 30000 \
  --max-eval-per-label 3000 \
  --early-stop-when-balanced \
  --reverse-positive-as-negative \
  --min-global-step 0 \
  --max-global-step "${TARGET_STEP}" \
  2>&1 | tee -a "${PIPELINE_LOG}"

log "dataset export finished"
log "dataset_info=${DATASET_DIR}/dataset_info.json"
cat "${DATASET_DIR}/dataset_info.json" | tee -a "${PIPELINE_LOG}"

log "starting Qwen SFT"
SFT_LOG_DIR="${REPO_PATH}/logs/$(date +'%Y%m%d-%H:%M:%S')-${CONFIG_NAME}"
mkdir -p "${SFT_LOG_DIR}"
SFT_LOG="${SFT_LOG_DIR}/run_vlm_sft.log"
CMD=(
  "${REPO_PATH}/.venv/bin/python"
  "${REPO_PATH}/examples/sft/train_vlm_sft.py"
  --config-path "${REPO_PATH}/examples/sft/config/"
  --config-name "${CONFIG_NAME}"
  "runner.logger.log_path=${SFT_LOG_DIR}"
)
printf '%q ' "${CMD[@]}" | tee "${SFT_LOG}"
echo | tee -a "${SFT_LOG}"
"${CMD[@]}" 2>&1 | tee -a "${SFT_LOG}" &
SFT_PID=$!
log "sft_pid=${SFT_PID} sft_log=${SFT_LOG}"

CHECKPOINT_ROOT="${SFT_LOG_DIR}/${CONFIG_NAME}/checkpoints"
while kill -0 "${SFT_PID}" 2>/dev/null; do
  first_ckpt="$(find "${CHECKPOINT_ROOT}" -maxdepth 1 -type d -name 'global_step_*' 2>/dev/null | sort | head -1 || true)"
  if [[ -n "${first_ckpt}" ]]; then
    log "first checkpoint detected: ${first_ckpt}"
    wait "${SFT_PID}"
    exit $?
  fi
  log "waiting for first checkpoint under ${CHECKPOINT_ROOT}"
  sleep 120
done

wait "${SFT_PID}"

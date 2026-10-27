#!/usr/bin/env bash
set -euo pipefail

# One-command Sokoban experiment:
# random offline data -> World Model -> Reward Head -> Critic -> PPO.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${DWELL_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
ENGINE_ROOT="${MBRL_ENGINE_ROOT:-${ROOT}}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT}/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi
MODEL_PATH="${MODEL_PATH:-${ROOT}/../../models/Qwen2.5-VL-3B-Instruct}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-1}"
PHASE="${PHASE:-all}"
WORLD_MODEL_MODE="${WORLD_MODEL_MODE:-frozen_wm}"

DATA_EPISODES="${DATA_EPISODES:-50000}"
DATA_SEED_START="${DATA_SEED_START:-1}"
RAW_DATA_DIR="${RAW_DATA_DIR:-${ROOT}/data/sokoban/random_rgb_seed${DATA_SEED_START}_${DATA_EPISODES}}"
PAIRED_ROOT="${PAIRED_ROOT:-${ROOT}/data/sokoban/random_qwen_vjepa_teacher_3B_seed${DATA_SEED_START}_${DATA_EPISODES}}"
DATA_DIR="${DATA_DIR:-${PAIRED_ROOT}/tokenized}"
DATA_REPORT="${DATA_REPORT:-${PAIRED_ROOT}/dataset_contract_report.json}"

EXP_ROOT="${EXP_ROOT:-${ROOT}/checkpoints/sokoban/random_qwen_vjepa_seed${SEED}}"
BASE_WM_DIR="${BASE_WM_DIR:-${EXP_ROOT}/stage1_wm}"
PRIOR_WM_DIR="${PRIOR_WM_DIR:-${EXP_ROOT}/stage1_prior_repair_lora}"
FINAL_WM_DIR="${FINAL_WM_DIR:-${PRIOR_WM_DIR}}"
STAGE1_GATE_REPORT="${STAGE1_GATE_REPORT:-${FINAL_WM_DIR}/stage1_semantic_gate_report.json}"
REWARD_DIR="${REWARD_DIR:-${EXP_ROOT}/stage2_reward_head}"
CRITIC_DIR="${CRITIC_DIR:-${EXP_ROOT}/stage3_critic_warmup}"
PPO_DIR="${PPO_DIR:-${EXP_ROOT}/stage4_ppo_${WORLD_MODEL_MODE}}"
H2_CACHE="${H2_CACHE:-${ROOT}/data/sokoban/critic_h2_cache/random_qwen_vjepa_seed${SEED}/cache.pt}"
REWARD_FEATURE_CACHE="${REWARD_FEATURE_CACHE:-${ROOT}/data/sokoban/reward_head_feature_cache/random_qwen_vjepa_seed${SEED}}"
EVAL_LEVELS_FILE="${EVAL_LEVELS_FILE:-${ROOT}/data/vagen_mirror_testset/sokoban_256.json}"

case "${PHASE}" in
  data|base_wm|prior|wm|reward|critic|ppo|rl|all) ;;
  *)
    echo "PHASE must be data, base_wm, prior, wm, reward, critic, ppo, rl, or all; got ${PHASE}" >&2
    exit 2
    ;;
esac
case "${WORLD_MODEL_MODE}" in
  frozen_wm|alternating_wm) ;;
  *) echo "WORLD_MODEL_MODE must be frozen_wm or alternating_wm" >&2; exit 2 ;;
esac

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing Python: ${PYTHON_BIN}" >&2; exit 2; }
[[ -f "${MODEL_PATH}/config.json" ]] || { echo "Missing Qwen model: ${MODEL_PATH}" >&2; exit 2; }
[[ -f "${EVAL_LEVELS_FILE}" ]] || { echo "Missing fixed Sokoban evaluation levels: ${EVAL_LEVELS_FILE}" >&2; exit 2; }

export DWELL_ROOT="${ROOT}"
export MBRL_ENGINE_ROOT="${ENGINE_ROOT}"
export PYTHON_BIN MODEL_PATH GPU_ID SEED
export PYTHONPATH="${ROOT}:${ENGINE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${GPU_ID}"
export PYTHONUNBUFFERED=1

require_file() {
  [[ -f "$1" ]] || { echo "Missing required artifact: $1" >&2; exit 2; }
}

validate_data() {
  "${PYTHON_BIN}" -u "${ROOT}/scripts/train/validate_qwen_vjepa_dataset.py" \
    --data-dir "${DATA_DIR}" \
    --env-id sokoban \
    --expected-episodes "${DATA_EPISODES}" \
    --expected-seed-start "${DATA_SEED_START}" \
    --check-samples "${DATA_CHECK_SAMPLES:-16}" \
    --output "${DATA_REPORT}" \
    --enforce
}

reward_floor() {
  if [[ -n "${REWARD_CONFIDENCE_FLOOR:-}" ]]; then
    printf '%s\n' "${REWARD_CONFIDENCE_FLOOR}"
    return
  fi
  "${PYTHON_BIN}" -c \
    'import json,sys; print(float(json.load(open(sys.argv[1]))["decision_threshold"]))' \
    "${REWARD_DIR}/reward_head_gate_report.json"
}

run_data() {
  local paired_count=0
  if [[ -f "${DATA_DIR}/manifest.jsonl" ]]; then
    paired_count="$(wc -l < "${DATA_DIR}/manifest.jsonl")"
    if [[ "${paired_count}" == "${DATA_EPISODES}" ]]; then
      echo "Paired replay already contains ${paired_count} episodes; validating it."
      validate_data
      return
    fi
    echo "Paired replay is partial (${paired_count}/${DATA_EPISODES}); resuming extraction."
  fi

  echo "=== Collect/resume real Sokoban random-policy RGB replay ==="
  "${PYTHON_BIN}" -u "${ROOT}/scripts/train/collect_offline_dataset.py" \
    --env-id sokoban \
    --require-real-env \
    --output-dir "${RAW_DATA_DIR}" \
    --num-episodes "${DATA_EPISODES}" \
    --seed-start "${DATA_SEED_START}" \
    --max-steps 25 \
    --split train \
    --strategy random:1

  echo "=== Extract native Qwen observations and frozen V-JEPA teachers ==="
  "${PYTHON_BIN}" -u "${ROOT}/scripts/train/build_qwen_vjepa_teacher_dataset.py" \
    --raw-data-dir "${RAW_DATA_DIR}" \
    --output-dir "${PAIRED_ROOT}" \
    --backbone-model "${MODEL_PATH}" \
    --hidden-dim 2048 \
    --belief-slots 36 \
    --device cuda:0 \
    --frame-batch-size "${TOKENIZE_BATCH_SIZE:-8}"
  validate_data
}

run_base_wm() {
  validate_data
  PHASE=train \
  RAW_DATA_DIR="${RAW_DATA_DIR}" PAIRED_ROOT="${PAIRED_ROOT}" DATA_DIR="${DATA_DIR}" \
  WM_DIR="${BASE_WM_DIR}" WM_STEPS="${WM_STEPS:-2000}" \
  GPU_ID="${GPU_ID}" SEED="${SEED}" NO_WANDB="${NO_WANDB:-0}" \
  WANDB_PROJECT="${WANDB_PROJECT:-jiayu-mbrl}" \
  WANDB_GROUP="${WANDB_GROUP_WM:-sokoban_qwen_vjepa_wm}" \
  WANDB_RUN_NAME="sokoban_qwen_vjepa_wm_seed${SEED}" \
    bash "${ROOT}/scripts/train/run_qwen_vjepa_teacher_wm.sh"
}

run_prior() {
  validate_data
  require_file "${BASE_WM_DIR}/best.pt"
  PHASE=all SOURCE_WM="${BASE_WM_DIR}/best.pt" DATA_DIR="${DATA_DIR}" \
  OUTPUT_DIR="${PRIOR_WM_DIR}" REPAIR_STEPS="${PRIOR_REPAIR_STEPS:-3000}" \
  GPU_ID="${GPU_ID}" SEED="${SEED}" NO_WANDB="${NO_WANDB:-0}" \
  WANDB_PROJECT="${WANDB_PROJECT:-jiayu-mbrl}" \
  WANDB_GROUP="${WANDB_GROUP_PRIOR:-sokoban_qwen_vjepa_prior}" \
  WANDB_RUN_NAME="sokoban_qwen_vjepa_prior_seed${SEED}" \
    bash "${ROOT}/scripts/train/run_qwen_vjepa_prior_repair.sh"
}

run_reward() {
  validate_data
  require_file "${FINAL_WM_DIR}/best.pt"
  require_file "${STAGE1_GATE_REPORT}"
  ENV_ID=sokoban SOURCE_DIR="${FINAL_WM_DIR}" SOURCE_WM="${FINAL_WM_DIR}/best.pt" \
  STAGE1_GATE_REPORT="${STAGE1_GATE_REPORT}" DATA_DIR="${DATA_DIR}" \
  OUTPUT_DIR="${REWARD_DIR}" FEATURE_CACHE_DIR="${REWARD_FEATURE_CACHE}" \
  GPU_ID="${GPU_ID}" SEED="${SEED}" NO_WANDB="${NO_WANDB:-0}" \
  WANDB_PROJECT="${WANDB_PROJECT:-jiayu-mbrl}" \
  WANDB_GROUP="${WANDB_GROUP_REWARD:-sokoban_qwen_vjepa_reward}" \
  WANDB_RUN_NAME="sokoban_qwen_vjepa_reward_seed${SEED}" \
    bash "${ROOT}/scripts/train/run_qwen_vjepa_reward_head.sh"
}

run_critic() {
  validate_data
  require_file "${REWARD_DIR}/best.pt"
  require_file "${REWARD_DIR}/reward_head_gate_report.json"
  local floor
  floor="$(reward_floor)"
  ENV_ID=sokoban PHASE=all RESUME="${CRITIC_RESUME:-0}" \
  SOURCE_WM="${REWARD_DIR}/best.pt" REWARD_DIR="${REWARD_DIR}" \
  DATA_DIR="${DATA_DIR}" H2_CACHE="${H2_CACHE}" OUTPUT_DIR="${CRITIC_DIR}" \
  REWARD_CONFIDENCE_FLOOR="${floor}" \
  GPU_ID="${GPU_ID}" SEED="${SEED}" NO_WANDB="${NO_WANDB:-0}" \
  WANDB_PROJECT="${WANDB_PROJECT:-jiayu-mbrl}" \
  WANDB_GROUP="${WANDB_GROUP_CRITIC:-sokoban_qwen_vjepa_critic}" \
  WANDB_RUN_NAME="sokoban_qwen_vjepa_critic_seed${SEED}" \
    bash "${ROOT}/scripts/train/run_qwen_vjepa_critic_warmup.sh"
}

run_ppo() {
  validate_data
  require_file "${REWARD_DIR}/best.pt"
  require_file "${CRITIC_DIR}/best.pt"
  require_file "${CRITIC_DIR}/warmup_report.json"
  local floor
  floor="$(reward_floor)"
  ENV_ID=sokoban WORLD_MODEL_MODE="${WORLD_MODEL_MODE}" RESUME="${PPO_RESUME:-0}" \
  REWARD_CKPT="${REWARD_DIR}/best.pt" \
  CRITIC_DIR="${CRITIC_DIR}" CRITIC_CKPT="${CRITIC_DIR}/best.pt" \
  CRITIC_REPORT="${CRITIC_DIR}/warmup_report.json" DATA_DIR="${DATA_DIR}" \
  VAGEN_LEVELS_FILE="${EVAL_LEVELS_FILE}" OUTPUT_DIR="${PPO_DIR}" \
  REWARD_CONFIDENCE_FLOOR="${floor}" ACTOR_UPDATES="${ACTOR_UPDATES:-400}" \
  TOTAL_UPDATES="${TOTAL_UPDATES:-2020}" EVAL_EVERY="${EVAL_EVERY:-10}" \
  CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-50}" ACTOR_LR="${ACTOR_LR:-2e-6}" \
  GPU_ID="${GPU_ID}" SEED="${SEED}" NO_WANDB="${NO_WANDB:-0}" \
  WANDB_PROJECT="${WANDB_PROJECT:-jiayu-mbrl}" \
  WANDB_GROUP="${WANDB_GROUP_PPO:-sokoban_qwen_vjepa_ppo}" \
  WANDB_RUN_NAME="sokoban_qwen_vjepa_ppo_${WORLD_MODEL_MODE}_seed${SEED}" \
    bash "${ROOT}/scripts/train/run_qwen_vjepa_ppo.sh"
}

run_world_model() {
  run_base_wm
  run_prior
}

run_rl() {
  run_critic
  run_ppo
}

echo "========================================================"
echo " Sokoban full training pipeline"
echo "========================================================"
echo "phase         : ${PHASE}"
echo "offline data  : random policy, ${DATA_EPISODES} episodes"
echo "seed range    : ${DATA_SEED_START}..$((DATA_SEED_START + DATA_EPISODES - 1))"
echo "GPU           : ${GPU_ID}"
echo "experiment    : ${EXP_ROOT}"

case "${PHASE}" in
  data) run_data ;;
  base_wm) run_base_wm ;;
  prior) run_prior ;;
  wm) run_world_model ;;
  reward) run_reward ;;
  critic) run_critic ;;
  ppo) run_ppo ;;
  rl) run_rl ;;
  all)
    run_data
    run_world_model
    run_reward
    run_rl
    ;;
esac

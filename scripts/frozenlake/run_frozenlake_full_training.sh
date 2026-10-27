#!/usr/bin/env bash
set -euo pipefail

# One-command FrozenLake experiment:
# random offline data -> World Model -> Reward Head -> Critic -> PPO.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${DWELL_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -x "${ROOT}/.venv/bin/python" ]]; then
    PYTHON_BIN="${ROOT}/.venv/bin/python"
  else
    PYTHON_BIN="$(command -v python3 || command -v python)"
  fi
fi

PHASE="${PHASE:-all}"
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-11}"
DATA_EPISODES="${DATA_EPISODES:-5000}"
DATA_SEED_START="${DATA_SEED_START:-1}"
WORLD_MODEL_MODE="${WORLD_MODEL_MODE:-frozen_wm}"
ACTOR_UPDATES="${ACTOR_UPDATES:-400}"

RAW_DATA_DIR="${RAW_DATA_DIR:-${ROOT}/data/frozenlake/random_rgb_seed${DATA_SEED_START}_${DATA_EPISODES}}"
PAIRED_ROOT="${PAIRED_ROOT:-${ROOT}/data/frozenlake/random_qwen_vjepa_teacher_3B_seed${DATA_SEED_START}_${DATA_EPISODES}}"
DATA_DIR="${DATA_DIR:-${PAIRED_ROOT}/tokenized}"
DATA_REPORT="${DATA_REPORT:-${PAIRED_ROOT}/dataset_contract_report.json}"
EXP_ROOT="${EXP_ROOT:-${ROOT}/checkpoints/frozenlake/random_qwen_vjepa_seed${SEED}}"

case "${PHASE}" in
  data|base_wm|prior|wm|reward|critic|ppo|rl|all) ;;
  *)
    echo "PHASE must be data, base_wm, prior, wm, reward, critic, ppo, rl, or all; got ${PHASE}" >&2
    exit 2
    ;;
esac

export DWELL_ROOT="${ROOT}"
export GPU_ID SEED DATA_EPISODES DATA_SEED_START WORLD_MODEL_MODE ACTOR_UPDATES
export RAW_DATA_DIR PAIRED_ROOT DATA_DIR DATA_REPORT EXP_ROOT
export DATA_COLLECTION_STRATEGY=random
export PYTHON_BIN
export MODEL_PATH="${MODEL_PATH:-${ROOT}/../../models/Qwen2.5-VL-3B-Instruct}"
export VAGEN_ROOT="${VAGEN_ROOT:-${ROOT}/../VAGEN-torch28}"

run_stage() {
  local stage="$1"
  echo ""
  echo "######## FrozenLake stage: ${stage} ########"
  PHASE="${stage}" bash "${ROOT}/scripts/frozenlake/run_frozenlake_training.sh"
}

run_world_model() {
  run_stage wm
  run_stage prior
  run_stage spatial_prior
  run_stage wm_audit
}

run_rl() {
  run_stage critic
  run_stage ppo
}

echo "========================================================"
echo " FrozenLake full training pipeline"
echo "========================================================"
echo "phase         : ${PHASE}"
echo "offline data  : random policy, ${DATA_EPISODES} episodes"
echo "seed range    : ${DATA_SEED_START}..$((DATA_SEED_START + DATA_EPISODES - 1))"
echo "GPU           : ${GPU_ID}"
echo "experiment    : ${EXP_ROOT}"

case "${PHASE}" in
  data) run_stage prepare ;;
  base_wm) run_stage wm ;;
  prior)
    run_stage prior
    run_stage spatial_prior
    run_stage wm_audit
    ;;
  wm) run_world_model ;;
  reward) run_stage reward ;;
  critic) run_stage critic ;;
  ppo) run_stage ppo ;;
  rl) run_rl ;;
  all)
    run_stage prepare
    run_world_model
    run_stage reward
    run_rl
    ;;
esac

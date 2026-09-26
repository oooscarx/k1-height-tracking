#!/usr/bin/env bash
set -uo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_DIR="${RUN_DIR:?Set RUN_DIR to the RSL-RL run containing model_*.pt files}"
TASK="${TASK:-Booster-K1-Fall-Recovery-Discovery-FaceUp-v0}"
MIN_ITERATION="${MIN_ITERATION:-500}"
POLL_SECONDS="${POLL_SECONDS:-30}"
MAX_STEPS="${MAX_STEPS:-500}"
UV="${UV:-$HOME/.local/bin/uv}"

case "${RUN_DIR}" in
  /*) ;;
  *) RUN_DIR="${ROOT_DIR}/${RUN_DIR}" ;;
esac

RUN_NAME="$(basename "${RUN_DIR}")"
STATE_DIR="${ROOT_DIR}/logs/supervision/${RUN_NAME}"
mkdir -p "${STATE_DIR}"

echo "[supervisor] task=${TASK}"
echo "[supervisor] run=${RUN_DIR}"
echo "[supervisor] state=${STATE_DIR}"

evaluate_checkpoint() {
  local checkpoint="$1"
  local iteration="$2"
  local seed="$3"
  local output="${STATE_DIR}/model_${iteration}_seed${seed}.npz"
  local log="${STATE_DIR}/model_${iteration}_seed${seed}.log"

  echo "[$(date -Is)] evaluating model_${iteration}.pt seed=${seed}" | tee "${log}"
  timeout 180 "${UV}" run --frozen python scripts/rsl_rl/export_discovery_trajectory.py \
    --task "${TASK}" \
    --headless \
    --device cuda:0 \
    --seed "${seed}" \
    --checkpoint "${checkpoint}" \
    --output "${output}" \
    --max_steps "${MAX_STEPS}" >>"${log}" 2>&1
  local status=$?

  local summary
  summary="$(grep -E 'RuntimeError:|Exported [0-9]+' "${log}" | tail -n 1)"
  echo "[$(date -Is)] status=${status} ${summary}" | tee -a "${log}"
  return "${status}"
}

while true; do
  while IFS= read -r checkpoint; do
    filename="$(basename "${checkpoint}")"
    iteration="${filename#model_}"
    iteration="${iteration%.pt}"
    [[ "${iteration}" =~ ^[0-9]+$ ]] || continue
    (( iteration >= MIN_ITERATION )) || continue

    marker="${STATE_DIR}/model_${iteration}.checked"
    [[ -e "${marker}" ]] && continue

    if evaluate_checkpoint "${checkpoint}" "${iteration}" 1; then
      robust=true
      for seed in 2 3 4 5; do
        if ! evaluate_checkpoint "${checkpoint}" "${iteration}" "${seed}"; then
          robust=false
          break
        fi
      done
      if ${robust}; then
        echo "[$(date -Is)] ROBUST_SUCCESS model_${iteration}.pt" | tee "${STATE_DIR}/SUCCESS"
      fi
    fi
    touch "${marker}"
  done < <(find "${RUN_DIR}" -maxdepth 1 -type f -name 'model_*.pt' | sort -V)

  sleep "${POLL_SECONDS}"
done

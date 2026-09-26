#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

usage="usage: watch_k1_wbc_mastery_milestones.sh RUN_DIR TRAINING_PID CACHE_PATH BASELINE_CHECKPOINT"
run_dir="${1:?$usage}"
training_pid="${2:?$usage}"
cache_path="${3:?$usage}"
baseline_checkpoint="${4:?$usage}"
milestones="${MASTERY_MILESTONES:-500 1000 1500 2000 3000 4000 5000 6000 7000 8000 9000 9999}"
num_envs="${NUM_ENVS:-128}"
modes="${WBC_MODES:-face_up face_down left_side right_side other}"
baseline_dir="$run_dir/evaluations/baseline_source"
baseline_summary="$baseline_dir/summary_lift_0_terrain_0.json"

echo "[WBC-MASTERY] evaluating source baseline: $baseline_checkpoint"
TRAINING_PID="$training_pid" PAUSE_TRAINING=1 NUM_ENVS="$num_envs" \
  LIFT_SCALE=0 TERRAIN_LEVEL=0 WBC_CACHE="$cache_path" \
  WBC_MODES="$modes" WBC_EVAL_DIR="$baseline_dir" \
  DISABLE_DOMAIN_RANDOMIZATION=1 \
  "$repo_root/scripts/rsl_rl/watch_k1_wbc_checkpoint.sh" "$baseline_checkpoint"

for iteration in $milestones; do
  checkpoint="$run_dir/model_${iteration}.pt"
  echo "[WBC-MASTERY] waiting for $checkpoint"
  TRAINING_PID="$training_pid" PAUSE_TRAINING=1 NUM_ENVS="$num_envs" \
    LIFT_SCALE=0 TERRAIN_LEVEL=0 WBC_CACHE="$cache_path" \
    WBC_MODES="$modes" DISABLE_DOMAIN_RANDOMIZATION=1 \
    "$repo_root/scripts/rsl_rl/watch_k1_wbc_checkpoint.sh" "$checkpoint"
  selected="$(
    "$repo_root/.venv/bin/python" \
      "$repo_root/scripts/rsl_rl/select_best_k1_wbc_checkpoint.py" \
      "$run_dir" \
      --lift-scale 0 \
      --terrain-level 0 \
      --modes $modes \
      --baseline-checkpoint "$baseline_checkpoint" \
      --baseline-summary "$baseline_summary" \
      --output "$run_dir/evaluations/checkpoint_selection_latest.json"
  )"
  echo "[WBC-MASTERY] iteration=$iteration selected=$selected"
  gate_output="$run_dir/evaluations/mastery_gate_latest.json"
  if "$repo_root/.venv/bin/python" \
    "$repo_root/scripts/rsl_rl/check_k1_wbc_mastery_gate.py" \
    "$run_dir/evaluations/checkpoint_selection_latest.json" \
    --output "$gate_output"; then
    echo "[WBC-MASTERY] terrain-0 confidence gate passed at iteration=$iteration"
  else
    echo "[WBC-MASTERY] terrain-0 confidence gate not yet passed at iteration=$iteration"
  fi
done

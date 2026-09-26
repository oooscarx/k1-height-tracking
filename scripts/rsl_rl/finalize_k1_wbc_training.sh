#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

checkpoint="${1:?usage: TRAINING_PID=<pid> finalize_k1_wbc_training.sh CHECKPOINT}"
training_pid="${TRAINING_PID:?set TRAINING_PID to the active training process}"
poll_seconds="${POLL_SECONDS:-5}"
stop_timeout_seconds="${STOP_TIMEOUT_SECONDS:-120}"
record_videos="${RECORD_VIDEOS:-1}"
package_artifacts="${PACKAGE_ARTIFACTS:-1}"
run_robustness_evaluation="${RUN_ROBUSTNESS_EVALUATION:-1}"
run_mastery_gate="${RUN_MASTERY_GATE:-1}"
minimum_aggregate_wilson="${MINIMUM_AGGREGATE_WILSON:-0.9}"
minimum_weakest_mode_wilson="${MINIMUM_WEAKEST_MODE_WILSON:-0.9}"
checkpoint_stable_seconds="${CHECKPOINT_STABLE_SECONDS:-2}"
checkpoint_stable_checks="${CHECKPOINT_STABLE_CHECKS:-3}"
selection_maximum_passes="${SELECTION_MAXIMUM_PASSES:-10}"

if [[ "$record_videos" != "0" && "$record_videos" != "1" ]]; then
  echo "RECORD_VIDEOS must be 0 or 1" >&2
  exit 2
fi
if [[ "$package_artifacts" != "0" && "$package_artifacts" != "1" ]]; then
  echo "PACKAGE_ARTIFACTS must be 0 or 1" >&2
  exit 2
fi
if [[ "$run_robustness_evaluation" != "0" && "$run_robustness_evaluation" != "1" ]]; then
  echo "RUN_ROBUSTNESS_EVALUATION must be 0 or 1" >&2
  exit 2
fi
if [[ "$run_mastery_gate" != "0" && "$run_mastery_gate" != "1" ]]; then
  echo "RUN_MASTERY_GATE must be 0 or 1" >&2
  exit 2
fi
if (( selection_maximum_passes <= 0 )); then
  echo "SELECTION_MAXIMUM_PASSES must be positive" >&2
  exit 2
fi

while [[ ! -s "$checkpoint" ]]; do
  if ! kill -0 "$training_pid" 2>/dev/null; then
    echo "training process exited before final checkpoint was ready: $training_pid" >&2
    exit 2
  fi
  sleep "$poll_seconds"
done

checkpoint_size=-1
stable_checks=0
while (( stable_checks < checkpoint_stable_checks )); do
  current_size="$(stat -c %s "$checkpoint")"
  if [[ "$current_size" == "$checkpoint_size" ]]; then
    ((stable_checks += 1))
  else
    checkpoint_size="$current_size"
    stable_checks=0
  fi
  sleep "$checkpoint_stable_seconds"
done

"$repo_root/.venv/bin/python" - "$checkpoint" <<'PY'
import json
import sys
from pathlib import Path

from scripts.rsl_rl.monitor_k1_wbc_training import health_alerts, inspect_checkpoint

checkpoint_path = Path(sys.argv[1])
summary = inspect_checkpoint(checkpoint_path)
expected_iteration = int(checkpoint_path.stem.rsplit("_", maxsplit=1)[-1])
if summary["checkpoint_iteration"] != expected_iteration:
    raise RuntimeError(
        "checkpoint iteration mismatch: "
        f"expected={expected_iteration} actual={summary['checkpoint_iteration']}"
    )
alerts = health_alerts(summary)
if alerts:
    raise RuntimeError(f"checkpoint failed pre-stop validation: {alerts}")
print(f"[WBC-FINALIZE] pre-stop validation={json.dumps(summary, sort_keys=True)}")
PY

echo "[WBC-FINALIZE] checkpoint ready: $checkpoint"
if kill -0 "$training_pid" 2>/dev/null; then
  echo "[WBC-FINALIZE] stopping trainer with SIGINT: $training_pid"
  kill -INT "$training_pid"

  for ((elapsed = 0; elapsed < stop_timeout_seconds; elapsed++)); do
    if ! kill -0 "$training_pid" 2>/dev/null; then
      break
    fi
    sleep 1
  done
  if kill -0 "$training_pid" 2>/dev/null; then
    echo "[WBC-FINALIZE] trainer did not exit after SIGINT; sending SIGTERM" >&2
    kill -TERM "$training_pid"
    sleep 10
  fi
  if kill -0 "$training_pid" 2>/dev/null; then
    echo "[WBC-FINALIZE] trainer is still alive after SIGTERM: $training_pid" >&2
    exit 2
  fi
else
  echo "[WBC-FINALIZE] trainer already exited after writing its final checkpoint"
fi

echo "[WBC-FINALIZE] trainer stopped; validating and evaluating final checkpoint"
TRAINING_PID="$training_pid" PAUSE_TRAINING=0 \
  "$repo_root/scripts/rsl_rl/watch_k1_wbc_checkpoint.sh" "$checkpoint"

run_dir="$(dirname "$checkpoint")"
checkpoint="$(realpath "$checkpoint")"
lift_scale="${LIFT_SCALE:-0}"
terrain_level="${TERRAIN_LEVEL:-0}"
evaluation_num_envs="${NUM_ENVS:-128}"
robustness_num_envs="${ROBUSTNESS_NUM_ENVS:-$evaluation_num_envs}"
robustness_terrain_level="${ROBUSTNESS_TERRAIN_LEVEL:--1}"
package_robustness_terrain_level=""
if (( run_robustness_evaluation )); then
  package_robustness_terrain_level="$robustness_terrain_level"
fi
read -r -a evaluation_modes <<< "${WBC_MODES:-face_up face_down left_side right_side other}"
expected_evaluation_envs=$((evaluation_num_envs * ${#evaluation_modes[@]}))
selection_report="$run_dir/evaluations/checkpoint_selection_lift_${lift_scale}_terrain_${terrain_level}.json"
robustness_report=""
mastery_report=""
robustness_mastery_report=""
selected_checkpoint=""

for ((selection_pass = 1; selection_pass <= selection_maximum_passes; selection_pass++)); do
  selected_checkpoint="$(
    "$repo_root/.venv/bin/python" \
      "$repo_root/scripts/rsl_rl/select_best_k1_wbc_checkpoint.py" \
      "$run_dir" \
      --lift-scale "$lift_scale" \
      --terrain-level "$terrain_level" \
      --modes "${evaluation_modes[@]}" \
      --output "$selection_report"
  )"
  selected_num_envs="$(
    "$repo_root/.venv/bin/python" - "$selection_report" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as report_file:
    report = json.load(report_file)
print(report["selected"]["num_envs"])
PY
  )"
  echo "[WBC-FINALIZE] selection pass=$selection_pass checkpoint=$selected_checkpoint num_envs=$selected_num_envs"
  if (( selected_num_envs >= expected_evaluation_envs )); then
    break
  fi
  echo "[WBC-FINALIZE] re-evaluating selected checkpoint with $evaluation_num_envs environments per mode"
  NUM_ENVS="$evaluation_num_envs" LIFT_SCALE="$lift_scale" TERRAIN_LEVEL="$terrain_level" \
    WBC_MODES="${evaluation_modes[*]}" \
    "$repo_root/scripts/rsl_rl/evaluate_k1_wbc_checkpoint.sh" "$selected_checkpoint"
  selected_checkpoint=""
done
if [[ -z "$selected_checkpoint" ]]; then
  echo "checkpoint selection did not stabilize after $selection_maximum_passes passes" >&2
  exit 2
fi
echo "[WBC-FINALIZE] selected validated checkpoint: $selected_checkpoint"

if (( run_mastery_gate )); then
  mastery_report="$run_dir/evaluations/checkpoint_mastery_gate_lift_${lift_scale}_terrain_${terrain_level}.json"
  echo "[WBC-FINALIZE] checking mastery confidence gate"
  "$repo_root/.venv/bin/python" \
    "$repo_root/scripts/rsl_rl/check_k1_wbc_mastery_gate.py" \
    "$selection_report" \
    --minimum-aggregate-wilson "$minimum_aggregate_wilson" \
    --minimum-weakest-mode-wilson "$minimum_weakest_mode_wilson" \
    --output "$mastery_report"
  echo "[WBC-FINALIZE] mastery confidence gate passed: $mastery_report"
fi

if (( run_robustness_evaluation )); then
  echo "[WBC-FINALIZE] evaluating selected checkpoint across terrain levels"
  NUM_ENVS="$robustness_num_envs" LIFT_SCALE="$lift_scale" \
    TERRAIN_LEVEL="$robustness_terrain_level" WBC_MODES="${evaluation_modes[*]}" \
    "$repo_root/scripts/rsl_rl/evaluate_k1_wbc_checkpoint.sh" "$selected_checkpoint"
  robustness_report="$run_dir/evaluations/checkpoint_robustness_lift_${lift_scale}_terrain_${robustness_terrain_level}.json"
  robustness_selected="$(
    "$repo_root/.venv/bin/python" \
      "$repo_root/scripts/rsl_rl/select_best_k1_wbc_checkpoint.py" \
      "$run_dir" \
      --lift-scale "$lift_scale" \
      --terrain-level "$robustness_terrain_level" \
      --modes "${evaluation_modes[@]}" \
      --output "$robustness_report"
  )"
  if [[ "$(realpath "$robustness_selected")" != "$selected_checkpoint" ]]; then
    echo "all-terrain safety validation selected a different checkpoint: $robustness_selected" >&2
    exit 2
  fi
  robustness_evaluation_envs="$(
    "$repo_root/.venv/bin/python" - "$robustness_report" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as report_file:
    report = json.load(report_file)
print(report["selected"]["num_envs"])
PY
  )"
  expected_robustness_envs=$((robustness_num_envs * ${#evaluation_modes[@]}))
  if (( robustness_evaluation_envs < expected_robustness_envs )); then
    echo "all-terrain safety validation has too few environments: $robustness_evaluation_envs < $expected_robustness_envs" >&2
    exit 2
  fi
  if (( run_mastery_gate )); then
    robustness_mastery_report="$run_dir/evaluations/checkpoint_mastery_gate_lift_${lift_scale}_terrain_${robustness_terrain_level}.json"
    echo "[WBC-FINALIZE] checking all-terrain mastery confidence gate"
    "$repo_root/.venv/bin/python" \
      "$repo_root/scripts/rsl_rl/check_k1_wbc_mastery_gate.py" \
      "$robustness_report" \
      --minimum-aggregate-wilson "$minimum_aggregate_wilson" \
      --minimum-weakest-mode-wilson "$minimum_weakest_mode_wilson" \
      --terrain-level "$robustness_terrain_level" \
      --output "$robustness_mastery_report"
    echo "[WBC-FINALIZE] all-terrain mastery confidence gate passed: $robustness_mastery_report"
  fi
  echo "[WBC-FINALIZE] all-terrain safety validation passed: $robustness_report"
fi

if (( record_videos )); then
  echo "[WBC-FINALIZE] recording verified successful videos for selected checkpoint"
  LIFT_SCALE="$lift_scale" TERRAIN_LEVEL="$terrain_level" \
    "$repo_root/scripts/rsl_rl/record_successful_k1_wbc_videos.sh" "$selected_checkpoint"
fi

if (( package_artifacts )); then
  echo "[WBC-FINALIZE] packaging validated checkpoint artifacts"
  LIFT_SCALE="$lift_scale" TERRAIN_LEVEL="$terrain_level" \
    ROBUSTNESS_TERRAIN_LEVEL="$package_robustness_terrain_level" \
    ROBUSTNESS_REPORT="$robustness_report" SELECTION_REPORT="$selection_report" \
    MASTERY_REPORT="$mastery_report" \
    ROBUSTNESS_MASTERY_REPORT="$robustness_mastery_report" \
    FINAL_CHECKPOINT="$checkpoint" \
    "$repo_root/scripts/rsl_rl/package_k1_wbc_checkpoint.sh" "$selected_checkpoint"
fi

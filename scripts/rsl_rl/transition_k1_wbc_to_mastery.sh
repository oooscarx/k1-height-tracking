#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

usage="usage: LEGACY_TRAINING_PID=pid transition_k1_wbc_to_mastery.sh MODEL_CHECKPOINT CACHE_PATH"
legacy_checkpoint="${1:?$usage}"
cache_path="${2:?$usage}"
legacy_pid="${LEGACY_TRAINING_PID:?set LEGACY_TRAINING_PID to the active trainer}"
poll_seconds="${POLL_SECONDS:-5}"
stop_timeout_seconds="${STOP_TIMEOUT_SECONDS:-120}"
mastery_num_envs="${MASTERY_NUM_ENVS:-4096}"
mastery_iterations="${MASTERY_ITERATIONS:-10000}"
checkpoint_stem="$(basename "${legacy_checkpoint%.pt}")"
if [[ ! "$checkpoint_stem" =~ ^model_([0-9]+)$ ]]; then
  echo "legacy checkpoint must be named model_<iteration>.pt: $legacy_checkpoint" >&2
  exit 2
fi
target_iteration="${BASH_REMATCH[1]}"
mastery_run_name="${MASTERY_RUN_NAME:-terrain0_mastery_after_legacy${target_iteration}_$(date +%Y%m%d_%H%M%S)}"
mastery_session="${MASTERY_SESSION:-k1-wbc-mastery-stage0}"
health_session="${MASTERY_HEALTH_SESSION:-k1-wbc-mastery-health}"
milestone_session="${MASTERY_MILESTONE_SESSION:-k1-wbc-mastery-milestones}"
console_log="logs/k1_wbc_${mastery_run_name}.console.log"
pid_file="logs/k1_wbc_${mastery_run_name}.pid"
transition_report="$(dirname "$legacy_checkpoint")/evaluations/checkpoint_selection_at_${target_iteration}.json"

if [[ ! -f "$cache_path" ]]; then
  echo "fallen-state cache does not exist: $cache_path" >&2
  exit 2
fi

legacy_paused=0
resume_legacy_on_error() {
  status=$?
  if (( status != 0 && legacy_paused )) && kill -0 "$legacy_pid" 2>/dev/null; then
    echo "[WBC-TRANSITION] resuming legacy trainer after transition failure" >&2
    kill -CONT "$legacy_pid" 2>/dev/null || true
  fi
  exit "$status"
}
trap resume_legacy_on_error EXIT

while [[ ! -s "$legacy_checkpoint" ]]; do
  if ! kill -0 "$legacy_pid" 2>/dev/null; then
    echo "legacy trainer exited before $legacy_checkpoint was ready" >&2
    exit 2
  fi
  sleep "$poll_seconds"
done

checkpoint_size="$(stat -c %s "$legacy_checkpoint")"
sleep 10
while [[ "$(stat -c %s "$legacy_checkpoint")" != "$checkpoint_size" ]]; do
  checkpoint_size="$(stat -c %s "$legacy_checkpoint")"
  sleep 10
done

echo "[WBC-TRANSITION] pausing legacy trainer at model_${target_iteration}"
kill -STOP "$legacy_pid"
legacy_paused=1

TRAINING_PID="$legacy_pid" PAUSE_TRAINING=0 NUM_ENVS=128 \
  LIFT_SCALE=0 TERRAIN_LEVEL=0 WBC_CACHE="$cache_path" \
  WBC_MODES="face_up face_down left_side right_side other" \
  "$repo_root/scripts/rsl_rl/watch_k1_wbc_checkpoint.sh" "$legacy_checkpoint"

legacy_run_dir="$(dirname "$legacy_checkpoint")"
selected_checkpoint="$(
  "$repo_root/.venv/bin/python" \
    "$repo_root/scripts/rsl_rl/select_best_k1_wbc_checkpoint.py" \
    "$legacy_run_dir" \
    --lift-scale 0 \
    --terrain-level 0 \
    --modes face_up face_down left_side right_side other \
    --output "$transition_report"
)"
echo "[WBC-TRANSITION] selected source checkpoint: $selected_checkpoint"

echo "[WBC-TRANSITION] stopping legacy trainer"
kill -CONT "$legacy_pid"
legacy_paused=0
kill -INT "$legacy_pid"
for ((elapsed = 0; elapsed < stop_timeout_seconds; elapsed++)); do
  if ! kill -0 "$legacy_pid" 2>/dev/null; then
    break
  fi
  sleep 1
done
if kill -0 "$legacy_pid" 2>/dev/null; then
  kill -TERM "$legacy_pid"
  sleep 10
fi
if kill -0 "$legacy_pid" 2>/dev/null; then
  echo "legacy trainer is still alive after SIGTERM: $legacy_pid" >&2
  exit 2
fi

for session in "$mastery_session" "$health_session" "$milestone_session"; do
  tmux kill-session -t "$session" 2>/dev/null || true
done
rm -f "$pid_file"

training_args=(
  "$repo_root/.venv/bin/python"
  scripts/rsl_rl/train.py
  --task Booster-K1-Stand-Up-WBC-Mastery-v0
  --num_envs "$mastery_num_envs"
  --max_iterations "$mastery_iterations"
  --headless
  --device cuda:0
  --resume
  --resume_path "$selected_checkpoint"
  --wbc_fallen_cache "$cache_path"
  --wbc_max_terrain_level 0
  --run_name "$mastery_run_name"
)
printf -v training_command "%q " "${training_args[@]}"
printf -v quoted_repo "%q" "$repo_root"
printf -v quoted_pid_file "%q" "$pid_file"
printf -v quoted_console "%q" "$console_log"
tmux new-session -d -s "$mastery_session" \
  "cd $quoted_repo && { $training_command >> $quoted_console 2>&1 & trainer_pid=\$!; echo \$trainer_pid > $quoted_pid_file; wait \$trainer_pid; }"

for _ in {1..120}; do
  if [[ -s "$pid_file" ]]; then
    mastery_pid="$(<"$pid_file")"
    if kill -0 "$mastery_pid" 2>/dev/null; then
      break
    fi
  fi
  sleep 1
done
if [[ -z "${mastery_pid:-}" ]] || ! kill -0 "$mastery_pid" 2>/dev/null; then
  echo "mastery trainer failed to start; inspect $console_log" >&2
  exit 2
fi

mastery_run_dir=""
for _ in {1..120}; do
  mastery_run_dir="$(
    find logs/rsl_rl/k1_wbc_stand_up_mastery -mindepth 1 -maxdepth 1 \
      -type d -name "*_${mastery_run_name}" -printf '%T@ %p\n' 2>/dev/null \
      | sort -nr | head -n 1 | cut -d' ' -f2-
  )"
  if [[ -n "$mastery_run_dir" ]]; then
    break
  fi
  sleep 1
done
if [[ -z "$mastery_run_dir" ]]; then
  echo "mastery run directory was not created; inspect $console_log" >&2
  exit 2
fi

health_output="logs/k1_wbc_health_${mastery_run_name}.jsonl"
tmux new-session -d -s "$health_session" \
  "cd $quoted_repo && exec $repo_root/.venv/bin/python scripts/rsl_rl/monitor_k1_wbc_training.py --console-log $console_log --run-dir $mastery_run_dir --training-pid $mastery_pid --poll-seconds 60 --output $health_output --lift-start-step -1 --lift-num-steps 1 --lift-linear >> logs/k1_wbc_health_${mastery_run_name}.console.log 2>&1"

tmux new-session -d -s "$milestone_session" \
  "cd $quoted_repo && exec env NUM_ENVS=128 scripts/rsl_rl/watch_k1_wbc_mastery_milestones.sh $mastery_run_dir $mastery_pid $cache_path $selected_checkpoint >> logs/k1_wbc_milestones_${mastery_run_name}.log 2>&1"

echo "[WBC-TRANSITION] mastery_pid=$mastery_pid"
echo "[WBC-TRANSITION] mastery_run_dir=$mastery_run_dir"
echo "[WBC-TRANSITION] selection_report=$transition_report"
trap - EXIT

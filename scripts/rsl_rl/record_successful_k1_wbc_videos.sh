#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

checkpoint="${1:?usage: record_successful_k1_wbc_videos.sh CHECKPOINT}"
python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
lift_scale="${LIFT_SCALE:-0}"
terrain_level="${TERRAIN_LEVEL:-0}"
video_length="${VIDEO_LENGTH:-1000}"
seed_start="${SEED_START:-0}"
max_seed="${MAX_SEED:-31}"
modes="${WBC_VIDEO_MODES:-face_up face_down left_side right_side other}"
cache_path="${WBC_CACHE:-}"

if [[ -z "$cache_path" ]]; then
  cache_path="$(ls -S fallen_states_cache/*.pt | head -n 1)"
fi
if [[ ! -s "$checkpoint" ]]; then
  echo "checkpoint does not exist or is empty: $checkpoint" >&2
  exit 1
fi
if [[ ! -s "$cache_path" ]]; then
  echo "fallen-state cache does not exist or is empty: $cache_path" >&2
  exit 1
fi

checkpoint_name="$(basename "${checkpoint%.pt}")"
evaluation_dir="${WBC_EVAL_DIR:-$(dirname "$checkpoint")/evaluations/$checkpoint_name}"
output_root="${VIDEO_OUTPUT_DIR:-$evaluation_dir/success_videos}"
mkdir -p "$output_root"

export PYTHONPATH="$repo_root/third_party/wbc_agile_rsl_rl:$repo_root/source/booster_train:$repo_root/third_party/booster_assets${PYTHONPATH:+:$PYTHONPATH}"

work_dir="$(mktemp -d)"
trap 'rm -rf "$work_dir"' EXIT

for mode in $modes; do
  recorded=0
  for ((seed = seed_start; seed <= max_seed; seed++)); do
    attempt_dir="$work_dir/${mode}_seed_${seed}"
    result_path="$attempt_dir/result.json"
    mkdir -p "$attempt_dir"
    echo "[WBC-SUCCESS-VIDEO] mode=$mode seed=$seed"
    "$python_bin" scripts/rsl_rl/play.py \
      --task Booster-K1-Stand-Up-WBC-v0 \
      --checkpoint "$checkpoint" \
      --num_envs 1 \
      --seed "$seed" \
      --video \
      --video_length "$video_length" \
      --video_output_dir "$attempt_dir" \
      --wbc_fallen_cache "$cache_path" \
      --wbc_reset_mode "$mode" \
      --wbc_lift_scale "$lift_scale" \
      --wbc_terrain_level "$terrain_level" \
      --wbc_disable_external_disturbances \
      --wbc_eval_output "$result_path" \
      --headless \
      >"$attempt_dir/play.log" 2>&1

    if "$python_bin" - "$result_path" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as result_file:
    result = json.load(result_file)
raise SystemExit(0 if result["success_count"] == result["num_envs"] else 1)
PY
    then
      video_path="$(find "$attempt_dir/$mode" -maxdepth 1 -type f -name '*.mp4' -size +0c -print -quit)"
      if [[ -z "$video_path" ]]; then
        echo "successful rollout did not produce a video: mode=$mode seed=$seed" >&2
        exit 1
      fi
      output_stem="${checkpoint_name}_${mode}_seed${seed}_lift${lift_scale}"
      cp "$video_path" "$output_root/${output_stem}.mp4"
      cp "$result_path" "$output_root/${output_stem}.json"
      cp "$attempt_dir/play.log" "$output_root/${output_stem}.log"
      echo "[WBC-SUCCESS-VIDEO] recorded=$output_root/${output_stem}.mp4"
      recorded=1
      break
    fi
  done
  if (( ! recorded )); then
    echo "no successful rollout found for mode=$mode seeds=$seed_start..$max_seed" >&2
    exit 1
  fi
done

echo "[WBC-SUCCESS-VIDEO] output=$output_root"

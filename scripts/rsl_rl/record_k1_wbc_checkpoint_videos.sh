#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

checkpoint="${1:?usage: record_k1_wbc_checkpoint_videos.sh CHECKPOINT}"
python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
num_envs="${VIDEO_NUM_ENVS:-4}"
video_length="${VIDEO_LENGTH:-750}"
lift_scale="${LIFT_SCALE:-0}"
terrain_level="${TERRAIN_LEVEL:-0}"
modes="${WBC_VIDEO_MODES:-face_up face_down left_side right_side other}"
cache_path="${WBC_CACHE:-}"

if [[ -z "$cache_path" ]]; then
  cache_path="$(ls -S fallen_states_cache/*.pt | head -n 1)"
fi
if [[ ! -f "$checkpoint" ]]; then
  echo "checkpoint does not exist: $checkpoint" >&2
  exit 1
fi
if [[ ! -f "$cache_path" ]]; then
  echo "fallen-state cache does not exist: $cache_path" >&2
  exit 1
fi

checkpoint_name="$(basename "${checkpoint%.pt}")"
evaluation_dir="${WBC_EVAL_DIR:-$(dirname "$checkpoint")/evaluations/$checkpoint_name}"
video_root="$evaluation_dir/videos"
mkdir -p "$video_root"

export PYTHONPATH="$repo_root/third_party/wbc_agile_rsl_rl:$repo_root/source/booster_train:$repo_root/third_party/booster_assets${PYTHONPATH:+:$PYTHONPATH}"

for mode in $modes; do
  log_path="$video_root/${mode}.log"
  echo "[WBC-VIDEO] mode=$mode lift=$lift_scale terrain=$terrain_level length=$video_length"
  "$python_bin" scripts/rsl_rl/play.py \
    --task Booster-K1-Stand-Up-WBC-v0 \
    --checkpoint "$checkpoint" \
    --num_envs "$num_envs" \
    --seed 42 \
    --video \
    --video_length "$video_length" \
    --video_output_dir "$video_root" \
    --wbc_fallen_cache "$cache_path" \
    --wbc_reset_mode "$mode" \
    --wbc_lift_scale "$lift_scale" \
    --wbc_terrain_level "$terrain_level" \
    --wbc_disable_external_disturbances \
    --headless \
    >"$log_path" 2>&1

  if ! find "$video_root/$mode" -maxdepth 1 -type f -name '*.mp4' -size +0c -print -quit | grep -q .; then
    echo "video was not created for mode=$mode" >&2
    exit 1
  fi
done

echo "[WBC-VIDEO] videos=$video_root"

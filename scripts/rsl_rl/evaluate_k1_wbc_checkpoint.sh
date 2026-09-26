#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

checkpoint="${1:?usage: evaluate_k1_wbc_checkpoint.sh CHECKPOINT}"
python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
num_envs="${NUM_ENVS:-128}"
lift_scale="${LIFT_SCALE:-0}"
terrain_level="${TERRAIN_LEVEL:-0}"
eval_seed="${EVAL_SEED:-42}"
modes="${WBC_MODES:-face_up face_down left_side right_side other}"
cache_path="${WBC_CACHE:-}"
disable_domain_randomization="${DISABLE_DOMAIN_RANDOMIZATION:-0}"

if [[ "$disable_domain_randomization" != "0" && "$disable_domain_randomization" != "1" ]]; then
  echo "DISABLE_DOMAIN_RANDOMIZATION must be 0 or 1" >&2
  exit 2
fi
if [[ ! "$eval_seed" =~ ^-?[0-9]+$ ]]; then
  echo "EVAL_SEED must be an integer" >&2
  exit 2
fi

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
mkdir -p "$evaluation_dir"

export PYTHONPATH="$repo_root/third_party/wbc_agile_rsl_rl:$repo_root/source/booster_train:$repo_root/third_party/booster_assets${PYTHONPATH:+:$PYTHONPATH}"

domain_randomization_args=()
if (( disable_domain_randomization )); then
  domain_randomization_args+=(--wbc_disable_domain_randomization)
fi

for mode in $modes; do
  output_path="$evaluation_dir/${mode}_lift_${lift_scale}_terrain_${terrain_level}.json"
  log_path="$evaluation_dir/${mode}_lift_${lift_scale}_terrain_${terrain_level}.log"
  echo "[WBC-EVAL] mode=$mode lift=$lift_scale terrain=$terrain_level"
  "$python_bin" scripts/rsl_rl/play.py \
    --task Booster-K1-Stand-Up-WBC-v0 \
    --checkpoint "$checkpoint" \
    --num_envs "$num_envs" \
    --seed "$eval_seed" \
    --wbc_fallen_cache "$cache_path" \
    --wbc_reset_mode "$mode" \
    --wbc_lift_scale "$lift_scale" \
    --wbc_terrain_level "$terrain_level" \
    --wbc_disable_external_disturbances \
    "${domain_randomization_args[@]}" \
    --wbc_eval_output "$output_path" \
    --headless \
    >"$log_path" 2>&1
  "$python_bin" - "$output_path" "$lift_scale" "$mode" "$eval_seed" <<'PY'
import json
import math
import sys

output_path, expected_lift, expected_mode, expected_seed = sys.argv[1:]
with open(output_path, encoding="utf-8") as output_file:
    result = json.load(output_file)

actual_lift = float(result["lift_force_scale"])
if not math.isclose(actual_lift, float(expected_lift), abs_tol=1.0e-8):
    raise RuntimeError(
        f"evaluation lift mismatch: expected={expected_lift}, actual={actual_lift}"
    )
if result["orientation_mode"] != expected_mode:
    raise RuntimeError(
        "evaluation orientation mismatch: "
        f"expected={expected_mode}, actual={result['orientation_mode']}"
    )
if int(result["seed"]) != int(expected_seed):
    raise RuntimeError(
        f"evaluation seed mismatch: expected={expected_seed}, actual={result['seed']}"
    )
if not math.isclose(float(result["initial_mode_match_rate"]), 1.0, abs_tol=1.0e-6):
    raise RuntimeError(
        "evaluation reset did not use only the requested orientation: "
        f"match_rate={result['initial_mode_match_rate']}"
    )

print(
    {
        key: result[key]
        for key in (
            "orientation_mode",
            "seed",
            "lift_force_scale",
            "success_rate",
            "mean_recovery_time",
            "no_progress_failure_count",
            "timeout_failure_count",
            "parallel_fallback_count",
            "nonfinite_action_count",
        )
    }
)
PY
done

summary_path="$evaluation_dir/summary_lift_${lift_scale}_terrain_${terrain_level}.json"
"$python_bin" scripts/rsl_rl/summarize_k1_wbc_evaluation.py \
  "$evaluation_dir" \
  --lift-scale "$lift_scale" \
  --terrain-level "$terrain_level" \
  --modes $modes \
  --output "$summary_path"
echo "[WBC-EVAL] summary=$summary_path"

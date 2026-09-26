#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CHECKPOINT="${1:?usage: start_k1_height_progressive_robustness.sh /absolute/path/to/model.pt}"
CACHE="${WBC_FALLEN_CACHE:-$ROOT/height_tracking_states_cache/fallen_states_v7_Booster_K1_Height_Tracking_v0_548cc4ff_97a95488.pt}"

cd "$ROOT"
exec .venv/bin/python3 -u scripts/rsl_rl/train.py \
  --task Booster-K1-Height-Tracking-ProgressiveRobustness-v0 \
  --num_envs "${NUM_ENVS:-10240}" \
  --max_iterations "${MAX_ITERATIONS:-5000}" \
  --headless --device "${DEVICE:-cuda:0}" \
  --resume --resume_path "$CHECKPOINT" \
  --wbc_fallen_cache "$CACHE" \
  --wbc_initial_lift_scale 0

#!/usr/bin/env bash
set -euo pipefail

NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-20000}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-k1_recovery_5090}"
TASK="${TASK:-Booster-K1-Fall-Recovery-v0}"

exec uv run --frozen python scripts/rsl_rl/train.py \
  --task "${TASK}" \
  --headless \
  --device "${DEVICE}" \
  --num_envs "${NUM_ENVS}" \
  --max_iterations "${MAX_ITERATIONS}" \
  --seed "${SEED}" \
  --run_name "${RUN_NAME}" \
  "$@"

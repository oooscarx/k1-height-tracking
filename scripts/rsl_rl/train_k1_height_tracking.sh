#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

exec .venv/bin/python3 -u scripts/rsl_rl/train.py \
  --task Booster-K1-Height-Tracking-v0 \
  --num_envs "${NUM_ENVS:-4096}" \
  --max_iterations "${MAX_ITERATIONS:-100000}" \
  --headless --device "${DEVICE:-cuda:0}" "$@"

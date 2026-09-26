#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

export OMNI_KIT_ACCEPT_EULA="${OMNI_KIT_ACCEPT_EULA:-YES}"
export XDG_RUNTIME_DIR="${XDG_RUNTIME_DIR:-/tmp/k1-height-xdg-runtime}"
install -d -m 700 "$XDG_RUNTIME_DIR"

exec .venv/bin/torchrun \
  --standalone \
  --nproc-per-node="${NUM_GPUS:-2}" \
  --no-python \
  scripts/rsl_rl/numa_rank.sh \
  .venv/bin/python -u scripts/rsl_rl/train.py \
  --task Booster-K1-Height-Tracking-v0 \
  --num_envs "${NUM_ENVS_PER_GPU:-8192}" \
  --max_iterations "${MAX_ITERATIONS:-100000}" \
  --headless --distributed "$@"

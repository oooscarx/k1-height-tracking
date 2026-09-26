#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

num_envs="${NUM_ENVS:-4096}"
max_iterations="${MAX_ITERATIONS:-100000}"
device="${DEVICE:-cuda:0}"
python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"

export PYTHONPATH="$repo_root/third_party/wbc_agile_rsl_rl:$repo_root/source/booster_train:$repo_root/third_party/booster_assets${PYTHONPATH:+:$PYTHONPATH}"

exec "$python_bin" scripts/rsl_rl/train.py \
  --task Booster-K1-Stand-Up-WBC-v0 \
  --num_envs "$num_envs" \
  --max_iterations "$max_iterations" \
  --headless \
  --device "$device" \
  "$@"

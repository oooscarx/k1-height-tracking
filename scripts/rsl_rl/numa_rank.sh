#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${LOCAL_RANK:-}" ]]; then
  echo "LOCAL_RANK is required" >&2
  exit 2
fi

exec numactl --cpunodebind="${LOCAL_RANK}" "$@"

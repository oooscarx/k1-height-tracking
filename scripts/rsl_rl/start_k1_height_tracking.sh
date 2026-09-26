#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_SESSION="${TRAIN_SESSION:-k1-height-training}"
MONITOR_SESSION="${MONITOR_SESSION:-k1-height-monitor}"
CONSOLE_LOG="$ROOT/logs/height_tracking_train.console.log"
MONITOR_LOG="$ROOT/logs/height_tracking_monitor.console.log"

mkdir -p "$ROOT/logs"
if tmux has-session -t "$TRAIN_SESSION" 2>/dev/null; then
  echo "training session already exists: $TRAIN_SESSION" >&2
  exit 1
fi

tmux new-session -d -s "$TRAIN_SESSION" \
  "cd '$ROOT' && set -o pipefail && exec ./scripts/rsl_rl/train_k1_height_tracking.sh 2>&1 | tee -a '$CONSOLE_LOG'"

if ! tmux has-session -t "$MONITOR_SESSION" 2>/dev/null; then
  tmux new-session -d -s "$MONITOR_SESSION" \
    "cd '$ROOT' && exec .venv/bin/python3 -u scripts/rsl_rl/monitor_k1_height_tracking.py --interval '${MONITOR_INTERVAL:-900}' 2>&1 | tee -a '$MONITOR_LOG'"
fi

echo "training: tmux attach -t $TRAIN_SESSION"
echo "monitor:  tail -f $ROOT/logs/height_tracking_health.jsonl"

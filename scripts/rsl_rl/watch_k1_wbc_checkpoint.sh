#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

checkpoint="${1:?usage: TRAINING_PID=<pid> watch_k1_wbc_checkpoint.sh CHECKPOINT}"
training_pid="${TRAINING_PID:?set TRAINING_PID to the active training process}"
poll_seconds="${POLL_SECONDS:-300}"
pause_training="${PAUSE_TRAINING:-1}"

if [[ "$pause_training" != "0" && "$pause_training" != "1" ]]; then
  echo "PAUSE_TRAINING must be 0 or 1" >&2
  exit 2
fi

while [[ ! -s "$checkpoint" ]]; do
  if ! kill -0 "$training_pid" 2>/dev/null; then
    echo "training process exited before checkpoint was ready: $training_pid" >&2
    exit 2
  fi
  sleep "$poll_seconds"
done

# Avoid reading a checkpoint while torch.save is still replacing its contents.
checkpoint_size="$(stat -c %s "$checkpoint")"
sleep 10
if [[ "$(stat -c %s "$checkpoint")" != "$checkpoint_size" ]]; then
  sleep 10
fi

"$repo_root/.venv/bin/python" - "$checkpoint" <<'PY'
import math
import sys

import torch

checkpoint_path = sys.argv[1]
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

for section_name in ("model_state_dict", "reward_norm_state_dict"):
    section = checkpoint.get(section_name)
    if not isinstance(section, dict):
        raise RuntimeError(f"checkpoint is missing {section_name}: {checkpoint_path}")
    nonfinite = [
        name
        for name, value in section.items()
        if torch.is_tensor(value) and not bool(torch.isfinite(value).all())
    ]
    if nonfinite:
        raise RuntimeError(
            f"checkpoint has non-finite {section_name} tensors: {nonfinite}"
        )

normalizer = checkpoint["reward_norm_state_dict"]
summary = {}
for name in ("_mean", "_var", "_std", "_return_correction"):
    value = normalizer.get(name)
    if not torch.is_tensor(value) or value.numel() != 1:
        raise RuntimeError(f"reward normalizer has invalid {name}: {value!r}")
    summary[name] = float(value.item())

if summary["_var"] < 0.0 or summary["_std"] <= 0.0:
    raise RuntimeError(f"reward normalizer scale is invalid: {summary}")
if summary["_return_correction"] <= 0.0:
    raise RuntimeError(f"reward normalizer correction is invalid: {summary}")
if not all(math.isfinite(value) for value in summary.values()):
    raise RuntimeError(f"reward normalizer is non-finite: {summary}")

policy_std = checkpoint["model_state_dict"].get("std")
if not torch.is_tensor(policy_std) or policy_std.ndim != 1:
    raise RuntimeError(f"checkpoint has invalid policy std: {policy_std!r}")
if not bool((policy_std > 0.0).all()):
    raise RuntimeError("checkpoint policy std must remain positive")
policy_std_summary = {
    "minimum": float(policy_std.min().item()),
    "mean": float(policy_std.mean().item()),
    "maximum": float(policy_std.max().item()),
}

print(
    "[WBC-CHECKPOINT]"
    f" path={checkpoint_path}"
    f" iter={checkpoint.get('iter')}"
    f" reward_normalizer={summary}"
    f" policy_std={policy_std_summary}",
    flush=True,
)
PY

training_paused=0
resume_training() {
  if (( training_paused )); then
    kill -CONT "$training_pid" 2>/dev/null || true
  fi
}
trap resume_training EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

if (( pause_training )); then
  kill -STOP "$training_pid"
  training_paused=1
elif kill -0 "$training_pid" 2>/dev/null; then
  echo "[WBC-CHECKPOINT] evaluating while training remains active"
else
  echo "[WBC-CHECKPOINT] evaluating after training has exited"
fi

"$repo_root/scripts/rsl_rl/evaluate_k1_wbc_checkpoint.sh" "$checkpoint"

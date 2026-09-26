#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

checkpoint="${1:?usage: package_k1_wbc_checkpoint.sh CHECKPOINT}"
python_bin="${PYTHON_BIN:-$repo_root/.venv/bin/python}"
lift_scale="${LIFT_SCALE:-0}"
terrain_level="${TERRAIN_LEVEL:-0}"
robustness_terrain_level="${ROBUSTNESS_TERRAIN_LEVEL:-}"

if [[ ! -s "$checkpoint" ]]; then
  echo "checkpoint does not exist or is empty: $checkpoint" >&2
  exit 1
fi

checkpoint_name="$(basename "${checkpoint%.pt}")"
run_dir="$(dirname "$checkpoint")"
evaluation_dir="${WBC_EVAL_DIR:-$run_dir/evaluations/$checkpoint_name}"
summary="$evaluation_dir/summary_lift_${lift_scale}_terrain_${terrain_level}.json"
robustness_summary=""
if [[ -n "$robustness_terrain_level" ]]; then
  robustness_summary="$evaluation_dir/summary_lift_${lift_scale}_terrain_${robustness_terrain_level}.json"
fi
output="${PACKAGE_OUTPUT:-$run_dir/artifacts/${checkpoint_name}_validated_bundle.zip}"
health_log="${HEALTH_LOG:-}"
selection_report="${SELECTION_REPORT:-}"
robustness_report="${ROBUSTNESS_REPORT:-}"
mastery_report="${MASTERY_REPORT:-}"
robustness_mastery_report="${ROBUSTNESS_MASTERY_REPORT:-}"
final_checkpoint="${FINAL_CHECKPOINT:-}"
export_dir="$run_dir/exported"
export_manifest="$export_dir/k1_wbc_export_manifest.json"

if [[ ! -s "$summary" ]]; then
  echo "evaluation summary does not exist or is empty: $summary" >&2
  exit 1
fi
if [[ -n "$robustness_summary" && ! -s "$robustness_summary" ]]; then
  echo "robustness summary does not exist or is empty: $robustness_summary" >&2
  exit 1
fi
if [[ ! -s "$export_manifest" ]]; then
  echo "export provenance manifest does not exist or is empty: $export_manifest" >&2
  exit 1
fi
"$python_bin" "$repo_root/scripts/rsl_rl/k1_wbc_export_manifest.py" \
  verify "$checkpoint" "$export_dir"
"$python_bin" "$repo_root/scripts/rsl_rl/validate_k1_wbc_exports.py" \
  "$checkpoint" "$export_dir"

staging="$(mktemp -d)"
trap 'rm -rf "$staging"' EXIT
mkdir -p "$staging/evaluation" "$(dirname "$output")"

cp "$checkpoint" "$staging/"
find "$evaluation_dir" -maxdepth 1 -type f -name '*.json' -exec cp {} "$staging/evaluation/" \;
if [[ -n "$selection_report" ]]; then
  if [[ ! -s "$selection_report" ]]; then
    echo "selection report does not exist or is empty: $selection_report" >&2
    exit 1
  fi
  cp "$selection_report" "$staging/evaluation/"
fi
if [[ -n "$robustness_report" ]]; then
  if [[ ! -s "$robustness_report" ]]; then
    echo "robustness report does not exist or is empty: $robustness_report" >&2
    exit 1
  fi
  cp "$robustness_report" "$staging/evaluation/"
fi
if [[ -n "$mastery_report" ]]; then
  if [[ ! -s "$mastery_report" ]]; then
    echo "mastery report does not exist or is empty: $mastery_report" >&2
    exit 1
  fi
  cp "$mastery_report" "$staging/evaluation/"
fi
if [[ -n "$robustness_mastery_report" ]]; then
  if [[ ! -s "$robustness_mastery_report" ]]; then
    echo "robustness mastery report does not exist or is empty: $robustness_mastery_report" >&2
    exit 1
  fi
  cp "$robustness_mastery_report" "$staging/evaluation/"
fi
if [[ -n "$final_checkpoint" ]]; then
  if [[ ! -s "$final_checkpoint" ]]; then
    echo "final checkpoint does not exist or is empty: $final_checkpoint" >&2
    exit 1
  fi
  if [[ "$(realpath "$final_checkpoint")" != "$(realpath "$checkpoint")" ]]; then
    mkdir -p "$staging/final_training_checkpoint"
    cp "$final_checkpoint" "$staging/final_training_checkpoint/"
  fi
fi
if [[ -d "$evaluation_dir/videos" ]]; then
  cp -a "$evaluation_dir/videos" "$staging/"
fi
if [[ -d "$evaluation_dir/success_videos" ]]; then
  cp -a "$evaluation_dir/success_videos" "$staging/"
fi
if [[ -d "$run_dir/params" ]]; then
  cp -a "$run_dir/params" "$staging/"
fi
cp -a "$export_dir" "$staging/"
if [[ -n "$health_log" ]]; then
  if [[ ! -s "$health_log" ]]; then
    echo "health log does not exist or is empty: $health_log" >&2
    exit 1
  fi
  mkdir -p "$staging/monitoring"
  cp "$health_log" "$staging/monitoring/"
fi

cat >"$staging/README.txt" <<EOF
Booster K1 WBC stand-up validated checkpoint bundle

Checkpoint: $(basename "$checkpoint")
Evaluation: zero lift, terrain level $terrain_level
Summary: evaluation/$(basename "$summary")
Robustness terrain: ${robustness_terrain_level:-not requested}
Robustness summary: ${robustness_summary:+evaluation/$(basename "$robustness_summary")}
Robustness report: ${robustness_report:+evaluation/$(basename "$robustness_report")}
Selection report: ${selection_report:+evaluation/$(basename "$selection_report")}
Mastery report: ${mastery_report:+evaluation/$(basename "$mastery_report")}
Robustness mastery report: ${robustness_mastery_report:+evaluation/$(basename "$robustness_mastery_report")}
Final training checkpoint: ${final_checkpoint:+$(basename "$final_checkpoint")}

The exported/k1_wbc_export_manifest.json file proves that the bundled JIT and
ONNX policies were exported from the selected checkpoint. The
exported/k1_wbc_export_validation.json report records their deterministic
numerical parity check. The evaluation directory contains the per-orientation
metrics and aggregate
summary. The success_videos directory contains one zero-lift rollout per fall
orientation whose matching JSON result proves that rollout recovered. The
optional videos directory contains deterministic batch recordings. The params
and exported directories contain the exact training configuration and deployable
JIT/ONNX policy. The monitoring directory contains the training health history
when supplied. If selection favored an earlier model, final_training_checkpoint
contains the untouched last model from the completed run. Verify file integrity
with SHA256SUMS.
EOF

(
  cd "$staging"
  find . -type f ! -name SHA256SUMS -print0 \
    | sort -z \
    | xargs -0 sha256sum >SHA256SUMS
)

"$python_bin" - "$staging" "$output" <<'PY'
import sys
import zipfile
from pathlib import Path

staging = Path(sys.argv[1])
output = Path(sys.argv[2])
with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(staging.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(staging))

with zipfile.ZipFile(output) as archive:
    corrupt = archive.testzip()
    if corrupt is not None:
        raise RuntimeError(f"corrupt ZIP entry: {corrupt}")
    required = {"README.txt", "SHA256SUMS"}
    names = set(archive.namelist())
    missing = required - names
    if missing:
        raise RuntimeError(f"ZIP is missing required entries: {sorted(missing)}")
PY

echo "[WBC-PACKAGE] output=$output size=$(stat -c %s "$output")"

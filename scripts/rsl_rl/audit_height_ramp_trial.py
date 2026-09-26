"""Read-only TensorBoard comparison; does not launch an evaluation or touch training."""

import argparse
import json
import math
from pathlib import Path
import shutil
import subprocess
import time

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--window", type=int, default=250)
parser.add_argument("--output", type=Path)
args = parser.parse_args()
root = Path(__file__).resolve().parents[2]
manifest = json.loads((root / "logs/height_clock_ramp_20260915/manifest.json").read_text())
run = Path(manifest["run"])
events = EventAccumulator(str(run), size_guidance={"scalars": 0})
events.Reload()
latest = events.Scalars("Curriculum/adaptive_lift")[-1]
# A full 15s episode takes just over 31 iterations (24 policy steps/iteration).
begin = max(manifest["source_iteration"] + 32, latest.step - args.window)
metrics = {}
for tag in ("Curriculum/adaptive_lift", "Metrics/height/height_error",
            "Metrics/height/high_height_error", "Metrics/height/high_height_success",
            "Metrics/height/delta_limit", "Metrics/height/delta_frontier_success",
            "Metrics/height/delta_frontier_error", "Metrics/height/delta_frontier_count",
            "Episode_Termination/invalid_state"):
    values = [v for v in events.Scalars(tag) if v.step > begin]
    if not values:
        continue
    mean = sum(v.value for v in values) / len(values)
    # Old delta outcomes were successful substeps; ramp outcomes require final goals.
    baseline = None if tag.startswith("Metrics/height/delta_") else manifest["baseline"].get(tag)
    metrics[tag] = dict(mean=mean, last=values[-1].value, count=len(values),
                        baseline_mean=baseline["mean"] if baseline else None,
                        change=mean-baseline["mean"] if baseline else None)
deltas = {}
for name in ("ended", "target_arrived", "observed", "success", "censored", "terminated", "timeout", "error_sum", "valid_count"):
    values = [v for v in events.Scalars("Metrics/height/ramp_run_" + name) if v.step >= begin]
    if len(values) >= 2:
        deltas[name] = values[-1].value - values[0].value
rates = {}
if deltas.get("ended", 0) > 0:
    for key in ("target_arrived", "observed", "success", "censored", "terminated", "timeout"):
        rates[key] = deltas[key] / deltas["ended"]
    rates["success_among_observed"] = deltas["success"] / max(deltas["observed"], 1)
    rates["final_frame_error"] = deltas["error_sum"] / max(deltas["valid_count"], 1)
finite = all(math.isfinite(v.value) for tag in events.Tags()["scalars"]
             for v in events.Scalars(tag) if v.step > begin)
result = dict(time=time.time(), run=str(run), iteration=latest.step,
    event_age_s=time.time()-latest.wall_time, start_exclusive=begin,
    metrics=metrics, ramp_deltas=deltas, ramp_rates=rates, all_scalars_finite=finite,
    disk_free_gib=shutil.disk_usage(root).free/2**30,
    gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", "--format=csv,noheader"],text=True).strip(),
    sessions=subprocess.check_output(["tmux", "list-sessions"],text=True).splitlines(),
    caution="Ramp arrival is command progress, not physical success; original-cohort metrics determine improvement. Counter differences use rollout averages.")
if args.output:
    args.output.write_text(json.dumps(result, indent=2))
print(json.dumps(result))

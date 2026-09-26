"""Read-only live-PID audit of the shared-governor trial."""

import json
import math
from pathlib import Path
import shutil
import subprocess
import time

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path(__file__).resolve().parents[2]
state = json.loads((root / "logs/height_mastery_lift_schedule_state_k1_scaled_v5_learnable_std.json").read_text())
assert state["experiment_status"] == "shared_governor_trial"
pid = int(subprocess.check_output(["tmux", "display-message", "-p", "-t", "k1-height-training", "#{pane_pid}"], text=True))
assert state["checkpoint"].encode() in Path(f"/proc/{pid}/cmdline").read_bytes()
files = list((root / "logs/rsl_rl/height_tracking_k1").glob(f"*/events.out.tfevents.*.{pid}.*"))
run = max(files, key=lambda p: p.stat().st_mtime).parent
events = EventAccumulator(str(run), size_guidance={"scalars": 0})
events.Reload()
tags = events.Tags()["scalars"]
last = events.Scalars("Curriculum/adaptive_lift")[-1]
start = max(state["checkpoint_iteration"] + 67, last.step - 250)
metrics = {}
for tag in tags:
    if tag in ("Curriculum/adaptive_lift", "Episode_Termination/invalid_state") or (
            tag.startswith("Metrics/height/") and "governor_" not in tag):
        values = [e.value for e in events.Scalars(tag) if e.step > start]
        if values:
            metrics[tag] = dict(mean=sum(values)/len(values), count=len(values))
outcomes = {}
for group in ("governed", "jump"):
    counts = {}
    for key in ("ended", "observed", "success", "timeout", "terminated", "success_time_sum", "censored"):
        values = [e.value for e in events.Scalars(f"Metrics/height/governor_{group}_{key}") if e.step >= start]
        if len(values) >= 2:
            counts[key] = values[-1] - values[0]
    if counts.get("ended", 0) > 0:
        outcomes[group] = dict(count_deltas=counts,
            success_all_ended=counts["success"]/counts["ended"],
            success_observed=counts["success"]/counts["observed"] if counts["observed"] else None,
            censored_fraction=counts["censored"]/counts["ended"],
            success_time_s=counts["success_time_sum"]/counts["success"] if counts["success"] else None)
report = dict(run=str(run), trainer_pid=pid, iteration=last.step, lift=last.value,
    event_age_s=time.time()-last.wall_time, mature=last.step > start,
    metrics=metrics, outcomes=outcomes,
    finite=all(math.isfinite(e.value) for tag in tags for e in events.Scalars(tag) if e.step > start),
    disk_free_gib=shutil.disk_usage(root).free/2**30,
    gpu=subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
                                 "--format=csv,noheader"], text=True).strip(),
    caution="Counters are rollout averaged; use differences within one process. Both cohorts have 32s episodes; no direct causal comparison to the old 15s baseline.")
output = Path(state["experiment_manifest"]).parent / "monitor_latest.json"
output.write_text(json.dumps(report, indent=2))
print(json.dumps(report))

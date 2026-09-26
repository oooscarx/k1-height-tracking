"""One-time user-authorized switch from capped to sequential height commands."""

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path("/home/bravo/xsb/k1-height-tracking")
OUT = ROOT / "logs/height_staircase_20260915"
STATE = ROOT / "logs/height_mastery_lift_schedule_state_k1_scaled_v5_learnable_std.json"
RUN = ROOT / "logs/rsl_rl/height_tracking_k1/2026-09-14_20-46-40_height_tracking_k1"
if (OUT / "manifest.json").exists():
    raise RuntimeError("trial already prepared/dispatched; inspect before any retry")

source = max(RUN.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
checkpoint = torch.load(source, map_location="cpu", weights_only=False)
iteration = int(checkpoint["iter"])
assert source.name == f"model_{iteration}.pt"


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


assert finite(checkpoint), "checkpoint is nonfinite; trainer has not been stopped"
delta = checkpoint["infos"]["height_delta_curriculum"]
assert delta["limit"] > 0 and delta["steps"] > 0
events = EventAccumulator(str(RUN), size_guidance={"scalars": 0})
events.Reload()
baseline = {}
for tag in ("Curriculum/adaptive_lift", "Metrics/height/height_error",
            "Metrics/height/high_height_error", "Metrics/height/high_height_success"):
    values = [e for e in events.Scalars(tag) if e.step <= iteration][-250:]
    assert len(values) == 250 and all(math.isfinite(e.value) for e in values)
    baseline[tag] = dict(mean=sum(e.value for e in values) / len(values),
                         last=values[-1].value, step=values[-1].step)
lift = baseline["Curriculum/adaptive_lift"]["last"]
ema = baseline["Metrics/height/height_error"]["mean"]
state = json.loads(STATE.read_text())
supervisor_command = json.loads((ROOT / "logs/ppo_kl_early_stop_20260914/manifest.json").read_text())["supervisor_command"]
spec = importlib.util.spec_from_file_location("supervise", ROOT / "scripts/rsl_rl/supervise_k1_height_mastery_lift.py")
supervise = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = supervise
spec.loader.exec_module(supervise)
pid = int(subprocess.check_output(["tmux", "display-message", "-p", "-t", "k1-height-training", "#{pane_pid}"], text=True))
cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
assert str(ROOT / "scripts/rsl_rl/train.py").encode() in cmdline
assert str(RUN.parent / "2026-09-14_19-20-20_height_tracking_k1/model_225000.pt").encode() in cmdline
identity = Path(f"/proc/{pid}/stat").read_text().split()[21]
command = list(state["command"])
for flag, value in (("--resume_path", str(source)), ("--max_iterations", "5000"),
                    ("--wbc_initial_lift_scale", str(lift)), ("--wbc_initial_lift_ema", str(ema))):
    command[command.index(flag) + 1] = value
manifest = dict(status="prepared", source_checkpoint=str(source), source_iteration=iteration,
                checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source_delta_steps=delta["steps"], source_delta_stage=delta["stage"],
                source_delta_limit=delta["limit"], lift=lift, initial_ema=ema,
                command=command, baseline=baseline, expected_end=iteration + 5000,
                reviews=[iteration + 250, iteration + 1000, iteration + 2500, iteration + 5000],
                intervention="sequential intermediate height targets retaining sampled final goal",
                step_tolerance_m=0.04, stable_time_s=0.5, reference_fraction=0.2,
                hold_time_s=[1, 7], tests_passed=40, original_training_pid=pid,
                diagnostic_counters="process lifetime; high mean vs intermittent vs terminated; timeout censored",
                restore_note="model/PPO/delta state restored; lift restored via args, EMA initialized from last250; physics and active command segments reset; existing action-std optimizer reset behavior unchanged")
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "state_before.json").write_text(json.dumps(state, indent=2))
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
subprocess.run(["tmux", "kill-session", "-t", "k1-height-supervisor"], check=True)
os.kill(pid, signal.SIGINT)
for _ in range(90):
    stat = Path(f"/proc/{pid}/stat")
    if not stat.exists() or stat.read_text().split()[21] != identity:
        break
    time.sleep(1)
else:
    raise RuntimeError("trainer did not exit after SIGINT; refusing duplicate launch")
assert not Path(f"/proc/{pid}").exists()
state.update(checkpoint=str(source), checkpoint_iteration=iteration, event_iteration=iteration,
             expected_end=iteration + 5000, command=command, lift_scale=lift, initial_lift_ema=ema,
             experiment_status="sequential_height_staircase_trial",
             experiment_manifest=str(OUT / "manifest.json"), resume_reason="user_authorized_sequential_height_staircase",
             phase="normal", status="launching", task="Booster-K1-Height-Tracking-v0",
             staircase={"enabled": True, "tolerance": .04, "stable_time_s": .5, "preserve_hold_time": True})
supervise._write_state(STATE, state)
supervise._launch_tmux("k1-height-training", ROOT, command)
supervise._launch_monitor(ROOT, "k1-height-monitor", "k1-height-training", "normal", iteration + 5000)
supervise._launch_tmux("k1-height-supervisor", ROOT, supervisor_command)
state["status"] = "training"
supervise._write_state(STATE, state)
manifest["status"] = "dispatched"
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest), flush=True)

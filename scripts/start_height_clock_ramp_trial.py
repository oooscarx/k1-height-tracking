"""One-time guarded switch to policy-independent height-command ramps."""

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
OUT = ROOT / "logs/height_clock_ramp_20260915"
STATE = ROOT / "logs/height_mastery_lift_schedule_state_k1_scaled_v5_learnable_std.json"
RUN = ROOT / "logs/rsl_rl/height_tracking_k1/2026-09-15_14-08-56_height_tracking_k1"
if (OUT / "manifest.json").exists():
    raise RuntimeError("trial already prepared; inspect state before retrying")
state = json.loads(STATE.read_text())
assert state["experiment_status"] == "staircase_accounting_v2_trial"
source = max(RUN.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
assert time.time() - source.stat().st_mtime > 10, "checkpoint may still be saving"
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


assert finite(checkpoint), "nonfinite checkpoint; training has not been stopped"
delta = checkpoint["infos"]["height_delta_curriculum"]
assert checkpoint["infos"]["height_staircase_accounting"]["version"] == 2
events = EventAccumulator(str(RUN), size_guidance={"scalars": 0})
events.Reload()
baseline = {}
for tag in ("Curriculum/adaptive_lift", "Metrics/height/height_error",
            "Metrics/height/high_height_error", "Metrics/height/high_height_success",
            "Metrics/height/delta_frontier_success", "Episode_Termination/invalid_state"):
    values = [e for e in events.Scalars(tag) if e.step <= iteration][-250:]
    assert len(values) == 250 and all(math.isfinite(e.value) for e in values)
    baseline[tag] = dict(mean=sum(e.value for e in values) / len(values),
                         last=values[-1].value, step=values[-1].step)
lift = baseline["Curriculum/adaptive_lift"]["last"]
ema = baseline["Metrics/height/height_error"]["mean"]


def difference(name):
    values = [e for e in events.Scalars("Metrics/height/stair_run_" + name) if e.step <= iteration][-251:]
    return values[-1].value - values[0].value


ended = difference("end_total")
diagnosis = {name: difference("end_" + name) / ended for name in
             ("no_advance", "midway", "final_unsettled", "final_success", "resample", "timeout")}


def reward(error, settled):
    scale = .72 / .92
    return sum(w * math.exp(-(error / (std * scale)) ** 2)
               for w, std in ((4, .5), (8, .3), (16, .2))) + (
                   8 * math.exp(-(error / (.2 * scale)) ** 2) if settled else 0)


diagnosis["height_reward_rate_counterexample"] = dict(
    hovering_4p1cm=reward(.041, True), before_success_3p9cm=reward(.039, True),
    after_success_target_plus_12cm=reward(.159, False),
    note="height reward components only, before dt scaling; not a measured total-policy return")
supervisor_command = json.loads((ROOT / "logs/ppo_kl_early_stop_20260914/manifest.json").read_text())["supervisor_command"]
spec = importlib.util.spec_from_file_location("supervise", ROOT / "scripts/rsl_rl/supervise_k1_height_mastery_lift.py")
supervise = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = supervise
spec.loader.exec_module(supervise)
pid = int(subprocess.check_output(["tmux", "display-message", "-p", "-t", "k1-height-training", "#{pane_pid}"], text=True))
cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
assert str(ROOT / "scripts/rsl_rl/train.py").encode() in cmdline
assert state["checkpoint"].encode() in cmdline
identity = Path(f"/proc/{pid}/stat").read_text().split()[21]
command = list(state["command"])
for flag, value in (("--resume_path", str(source)), ("--max_iterations", "5000"),
                    ("--wbc_initial_lift_scale", str(lift)), ("--wbc_initial_lift_ema", str(ema))):
    command[command.index(flag) + 1] = value
manifest = dict(status="prepared", source_checkpoint=str(source), source_iteration=iteration,
    checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), source_delta_steps=delta["steps"],
    source_delta_stage=delta["stage"], source_delta_limit=delta["limit"], lift=lift, initial_ema=ema,
    command=command, baseline=baseline, diagnosis=diagnosis, expected_end=iteration + 5000,
    reviews=[iteration + 250, iteration + 1000, iteration + 2500, iteration + 5000],
    intervention="Replace success-contingent staircase with clock-driven upward command ramp",
    ramp_step_duration_s=.5, initial_ramp_speed_m_s=delta["limit"] / .5,
    reference_fraction=.2, hold_time_s=[1, 7], tests_passed=54, original_training_pid=pid,
    restore_note="model/PPO and delta limit/stage/clock restored; incompatible outcome window cleared; lift inherited via args; EMA initialized from last250; physics reset and existing std optimizer reset unchanged",
    acceptance="Original reference height error and lift must improve; ramp target-arrival is not robot success")
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
    raise RuntimeError("old trainer did not exit; refusing duplicate launch")
assert not Path(f"/proc/{pid}").exists()
state.update(checkpoint=str(source), checkpoint_iteration=iteration, event_iteration=iteration,
    expected_end=iteration + 5000, command=command, lift_scale=lift, initial_lift_ema=ema,
    experiment_status="clock_ramp_trial", experiment_manifest=str(OUT / "manifest.json"),
    resume_reason="user_authorized_fix_success_contingent_command_reward_conflict",
    phase="normal", status="launching", task="Booster-K1-Height-Tracking-v0",
    staircase={"enabled": False},
    ramp={"enabled": True, "step_duration_s": .5, "initial_speed_m_s": delta["limit"] / .5,
          "delta_meaning": "command increase per 0.5s, not a bound on physical tracking error",
          "success": "final held target, 2s settle then >=1s observations, >=90% within8cm"})
supervise._write_state(STATE, state)
supervise._launch_tmux("k1-height-training", ROOT, command)
supervise._launch_monitor(ROOT, "k1-height-monitor", "k1-height-training", "normal", iteration + 5000)
supervise._launch_tmux("k1-height-supervisor", ROOT, supervisor_command)
state["status"] = "training"
supervise._write_state(STATE, state)
manifest["status"] = "dispatched"
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest), flush=True)

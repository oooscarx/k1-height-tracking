"""Guarded restart for the authorized governed-8cm / jump-12cm lift gate."""

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = Path("/home/bravo/xsb/k1-height-tracking")
OUT = ROOT / "logs/height_dual_gate_20260917"
STATE = ROOT / "logs/height_mastery_lift_schedule_state_k1_scaled_v5_learnable_std.json"
assert not (OUT / "manifest.json").exists(), "Already prepared; inspect before retrying"
candidate = OUT / "candidate"
files = [
    "source/booster_train/booster_train/tasks/manager_based/height_tracking/commands.py",
    "source/booster_train/booster_train/tasks/manager_based/height_tracking/curriculums.py",
    "source/booster_train/booster_train/tasks/manager_based/height_tracking/robots/k1/env_cfg.py",
    "tests/test_height_governor.py", "tests/test_height_tracking_curriculum.py",
    "tests/test_k1_height_tracking.py",
]
tests = ["tests/test_height_governor.py", "tests/test_height_tracking_curriculum.py",
         "tests/test_k1_height_tracking.py", "tests/test_checkpoint_loading.py",
         "tests/test_monitor_k1_height_tracking.py", "tests/test_supervise_k1_height_extension.py"]
for rel in files:
    compile((candidate / rel).read_text(), rel, "exec")
preflight = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests"],
                           cwd=candidate, capture_output=True, text=True)
(OUT / "preflight.txt").write_text(preflight.stdout + preflight.stderr)
assert preflight.returncode == 0, preflight.stdout + preflight.stderr
state = json.loads(STATE.read_text())
assert state["experiment_status"] == "shared_governor_trial"


def pane_pid(session):
    return int(subprocess.check_output(
        ["tmux", "display-message", "-p", "-t", session, "#{pane_pid}"], text=True))


pid = pane_pid("k1-height-training")
cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
assert state["checkpoint"].encode() in cmdline and b"scripts/rsl_rl/train.py" in cmdline
identity = Path(f"/proc/{pid}/stat").read_text().split()[21]
supervisor_pid = pane_pid("k1-height-supervisor")
supervisor_command = Path(f"/proc/{supervisor_pid}/cmdline").read_bytes().decode().strip("\0").split("\0")
assert any("supervise_k1_height_mastery_lift.py" in x for x in supervisor_command)
event_file = max((ROOT / "logs/rsl_rl/height_tracking_k1").glob(f"*/events.out.tfevents.*.{pid}.*"),
                 key=lambda p: p.stat().st_mtime)
run = event_file.parent
source = max((p for p in run.glob("model_*.pt") if time.time() - p.stat().st_mtime > 15),
             key=lambda p: int(p.stem.split("_")[-1]))
checkpoint = torch.load(source, map_location="cpu", weights_only=False)


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


assert finite(checkpoint)
iteration = checkpoint["iter"]
assert source.name == f"model_{iteration}.pt"
events = EventAccumulator(str(run), size_guidance={"scalars": 0})
events.Reload()
baseline = {}
for tag in ("Curriculum/adaptive_lift", "Metrics/height/height_error",
            "Metrics/height/governed_height_error"):
    values = [e for e in events.Scalars(tag) if e.step <= iteration][-250:]
    assert len(values) == 250 and all(math.isfinite(e.value) for e in values)
    baseline[tag] = {"mean": sum(e.value for e in values) / len(values), "last": values[-1].value}
lift = baseline["Curriculum/adaptive_lift"]["last"]
ema = baseline["Metrics/height/height_error"]["mean"]
command = list(state["command"])
for flag, value in (("--resume_path", str(source)), ("--max_iterations", "5000"),
                    ("--wbc_initial_lift_scale", str(lift)), ("--wbc_initial_lift_ema", str(ema))):
    command[command.index(flag) + 1] = value
hashes = {}
for rel in files:
    backup = OUT / "before" / rel
    backup.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / rel, backup)
    hashes[rel] = {"before": hashlib.sha256(backup.read_bytes()).hexdigest(),
                   "after": hashlib.sha256((candidate / rel).read_bytes()).hexdigest()}
(OUT / "state_before.json").write_text(json.dumps(state, indent=2))
manifest = dict(status="prepared", source_checkpoint=str(source), source_iteration=iteration,
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), checkpoint_finite=True,
    previous_run=str(run), baseline=baseline, lift=lift, initial_ema=ema,
    command=command, supervisor_command=supervisor_command,
    expected_end=iteration+5000, hashes=hashes,
    thresholds={"governed": .08, "jump": .12, "operator": "AND, strict less-than"},
    resume_note="Lift inherited. Jump EMA seeded from last250; governed EMA from first valid post-warmup reset. Physics and in-flight goals reset as before.")


def save():
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))


save()
subprocess.run(["tmux", "kill-session", "-t", "=k1-height-supervisor"], check=True)
os.kill(pid, signal.SIGINT)
for _ in range(120):
    stat = Path(f"/proc/{pid}/stat")
    if not stat.exists() or stat.read_text().split()[21] != identity:
        break
    time.sleep(1)
else:
    raise RuntimeError("Trainer still alive; refusing duplicate launch")
assert not Path(f"/proc/{pid}").exists()
subprocess.run(["tmux", "kill-session", "-t", "=k1-height-monitor"], check=True)
for rel in files:
    shutil.copy2(candidate / rel, ROOT / rel)
test = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests],
                      cwd=ROOT, capture_output=True, text=True)
(OUT / "tests.txt").write_text(test.stdout + test.stderr)
assert test.returncode == 0, test.stdout + test.stderr
manifest["tests"] = test.stdout.strip()
spec = importlib.util.spec_from_file_location("supervise", ROOT / "scripts/rsl_rl/supervise_k1_height_mastery_lift.py")
supervise = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = supervise
spec.loader.exec_module(supervise)
state.update(command=command, checkpoint=str(source), checkpoint_iteration=iteration,
    event_iteration=iteration, expected_end=iteration+5000, lift_scale=lift, initial_lift_ema=ema,
    experiment_manifest=str(OUT / "manifest.json"), lift_gate_thresholds=manifest["thresholds"],
    resume_reason="user_authorized_dual_cohort_lift_gate", status="launching")
state["governor"]["lift_input"] = "governed EMA < 0.08 AND jump EMA < 0.12; decay only on valid jump resets"
supervise._write_state(STATE, state)
supervise._launch_tmux("k1-height-training", ROOT, command)
supervise._launch_monitor(ROOT, "k1-height-monitor", "k1-height-training", "normal", iteration+5000)
supervise._launch_tmux("k1-height-supervisor", ROOT, supervisor_command)
state["status"] = "training"
supervise._write_state(STATE, state)
manifest["status"] = "dispatched"
save()
print(json.dumps(manifest), flush=True)

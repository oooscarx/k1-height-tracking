"""Guarded launch of the user-authorized 80/20 shared-governor experiment."""

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
OUT = ROOT / "logs/height_governor_20260917"
STATE = ROOT / "logs/height_mastery_lift_schedule_state_k1_scaled_v5_learnable_std.json"
if (OUT / "manifest.json").exists():
    raise RuntimeError("Trial already prepared; inspect before retrying")
candidate = OUT / "candidate"
candidate.mkdir(parents=True, exist_ok=True)
subprocess.run(["tar", "-xzf", "/tmp/k1-governor-candidate-20260917.tar.gz", "-C", str(candidate)], check=True)
files = [p for p in candidate.rglob("*.py") if not p.name.startswith("._")]
assert len(files) == 8
for path in files:
    compile(path.read_text(), str(path), "exec")
pretest = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_height_governor.py",
                          "-k", "not ppo_remains_full_update"], cwd=candidate, capture_output=True, text=True)
(OUT / "preflight_tests.txt").write_text(pretest.stdout + pretest.stderr)
assert pretest.returncode == 0, pretest.stdout + pretest.stderr
state = json.loads(STATE.read_text())
assert state["experiment_status"] == "baseline_224000_restored"
pid = int(subprocess.check_output(["tmux", "display-message", "-p", "-t", "k1-height-training", "#{pane_pid}"], text=True))
cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
assert state["checkpoint"].encode() in cmdline and b"scripts/rsl_rl/train.py" in cmdline
identity = Path(f"/proc/{pid}/stat").read_text().split()[21]
events_paths = list((ROOT / "logs/rsl_rl/height_tracking_k1").glob(f"*/events.out.tfevents.*.{pid}.*"))
run = max(events_paths, key=lambda p: p.stat().st_mtime).parent
source = max(run.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
assert time.time() - source.stat().st_mtime > 15
checkpoint = torch.load(source, map_location="cpu", weights_only=False)


def finite(x):
    if torch.is_tensor(x):
        return bool(torch.isfinite(x).all())
    if isinstance(x, dict):
        return all(finite(v) for v in x.values())
    if isinstance(x, (tuple, list)):
        return all(finite(v) for v in x)
    return not isinstance(x, float) or math.isfinite(x)


assert finite(checkpoint)
iteration = checkpoint["iter"]
assert source.name == f"model_{iteration}.pt"
e = EventAccumulator(str(run), size_guidance={"scalars": 0})
e.Reload()
baseline = {}
for tag in ("Curriculum/adaptive_lift", "Metrics/height/height_error",
            "Metrics/height/high_height_error", "Metrics/height/high_height_success"):
    values = [x for x in e.Scalars(tag) if x.step <= iteration][-750:]
    assert len(values) == 750 and all(math.isfinite(x.value) for x in values)
    baseline[tag] = dict(mean=sum(x.value for x in values)/len(values), last=values[-1].value,
                         step=values[-1].step, count=len(values))
lift = baseline["Curriculum/adaptive_lift"]["last"]
ema = baseline["Metrics/height/height_error"]["mean"]
command = list(state["command"])
for flag, value in (("--resume_path", str(source)), ("--max_iterations", "5000"),
                    ("--wbc_initial_lift_scale", str(lift)), ("--wbc_initial_lift_ema", str(ema))):
    command[command.index(flag)+1] = value
before_hashes = {}
after_hashes = {}
for path in files:
    rel = path.relative_to(candidate)
    target = ROOT / rel
    if target.exists():
        backup = OUT / "before" / rel
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(target, backup)
        before_hashes[str(rel)] = hashlib.sha256(target.read_bytes()).hexdigest()
    after_hashes[str(rel)] = hashlib.sha256(path.read_bytes()).hexdigest()
manifest = dict(status="prepared", source_checkpoint=str(source), source_iteration=iteration,
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), checkpoint_finite=True,
    previous_run=str(run), original_training_pid=pid, baseline=baseline, lift=lift, initial_ema=ema,
    command=command, expected_end=iteration+5000, before_hashes=before_hashes, after_hashes=after_hashes,
    reviews=[iteration+250, iteration+1000, iteration+3000, iteration+5000],
    governor=dict(upward_step_m=.12, stage_hold_s=4., reference_fraction=.2, episode_s=32.,
                  progression="fixed clock, never success-triggered", reward_age="whole request",
                  metric_age="current intermediate target", lift_input="jump cohort only with sample-count EMA",
                  maximum_request_duration_s=27.),
    resume_note="Policy/optimizer/normalizers resume normally; lift explicitly inherited; EMA seeded from last750; physics, in-flight requests and process counters reset.")
(OUT / "state_before.json").write_text(json.dumps(state, indent=2))


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
manifest["status"] = "stopped_for_install"
save()
for path in files:
    target = ROOT / path.relative_to(candidate)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
test = subprocess.run([sys.executable, "-m", "pytest", "-q", "tests/test_height_governor.py",
                       "tests/test_k1_height_tracking.py", "tests/test_height_tracking_curriculum.py",
                       "tests/test_checkpoint_loading.py", "tests/test_monitor_k1_height_tracking.py",
                       "tests/test_supervise_k1_height_extension.py"], cwd=ROOT, capture_output=True, text=True)
(OUT / "tests.txt").write_text(test.stdout + test.stderr)
assert test.returncode == 0, test.stdout + test.stderr
manifest["tests"] = test.stdout.strip()
state.update(command=command, checkpoint=str(source), checkpoint_iteration=iteration,
    event_iteration=iteration, expected_end=iteration+5000, lift_scale=lift, initial_lift_ema=ema,
    experiment_status="shared_governor_trial", experiment_manifest=str(OUT / "manifest.json"),
    resume_reason="user_authorized_shared_governor_with_20pct_jump_exploration", status="launching",
    governor=manifest["governor"])
spec = importlib.util.spec_from_file_location("supervise", ROOT / "scripts/rsl_rl/supervise_k1_height_mastery_lift.py")
supervise = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = supervise
spec.loader.exec_module(supervise)
supervise._write_state(STATE, state)
supervise._launch_tmux("k1-height-training", ROOT, command)
supervise._launch_monitor(ROOT, "k1-height-monitor", "k1-height-training", "normal", iteration+5000)
original = json.loads((ROOT / "logs/exploration_std008_20260914T030010Z/manifest.json").read_text())
supervise._launch_tmux("k1-height-supervisor", ROOT, original["supervisor_command"])
state["status"] = "training"
supervise._write_state(STATE, state)
manifest["status"] = "dispatched"
save()
print(json.dumps(manifest), flush=True)

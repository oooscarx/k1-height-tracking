"""Guarded, one-time restoration of the user-selected pre-acceleration baseline."""

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
OUT = ROOT / "logs/height_baseline_restore_20260916"
STATE = ROOT / "logs/height_mastery_lift_schedule_state_k1_scaled_v5_learnable_std.json"
BASE = ROOT / "logs/exploration_std008_20260914T030010Z"
if (OUT / "manifest.json").exists():
    raise RuntimeError("Restoration already prepared; inspect it before retrying")
original = json.loads((BASE / "manifest.json").read_text())
state = json.loads(STATE.read_text())
assert state["experiment_status"] == "clock_ramp_trial"
source = Path(original["checkpoint"])
checkpoint = torch.load(source, map_location="cpu", weights_only=False)


def finite(value):
    if torch.is_tensor(value):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


assert checkpoint["iter"] == 224000 and finite(checkpoint)
assert digest(source) == digest(BASE / "model_224000.pt")
events = EventAccumulator(str(source.parent), size_guidance={"scalars": 0})
events.Reload()
lift_samples = [v for v in events.Scalars("Curriculum/adaptive_lift") if v.step <= 224000]
assert lift_samples[-1].step == 224000
lift = lift_samples[-1].value
assert lift == original["initial_lift"]
ema = original["initial_ema"]
back = ROOT / "logs/height_progress_sampling_20260914/before"
restores = {str(p.relative_to(back)): p for p in back.rglob("*.py")}
kl_back = ROOT / "logs/ppo_kl_early_stop_20260914/before"
for rel in (
    "third_party/wbc_agile_rsl_rl/rsl_rl/algorithms/ppo.py",
    "third_party/wbc_agile_rsl_rl/rsl_rl/runners/on_policy_runner.py",
    "source/booster_train/booster_train/tasks/manager_based/fall_recovery/wbc_stand_up/rl.py",
):
    restores[rel] = kl_back / rel
assert len(restores) == 8
for rel, path in restores.items():
    compile(path.read_text(), rel, "exec")
task = "source/booster_train/booster_train/tasks/manager_based/height_tracking/"
assert "minimum_action_std: float | None = 0.04" in restores[task + "robots/k1/ppo_cfg.py"].read_text()
assert "kl_early_stop" not in restores[task + "robots/k1/ppo_cfg.py"].read_text()
assert "delta_curriculum" not in restores[task + "commands.py"].read_text()
pid = int(subprocess.check_output(
    ["tmux", "display-message", "-p", "-t", "k1-height-training", "#{pane_pid}"], text=True))
cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
assert str(ROOT / "scripts/rsl_rl/train.py").encode() in cmdline
assert state["checkpoint"].encode() in cmdline
identity = Path(f"/proc/{pid}/stat").read_text().split()[21]
command = list(original["command"])
assert command[command.index("--resume_path") + 1] == str(source)
assert command[command.index("--wbc_initial_lift_scale") + 1] == str(lift)
OUT.mkdir(parents=True, exist_ok=True)
manifest = dict(status="prepared", source_checkpoint=str(source), source_iteration=224000,
    checkpoint_sha256=digest(source), checkpoint_all_finite=True,
    lift=lift, initial_ema=ema, baseline=original["baseline"], command=command,
    expected_end=229000, reviews=[224250, 225000, 227000, 229000],
    previous_state=state, original_training_pid=pid,
    before_sha256={rel: digest(ROOT / rel) for rel in restores},
    restored_sha256={rel: digest(path) for rel, path in restores.items()},
    restoration="Original command distribution in all environments; original lift EMA; full PPO epochs; std floor 0.04",
    resume_caveat="Checkpoint weights/optimizer/normalizers restored using baseline loader. Exact live EMA/physics/RNG were not saved; EMA seeded from original last1000 mean; baseline std-optimizer reset retained.")


def save():
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))


save()
shutil.copy2(STATE, OUT / "state_before.json")
for rel in restores:
    target = OUT / "before" / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / rel, target)
subprocess.run(["tmux", "kill-session", "-t", "k1-height-supervisor"], check=True)
os.kill(pid, signal.SIGINT)
for _ in range(120):
    stat = Path(f"/proc/{pid}/stat")
    if not stat.exists() or stat.read_text().split()[21] != identity:
        break
    time.sleep(1)
else:
    raise RuntimeError("Trainer has not exited; refusing restoration or duplicate launch")
assert not Path(f"/proc/{pid}").exists()
subprocess.run(["tmux", "kill-session", "-t", "k1-height-monitor"], check=True)
manifest["status"] = "stopped_for_restore"
save()
for rel, archived in restores.items():
    # Restore verified archived files verbatim, rather than reconstruct old behavior.
    shutil.copy2(archived, ROOT / rel)
    assert digest(ROOT / rel) == manifest["restored_sha256"][rel]
result = subprocess.run(
    [str(ROOT / ".venv/bin/python3"), "-m", "pytest", "-q",
     "tests/test_k1_height_tracking.py", "tests/test_height_tracking_curriculum.py",
     "tests/test_checkpoint_loading.py", "tests/test_monitor_k1_height_tracking.py",
     "tests/test_supervise_k1_height_extension.py"],
    cwd=ROOT, capture_output=True, text=True)
(OUT / "tests.txt").write_text(result.stdout + result.stderr)
manifest["tests_exit_code"] = result.returncode
save()
if result.returncode:
    raise RuntimeError("Baseline regression tests failed; training remains stopped: " + result.stdout[-4000:])
restored_state = json.loads((ROOT / "logs/height_progress_sampling_20260914/state_before.json").read_text())
restored_state.update(command=command, checkpoint=str(source), checkpoint_iteration=224000,
    event_iteration=224000, expected_end=229000, lift_scale=lift, initial_lift_ema=ema,
    experiment_status="baseline_224000_restored", experiment_manifest=str(OUT / "manifest.json"),
    resume_reason="user_authorized_rollback_all_acceleration_trials", status="launching")
spec = importlib.util.spec_from_file_location("supervise", ROOT / "scripts/rsl_rl/supervise_k1_height_mastery_lift.py")
supervise = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = supervise
spec.loader.exec_module(supervise)
supervise._write_state(STATE, restored_state)
supervise._launch_tmux("k1-height-training", ROOT, command)
supervise._launch_monitor(ROOT, "k1-height-monitor", "k1-height-training", "normal", 229000)
supervise._launch_tmux("k1-height-supervisor", ROOT, original["supervisor_command"])
restored_state["status"] = "training"
supervise._write_state(STATE, restored_state)
manifest["status"] = "dispatched"
save()
print(json.dumps(manifest), flush=True)

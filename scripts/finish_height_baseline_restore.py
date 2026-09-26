"""Finish a prepared restoration after its regression tests have been updated."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path("/home/bravo/xsb/k1-height-tracking")
OUT = ROOT / "logs/height_baseline_restore_20260916"
STATE = ROOT / "logs/height_mastery_lift_schedule_state_k1_scaled_v5_learnable_std.json"
manifest = json.loads((OUT / "manifest.json").read_text())
assert manifest["status"] == "stopped_for_restore"
for session in ("k1-height-training", "k1-height-supervisor", "k1-height-monitor"):
    assert subprocess.run(["tmux", "has-session", "-t", session], capture_output=True).returncode != 0
for rel, expected in manifest["restored_sha256"].items():
    assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == expected, rel
source = Path(manifest["source_checkpoint"])
assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["checkpoint_sha256"]
result = subprocess.run(
    [str(ROOT / ".venv/bin/python3"), "-m", "pytest", "-q",
     "tests/test_k1_height_tracking.py", "tests/test_height_tracking_curriculum.py",
     "tests/test_checkpoint_loading.py", "tests/test_monitor_k1_height_tracking.py",
     "tests/test_supervise_k1_height_extension.py"], cwd=ROOT, capture_output=True, text=True)
(OUT / "tests_restored_baseline.txt").write_text(result.stdout + result.stderr)
assert result.returncode == 0, result.stdout + result.stderr
manifest["tests_exit_code"] = 0
manifest["tests"] = result.stdout.strip()
state = json.loads((ROOT / "logs/height_progress_sampling_20260914/state_before.json").read_text())
state.update(command=manifest["command"], checkpoint=str(source), checkpoint_iteration=224000,
    event_iteration=224000, expected_end=229000, lift_scale=manifest["lift"],
    initial_lift_ema=manifest["initial_ema"],
    experiment_status="baseline_224000_restored", experiment_manifest=str(OUT / "manifest.json"),
    resume_reason="user_authorized_rollback_all_acceleration_trials", status="launching")
spec = importlib.util.spec_from_file_location("supervise", ROOT / "scripts/rsl_rl/supervise_k1_height_mastery_lift.py")
supervise = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = supervise
spec.loader.exec_module(supervise)
supervise._write_state(STATE, state)
supervise._launch_tmux("k1-height-training", ROOT, manifest["command"])
supervise._launch_monitor(ROOT, "k1-height-monitor", "k1-height-training", "normal", 229000)
original = json.loads((ROOT / "logs/exploration_std008_20260914T030010Z/manifest.json").read_text())
supervise._launch_tmux("k1-height-supervisor", ROOT, original["supervisor_command"])
state["status"] = "training"
supervise._write_state(STATE, state)
manifest["status"] = "dispatched"
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps({"status": manifest["status"], "tests": manifest["tests"],
                  "checkpoint": str(source), "lift": manifest["lift"]}), flush=True)

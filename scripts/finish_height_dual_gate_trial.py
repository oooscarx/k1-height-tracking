"""Finish a prepared dual-gate restart after its pre-launch tests pass."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path("/home/bravo/xsb/k1-height-tracking")
OUT = ROOT / "logs/height_dual_gate_20260917"
STATE = ROOT / "logs/height_mastery_lift_schedule_state_k1_scaled_v5_learnable_std.json"
manifest = json.loads((OUT / "manifest.json").read_text())
assert manifest["status"] == "prepared"
for path in Path("/proc").glob("[0-9]*/cmdline"):
    try:
        command = path.read_bytes().split(b"\0")
    except (FileNotFoundError, PermissionError):
        continue
    assert not any(x.endswith(b"scripts/rsl_rl/train.py") for x in command) or not any(
        str(ROOT).encode() in x for x in command), "Trainer already running"
for rel, hashes in manifest["hashes"].items():
    assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == hashes["after"]
source = Path(manifest["source_checkpoint"])
assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_sha256"]
tests = ["tests/test_height_governor.py", "tests/test_height_tracking_curriculum.py",
         "tests/test_k1_height_tracking.py", "tests/test_checkpoint_loading.py",
         "tests/test_monitor_k1_height_tracking.py", "tests/test_supervise_k1_height_extension.py"]
result = subprocess.run([sys.executable, "-m", "pytest", "-q", *tests],
                        cwd=ROOT, capture_output=True, text=True)
(OUT / "tests.txt").write_text(result.stdout + result.stderr)
assert result.returncode == 0, result.stdout + result.stderr
state = json.loads(STATE.read_text())
assert state["experiment_status"] == "shared_governor_trial"
spec = importlib.util.spec_from_file_location("supervise", ROOT / "scripts/rsl_rl/supervise_k1_height_mastery_lift.py")
supervise = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = supervise
spec.loader.exec_module(supervise)
supervisor_command = manifest.get("supervisor_command")
if supervisor_command is None:
    supervisor_command = json.loads((ROOT / "logs/exploration_std008_20260914T030010Z/manifest.json").read_text())["supervisor_command"]
state.update(command=manifest["command"], checkpoint=str(source),
    checkpoint_iteration=manifest["source_iteration"], event_iteration=manifest["source_iteration"],
    expected_end=manifest["expected_end"], lift_scale=manifest["lift"], initial_lift_ema=manifest["initial_ema"],
    experiment_manifest=str(OUT / "manifest.json"), lift_gate_thresholds=manifest["thresholds"],
    resume_reason="user_authorized_dual_cohort_lift_gate", status="launching")
state["governor"]["lift_input"] = "governed EMA < 0.08 AND jump EMA < 0.12; decay only on valid jump resets"
supervise._write_state(STATE, state)
supervise._launch_tmux("k1-height-training", ROOT, manifest["command"])
supervise._launch_monitor(ROOT, "k1-height-monitor", "k1-height-training", "normal", manifest["expected_end"])
supervise._launch_tmux("k1-height-supervisor", ROOT, supervisor_command)
state["status"] = "training"
supervise._write_state(STATE, state)
manifest.update(status="dispatched", tests=result.stdout.strip(), supervisor_command=supervisor_command,
                test_fix="Old substring assertion matched 0.12 as 0.1; match full 0.1 literal instead.")
(OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest), flush=True)

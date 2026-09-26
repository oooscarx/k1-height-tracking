#!/usr/bin/env python3
"""Write sparse, machine-readable health snapshots for K1 height tracking."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

DEFAULT_TAGS = {
    "mean_reward": "Train/mean_reward",
    "mean_episode_length": "Train/mean_episode_length",
    "height_error": "Metrics/height/height_error",
    "high_height_error": "Metrics/height/high_height_error",
    "high_height_success": "Metrics/height/high_height_success",
    "lift_scale": "Curriculum/adaptive_lift",
    "terrain_level": "Curriculum/terrain_levels",
    "disturbance_scale": "Curriculum/disturbance_scale",
    "disturbance_jump_error_ema": "Metrics/height/disturbance_jump_error_ema",
    "disturbance_governed_error_ema": "Metrics/height/disturbance_governed_error_ema",
    "disturbance_gate_ready": "Metrics/height/disturbance_gate_ready",
    "disturbance_gate_passed": "Metrics/height/disturbance_gate_passed",
    "invalid_state": "Episode_Termination/invalid_state",
    "timeouts": "Episode_Termination/time_out",
    "entropy_loss": "Loss/entropy",
    "value_loss": "Loss/value_function",
    "policy_std": "Policy/mean_noise_std",
    "fps": "Perf/total_fps",
}
REFERENCE_NUM_ENVS = 4096
INVALID_STATE_WARNING_PER_REFERENCE_ENVS = 0.05


def latest_event_file(log_root: Path) -> Path | None:
    files = list(log_root.glob("**/events.out.tfevents.*"))
    return max(files, key=lambda path: path.stat().st_mtime) if files else None


def _recent_nonfinite_scalars(
    series_by_tag: dict[str, list],
    last_step: int,
    window: int = 250,
) -> list[dict[str, int | str]]:
    cutoff = max(0, last_step - window + 1)
    issues = [
        {"step": int(event.step), "tag": tag}
        for tag, events in series_by_tag.items()
        for event in events
        if event.step >= cutoff and not math.isfinite(float(event.value))
    ]
    return sorted(issues, key=lambda issue: (issue["step"], issue["tag"]))


def _recent_records(output: Path, count: int = 8) -> list[dict]:
    if not output.is_file():
        return []
    records = []
    for line in output.read_text(encoding="utf-8").splitlines()[-count:]:
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return records


def _training_session_alive(session: str) -> bool:
    result = subprocess.run(
        ["tmux", "list-panes", "-t", session, "-F", "#{pane_dead}"],
        capture_output=True,
        check=False,
        text=True,
        timeout=5,
    )
    return result.returncode == 0 and any(
        line.strip() == "0" for line in result.stdout.splitlines()
    )


def _training_num_envs(session: str) -> int | None:
    try:
        result = subprocess.run(
            ["tmux", "list-panes", "-t", session, "-F", "#{pane_pid}"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            cmdline = Path(f"/proc/{int(line.strip())}/cmdline").read_bytes().split(b"\0")
            args = [part.decode() for part in cmdline if part]
            for index, arg in enumerate(args):
                if arg == "--num_envs" and index + 1 < len(args):
                    return int(args[index + 1])
                if arg.startswith("--num_envs="):
                    return int(arg.partition("=")[2])
                if arg.startswith("NUM_ENVS_PER_GPU="):
                    return int(arg.partition("=")[2])
    except (FileNotFoundError, OSError, subprocess.SubprocessError, UnicodeDecodeError, ValueError):
        return None
    return None


def _gpu_process_memory_mib() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
        processes: dict[tuple[str, str], int] = {}
        for line in result.stdout.splitlines():
            if not line.strip():
                continue
            pid, gpu_uuid, used_memory = (part.strip() for part in line.split(",", 2))
            key = (pid, gpu_uuid)
            processes[key] = max(processes.get(key, 0), int(used_memory))
        return sum(processes.values())
    except (FileNotFoundError, subprocess.SubprocessError, ValueError):
        return None


def _gpu_count() -> int | None:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
        return sum(1 for line in result.stdout.splitlines() if line.strip())
    except (FileNotFoundError, subprocess.SubprocessError):
        return None


def _add_issue(result: dict, status: str, reason: str) -> None:
    severity = {"ok": 0, "waiting": 0, "complete": 0, "warning": 1, "critical": 2}
    if severity.get(status, 0) > severity.get(result["status"], 0):
        result["status"] = status
    result.setdefault("reasons", []).append(reason)


def _mark_complete_if_healthy(result: dict) -> None:
    if result["status"] == "ok":
        result["status"] = "complete"


def _same_run(records: list[dict], current: dict) -> list[dict]:
    return [record for record in records if record.get("run") == current.get("run")]


def _invalid_state_per_reference_envs(record: dict, fallback_num_envs: int) -> float:
    normalized = record.get("invalid_state_per_4096_envs")
    if normalized is not None:
        return float(normalized)
    value = float(record.get("invalid_state") or 0.0)
    num_envs = int(record.get("num_envs") or fallback_num_envs)
    return value * REFERENCE_NUM_ENVS / num_envs


def _add_training_quality_issues(result: dict, history: list[dict]) -> None:
    if len(history) < 3:
        return
    recent = history[-3:]
    if all(
        record.get("mean_reward") is not None and record["mean_reward"] < -500.0
        for record in recent
    ):
        _add_issue(result, "warning", "mean reward stayed below -500 for three samples")
    if all(
        record.get("height_error") is not None and record["height_error"] > 0.25
        for record in recent
    ):
        _add_issue(result, "warning", "height error stayed above 0.25 for three samples")
    current_num_envs = int(result.get("num_envs") or REFERENCE_NUM_ENVS)
    if all(
        _invalid_state_per_reference_envs(record, current_num_envs)
        > INVALID_STATE_WARNING_PER_REFERENCE_ENVS
        for record in recent
    ):
        _add_issue(
            result,
            "warning",
            "invalid-state term exceeded 0.05 per 4096 envs for three samples",
        )


def snapshot(
    log_root: Path,
    output: Path,
    training_session: str,
    target_iteration: int = 100_000,
) -> dict:
    event_file = latest_event_file(log_root)
    now_dt = datetime.now(timezone.utc)
    now = now_dt.isoformat()
    if event_file is None:
        return {"timestamp": now, "status": "waiting", "reason": "no TensorBoard event file"}
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    result = {
        "timestamp": now,
        "status": "ok",
        "run": event_file.parent.name,
        "event_file": str(event_file),
        "training_alive": _training_session_alive(training_session),
        "event_age_s": round(max(0.0, time.time() - event_file.stat().st_mtime), 1),
    }
    num_envs = _training_num_envs(training_session)
    if num_envs is not None:
        result["num_envs"] = num_envs
    last_step = 0
    scalar_series = {}
    for output_name, tag in DEFAULT_TAGS.items():
        if tag not in available:
            continue
        events = accumulator.Scalars(tag)
        scalar_series[tag] = events
        event = events[-1]
        last_step = max(last_step, event.step)
        value = float(event.value)
        result[output_name] = value if math.isfinite(value) else None
        if not math.isfinite(value):
            _add_issue(result, "critical", f"non-finite TensorBoard scalar: {tag}")
    recent_nonfinite = _recent_nonfinite_scalars(scalar_series, last_step)
    if recent_nonfinite:
        result["recent_nonfinite_scalars"] = recent_nonfinite
        _add_issue(result, "warning", "non-finite scalar occurred within the last 250 iterations")
    if result.get("invalid_state") is not None and num_envs:
        result["invalid_state_per_4096_envs"] = (
            result["invalid_state"] * REFERENCE_NUM_ENVS / num_envs
        )
    result["iteration"] = last_step
    disk = shutil.disk_usage(log_root.resolve())
    result["disk_total_gib"] = round(disk.total / 2**30, 1)
    result["disk_free_gib"] = round(disk.free / 2**30, 1)
    gpu_memory = _gpu_process_memory_mib()
    if gpu_memory is not None:
        result["gpu_process_memory_mib"] = gpu_memory
    gpu_count = _gpu_count()
    if gpu_count is not None:
        result["gpu_count"] = gpu_count

    previous = _same_run(_recent_records(output), result)
    history = (previous + [result])[-8:]
    iteration = int(result.get("iteration", 0))
    if iteration >= target_iteration - 1 and not result["training_alive"]:
        _mark_complete_if_healthy(result)
    elif not result["training_alive"]:
        _add_issue(
            result,
            "critical",
            f"training tmux session is missing before iteration {target_iteration - 1}",
        )
    if result["event_age_s"] > 3600:
        _add_issue(result, "critical", "TensorBoard event stream is stale for more than 60 minutes")
    elif result["event_age_s"] > 1800:
        _add_issue(result, "warning", "TensorBoard event stream is stale for more than 30 minutes")
    critical_disk_gib = min(20.0, result["disk_total_gib"] * 0.1)
    warning_disk_gib = min(50.0, result["disk_total_gib"] * 0.2)
    if result["disk_free_gib"] < critical_disk_gib:
        _add_issue(
            result,
            "critical",
            f"less than {critical_disk_gib:.1f} GiB disk space remains",
        )
    elif result["disk_free_gib"] < warning_disk_gib:
        _add_issue(
            result,
            "warning",
            f"less than {warning_disk_gib:.1f} GiB disk space remains",
        )
    gpu_memory_limit = 30_000 * (gpu_count or 1)
    if gpu_memory is None:
        _add_issue(result, "warning", "unable to read GPU compute-process memory")
    elif result["training_alive"] and gpu_memory < 1_024:
        _add_issue(
            result,
            "critical",
            "training session is alive but GPU compute-process memory is below 1024 MiB",
        )
    elif gpu_memory > gpu_memory_limit:
        _add_issue(
            result,
            "warning",
            f"GPU compute processes use more than {gpu_memory_limit} MiB",
        )

    value_losses = [record.get("value_loss") for record in history[-2:]]
    if any(value is not None and value > 1_000.0 for value in value_losses):
        _add_issue(result, "critical", "critic value loss exceeded 1000")
    elif len(value_losses) == 2 and all(value is not None and value > 10.0 for value in value_losses):
        _add_issue(result, "warning", "critic value loss exceeded 10 for two samples")
    _add_training_quality_issues(result, history)
    if len(history) >= 4 and len({record.get("iteration") for record in history[-4:]}) == 1:
        oldest = datetime.fromisoformat(history[-4]["timestamp"])
        if (now_dt - oldest).total_seconds() >= 2700:
            _add_issue(result, "critical", "training iteration did not advance for at least 45 minutes")

    # Height error covers settled non-negative commands. Sparse snapshots alone cannot prove
    # that the lift or terrain curriculum has stalled, so avoid inferring either here.
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-root", type=Path, default=Path("logs/rsl_rl/height_tracking_k1"))
    parser.add_argument("--output", type=Path, default=Path("logs/height_tracking_health.jsonl"))
    parser.add_argument("--interval", type=float, default=900.0)
    parser.add_argument("--training-session", default="k1-height-training")
    parser.add_argument("--target-iteration", type=int, default=100_000)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    while True:
        record = snapshot(
            args.log_root,
            args.output,
            args.training_session,
            args.target_iteration,
        )
        line = json.dumps(record, ensure_ascii=True, sort_keys=True)
        print(line, flush=True)
        with args.output.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        if args.once:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()

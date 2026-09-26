#!/usr/bin/env python3
"""Stage K1 high-height mastery lift, then resume normal lift decay."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import time
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


CHECKPOINT_RE = re.compile(r"model_(\d+)\.pt$")
METRIC_TAGS = {
    "height_error": "Metrics/height/height_error",
    "high_height_error": "Metrics/height/high_height_error",
    "high_height_success": "Metrics/height/high_height_success",
    "invalid_state": "Episode_Termination/invalid_state",
    "lift_scale": "Curriculum/adaptive_lift",
}


def _training_alive(session: str) -> bool:
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


def _checkpoint_iteration(path: Path) -> int:
    match = CHECKPOINT_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"not a numbered checkpoint: {path}")
    return int(match.group(1))


def _latest_event_file(log_root: Path) -> Path:
    files = list(log_root.glob("**/events.out.tfevents.*"))
    if not files:
        raise RuntimeError(f"no TensorBoard events under {log_root}")
    return max(files, key=lambda path: path.stat().st_mtime)


def _latest_checkpoint(run_dir: Path) -> tuple[Path, int]:
    candidates = []
    for path in run_dir.glob("model_*.pt"):
        try:
            candidates.append((_checkpoint_iteration(path), path))
        except ValueError:
            continue
    if not candidates:
        raise RuntimeError(f"no checkpoint in {run_dir}")
    iteration, checkpoint = max(candidates)
    return checkpoint, iteration


def _rolling_metrics(log_root: Path, window: int) -> tuple[Path, int, dict[str, float]]:
    event_file = _latest_event_file(log_root)
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    available = set(accumulator.Tags().get("scalars", []))
    missing = [tag for tag in METRIC_TAGS.values() if tag not in available]
    if missing:
        raise RuntimeError(f"missing TensorBoard metrics: {missing}")

    metrics: dict[str, float] = {}
    final_steps = set()
    for name, tag in METRIC_TAGS.items():
        events = accumulator.Scalars(tag)
        if len(events) < window:
            raise RuntimeError(f"{tag} has only {len(events)} points; need {window}")
        values = [float(event.value) for event in events[-window:]]
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError(f"non-finite values in {tag}")
        metrics[name] = sum(values) / len(values)
        metrics[f"{name}_last"] = values[-1]
        final_steps.add(events[-1].step)
    if len(final_steps) != 1:
        raise RuntimeError(f"TensorBoard metrics end at different steps: {sorted(final_steps)}")
    return event_file.parent, final_steps.pop(), metrics


def _write_state(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _launch_tmux(session: str, root: Path, command: list[str]) -> None:
    subprocess.run(["tmux", "kill-session", "-t", session], check=False, timeout=10)
    pane_command = f"cd {shlex.quote(str(root))} && exec {shlex.join(command)}"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, pane_command], check=True)
    subprocess.run(
        ["tmux", "set-option", "-t", session, "remain-on-exit", "on"],
        check=True,
    )


def _launch_monitor(
    root: Path,
    monitor_session: str,
    training_session: str,
    phase: str,
    expected_end: int,
) -> None:
    experiment = "height_tracking_k1_mastery" if phase == "mastery" else "height_tracking_k1"
    output = (
        "logs/height_tracking_mastery_health.jsonl"
        if phase == "mastery"
        else "logs/height_tracking_health.jsonl"
    )
    console = (
        "logs/height_tracking_mastery_monitor.console.log"
        if phase == "mastery"
        else "logs/height_tracking_monitor.console.log"
    )
    command = (
        f"cd {shlex.quote(str(root))} && exec {shlex.quote(str(root / '.venv/bin/python3'))} -u "
        f"scripts/rsl_rl/monitor_k1_height_tracking.py "
        f"--log-root logs/rsl_rl/{experiment} --output {output} --interval 900 "
        f"--training-session {shlex.quote(training_session)} --target-iteration {expected_end} "
        f">> {console} 2>&1"
    )
    subprocess.run(["tmux", "kill-session", "-t", monitor_session], check=False, timeout=10)
    subprocess.run(["tmux", "new-session", "-d", "-s", monitor_session, command], check=True)
    subprocess.run(
        ["tmux", "set-option", "-t", monitor_session, "remain-on-exit", "on"],
        check=True,
    )


def _training_command(
    root: Path,
    task: str,
    checkpoint: Path,
    segment_iterations: int,
    lift_scale: float,
    height_error: float,
    num_envs: int,
    fallen_cache: Path,
) -> list[str]:
    return [
        str(root / ".venv/bin/python3"),
        "-u",
        str(root / "scripts/rsl_rl/train.py"),
        "--task",
        task,
        "--num_envs",
        str(num_envs),
        "--max_iterations",
        str(segment_iterations),
        "--headless",
        "--device",
        "cuda:0",
        "--resume",
        "--resume_path",
        str(checkpoint),
        "--wbc_fallen_cache",
        str(fallen_cache),
        "--wbc_initial_lift_scale",
        repr(lift_scale),
        "--wbc_initial_lift_ema",
        repr(max(0.0, height_error)),
    ]


def _mastery_passed(metrics: dict[str, float], args: argparse.Namespace) -> bool:
    return (
        metrics["high_height_success"] >= args.min_success
        and metrics["high_height_error"] <= args.max_high_error
        and metrics["height_error"] <= args.max_height_error
        and metrics["invalid_state"] <= args.max_invalid_state
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--training-session", default="k1-height-training")
    parser.add_argument("--monitor-session", default="k1-height-monitor")
    parser.add_argument("--initial-checkpoint", type=Path, required=True)
    parser.add_argument("--initial-end", type=int, required=True)
    parser.add_argument("--lift-schedule", default="0.22,0.20,0.18,0.16,0.14272737503051758")
    parser.add_argument("--segment-iterations", type=int, default=5000)
    parser.add_argument("--window", type=int, default=250)
    parser.add_argument("--min-success", type=float, default=0.90)
    parser.add_argument("--max-high-error", type=float, default=0.08)
    parser.add_argument("--max-height-error", type=float, default=0.08)
    parser.add_argument("--max-invalid-state", type=float, default=0.05)
    parser.add_argument("--num-envs", type=int, default=16_384)
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument(
        "--state-path",
        type=Path,
        default=Path("logs/height_mastery_lift_schedule_state.json"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    state_path = args.state_path if args.state_path.is_absolute() else root / args.state_path
    initial_checkpoint = args.initial_checkpoint.resolve()
    if not initial_checkpoint.is_file():
        raise FileNotFoundError(initial_checkpoint)
    fallen_cache = (
        root
        / "height_tracking_states_cache"
        / "fallen_states_v7_Booster_K1_Height_Tracking_v0_548cc4ff_97a95488.pt"
    )
    if not fallen_cache.is_file():
        raise FileNotFoundError(fallen_cache)
    lift_schedule = [float(value) for value in args.lift_schedule.split(",")]
    if not lift_schedule or any(not 0.0 <= value <= 1.0 for value in lift_schedule):
        raise ValueError("lift schedule values must be within [0, 1]")
    if any(left <= right for left, right in zip(lift_schedule, lift_schedule[1:])):
        raise ValueError("lift schedule must be strictly decreasing")

    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {
            "status": "training",
            "phase": "mastery",
            "lift_index": 0,
            "lift_scale": lift_schedule[0],
            "expected_end": args.initial_end,
            "checkpoint": str(initial_checkpoint),
            "segment_iterations": args.segment_iterations,
        }
        _write_state(state_path, state)

    print(
        f"Watching {args.training_session}; phase={state['phase']} "
        f"lift={state['lift_scale']:.8f} target={state['expected_end']}",
        flush=True,
    )
    while True:
        if _training_alive(args.training_session):
            time.sleep(args.poll_interval)
            continue

        time.sleep(15.0)
        phase = state["phase"]
        experiment = "height_tracking_k1_mastery" if phase == "mastery" else "height_tracking_k1"
        log_root = root / "logs/rsl_rl" / experiment
        run_dir, event_iteration, metrics = _rolling_metrics(log_root, args.window)
        checkpoint, checkpoint_iteration = _latest_checkpoint(run_dir)
        expected_end = int(state["expected_end"])
        if event_iteration < expected_end - 1 or checkpoint_iteration < expected_end - 1:
            state.update(
                status="error",
                reason="training exited before expected segment endpoint",
                event_iteration=event_iteration,
                checkpoint_iteration=checkpoint_iteration,
            )
            _write_state(state_path, state)
            raise RuntimeError(state["reason"])

        next_phase = phase
        next_lift = float(metrics["lift_scale_last"])
        gate = None
        if phase == "mastery":
            passed = _mastery_passed(metrics, args)
            gate = {
                "passed": passed,
                "window": args.window,
                "min_success": args.min_success,
                "max_high_error": args.max_high_error,
                "max_height_error": args.max_height_error,
                "max_invalid_state": args.max_invalid_state,
                "metrics": metrics,
            }
            lift_index = int(state["lift_index"])
            if passed and lift_index + 1 < len(lift_schedule):
                lift_index += 1
                next_lift = lift_schedule[lift_index]
            elif passed:
                next_phase = "normal"
                next_lift = lift_schedule[-1]
            else:
                next_lift = lift_schedule[lift_index]
            state["lift_index"] = lift_index

        if phase == "normal" and next_lift <= 0.0:
            state.update(
                status="complete",
                checkpoint=str(checkpoint),
                checkpoint_iteration=checkpoint_iteration,
                event_iteration=event_iteration,
                lift_scale=0.0,
                metrics=metrics,
            )
            _write_state(state_path, state)
            print(f"Lift reached zero at iteration {event_iteration}", flush=True)
            return

        task = (
            "Booster-K1-Height-Tracking-Mastery-v0"
            if next_phase == "mastery"
            else "Booster-K1-Height-Tracking-v0"
        )
        expected_end = checkpoint_iteration + args.segment_iterations
        command = _training_command(
            root,
            task,
            checkpoint,
            args.segment_iterations,
            next_lift,
            metrics["height_error"],
            args.num_envs,
            fallen_cache,
        )
        state.update(
            status="launching",
            phase=next_phase,
            lift_scale=next_lift,
            expected_end=expected_end,
            checkpoint=str(checkpoint),
            checkpoint_iteration=checkpoint_iteration,
            event_iteration=event_iteration,
            last_gate=gate,
            command=command,
        )
        _write_state(state_path, state)
        _launch_tmux(args.training_session, root, command)
        _launch_monitor(
            root,
            args.monitor_session,
            args.training_session,
            next_phase,
            expected_end,
        )
        time.sleep(20.0)
        if not _training_alive(args.training_session):
            state.update(status="error", reason="next training segment failed during startup")
            _write_state(state_path, state)
            raise RuntimeError(state["reason"])
        state["status"] = "training"
        _write_state(state_path, state)
        print(
            f"Resumed {checkpoint.name} toward {expected_end}; "
            f"phase={next_phase} lift={next_lift:.8f} gate={gate}",
            flush=True,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

"""Pause an active training process and evaluate a checkpoint once it appears."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path


def is_training_command(executable_path: str, arguments: list[str], token: str) -> bool:
    if not arguments:
        return False
    executable = Path(executable_path).name
    return (
        executable.startswith("python")
        and "scripts/skrl/train.py" in arguments[1:]
        and token in arguments
    )


def reference_clip_arguments(indices: list[int] | None) -> list[str]:
    if indices is None:
        return []
    if any(index < 0 for index in indices):
        raise ValueError("--reference-clip-indices must be non-negative")
    if len(set(indices)) != len(indices):
        raise ValueError("--reference-clip-indices must be unique")
    return ["--reference_clip_indices", *(str(index) for index in indices)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--train-process-token", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--phase-bins", type=float, nargs="+", required=True)
    parser.add_argument("--reference-clip-indices", type=int, nargs="+", default=None)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--episodes-per-mode", type=int, default=1024)
    parser.add_argument("--joint-state-tolerance", type=float, default=0.05)
    parser.add_argument("--parallel-state-tolerance", type=float, default=0.05)
    parser.add_argument("--task-reward-scale", type=float, default=None)
    parser.add_argument("--style-reward-scale", type=float, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument(
        "--record-reference-phase",
        type=float,
        nargs=2,
        action="append",
        default=[],
        metavar=("MINIMUM", "MAXIMUM"),
    )
    parser.add_argument("--video-output-root", type=Path, default=None)
    parser.add_argument("--video-length", type=int, default=300)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser.parse_args()


def find_training_pid(token: str) -> int | None:
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            executable_path = os.readlink(entry / "exe")
            arguments = [
                value.decode()
                for value in (entry / "cmdline").read_bytes().split(b"\0")
                if value
            ]
        except (FileNotFoundError, PermissionError, ProcessLookupError, UnicodeDecodeError, OSError):
            continue
        if is_training_command(executable_path, arguments, token):
            return int(entry.name)
    return None


def main() -> int:
    args = parse_args()
    if args.checkpoint_step < 1:
        raise ValueError("--checkpoint-step must be positive")
    if args.poll_seconds <= 0.0:
        raise ValueError("--poll-seconds must be positive")
    if len(args.phase_bins) < 2:
        raise ValueError("--phase-bins requires at least two boundaries")
    if args.video_length < 1:
        raise ValueError("--video-length must be positive")
    for name in ("task_reward_scale", "style_reward_scale"):
        value = getattr(args, name)
        if value is not None and value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be nonnegative")
    for minimum, maximum in args.record_reference_phase:
        if not 0.0 <= minimum <= maximum <= 1.0:
            raise ValueError("recorded reference phases must satisfy 0 <= minimum <= maximum <= 1")
    if args.record_reference_phase and args.video_output_root is None:
        raise ValueError("--video-output-root is required when recording reference phases")

    checkpoint = args.run_dir / "checkpoints" / f"agent_{args.checkpoint_step}.pt"
    while not checkpoint.is_file():
        if find_training_pid(args.train_process_token) is None:
            args.log.parent.mkdir(parents=True, exist_ok=True)
            args.log.write_text(
                f"Training ended before checkpoint appeared: {checkpoint}\n",
                encoding="utf-8",
            )
            return 1
        time.sleep(args.poll_seconds)

    training_pid = find_training_pid(args.train_process_token)
    if training_pid is None:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        args.log.write_text("Training process not found; checkpoint was not evaluated.\n", encoding="utf-8")
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log.parent.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "scripts/skrl/evaluate.py",
        "--task",
        args.task,
        "--checkpoint",
        str(checkpoint),
        "--num_envs",
        str(args.num_envs),
        "--episodes_per_mode",
        str(args.episodes_per_mode),
        "--reference_phase_bins",
        *(str(value) for value in args.phase_bins),
        "--joint_state_tolerance",
        str(args.joint_state_tolerance),
        "--parallel_state_tolerance",
        str(args.parallel_state_tolerance),
        "--output",
        str(args.output),
        "--headless",
    ]
    command.extend(reference_clip_arguments(args.reference_clip_indices))
    for name in ("task_reward_scale", "style_reward_scale"):
        value = getattr(args, name)
        if value is not None:
            command.extend([f"--{name}", str(value)])

    os.kill(training_pid, signal.SIGSTOP)
    try:
        with args.log.open("w", encoding="utf-8") as log:
            log.write(f"Paused training PID {training_pid}; evaluating {checkpoint}\n")
            log.flush()
            status = subprocess.run(
                command,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            ).returncode
            if status != 0:
                return status
            for minimum, maximum in args.record_reference_phase:
                phase_name = f"phase_{minimum:.3f}_{maximum:.3f}"
                video_command = [
                    sys.executable,
                    "scripts/skrl/play.py",
                    "--task",
                    args.task,
                    "--checkpoint",
                    str(checkpoint),
                    "--reset_mode",
                    "reference",
                    "--reference_phase_min",
                    str(minimum),
                    "--reference_phase_max",
                    str(maximum),
                    "--joint_state_tolerance",
                    str(args.joint_state_tolerance),
                    "--parallel_state_tolerance",
                    str(args.parallel_state_tolerance),
                    "--video_length",
                    str(args.video_length),
                    "--output_dir",
                    str(args.video_output_root / phase_name),
                    "--headless",
                ]
                log.write(f"Recording {phase_name}\n")
                log.flush()
                status = subprocess.run(
                    video_command,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                ).returncode
                if status != 0:
                    return status
            return 0
    finally:
        try:
            os.kill(training_pid, signal.SIGCONT)
        except ProcessLookupError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Continue the formal K1 height run until the lift curriculum reaches zero."""

from __future__ import annotations

import argparse
import json
import math
import re
import shlex
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

CHECKPOINT_RE = re.compile(r"model_(\d+)\.pt$")


@dataclass(frozen=True)
class TrainingState:
    run_dir: Path
    iteration: int
    lift_scale: float
    height_error: float


def _training_alive(session: str) -> bool:
    result = subprocess.run(
        ["tmux", "list-panes", "-t", session, "-F", "#{pane_dead} #{pane_pid}"],
        capture_output=True,
        check=False,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        return False
    return any(line.split(maxsplit=1)[0] == "0" for line in result.stdout.splitlines() if line.strip())


def _latest_training_state(log_root: Path) -> TrainingState:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    event_files = list(log_root.glob("**/events.out.tfevents.*"))
    if not event_files:
        raise RuntimeError(f"no TensorBoard events under {log_root}")
    event_file = max(event_files, key=lambda path: path.stat().st_mtime)
    accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
    accumulator.Reload()
    lift = accumulator.Scalars("Curriculum/adaptive_lift")[-1]
    height = accumulator.Scalars("Metrics/height/height_error")[-1]
    if lift.step != height.step:
        raise RuntimeError(f"curriculum and height metrics disagree: {lift.step} != {height.step}")
    if not math.isfinite(lift.value) or not math.isfinite(height.value):
        raise RuntimeError("cannot resume from non-finite curriculum state")
    return TrainingState(event_file.parent, lift.step, float(lift.value), float(height.value))


def _checkpoint_iteration(path: Path) -> int:
    match = CHECKPOINT_RE.search(path.name)
    if match is None:
        raise ValueError(f"not a numbered checkpoint: {path}")
    return int(match.group(1))


def _latest_checkpoint(run_dir: Path) -> tuple[Path, int]:
    checkpoints = []
    for path in run_dir.glob("model_*.pt"):
        try:
            checkpoints.append((_checkpoint_iteration(path), path))
        except ValueError:
            continue
    if not checkpoints:
        raise RuntimeError(f"no checkpoint in {run_dir}")
    iteration, path = max(checkpoints)
    return path, iteration


def _next_end(expected_end: int, target_end: int, extension: int, lift_scale: float) -> int | None:
    if expected_end < target_end:
        return target_end
    if lift_scale > 0.0:
        return expected_end + extension
    return None


def _training_command(
    root: Path,
    checkpoint: Path,
    checkpoint_iteration: int,
    next_end: int,
    lift_scale: float,
    height_error: float,
    num_envs: int,
    fallen_cache: Path,
    distributed: bool = False,
    num_gpus: int = 2,
    num_envs_per_gpu: int = 8192,
) -> list[str]:
    remaining = next_end - checkpoint_iteration
    if remaining <= 0:
        raise ValueError(f"target {next_end} must exceed checkpoint {checkpoint_iteration}")
    resume_arguments = [
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
    if distributed:
        return [
            "env",
            f"NUM_GPUS={num_gpus}",
            f"NUM_ENVS_PER_GPU={num_envs_per_gpu}",
            f"MAX_ITERATIONS={remaining}",
            str(root / "scripts/rsl_rl/train_k1_height_tracking_distributed.sh"),
            *resume_arguments,
        ]
    return [
        str(root / ".venv/bin/python3"),
        "-u",
        str(root / "scripts/rsl_rl/train.py"),
        "--task",
        "Booster-K1-Height-Tracking-v0",
        "--num_envs",
        str(num_envs),
        "--max_iterations",
        str(remaining),
        "--headless",
        "--device",
        "cuda:0",
        *resume_arguments,
    ]


def _write_state(path: Path, expected_end: int, **extra: object) -> None:
    payload = {"expected_end": expected_end, **extra}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def _launch(session: str, root: Path, command: list[str]) -> None:
    subprocess.run(["tmux", "kill-session", "-t", session], check=False, timeout=10)
    pane_command = f"cd {shlex.quote(str(root))} && exec {shlex.join(command)}"
    subprocess.run(["tmux", "new-session", "-d", "-s", session, pane_command], check=True, timeout=10)
    subprocess.run(
        ["tmux", "set-option", "-t", session, "remain-on-exit", "on"],
        check=True,
        timeout=10,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--training-session", default="k1-height-training")
    parser.add_argument("--initial-end", type=int, default=100_000)
    parser.add_argument("--target-end", type=int, default=700_000)
    parser.add_argument("--extension", type=int, default=100_000)
    parser.add_argument("--num-envs", type=int, default=16_384)
    parser.add_argument("--distributed", action="store_true")
    parser.add_argument("--num-gpus", type=int, default=2)
    parser.add_argument("--num-envs-per-gpu", type=int, default=8192)
    parser.add_argument("--poll-interval", type=float, default=60.0)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root = args.root.resolve()
    log_root = root / "logs/rsl_rl/height_tracking_k1"
    state_path = root / "logs/height_training_extension_state.json"
    fallen_cache = (
        root
        / "height_tracking_states_cache"
        / "fallen_states_v7_Booster_K1_Height_Tracking_v0_548cc4ff_97a95488.pt"
    )
    if not fallen_cache.is_file():
        raise FileNotFoundError(fallen_cache)
    expected_end = args.initial_end
    if state_path.is_file():
        expected_end = int(json.loads(state_path.read_text(encoding="utf-8"))["expected_end"])

    if args.dry_run:
        state = _latest_training_state(log_root)
        checkpoint, checkpoint_iteration = _latest_checkpoint(state.run_dir)
        print(
            json.dumps(
                {
                    "training_alive": _training_alive(args.training_session),
                    "expected_end": expected_end,
                    "latest": {**asdict(state), "run_dir": str(state.run_dir)},
                    "checkpoint": str(checkpoint),
                    "checkpoint_iteration": checkpoint_iteration,
                },
                sort_keys=True,
            )
        )
        return

    print(
        f"Watching {args.training_session}; target={args.target_end}, extension={args.extension}",
        flush=True,
    )
    while True:
        if _training_alive(args.training_session):
            time.sleep(args.poll_interval)
            continue

        time.sleep(10.0)
        state = _latest_training_state(log_root)
        checkpoint, checkpoint_iteration = _latest_checkpoint(state.run_dir)
        if state.iteration < expected_end - 1 or checkpoint_iteration < state.iteration:
            raise RuntimeError(
                "training exited before its expected endpoint: "
                f"expected={expected_end}, event={state.iteration}, checkpoint={checkpoint_iteration}"
            )
        next_end = _next_end(expected_end, args.target_end, args.extension, state.lift_scale)
        if next_end is None:
            _write_state(
                state_path,
                expected_end,
                status="complete",
                lift_scale=state.lift_scale,
                checkpoint=str(checkpoint),
            )
            print(f"Lift reached zero at iteration {state.iteration}; extension complete", flush=True)
            return

        command = _training_command(
            root,
            checkpoint,
            checkpoint_iteration,
            next_end,
            state.lift_scale,
            state.height_error,
            args.num_envs,
            fallen_cache,
            args.distributed,
            args.num_gpus,
            args.num_envs_per_gpu,
        )
        _write_state(
            state_path,
            next_end,
            status="launching",
            previous_iteration=state.iteration,
            lift_scale=state.lift_scale,
            checkpoint=str(checkpoint),
            command=command,
        )
        _launch(args.training_session, root, command)
        time.sleep(10.0)
        if not _training_alive(args.training_session):
            raise RuntimeError("extended training did not survive startup")
        expected_end = next_end
        _write_state(
            state_path,
            expected_end,
            status="training",
            lift_scale=state.lift_scale,
            checkpoint=str(checkpoint),
            command=command,
        )
        print(
            f"Resumed checkpoint {checkpoint_iteration} toward {next_end} "
            f"with lift={state.lift_scale:.8f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

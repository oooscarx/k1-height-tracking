#!/usr/bin/env python3

"""Run long, gated reverse-phase AMP stages until the ground frontier is learned."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from check_recovery_acceptance import evaluate_acceptance


def phase_name(value: float) -> str:
    return f"{value:.3f}"


def stage_bins(minimum: float, maximum: float) -> list[float]:
    lower_context = max(0.0, minimum - 0.1)
    upper_context = min(1.0, maximum + 0.1)
    values = [lower_context, minimum, minimum + 0.1, maximum, upper_context]
    return sorted({round(value, 6) for value in values})


def required_stage_modes(minimum: float, maximum: float) -> list[str]:
    midpoint = round(minimum + 0.1, 6)
    return [
        f"reference_phase_{minimum:.3f}_{midpoint:.3f}",
        f"reference_phase_{midpoint:.3f}_{maximum:.3f}",
    ]


def checkpoint_step(path: Path) -> int:
    return int(path.stem.removeprefix("agent_"))


def latest_checkpoint(run_dir: Path) -> Path:
    checkpoints = list((run_dir / "checkpoints").glob("agent_*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"No agent checkpoint found under {run_dir}")
    return max(checkpoints, key=checkpoint_step)


def latest_checkpoint_or_recorded(
    run_dir: Path,
    recorded_checkpoint: str | None,
) -> Path:
    """Use the recorded seed when an interrupted run produced no checkpoint."""
    try:
        return latest_checkpoint(run_dir)
    except FileNotFoundError:
        if recorded_checkpoint is None:
            raise
        checkpoint = Path(recorded_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise
        return checkpoint


def wait_for_file(path: Path, poll_seconds: float) -> None:
    while not path.is_file() or path.stat().st_size == 0:
        time.sleep(poll_seconds)


def run_logged(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + " ".join(command) + "\n")
        log.flush()
        status = subprocess.run(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        ).returncode
    if status != 0:
        raise RuntimeError(f"Command failed with status {status}; see {log_path}")


def newest_run(log_root: Path, experiment_name: str, previous: set[Path]) -> Path:
    matches = {
        path.resolve()
        for path in log_root.glob(f"*_amp_torch_{experiment_name}")
        if path.is_dir()
    }
    created = matches - previous
    if len(created) != 1:
        raise RuntimeError(
            f"Expected one new run for {experiment_name}, found {sorted(created)}"
        )
    return created.pop()


def training_command_matches(arguments: list[str], experiment_name: str) -> bool:
    """Identify the actual skrl trainer for one curriculum attempt."""
    try:
        script_index = arguments.index("scripts/skrl/train.py")
        experiment_index = arguments.index("--experiment_name")
    except ValueError:
        return False
    return (
        script_index > 0
        and experiment_index + 1 < len(arguments)
        and arguments[experiment_index + 1] == experiment_name
    )


def training_process_active(experiment_name: str) -> bool:
    proc = Path("/proc")
    if not proc.is_dir():
        return False
    for command_path in proc.glob("[0-9]*/cmdline"):
        try:
            raw = command_path.read_bytes()
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        arguments = [
            value.decode("utf-8", errors="replace")
            for value in raw.split(b"\0")
            if value
        ]
        if training_command_matches(arguments, experiment_name):
            return True
    return False


def existing_run(log_root: Path, experiment_name: str) -> Path:
    matches = [
        path.resolve()
        for path in log_root.glob(f"*_amp_torch_{experiment_name}")
        if path.is_dir()
    ]
    if not matches:
        raise RuntimeError(f"No existing run found for {experiment_name}")
    return max(matches, key=lambda path: path.stat().st_mtime_ns)


def wait_for_interrupted_training(
    log_root: Path,
    experiment_name: str,
    poll_seconds: float,
) -> Path:
    """Wait for a trainer left alive by a supervisor restart."""
    while training_process_active(experiment_name):
        time.sleep(poll_seconds)
    return existing_run(log_root, experiment_name)


def expected_final_checkpoint_step(
    run_dir: Path,
    iterations: int,
    checkpoint_interval: int,
) -> int:
    checkpoints = sorted(
        (run_dir / "checkpoints").glob("agent_*.pt"),
        key=checkpoint_step,
    )
    if not checkpoints:
        raise FileNotFoundError(f"No periodic checkpoint found under {run_dir}")
    first_step = checkpoint_step(checkpoints[0])
    if first_step % checkpoint_interval != 0:
        raise RuntimeError(
            f"Cannot infer rollout length from {checkpoints[0]} and interval "
            f"{checkpoint_interval}"
        )
    rollout_length = first_step // checkpoint_interval
    return iterations * rollout_length


def interrupted_checkpoint_or_seed(
    *,
    run_dir: Path,
    seed_checkpoint: Path,
    iterations: int,
    checkpoint_interval: int,
) -> tuple[Path, int | None, bool]:
    """Recover a partial checkpoint, or retain the seed if none was written."""
    try:
        checkpoint = latest_checkpoint(run_dir)
    except FileNotFoundError:
        return seed_checkpoint, None, False
    return (
        checkpoint,
        expected_final_checkpoint_step(run_dir, iterations, checkpoint_interval),
        True,
    )


def interrupted_attempt_state(
    *,
    minimum: float,
    maximum: float,
    attempt: int,
    run_dir: Path,
    checkpoint: Path,
    expected_checkpoint_step: int | None,
    task_reward_scale: float,
    style_reward_scale: float,
) -> dict:
    """Record a partial run so its checkpoint can seed a clean retry."""
    return {
        "minimum": minimum,
        "maximum": maximum,
        "attempt": attempt,
        "run_dir": str(run_dir),
        "checkpoint": str(checkpoint),
        "expected_checkpoint_step": expected_checkpoint_step,
        "interrupted": True,
        "task_reward_scale": task_reward_scale,
        "style_reward_scale": style_reward_scale,
    }


def retry_seed_checkpoint(
    *,
    current_checkpoint: Path,
    stage_start_checkpoint: Path,
    attempt: int,
    retry_from_stage_start: bool,
) -> Path:
    """Choose a clean stage seed for retries when an attempt has regressed."""
    if attempt > 1 and retry_from_stage_start:
        return stage_start_checkpoint
    return current_checkpoint


def acceptance_summary(
    evaluation_path: Path,
    minimum: float,
    maximum: float,
    *,
    minimum_success_rate: float,
    maximum_joint_limit_rate: float,
    maximum_parallel_ankle_rate: float,
    maximum_parallel_state_violation_fraction: float,
    maximum_p90_recovery_time_s: float,
    minimum_episodes_per_mode: int,
    minimum_amp_style_reward_ratio: float,
    minimum_weighted_amp_style_fraction: float,
) -> dict:
    results = json.loads(evaluation_path.read_text(encoding="utf-8"))
    return evaluate_acceptance(
        results,
        required_stage_modes(minimum, maximum),
        minimum_success_rate=minimum_success_rate,
        maximum_joint_limit_rate=maximum_joint_limit_rate,
        maximum_parallel_ankle_rate=maximum_parallel_ankle_rate,
        maximum_nonfinite_action_rate=0.0,
        maximum_parallel_state_violation_fraction=maximum_parallel_state_violation_fraction,
        maximum_p90_recovery_time_s=maximum_p90_recovery_time_s,
        minimum_episodes_per_mode=minimum_episodes_per_mode,
        minimum_amp_style_reward_ratio=minimum_amp_style_reward_ratio,
        minimum_weighted_amp_style_fraction=minimum_weighted_amp_style_fraction,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-run-dir", type=Path, required=True)
    parser.add_argument("--initial-evaluation", type=Path, required=True)
    parser.add_argument(
        "--phase-minima",
        type=float,
        nargs="+",
        default=[0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    )
    parser.add_argument("--phase-width", type=float, default=0.2)
    parser.add_argument(
        "--task",
        default="Booster-K1-Fall-Recovery-AMP-FaceUp-Native-TaskDriven-v0",
    )
    parser.add_argument("--num-envs", type=int, default=16384)
    parser.add_argument("--iterations-per-stage", type=int, default=2400)
    parser.add_argument("--checkpoint-interval", type=int, default=400)
    parser.add_argument("--task-reward-scale", type=float, default=5.0)
    parser.add_argument("--style-reward-scale", type=float, default=5.0)
    parser.add_argument("--agent-learning-rate", type=float, default=None)
    parser.add_argument(
        "--preserve-checkpoint-value",
        action="store_true",
        help="Retain the checkpoint critic when continuing the same task and gate",
    )
    parser.add_argument("--maximum-amp-style-fraction", type=float, default=None)
    parser.add_argument("--minimum-amp-style-reward-scale", type=float, default=0.0)
    parser.add_argument("--handoff-curriculum-initial-progress", type=float, default=None)
    parser.add_argument("--evaluation-num-envs", type=int, default=1024)
    parser.add_argument("--evaluation-episodes", type=int, default=2048)
    parser.add_argument("--maximum-attempts-per-stage", type=int, default=3)
    parser.add_argument(
        "--retry-from-stage-start",
        action="store_true",
        help="Seed every retry from the checkpoint that entered the current stage",
    )
    parser.add_argument("--joint-state-tolerance", type=float, default=0.05)
    parser.add_argument("--parallel-state-tolerance", type=float, default=0.0)
    parser.add_argument("--minimum-success-rate", type=float, default=0.90)
    parser.add_argument("--maximum-joint-limit-rate", type=float, default=0.001)
    parser.add_argument("--maximum-parallel-ankle-rate", type=float, default=0.001)
    parser.add_argument(
        "--maximum-parallel-state-violation-fraction",
        type=float,
        default=0.001,
    )
    parser.add_argument("--maximum-p90-recovery-time-s", type=float, default=4.0)
    parser.add_argument("--minimum-amp-style-reward-ratio", type=float, default=0.10)
    parser.add_argument("--minimum-weighted-amp-style-fraction", type=float, default=0.04)
    parser.add_argument("--video-length", type=int, default=300)
    parser.add_argument("--artifacts-dir", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60.0)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.phase_width <= 0.0:
        raise ValueError("--phase-width must be positive")
    if args.iterations_per_stage < 1 or args.checkpoint_interval < 1:
        raise ValueError("training iteration counts must be positive")
    if args.task_reward_scale < 0.0 or args.style_reward_scale < 0.0:
        raise ValueError("AMP reward scales must be nonnegative")
    if args.agent_learning_rate is not None and args.agent_learning_rate <= 0.0:
        raise ValueError("--agent-learning-rate must be positive")
    if (
        args.maximum_amp_style_fraction is not None
        and not 0.0 < args.maximum_amp_style_fraction < 1.0
    ):
        raise ValueError("--maximum-amp-style-fraction must be inside (0, 1)")
    if not 0.0 <= args.minimum_amp_style_reward_scale <= args.style_reward_scale:
        raise ValueError(
            "--minimum-amp-style-reward-scale must be between zero and the configured style scale"
        )
    if (
        args.handoff_curriculum_initial_progress is not None
        and not 0.0 <= args.handoff_curriculum_initial_progress <= 1.0
    ):
        raise ValueError("--handoff-curriculum-initial-progress must be in [0, 1]")
    for name in (
        "minimum_amp_style_reward_ratio",
        "minimum_weighted_amp_style_fraction",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")
    if args.maximum_attempts_per_stage < 1:
        raise ValueError("--maximum-attempts-per-stage must be positive")
    if args.poll_seconds <= 0.0:
        raise ValueError("--poll-seconds must be positive")
    previous = float("inf")
    for minimum in args.phase_minima:
        maximum = minimum + args.phase_width
        if not 0.0 <= minimum < maximum <= 1.0:
            raise ValueError("phase stages must fit in [0, 1]")
        if minimum >= previous:
            raise ValueError("--phase-minima must be strictly descending")
        previous = minimum


def write_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(state, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def record_video(
    args: argparse.Namespace,
    checkpoint: Path,
    minimum: float,
    maximum: float,
    stage_label: str,
) -> None:
    output_dir = args.artifacts_dir / "videos" / stage_label
    command = [
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
        str(output_dir),
        "--headless",
    ]
    run_logged(command, args.artifacts_dir / f"{stage_label}_video.log")


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.initial_run_dir = args.initial_run_dir.expanduser().resolve()
    args.initial_evaluation = args.initial_evaluation.expanduser().resolve()
    args.artifacts_dir = args.artifacts_dir.expanduser().resolve()
    args.state = args.state.expanduser().resolve()
    log_root = (Path("logs") / "skrl" / "k1_fall_recovery_amp").resolve()

    state = (
        json.loads(args.state.read_text(encoding="utf-8"))
        if args.state.is_file()
        else {
            "status": "waiting_for_initial_evaluation",
            "stages": [],
            "accepted_stages": [],
            "current_run_dir": str(args.initial_run_dir),
            "current_evaluation": str(args.initial_evaluation),
        }
    )
    state.setdefault("accepted_stages", [])
    state.setdefault("stage_start_checkpoints", {})
    state["configured_task_reward_scale"] = args.task_reward_scale
    state["configured_style_reward_scale"] = args.style_reward_scale
    state["configured_agent_learning_rate"] = args.agent_learning_rate
    state["configured_preserve_checkpoint_value"] = args.preserve_checkpoint_value
    state["configured_maximum_amp_style_fraction"] = args.maximum_amp_style_fraction
    state["configured_minimum_amp_style_reward_scale"] = (
        args.minimum_amp_style_reward_scale
    )
    state["configured_handoff_curriculum_initial_progress"] = (
        args.handoff_curriculum_initial_progress
    )
    state["configured_minimum_amp_style_reward_ratio"] = (
        args.minimum_amp_style_reward_ratio
    )
    state["configured_minimum_weighted_amp_style_fraction"] = (
        args.minimum_weighted_amp_style_fraction
    )
    write_state(args.state, state)
    wait_for_file(args.initial_evaluation, args.poll_seconds)

    current_run = Path(state["current_run_dir"])
    current_evaluation = Path(state["current_evaluation"])
    current_checkpoint = latest_checkpoint_or_recorded(
        current_run,
        state.get("current_checkpoint"),
    )

    for stage_index, minimum in enumerate(args.phase_minima):
        maximum = round(minimum + args.phase_width, 6)
        accepted_key = f"{minimum:.3f}_{maximum:.3f}"
        if accepted_key in state["accepted_stages"]:
            continue
        if accepted_key not in state["stage_start_checkpoints"]:
            state["stage_start_checkpoints"][accepted_key] = str(current_checkpoint)
            write_state(args.state, state)
        stage_start_checkpoint = Path(
            state["stage_start_checkpoints"][accepted_key]
        )
        attempts = [
            item
            for item in state["stages"]
            if item["minimum"] == minimum and item["maximum"] == maximum
        ]
        evaluation = current_evaluation if stage_index == 0 and not attempts else None
        while True:
            if evaluation is not None:
                summary = acceptance_summary(
                    evaluation,
                    minimum,
                    maximum,
                    minimum_success_rate=args.minimum_success_rate,
                    maximum_joint_limit_rate=args.maximum_joint_limit_rate,
                    maximum_parallel_ankle_rate=args.maximum_parallel_ankle_rate,
                    maximum_parallel_state_violation_fraction=(
                        args.maximum_parallel_state_violation_fraction
                    ),
                    maximum_p90_recovery_time_s=args.maximum_p90_recovery_time_s,
                    minimum_episodes_per_mode=args.evaluation_episodes,
                    minimum_amp_style_reward_ratio=(
                        args.minimum_amp_style_reward_ratio
                    ),
                    minimum_weighted_amp_style_fraction=(
                        args.minimum_weighted_amp_style_fraction
                    ),
                )
                acceptance_path = evaluation.with_name(
                    evaluation.stem + "_acceptance.json"
                )
                acceptance_path.write_text(
                    json.dumps(summary, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                if summary["accepted"]:
                    stage_label = (
                        f"phase_{phase_name(minimum)}_{phase_name(maximum)}"
                        f"_accepted_{checkpoint_step(current_checkpoint)}"
                    )
                    record_video(
                        args,
                        current_checkpoint,
                        minimum,
                        min(minimum + 0.1, maximum),
                        stage_label,
                    )
                    state["status"] = "stage_accepted"
                    state["accepted_minimum"] = minimum
                    state["accepted_maximum"] = maximum
                    state["current_checkpoint"] = str(current_checkpoint)
                    state["accepted_stages"].append(accepted_key)
                    write_state(args.state, state)
                    break
                if len(attempts) >= args.maximum_attempts_per_stage:
                    state["status"] = "review_required"
                    state["failed_minimum"] = minimum
                    state["failed_maximum"] = maximum
                    state["failures"] = summary["failures"]
                    write_state(args.state, state)
                    return 2

            attempt = len(attempts) + 1
            experiment_name = (
                f"reverse_phase_{int(round(minimum * 100)):03d}_"
                f"{int(round(maximum * 100)):03d}_attempt{attempt}"
            )
            seed_checkpoint = retry_seed_checkpoint(
                current_checkpoint=current_checkpoint,
                stage_start_checkpoint=stage_start_checkpoint,
                attempt=attempt,
                retry_from_stage_start=args.retry_from_stage_start,
            )
            train_log = args.artifacts_dir / f"{experiment_name}.log"
            resuming_training = (
                state.get("status")
                in {"training", "waiting_for_interrupted_training"}
                and state.get("training_minimum") == minimum
                and state.get("training_maximum") == maximum
                and state.get("training_attempt") == attempt
            )
            if resuming_training:
                state["status"] = "waiting_for_interrupted_training"
                state["resumed_experiment_name"] = experiment_name
                write_state(args.state, state)
                current_run = wait_for_interrupted_training(
                    log_root,
                    experiment_name,
                    args.poll_seconds,
                )
                (
                    interrupted_checkpoint,
                    expected_step,
                    produced_checkpoint,
                ) = interrupted_checkpoint_or_seed(
                    run_dir=current_run,
                    seed_checkpoint=seed_checkpoint,
                    iterations=args.iterations_per_stage,
                    checkpoint_interval=args.checkpoint_interval,
                )
                completed_step = (
                    checkpoint_step(interrupted_checkpoint)
                    if produced_checkpoint
                    else 0
                )
                if expected_step is None or completed_step < expected_step:
                    attempt_state = interrupted_attempt_state(
                        minimum=minimum,
                        maximum=maximum,
                        attempt=attempt,
                        run_dir=current_run,
                        checkpoint=interrupted_checkpoint,
                        expected_checkpoint_step=expected_step,
                        task_reward_scale=args.task_reward_scale,
                        style_reward_scale=args.style_reward_scale,
                    )
                    attempt_state["produced_checkpoint"] = produced_checkpoint
                    state["stages"].append(attempt_state)
                    state["current_run_dir"] = str(current_run)
                    state["current_checkpoint"] = str(interrupted_checkpoint)
                    state["status"] = "interrupted_training_recovered"
                    state.pop("failures", None)
                    attempts.append(attempt_state)
                    current_checkpoint = interrupted_checkpoint
                    write_state(args.state, state)
                    evaluation = None
                    continue
            else:
                previous_runs = {
                    path.resolve() for path in log_root.iterdir() if path.is_dir()
                }
                state["status"] = "training"
                state["training_minimum"] = minimum
                state["training_maximum"] = maximum
                state["training_attempt"] = attempt
                write_state(args.state, state)
                train_command = [
                    sys.executable,
                    "scripts/skrl/train.py",
                    "--task",
                    args.task,
                    "--checkpoint",
                    str(seed_checkpoint),
                    "--reset_checkpoint_optimizer",
                    "--reset_checkpoint_learning_rate",
                    "--task_reward_scale",
                    str(args.task_reward_scale),
                    "--style_reward_scale",
                    str(args.style_reward_scale),
                    "--num_envs",
                    str(args.num_envs),
                    "--max_iterations",
                    str(args.iterations_per_stage),
                    "--checkpoint_interval",
                    str(args.checkpoint_interval),
                    "--experiment_name",
                    experiment_name,
                    "--reference_reset_probability",
                    "1.0",
                    "--failure_reset_probability_start",
                    "0.0",
                    "--failure_reset_probability_end",
                    "0.0",
                    "--reference_phase_min",
                    str(minimum),
                    "--reference_phase_max",
                    str(maximum),
                    "--joint_state_tolerance",
                    str(args.joint_state_tolerance),
                    "--parallel_state_tolerance",
                    str(args.parallel_state_tolerance),
                    "--headless",
                ]
                if not args.preserve_checkpoint_value:
                    train_command.append("--reset_checkpoint_value")
                if args.maximum_amp_style_fraction is not None:
                    train_command.extend(
                        [
                            "--maximum_amp_style_fraction",
                            str(args.maximum_amp_style_fraction),
                            "--minimum_amp_style_reward_scale",
                            str(args.minimum_amp_style_reward_scale),
                        ]
                    )
                if args.agent_learning_rate is not None:
                    train_command.extend(
                        ["--agent_learning_rate", str(args.agent_learning_rate)]
                    )
                if args.handoff_curriculum_initial_progress is not None:
                    train_command.extend(
                        [
                            "--handoff_curriculum_initial_progress",
                            str(args.handoff_curriculum_initial_progress),
                        ]
                    )
                run_logged(train_command, train_log)
                current_run = newest_run(
                    log_root,
                    experiment_name,
                    previous_runs,
                )
            current_checkpoint = latest_checkpoint(current_run)

            evaluation = args.artifacts_dir / f"{experiment_name}_evaluation.json"
            evaluation_log = args.artifacts_dir / f"{experiment_name}_evaluation.log"
            state["status"] = "evaluating"
            write_state(args.state, state)
            evaluation_command = [
                sys.executable,
                "scripts/skrl/evaluate.py",
                "--task",
                args.task,
                "--checkpoint",
                str(current_checkpoint),
                "--num_envs",
                str(args.evaluation_num_envs),
                "--episodes_per_mode",
                str(args.evaluation_episodes),
                "--reference_phase_bins",
                *(str(value) for value in stage_bins(minimum, maximum)),
                "--joint_state_tolerance",
                str(args.joint_state_tolerance),
                "--parallel_state_tolerance",
                str(args.parallel_state_tolerance),
                "--task_reward_scale",
                str(args.task_reward_scale),
                "--style_reward_scale",
                str(args.style_reward_scale),
                "--output",
                str(evaluation),
                "--headless",
            ]
            run_logged(evaluation_command, evaluation_log)
            attempt_state = {
                "minimum": minimum,
                "maximum": maximum,
                "attempt": attempt,
                "run_dir": str(current_run),
                "checkpoint": str(current_checkpoint),
                "seed_checkpoint": str(seed_checkpoint),
                "evaluation": str(evaluation),
                "task_reward_scale": args.task_reward_scale,
                "style_reward_scale": args.style_reward_scale,
            }
            state["stages"].append(attempt_state)
            state["current_run_dir"] = str(current_run)
            state["current_evaluation"] = str(evaluation)
            attempts.append(attempt_state)
            write_state(args.state, state)

        current_evaluation = Path(state["current_evaluation"])

    state["status"] = "faceup_reference_curriculum_complete"
    state["current_checkpoint"] = str(current_checkpoint)
    write_state(args.state, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3

"""Train and gate one AMP policy from both get-up clips to arbitrary falls."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from check_recovery_acceptance import DEFAULT_MODES, evaluate_acceptance
from run_reverse_phase_curriculum import (
    checkpoint_step,
    interrupted_checkpoint_or_seed,
    latest_checkpoint,
    newest_run,
    run_logged,
    stage_bins,
    wait_for_interrupted_training,
    write_state,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
HARDWARE_CONFIG_PATH = (
    REPOSITORY_ROOT
    / "source"
    / "booster_train"
    / "booster_train"
    / "tasks"
    / "manager_based"
    / "fall_recovery"
    / "config"
    / "k1_fall_recovery.json"
)
DEFAULT_GROUND_STATE_PATH = (
    REPOSITORY_ROOT
    / "source"
    / "booster_train"
    / "booster_train"
    / "tasks"
    / "manager_based"
    / "fall_recovery"
    / "motions"
    / "random"
    / "k1_settled_random_ground_states.npz"
)


def load_handoff_contract() -> dict:
    config = json.loads(HARDWARE_CONFIG_PATH.read_text(encoding="utf-8"))
    return config["handoff_policy"]


def clip_stage_modes(
    clip_indices: list[int],
    minimum: float,
    maximum: float,
) -> list[str]:
    midpoint = round(minimum + 0.1, 6)
    return [
        f"reference_clip_{clip_index}_phase_{lower:.3f}_{upper:.3f}"
        for clip_index in clip_indices
        for lower, upper in ((minimum, midpoint), (midpoint, maximum))
    ]


def next_training_log_path(artifacts_dir: Path, experiment: str) -> Path:
    base = artifacts_dir / f"{experiment}.log"
    if not base.exists():
        return base
    restart = 1
    while True:
        candidate = artifacts_dir / f"{experiment}_restart{restart}.log"
        if not candidate.exists():
            return candidate
        restart += 1


def recover_interrupted_training(
    *,
    log_root: Path,
    experiment: str,
    seed_checkpoint: Path,
    iterations: int,
    checkpoint_interval: int,
    poll_seconds: float,
) -> tuple[Path, Path, bool]:
    run_dir = wait_for_interrupted_training(
        log_root,
        experiment,
        poll_seconds,
    )
    checkpoint, expected_step, produced_checkpoint = interrupted_checkpoint_or_seed(
        run_dir=run_dir,
        seed_checkpoint=seed_checkpoint,
        iterations=iterations,
        checkpoint_interval=checkpoint_interval,
    )
    completed = (
        produced_checkpoint
        and expected_step is not None
        and checkpoint_step(checkpoint) >= expected_step
    )
    return run_dir, checkpoint, completed


def wait_for_faceup_checkpoint(
    faceup_state_path: Path,
    poll_seconds: float,
) -> Path:
    while True:
        if faceup_state_path.is_file():
            state = json.loads(faceup_state_path.read_text(encoding="utf-8"))
            if state.get("status") == "faceup_reference_curriculum_complete":
                checkpoint = Path(state["current_checkpoint"])
                if checkpoint.is_file():
                    return checkpoint.resolve()
            if state.get("status") == "review_required":
                raise RuntimeError(
                    "Face-up curriculum requires review before unified training: "
                    + "; ".join(state.get("failures", []))
                )
        time.sleep(poll_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--faceup-state", type=Path, required=True)
    parser.add_argument("--task", default="Booster-K1-Fall-Recovery-AMP-TaskDriven-v0")
    parser.add_argument("--clip-indices", type=int, nargs="+", default=[0, 1])
    parser.add_argument(
        "--phase-minima",
        type=float,
        nargs="+",
        default=[0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    )
    parser.add_argument("--phase-width", type=float, default=0.2)
    parser.add_argument("--num-envs", type=int, default=16384)
    parser.add_argument("--bc-num-envs", type=int, default=2048)
    parser.add_argument("--bc-iterations", type=int, default=2000)
    parser.add_argument("--iterations-per-phase-stage", type=int, default=4000)
    parser.add_argument(
        "--iterations-per-random-stage",
        "--random-fall-iterations",
        dest="iterations_per_random_stage",
        type=int,
        default=4000,
    )
    parser.add_argument(
        "--random-fall-difficulties",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0],
    )
    parser.add_argument(
        "--maximum-domain-randomization-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--maximum-observation-noise-scale",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--maximum-push-randomization-scale",
        type=float,
        default=0.5,
    )
    parser.add_argument("--push-start-difficulty", type=float, default=0.5)
    parser.add_argument(
        "--initial-ground-state-file",
        type=Path,
        default=DEFAULT_GROUND_STATE_PATH,
        help="Settled off-reference ground states replayed from the first random-fall stage",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=400)
    parser.add_argument("--task-reward-scale", type=float, default=5.0)
    parser.add_argument("--style-reward-scale", type=float, default=5.0)
    parser.add_argument("--evaluation-num-envs", type=int, default=1024)
    parser.add_argument("--evaluation-episodes", type=int, default=2048)
    parser.add_argument("--final-evaluation-episodes", type=int, default=4096)
    parser.add_argument("--handoff-evaluation-num-envs", type=int, default=1024)
    parser.add_argument("--handoff-evaluation-episodes", type=int, default=4096)
    parser.add_argument("--maximum-attempts-per-stage", type=int, default=3)
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
    if not args.clip_indices or len(set(args.clip_indices)) != len(args.clip_indices):
        raise ValueError("--clip-indices must be a non-empty unique list")
    if any(index < 0 for index in args.clip_indices):
        raise ValueError("--clip-indices must be non-negative")
    if args.phase_width <= 0.0:
        raise ValueError("--phase-width must be positive")
    previous = float("inf")
    for minimum in args.phase_minima:
        maximum = minimum + args.phase_width
        if not 0.0 <= minimum < maximum <= 1.0:
            raise ValueError("phase stages must fit in [0, 1]")
        if minimum >= previous:
            raise ValueError("--phase-minima must be strictly descending")
        previous = minimum
    for name in (
        "num_envs",
        "bc_num_envs",
        "bc_iterations",
        "iterations_per_phase_stage",
        "iterations_per_random_stage",
        "checkpoint_interval",
        "evaluation_num_envs",
        "evaluation_episodes",
        "final_evaluation_episodes",
        "handoff_evaluation_num_envs",
        "handoff_evaluation_episodes",
        "maximum_attempts_per_stage",
        "video_length",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if (
        not args.random_fall_difficulties
        or any(
            not 0.0 < difficulty <= 1.0
            for difficulty in args.random_fall_difficulties
        )
        or any(
            first >= second
            for first, second in zip(
                args.random_fall_difficulties,
                args.random_fall_difficulties[1:],
            )
        )
        or args.random_fall_difficulties[-1] != 1.0
    ):
        raise ValueError(
            "--random-fall-difficulties must strictly increase inside (0, 1] "
            "and end at 1"
        )
    if args.poll_seconds <= 0.0:
        raise ValueError("--poll-seconds must be positive")
    if args.task_reward_scale < 0.0 or args.style_reward_scale < 0.0:
        raise ValueError("AMP reward scales must be nonnegative")
    for name in (
        "minimum_amp_style_reward_ratio",
        "minimum_weighted_amp_style_fraction",
        "maximum_domain_randomization_scale",
        "maximum_observation_noise_scale",
        "maximum_push_randomization_scale",
        "push_start_difficulty",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1]")


def randomization_scales(
    difficulty: float,
    *,
    maximum_domain_scale: float,
    maximum_observation_noise_scale: float,
    maximum_push_scale: float,
    push_start_difficulty: float,
) -> tuple[float, float, float]:
    domain_scale = maximum_domain_scale * difficulty
    observation_noise_scale = maximum_observation_noise_scale * difficulty
    if difficulty <= push_start_difficulty or push_start_difficulty >= 1.0:
        push_scale = 0.0
    else:
        push_progress = (difficulty - push_start_difficulty) / (
            1.0 - push_start_difficulty
        )
        push_scale = maximum_push_scale * push_progress
    return domain_scale, observation_noise_scale, push_scale


def acceptance(
    evaluation: Path,
    modes: list[str],
    args: argparse.Namespace,
    *,
    minimum_episodes_per_mode: int | None = None,
    require_amp_style: bool = False,
) -> dict:
    results = json.loads(evaluation.read_text(encoding="utf-8"))
    return evaluate_acceptance(
        results,
        modes,
        minimum_success_rate=args.minimum_success_rate,
        maximum_joint_limit_rate=args.maximum_joint_limit_rate,
        maximum_parallel_ankle_rate=args.maximum_parallel_ankle_rate,
        maximum_nonfinite_action_rate=0.0,
        maximum_parallel_state_violation_fraction=(
            args.maximum_parallel_state_violation_fraction
        ),
        maximum_p90_recovery_time_s=args.maximum_p90_recovery_time_s,
        minimum_episodes_per_mode=(
            args.evaluation_episodes
            if minimum_episodes_per_mode is None
            else minimum_episodes_per_mode
        ),
        minimum_amp_style_reward_ratio=(
            args.minimum_amp_style_reward_ratio if require_amp_style else None
        ),
        minimum_weighted_amp_style_fraction=(
            args.minimum_weighted_amp_style_fraction if require_amp_style else None
        ),
    )


def handoff_acceptance(
    evaluation: Path,
    modes: list[str],
    args: argparse.Namespace,
) -> dict:
    summary = acceptance(
        evaluation,
        modes,
        args,
        minimum_episodes_per_mode=getattr(
            args,
            "handoff_evaluation_episodes",
            1,
        ),
    )
    results = json.loads(evaluation.read_text(encoding="utf-8"))
    expected = load_handoff_contract()
    failures = list(summary["failures"])
    for mode in modes:
        result = results.get(mode)
        if result is None:
            continue
        contract = result.get("handoff_contract")
        if not isinstance(contract, dict):
            failures.append(f"{mode}: commanded locomotion handoff contract is missing")
            continue
        expected_contract = {
            "actor": "forward",
            "maximum_start_pose_error": expected["maximum_start_pose_error"],
            "maximum_start_linear_speed_mps": expected[
                "maximum_start_linear_speed_mps"
            ],
            "maximum_start_angular_speed_radps": expected[
                "maximum_start_angular_speed_radps"
            ],
            "maximum_start_body_joint_speed_radps": expected[
                "maximum_start_body_joint_speed_radps"
            ],
            "commanded_forward_velocity_mps": expected[
                "commanded_forward_velocity_mps"
            ],
            "commanded_validation_s": expected["commanded_validation_s"],
            "minimum_commanded_forward_progress_m": expected[
                "minimum_commanded_forward_progress_m"
            ],
            "maximum_commanded_lateral_drift_m": expected[
                "maximum_commanded_lateral_drift_m"
            ],
            "minimum_commanded_height_m": expected[
                "minimum_commanded_height_m"
            ],
            "maximum_commanded_gravity_z": expected[
                "maximum_commanded_gravity_z"
            ],
        }
        for name, value in expected_contract.items():
            if contract.get(name) != value:
                failures.append(
                    f"{mode}: handoff contract {name} "
                    f"{contract.get(name)!r} != {value!r}"
                )

        checks = (
            (
                "handoff_start_pose_error_rad",
                "maximum",
                lambda value: value
                <= float(expected["maximum_start_pose_error"]),
                "handoff starts outside the deployment pose threshold",
            ),
            (
                "handoff_start_linear_speed_mps",
                "maximum",
                lambda value: value
                <= float(expected["maximum_start_linear_speed_mps"]),
                "handoff starts above the deployment linear-speed threshold",
            ),
            (
                "handoff_start_angular_speed_radps",
                "maximum",
                lambda value: value
                <= float(expected["maximum_start_angular_speed_radps"]),
                "handoff starts above the deployment angular-speed threshold",
            ),
            (
                "handoff_start_body_joint_speed_radps",
                "maximum",
                lambda value: value
                <= float(expected["maximum_start_body_joint_speed_radps"]),
                "handoff starts above the deployment joint-speed threshold",
            ),
            (
                "zero_command_xy_drift_m",
                "maximum",
                lambda value: value
                <= float(expected["maximum_zero_command_xy_drift_m"]),
                "zero-command drift exceeds deployment threshold",
            ),
            (
                "commanded_forward_progress_m",
                "minimum",
                lambda value: value
                >= float(expected["minimum_commanded_forward_progress_m"]),
                "commanded forward progress is below deployment threshold",
            ),
            (
                "commanded_peak_lateral_drift_m",
                "maximum",
                lambda value: value
                <= float(expected["maximum_commanded_lateral_drift_m"]),
                "commanded lateral drift exceeds deployment threshold",
            ),
            (
                "commanded_minimum_height_m",
                "minimum",
                lambda value: value
                >= float(expected["minimum_commanded_height_m"]),
                "commanded trunk height is below deployment threshold",
            ),
            (
                "commanded_maximum_gravity_z",
                "maximum",
                lambda value: value
                <= float(expected["maximum_commanded_gravity_z"]),
                "commanded posture exceeds deployment tilt threshold",
            ),
        )
        for statistics_name, statistic, predicate, message in checks:
            statistics = result.get(statistics_name)
            value = (
                None
                if not isinstance(statistics, dict)
                else statistics.get(statistic)
            )
            if value is None or not predicate(float(value)):
                failures.append(f"{mode}: {message}")
    summary["failures"] = failures
    summary["accepted"] = not failures
    summary["commanded_handoff_contract"] = {
        "commanded_forward_velocity_mps": expected[
            "commanded_forward_velocity_mps"
        ],
        "commanded_validation_s": expected["commanded_validation_s"],
        "minimum_commanded_forward_progress_m": expected[
            "minimum_commanded_forward_progress_m"
        ],
        "maximum_commanded_lateral_drift_m": expected[
            "maximum_commanded_lateral_drift_m"
        ],
    }
    return summary


def evaluate_clip_stage(
    args: argparse.Namespace,
    checkpoint: Path,
    minimum: float,
    maximum: float,
    label: str,
) -> tuple[Path, dict]:
    output = args.artifacts_dir / f"{label}_evaluation.json"
    command = [
        sys.executable,
        "scripts/skrl/evaluate.py",
        "--task",
        args.task,
        "--checkpoint",
        str(checkpoint),
        "--num_envs",
        str(args.evaluation_num_envs),
        "--episodes_per_mode",
        str(args.evaluation_episodes),
        "--reference_clip_indices",
        *(str(index) for index in args.clip_indices),
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
        str(output),
        "--headless",
    ]
    run_logged(command, args.artifacts_dir / f"{label}_evaluation.log")
    summary = acceptance(
        output,
        clip_stage_modes(args.clip_indices, minimum, maximum),
        args,
        require_amp_style=True,
    )
    output.with_name(output.stem + "_acceptance.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output, summary


def bootstrap_unified_policy(
    args: argparse.Namespace,
    checkpoint: Path,
    minimum: float,
    maximum: float,
) -> tuple[Path, Path]:
    experiment = (
        f"unified_bootstrap_bc_phase_{int(round(minimum * 100)):03d}_"
        f"{int(round(maximum * 100)):03d}"
    )
    log_root = (Path("logs") / "skrl" / "k1_fall_recovery_amp").resolve()
    previous_runs = (
        {path.resolve() for path in log_root.iterdir() if path.is_dir()}
        if log_root.is_dir()
        else set()
    )
    command = [
        sys.executable,
        "scripts/skrl/train.py",
        "--task",
        args.task,
        "--checkpoint",
        str(checkpoint),
        "--reset_checkpoint_optimizer",
        "--reset_checkpoint_value",
        "--reset_checkpoint_discriminator",
        "--reset_checkpoint_learning_rate",
        "--task_reward_scale",
        str(args.task_reward_scale),
        "--style_reward_scale",
        str(args.style_reward_scale),
        "--bc_iterations",
        str(args.bc_iterations),
        "--bc_teacher",
        "motion",
        "--bc_reset_mode",
        "reference",
        "--bc_only",
        "--num_envs",
        str(args.bc_num_envs),
        "--experiment_name",
        experiment,
        "--reference_reset_probability",
        "1.0",
        "--standing_reset_probability_start",
        "0.0",
        "--standing_reset_probability_end",
        "0.0",
        "--random_fall_probability_start",
        "0.0",
        "--random_fall_probability_end",
        "0.0",
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
    run_logged(command, args.artifacts_dir / f"{experiment}.log")
    run_dir = newest_run(log_root, experiment, previous_runs)
    bc_checkpoint = run_dir / "checkpoints" / "bc_agent.pt"
    if not bc_checkpoint.is_file():
        raise FileNotFoundError(f"BC checkpoint was not written: {bc_checkpoint}")
    return run_dir, bc_checkpoint


def train_phase_attempt(
    args: argparse.Namespace,
    checkpoint: Path,
    minimum: float,
    maximum: float,
    attempt: int,
    resume: bool = False,
) -> tuple[Path, Path]:
    experiment = (
        f"unified_phase_{int(round(minimum * 100)):03d}_"
        f"{int(round(maximum * 100)):03d}_attempt{attempt}"
    )
    log_root = (Path("logs") / "skrl" / "k1_fall_recovery_amp").resolve()
    if resume:
        run_dir, checkpoint, completed = recover_interrupted_training(
            log_root=log_root,
            experiment=experiment,
            seed_checkpoint=checkpoint,
            iterations=args.iterations_per_phase_stage,
            checkpoint_interval=args.checkpoint_interval,
            poll_seconds=args.poll_seconds,
        )
        if completed:
            return run_dir, checkpoint
    previous_runs = {path.resolve() for path in log_root.iterdir() if path.is_dir()}
    command = [
        sys.executable,
        "scripts/skrl/train.py",
        "--task",
        args.task,
        "--checkpoint",
        str(checkpoint),
        "--reset_checkpoint_optimizer",
        "--reset_checkpoint_value",
        "--reset_checkpoint_learning_rate",
        "--task_reward_scale",
        str(args.task_reward_scale),
        "--style_reward_scale",
        str(args.style_reward_scale),
    ]
    command.extend(
        [
            "--num_envs",
            str(args.num_envs),
            "--max_iterations",
            str(args.iterations_per_phase_stage),
            "--checkpoint_interval",
            str(args.checkpoint_interval),
            "--experiment_name",
            experiment,
            "--reference_reset_probability",
            "1.0",
            "--standing_reset_probability_start",
            "0.0",
            "--standing_reset_probability_end",
            "0.0",
            "--random_fall_probability_start",
            "0.0",
            "--random_fall_probability_end",
            "0.0",
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
    )
    run_logged(command, next_training_log_path(args.artifacts_dir, experiment))
    run_dir = newest_run(log_root, experiment, previous_runs)
    return run_dir, latest_checkpoint(run_dir)


def random_reset_probabilities(
    difficulty: float,
    failure_probability: float = 0.0,
) -> tuple[float, float, float, float]:
    random_fall = 0.05 + 0.40 * difficulty
    standing = 0.05
    reference = 0.90 - random_fall - failure_probability
    if reference < 0.0:
        raise ValueError("failure replay leaves no valid reference-reset probability")
    return reference, standing, random_fall, failure_probability


def failure_state_count(path: Path) -> int:
    with np.load(path, allow_pickle=False) as archive:
        return int(archive["joint_position"].shape[0])


def latest_failure_state_file(
    state: dict,
    initial_ground_state_file: Path | None = None,
) -> Path | None:
    for attempt in reversed(state["random_fall_attempts"]):
        path_value = attempt.get("failure_states")
        if path_value and int(attempt.get("failure_state_count", 0)) > 0:
            path = Path(path_value)
            if path.is_file():
                return path
    return initial_ground_state_file


def train_random_fall_attempt(
    args: argparse.Namespace,
    checkpoint: Path,
    difficulty: float,
    attempt: int,
    failure_states: Path | None,
    resume: bool = False,
) -> tuple[Path, Path]:
    difficulty_code = int(round(difficulty * 100))
    experiment = (
        f"unified_arbitrary_falls_difficulty{difficulty_code:03d}"
        f"_attempt{attempt}"
    )
    log_root = (Path("logs") / "skrl" / "k1_fall_recovery_amp").resolve()
    if resume:
        run_dir, checkpoint, completed = recover_interrupted_training(
            log_root=log_root,
            experiment=experiment,
            seed_checkpoint=checkpoint,
            iterations=args.iterations_per_random_stage,
            checkpoint_interval=args.checkpoint_interval,
            poll_seconds=args.poll_seconds,
        )
        if completed:
            return run_dir, checkpoint
    previous_runs = {path.resolve() for path in log_root.iterdir() if path.is_dir()}
    failure_probability = 0.15 if failure_states is not None else 0.0
    (
        reference_probability,
        standing_probability,
        random_fall_probability,
        failure_probability,
    ) = random_reset_probabilities(
        difficulty,
        failure_probability,
    )
    domain_scale, observation_noise_scale, push_scale = randomization_scales(
        difficulty,
        maximum_domain_scale=args.maximum_domain_randomization_scale,
        maximum_observation_noise_scale=args.maximum_observation_noise_scale,
        maximum_push_scale=args.maximum_push_randomization_scale,
        push_start_difficulty=args.push_start_difficulty,
    )
    command = [
        sys.executable,
        "scripts/skrl/train.py",
        "--task",
        args.task,
        "--checkpoint",
        str(checkpoint),
        "--reset_checkpoint_optimizer",
        "--reset_checkpoint_value",
        "--reset_checkpoint_learning_rate",
        "--task_reward_scale",
        str(args.task_reward_scale),
        "--style_reward_scale",
        str(args.style_reward_scale),
        "--num_envs",
        str(args.num_envs),
        "--max_iterations",
        str(args.iterations_per_random_stage),
        "--checkpoint_interval",
        str(args.checkpoint_interval),
        "--experiment_name",
        experiment,
        "--reference_reset_probability",
        str(reference_probability),
        "--standing_reset_probability_start",
        str(standing_probability),
        "--standing_reset_probability_end",
        str(standing_probability),
        "--random_fall_probability_start",
        str(random_fall_probability),
        "--random_fall_probability_end",
        str(random_fall_probability),
        "--failure_reset_probability_start",
        str(failure_probability),
        "--failure_reset_probability_end",
        str(failure_probability),
        "--failure_state_blend_start",
        "1.0",
        "--failure_state_blend_end",
        "1.0",
        "--random_fall_difficulty_start",
        str(difficulty),
        "--random_fall_difficulty_end",
        str(difficulty),
        "--domain_randomization_scale",
        str(domain_scale),
        "--observation_noise_scale",
        str(observation_noise_scale),
        "--push_randomization_scale",
        str(push_scale),
        "--reference_phase_min",
        "0.0",
        "--reference_phase_max",
        "1.0",
        "--joint_state_tolerance",
        str(args.joint_state_tolerance),
        "--parallel_state_tolerance",
        str(args.parallel_state_tolerance),
        "--headless",
    ]
    if failure_states is not None:
        command.extend(["--failure_state_file", str(failure_states)])
    run_logged(command, next_training_log_path(args.artifacts_dir, experiment))
    run_dir = newest_run(log_root, experiment, previous_runs)
    return run_dir, latest_checkpoint(run_dir)


def evaluate_random_fall_stage(
    args: argparse.Namespace,
    checkpoint: Path,
    difficulty: float,
    attempt: int,
    *,
    robust: bool,
) -> tuple[Path, dict, Path, int]:
    difficulty_code = int(round(difficulty * 100))
    label = (
        f"unified_arbitrary_falls_difficulty{difficulty_code:03d}"
        f"_attempt{attempt}"
    )
    if robust:
        label += "_robust"
    output = args.artifacts_dir / f"{label}_evaluation.json"
    failure_states = args.artifacts_dir / f"{label}_failure_states.npz"
    evaluation_episodes = (
        args.final_evaluation_episodes
        if difficulty == 1.0
        else args.evaluation_episodes
    )
    command = [
        sys.executable,
        "scripts/skrl/evaluate.py",
        "--task",
        args.task,
        "--checkpoint",
        str(checkpoint),
        "--num_envs",
        str(args.evaluation_num_envs),
        "--episodes_per_mode",
        str(evaluation_episodes),
        "--modes",
        *DEFAULT_MODES,
        "--random_fall_difficulty",
        str(difficulty),
        "--joint_state_tolerance",
        str(args.joint_state_tolerance),
        "--parallel_state_tolerance",
        str(args.parallel_state_tolerance),
        "--task_reward_scale",
        str(args.task_reward_scale),
        "--style_reward_scale",
        str(args.style_reward_scale),
        "--output",
        str(output),
        "--failure_states_output",
        str(failure_states),
        "--headless",
    ]
    if robust:
        domain_scale, observation_noise_scale, push_scale = randomization_scales(
            difficulty,
            maximum_domain_scale=args.maximum_domain_randomization_scale,
            maximum_observation_noise_scale=(
                args.maximum_observation_noise_scale
            ),
            maximum_push_scale=args.maximum_push_randomization_scale,
            push_start_difficulty=args.push_start_difficulty,
        )
        command.extend(
            [
                "--domain_randomization_scale",
                str(domain_scale),
                "--observation_noise_scale",
                str(observation_noise_scale),
                "--push_randomization_scale",
                str(push_scale),
            ]
        )
    run_logged(command, args.artifacts_dir / f"{label}_evaluation.log")
    summary = acceptance(
        output,
        list(DEFAULT_MODES),
        args,
        minimum_episodes_per_mode=evaluation_episodes,
    )
    output.with_name(output.stem + "_acceptance.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output, summary, failure_states, failure_state_count(failure_states)


def evaluate_handoff(
    args: argparse.Namespace,
    checkpoint: Path,
    difficulty: float,
    attempt: int,
) -> tuple[Path, dict]:
    difficulty_code = int(round(difficulty * 100))
    output = (
        args.artifacts_dir
        / (
            f"unified_arbitrary_falls_difficulty{difficulty_code:03d}"
            f"_attempt{attempt}_handoff_evaluation.json"
        )
    )
    domain_scale, observation_noise_scale, push_scale = randomization_scales(
        difficulty,
        maximum_domain_scale=args.maximum_domain_randomization_scale,
        maximum_observation_noise_scale=args.maximum_observation_noise_scale,
        maximum_push_scale=args.maximum_push_randomization_scale,
        push_start_difficulty=args.push_start_difficulty,
    )
    command = [
        sys.executable,
        "scripts/skrl/evaluate_handoff.py",
        "--task",
        args.task,
        "--checkpoint",
        str(checkpoint),
        "--num_envs",
        str(args.handoff_evaluation_num_envs),
        "--episodes_per_mode",
        str(args.handoff_evaluation_episodes),
        "--modes",
        *DEFAULT_MODES,
        "--random-fall-difficulty",
        str(difficulty),
        "--joint-state-tolerance",
        str(args.joint_state_tolerance),
        "--parallel-state-tolerance",
        str(args.parallel_state_tolerance),
        "--domain-randomization-scale",
        str(domain_scale),
        "--observation-noise-scale",
        str(observation_noise_scale),
        "--push-randomization-scale",
        str(push_scale),
        "--output",
        str(output),
        "--headless",
    ]
    run_logged(
        command,
        args.artifacts_dir
        / (
            f"unified_arbitrary_falls_difficulty{difficulty_code:03d}"
            f"_attempt{attempt}_handoff_evaluation.log"
        ),
    )
    summary = handoff_acceptance(output, list(DEFAULT_MODES), args)
    output.with_name(output.stem + "_acceptance.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return output, summary


def record_random_fall_video(
    args: argparse.Namespace,
    checkpoint: Path,
    difficulty: float,
    attempt: int,
) -> Path:
    difficulty_code = int(round(difficulty * 100))
    output_dir = (
        args.artifacts_dir
        / "videos"
        / f"random_difficulty{difficulty_code:03d}_attempt{attempt}"
    )
    command = [
        sys.executable,
        "scripts/skrl/play.py",
        "--task",
        args.task,
        "--checkpoint",
        str(checkpoint),
        "--reset_mode",
        "random",
        "--random_fall_difficulty",
        str(difficulty),
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
    run_logged(
        command,
        args.artifacts_dir
        / f"random_difficulty{difficulty_code:03d}_attempt{attempt}_video.log",
    )
    return output_dir


def main() -> int:
    args = parse_args()
    validate_args(args)
    args.faceup_state = args.faceup_state.expanduser().resolve()
    args.artifacts_dir = args.artifacts_dir.expanduser().resolve()
    args.state = args.state.expanduser().resolve()
    if args.initial_ground_state_file is not None:
        args.initial_ground_state_file = args.initial_ground_state_file.expanduser().resolve()
        if not args.initial_ground_state_file.is_file():
            raise FileNotFoundError(
                f"initial ground-state dataset does not exist: {args.initial_ground_state_file}"
            )
    args.artifacts_dir.mkdir(parents=True, exist_ok=True)
    state = (
        json.loads(args.state.read_text(encoding="utf-8"))
        if args.state.is_file()
        else {
            "status": "waiting_for_faceup",
            "accepted_phase_stages": [],
            "phase_attempts": [],
            "accepted_random_fall_stages": [],
            "random_fall_attempts": [],
        }
    )
    state.setdefault("accepted_phase_stages", [])
    state.setdefault("phase_attempts", [])
    state.setdefault("accepted_random_fall_stages", [])
    state.setdefault("random_fall_attempts", [])
    state["configured_task_reward_scale"] = args.task_reward_scale
    state["configured_style_reward_scale"] = args.style_reward_scale
    state["configured_minimum_amp_style_reward_ratio"] = (
        args.minimum_amp_style_reward_ratio
    )
    state["configured_minimum_weighted_amp_style_fraction"] = (
        args.minimum_weighted_amp_style_fraction
    )
    state["configured_initial_ground_state_file"] = (
        None
        if args.initial_ground_state_file is None
        else str(args.initial_ground_state_file)
    )
    state["configured_randomization"] = {
        "maximum_domain_scale": args.maximum_domain_randomization_scale,
        "maximum_observation_noise_scale": args.maximum_observation_noise_scale,
        "maximum_push_scale": args.maximum_push_randomization_scale,
        "push_start_difficulty": args.push_start_difficulty,
    }
    write_state(args.state, state)

    if "current_checkpoint" in state:
        current_checkpoint = Path(state["current_checkpoint"])
    else:
        current_checkpoint = wait_for_faceup_checkpoint(
            args.faceup_state,
            args.poll_seconds,
        )
        state["source_faceup_checkpoint"] = str(current_checkpoint)
        state["current_checkpoint"] = str(current_checkpoint)
        state["status"] = "faceup_ready"
        write_state(args.state, state)

    if (
        not state["phase_attempts"]
        and not state["accepted_phase_stages"]
        and "bootstrap_checkpoint" not in state
    ):
        minimum = args.phase_minima[0]
        maximum = round(minimum + args.phase_width, 6)
        state["status"] = "training_unified_bootstrap_bc"
        write_state(args.state, state)
        bootstrap_run_dir, current_checkpoint = bootstrap_unified_policy(
            args,
            current_checkpoint,
            minimum,
            maximum,
        )
        state["bootstrap_run_dir"] = str(bootstrap_run_dir)
        state["bootstrap_checkpoint"] = str(current_checkpoint)
        state["current_checkpoint"] = str(current_checkpoint)
        state["status"] = "unified_bootstrap_bc_complete"
        write_state(args.state, state)
    elif "bootstrap_checkpoint" in state and not state["phase_attempts"]:
        current_checkpoint = Path(state["bootstrap_checkpoint"])

    for minimum in args.phase_minima:
        maximum = round(minimum + args.phase_width, 6)
        stage_key = f"{minimum:.3f}_{maximum:.3f}"
        if stage_key in state["accepted_phase_stages"]:
            continue
        attempts = [
            item for item in state["phase_attempts"] if item["stage"] == stage_key
        ]
        while len(attempts) < args.maximum_attempts_per_stage:
            attempt = len(attempts) + 1
            label = (
                f"unified_phase_{int(round(minimum * 100)):03d}_"
                f"{int(round(maximum * 100)):03d}_attempt{attempt}"
            )
            same_active_attempt = (
                state.get("active_stage") == stage_key
                and state.get("active_attempt") == attempt
            )
            resume_evaluation = (
                same_active_attempt
                and state.get("status") == "evaluating_reference_phase"
                and state.get("active_run_dir")
                and state.get("active_checkpoint")
            )
            if resume_evaluation:
                run_dir = Path(state["active_run_dir"])
                current_checkpoint = Path(state["active_checkpoint"])
            else:
                resume_training = same_active_attempt and state.get("status") in {
                    "training_reference_phase",
                    "waiting_for_reference_training",
                }
                state["status"] = (
                    "waiting_for_reference_training"
                    if resume_training
                    else "training_reference_phase"
                )
                state["active_stage"] = stage_key
                state["active_attempt"] = attempt
                write_state(args.state, state)
                run_dir, current_checkpoint = train_phase_attempt(
                    args,
                    current_checkpoint,
                    minimum,
                    maximum,
                    attempt,
                    resume=resume_training,
                )
                state["status"] = "evaluating_reference_phase"
                state["active_run_dir"] = str(run_dir)
                state["active_checkpoint"] = str(current_checkpoint)
                write_state(args.state, state)
            evaluation, summary = evaluate_clip_stage(
                args,
                current_checkpoint,
                minimum,
                maximum,
                label,
            )
            record = {
                "stage": stage_key,
                "attempt": attempt,
                "run_dir": str(run_dir),
                "checkpoint": str(current_checkpoint),
                "evaluation": str(evaluation),
                "accepted": summary["accepted"],
                "task_reward_scale": args.task_reward_scale,
                "style_reward_scale": args.style_reward_scale,
            }
            state["phase_attempts"].append(record)
            state["current_checkpoint"] = str(current_checkpoint)
            attempts.append(record)
            if summary["accepted"]:
                state["accepted_phase_stages"].append(stage_key)
                state["status"] = "reference_phase_accepted"
                write_state(args.state, state)
                break
            state["failures"] = summary["failures"]
            write_state(args.state, state)
        else:
            state["status"] = "review_required"
            write_state(args.state, state)
            return 2

    for difficulty in args.random_fall_difficulties:
        stage_key = f"{difficulty:.3f}"
        if stage_key in state["accepted_random_fall_stages"]:
            continue
        attempts = [
            item
            for item in state["random_fall_attempts"]
            if item.get("stage") == stage_key
        ]
        while len(attempts) < args.maximum_attempts_per_stage:
            attempt = len(attempts) + 1
            replay_states = latest_failure_state_file(
                state,
                args.initial_ground_state_file,
            )
            same_active_attempt = (
                state.get("active_random_fall_difficulty") == difficulty
                and state.get("active_attempt") == attempt
            )
            resume_evaluation = (
                same_active_attempt
                and state.get("status") == "evaluating_arbitrary_falls"
                and state.get("active_run_dir")
                and state.get("active_checkpoint")
            )
            if resume_evaluation:
                run_dir = Path(state["active_run_dir"])
                current_checkpoint = Path(state["active_checkpoint"])
            else:
                resume_training = same_active_attempt and state.get("status") in {
                    "training_arbitrary_falls",
                    "waiting_for_arbitrary_fall_training",
                }
                state["status"] = (
                    "waiting_for_arbitrary_fall_training"
                    if resume_training
                    else "training_arbitrary_falls"
                )
                state["active_random_fall_difficulty"] = difficulty
                state["active_attempt"] = attempt
                write_state(args.state, state)
                run_dir, current_checkpoint = train_random_fall_attempt(
                    args,
                    current_checkpoint,
                    difficulty,
                    attempt,
                    replay_states,
                    resume=resume_training,
                )
                state["status"] = "evaluating_arbitrary_falls"
                state["active_run_dir"] = str(run_dir)
                state["active_checkpoint"] = str(current_checkpoint)
                write_state(args.state, state)
            (
                nominal_evaluation,
                nominal_recovery_summary,
                _,
                _,
            ) = evaluate_random_fall_stage(
                args,
                current_checkpoint,
                difficulty,
                attempt,
                robust=False,
            )
            (
                robust_evaluation,
                robust_recovery_summary,
                failure_states,
                failed_state_count,
            ) = evaluate_random_fall_stage(
                args,
                current_checkpoint,
                difficulty,
                attempt,
                robust=True,
            )
            recovery_summary = {
                "accepted": (
                    nominal_recovery_summary["accepted"]
                    and robust_recovery_summary["accepted"]
                ),
                "failures": [
                    *(f"nominal: {failure}" for failure in nominal_recovery_summary["failures"]),
                    *(f"robust: {failure}" for failure in robust_recovery_summary["failures"]),
                ],
            }
            record = {
                "stage": stage_key,
                "difficulty": difficulty,
                "attempt": attempt,
                "run_dir": str(run_dir),
                "checkpoint": str(current_checkpoint),
                "evaluation": str(nominal_evaluation),
                "robust_evaluation": str(robust_evaluation),
                "failure_states": str(failure_states),
                "failure_state_count": failed_state_count,
                "replay_source": None if replay_states is None else str(replay_states),
                "nominal_recovery_accepted": nominal_recovery_summary["accepted"],
                "robust_recovery_accepted": robust_recovery_summary["accepted"],
                "recovery_accepted": recovery_summary["accepted"],
                "accepted": recovery_summary["accepted"],
                "task_reward_scale": args.task_reward_scale,
                "style_reward_scale": args.style_reward_scale,
            }
            final_stage = difficulty == 1.0
            handoff_summary = None
            if final_stage and recovery_summary["accepted"]:
                state["status"] = "evaluating_locomotion_handoff"
                write_state(args.state, state)
                handoff_evaluation, handoff_summary = evaluate_handoff(
                    args,
                    current_checkpoint,
                    difficulty,
                    attempt,
                )
                record["handoff_evaluation"] = str(handoff_evaluation)
                record["handoff_accepted"] = handoff_summary["accepted"]
                record["accepted"] = handoff_summary["accepted"]
            state["random_fall_attempts"].append(record)
            state["current_checkpoint"] = str(current_checkpoint)
            attempts.append(record)
            if record["accepted"]:
                video_dir = record_random_fall_video(
                    args,
                    current_checkpoint,
                    difficulty,
                    attempt,
                )
                record["video_dir"] = str(video_dir)
                state["accepted_random_fall_stages"].append(stage_key)
                state["status"] = "arbitrary_fall_stage_accepted"
                state.pop("failures", None)
                write_state(args.state, state)
                break
            state["failures"] = (
                recovery_summary["failures"]
                if handoff_summary is None
                else handoff_summary["failures"]
            )
            write_state(args.state, state)
        else:
            state["status"] = "review_required"
            write_state(args.state, state)
            return 2

    state["status"] = "unified_recovery_complete"
    state.pop("failures", None)
    write_state(args.state, state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

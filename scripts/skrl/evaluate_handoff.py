#!/usr/bin/env python3

"""Evaluate recovery followed by real K1 AMP standing and commanded locomotion."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Booster-K1-Fall-Recovery-AMP-TaskDriven-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--episodes_per_mode", type=int, default=4096)
parser.add_argument(
    "--modes",
    nargs="+",
    default=["faceup", "facedown", "side_left", "side_right", "random"],
    choices=[
        "faceup",
        "facedown",
        "side_left",
        "side_right",
        "random",
        "standing",
        "reference",
        "terminal_reference",
    ],
)
parser.add_argument("--amp-config", type=Path, default=None)
parser.add_argument(
    "--start-handoff-at-reset",
    action="store_true",
    help="Skip recovery and validate AMP from a standing/reset pose.",
)
parser.add_argument("--recovery-timeout-s", type=float, default=6.0)
parser.add_argument("--engage-ramp-s", type=float, default=None)
parser.add_argument("--zero-command-validation-s", type=float, default=None)
parser.add_argument("--commanded-validation-s", type=float, default=None)
parser.add_argument("--commanded-forward-velocity", type=float, default=None)
parser.add_argument("--handoff-grace-s", type=float, default=1.0)
parser.add_argument("--joint-state-tolerance", type=float, default=0.05)
parser.add_argument("--parallel-state-tolerance", type=float, default=0.05)
parser.add_argument("--recovery-handoff-progress", type=float, default=1.0)
parser.add_argument("--recovery-hold-steps", type=int, default=None)
parser.add_argument("--reference-phase-min", type=float, default=None)
parser.add_argument("--reference-phase-max", type=float, default=None)
parser.add_argument("--random-fall-difficulty", type=float, default=1.0)
parser.add_argument("--domain-randomization-scale", type=float, default=0.0)
parser.add_argument("--push-randomization-scale", type=float, default=0.0)
parser.add_argument("--observation-noise-scale", type=float, default=0.0)
parser.add_argument("--push-interval-min-s", type=float, default=0.8)
parser.add_argument("--push-interval-max-s", type=float, default=2.0)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import booster_train.tasks  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
import torch  # noqa: E402
from booster_train.tasks.manager_based.fall_recovery.amp import (  # noqa: E402
    quat_rotate,
    yaw_quaternion,
)
from booster_train.tasks.manager_based.fall_recovery.amp_locomotion import (  # noqa: E402
    TorchAmpZeroCommandPolicy,
    TorchNativeAmpActor,
    load_amp_locomotion_config,
)
from booster_train.tasks.manager_based.fall_recovery.hardware_config import (  # noqa: E402
    load_k1_hardware_config,
)
from booster_train.tasks.manager_based.fall_recovery.recovery_math import (  # noqa: E402
    recovery_success_config_at_progress,
    strict_recovery_success_config,
)
from booster_train.tasks.manager_based.fall_recovery.robots.k1.env_cfg import (  # noqa: E402
    configure_training_randomization,
)
from isaaclab_rl.skrl import SkrlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rollout_numeric_guard import configure_normalized_policy_action_bounds  # noqa: E402
from skrl.utils.runner.torch import Runner  # noqa: E402


def count_true(value: torch.Tensor) -> int:
    return int(torch.count_nonzero(value).item())


def tensor_stats(values: list[torch.Tensor]) -> dict[str, float] | None:
    if not values or sum(value.numel() for value in values) == 0:
        return None
    data = torch.cat(values).to(torch.float32)
    return {
        "mean": float(torch.mean(data).item()),
        "minimum": float(torch.amin(data).item()),
        "p10": float(torch.quantile(data, 0.1).item()),
        "p50": float(torch.quantile(data, 0.5).item()),
        "p90": float(torch.quantile(data, 0.9).item()),
        "maximum": float(torch.amax(data).item()),
    }


def tensor_column_stats(
    values: list[torch.Tensor],
    names: list[str],
) -> dict[str, dict[str, float]] | None:
    if not values or sum(value.shape[0] for value in values) == 0:
        return None
    data = torch.cat(values).to(torch.float32)
    if data.ndim != 2 or data.shape[1] != len(names):
        raise ValueError("column statistic names do not match tensor shape")
    return {
        name: tensor_stats([data[:, index]])
        for index, name in enumerate(names)
    }


def recovery_gate(
    task_env,
    goal: torch.Tensor,
    success_cfg: dict,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    robot = task_env.robot
    joint_ids = task_env._joint_ids
    height = robot.data.root_pos_w[:, 2] - task_env.scene.env_origins[:, 2]
    gravity_z = robot.data.projected_gravity_b[:, 2]
    linear_velocity = torch.linalg.vector_norm(robot.data.root_lin_vel_b, dim=-1)
    angular_velocity = torch.linalg.vector_norm(robot.data.root_ang_vel_b, dim=-1)
    joint_position = robot.data.joint_pos[:, joint_ids]
    joint_velocity = robot.data.joint_vel[:, joint_ids]
    pose_error = torch.amax(torch.abs(joint_position[:, 2:] - goal[2:]), dim=-1)
    body_joint_velocity = torch.amax(torch.abs(joint_velocity[:, 2:]), dim=-1)
    gate = (
        (height >= success_cfg["minimum_height"])
        & (gravity_z <= success_cfg["maximum_gravity_z"])
        & (linear_velocity <= success_cfg["maximum_linear_velocity"])
        & (angular_velocity <= success_cfg["maximum_angular_velocity"])
        & (pose_error <= success_cfg["maximum_body_pose_error"])
        & (body_joint_velocity <= success_cfg["maximum_body_joint_velocity"])
    )
    return gate, {
        "height": height,
        "gravity_z": gravity_z,
        "linear_velocity": linear_velocity,
        "angular_velocity": angular_velocity,
        "joint_position": joint_position,
        "joint_velocity": joint_velocity,
        "pose_error": pose_error,
        "body_joint_velocity": body_joint_velocity,
    }


def normalized_target(action_term, target: torch.Tensor) -> torch.Tensor:
    return torch.clamp(
        (target - action_term._center) / action_term._scale,
        -1.0,
        1.0,
    )


def configure_mode(task_env, mode: str) -> None:
    if mode == "standing":
        task_env.cfg.amp_reset_mode = "train"
        task_env.cfg.amp_reference_reset_probability_start = 0.0
        task_env.cfg.amp_reference_reset_probability_end = 0.0
        task_env.cfg.amp_standing_reset_probability_start = 1.0
        task_env.cfg.amp_standing_reset_probability_end = 1.0
        task_env.cfg.amp_random_fall_probability_start = 0.0
        task_env.cfg.amp_random_fall_probability_end = 0.0
        task_env.cfg.amp_failure_reset_probability_start = 0.0
        task_env.cfg.amp_failure_reset_probability_end = 0.0
    elif mode in {"reference", "terminal_reference"}:
        task_env.cfg.amp_reset_mode = "reference"
        task_env.cfg.amp_reference_terminal_reset_probability = (
            1.0 if mode == "terminal_reference" else 0.0
        )
    else:
        task_env.cfg.amp_reset_mode = mode


def main() -> None:
    if args_cli.num_envs < 1 or args_cli.episodes_per_mode < 1:
        raise ValueError("environment and episode counts must be positive")
    for name in (
        "recovery_timeout_s",
        "handoff_grace_s",
    ):
        if getattr(args_cli, name) <= 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    for name in ("joint_state_tolerance", "parallel_state_tolerance"):
        if getattr(args_cli, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if not 0.0 <= args_cli.random_fall_difficulty <= 1.0:
        raise ValueError("--random-fall-difficulty must be in [0, 1]")

    amp_config, amp_config_path = load_amp_locomotion_config(args_cli.amp_config)
    hardware_config = load_k1_hardware_config()
    handoff = hardware_config["handoff_policy"]
    engage_ramp_s = (
        float(handoff["engage_ramp_s"])
        if args_cli.engage_ramp_s is None
        else args_cli.engage_ramp_s
    )
    validation_s = (
        float(handoff["zero_command_validation_s"])
        if args_cli.zero_command_validation_s is None
        else args_cli.zero_command_validation_s
    )
    commanded_validation_s = (
        float(handoff["commanded_validation_s"])
        if args_cli.commanded_validation_s is None
        else args_cli.commanded_validation_s
    )
    commanded_forward_velocity = (
        float(handoff["commanded_forward_velocity_mps"])
        if args_cli.commanded_forward_velocity is None
        else args_cli.commanded_forward_velocity
    )
    if (
        engage_ramp_s < 0.0
        or validation_s <= 0.0
        or commanded_validation_s <= 0.0
        or commanded_forward_velocity <= 0.0
    ):
        raise ValueError(
            "engage ramp must be non-negative and validation values positive"
        )

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    configure_training_randomization(
        env_cfg,
        domain_scale=args_cli.domain_randomization_scale,
        push_scale=args_cli.push_randomization_scale,
        observation_noise_scale=args_cli.observation_noise_scale,
        push_interval_range_s=(
            args_cli.push_interval_min_s,
            args_cli.push_interval_max_s,
        ),
    )
    env_cfg.observations.policy.enable_corruption = (
        args_cli.observation_noise_scale > 0.0
    )
    env_cfg.amp_random_fall_difficulty_start = args_cli.random_fall_difficulty
    env_cfg.amp_random_fall_difficulty_end = args_cli.random_fall_difficulty
    if (args_cli.reference_phase_min is None) != (
        args_cli.reference_phase_max is None
    ):
        raise ValueError("reference phase minimum and maximum must be specified together")
    if args_cli.reference_phase_min is not None:
        if not (
            0.0
            <= args_cli.reference_phase_min
            <= args_cli.reference_phase_max
            <= 1.0
        ):
            raise ValueError("reference phase range must satisfy 0 <= min <= max <= 1")
        env_cfg.amp_reference_phase_minimum = args_cli.reference_phase_min
        env_cfg.amp_reference_phase_maximum = args_cli.reference_phase_max
    if not 0.0 <= args_cli.recovery_handoff_progress <= 1.0:
        raise ValueError("--recovery-handoff-progress must be in [0, 1]")
    if args_cli.recovery_handoff_progress == 1.0:
        recovery_success_cfg = strict_recovery_success_config(
            env_cfg.terminations.recovered.params
        )
    else:
        recovery_success_cfg = recovery_success_config_at_progress(
            env_cfg.terminations.recovered.params,
            args_cli.recovery_handoff_progress,
        )
    if args_cli.recovery_hold_steps is not None:
        if args_cli.recovery_hold_steps < 1:
            raise ValueError("--recovery-hold-steps must be positive")
        recovery_success_cfg["hold_steps"] = args_cli.recovery_hold_steps
    env_cfg.terminations.recovered = None
    env_cfg.episode_length_s = (
        args_cli.recovery_timeout_s
        + engage_ramp_s
        + validation_s
        + commanded_validation_s
        + args_cli.handoff_grace_s
        + 0.5
    )
    joint_params = env_cfg.terminations.joint_limit.params
    if "margin_start" in joint_params:
        joint_params["margin_start"] = args_cli.joint_state_tolerance
        joint_params["margin_end"] = args_cli.joint_state_tolerance
    else:
        joint_params["margin"] = args_cli.joint_state_tolerance
    ankle_params = env_cfg.terminations.parallel_ankle.params
    if "motor_tolerance_start" in ankle_params:
        ankle_params["motor_tolerance_start"] = args_cli.parallel_state_tolerance
        ankle_params["motor_tolerance_end"] = args_cli.parallel_state_tolerance
    else:
        ankle_params["motor_tolerance"] = args_cli.parallel_state_tolerance

    agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_amp_cfg_entry_point")
    env_cfg.seed = int(agent_cfg["seed"])
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(raw_env, ml_framework="torch")
    runner = Runner(env, agent_cfg)
    configure_normalized_policy_action_bounds(runner.agent)
    checkpoint = str(Path(args_cli.checkpoint).expanduser().resolve())
    print(f"[INFO] Loading recovery checkpoint: {checkpoint}")
    runner.agent.load(checkpoint)
    runner.agent.enable_training_mode(False, apply_to_models=True)

    task_env = raw_env.unwrapped
    robot = task_env.robot
    action_term = task_env.action_manager.get_term("joint_pos")
    joint_ids = task_env._joint_ids
    device = robot.data.joint_pos.device
    dtype = robot.data.joint_pos.dtype
    goal = torch.tensor(
        task_env.cfg.commands.recovery.standing_joint_position,
        dtype=dtype,
        device=device,
    )
    joint_names = list(task_env.cfg.commands.recovery.joint_names)
    if len(joint_names) != len(joint_ids):
        raise RuntimeError("recovery joint names do not match deployment joint ids")
    body_joint_names = list(task_env.cfg.commands.recovery.joint_names[2:])
    hold_steps_required = int(recovery_success_cfg.pop("hold_steps"))
    success_cfg = recovery_success_cfg
    success_cfg.pop("hold_steps", None)
    success_cfg.pop("asset_cfg", None)
    success_cfg.pop("goal_position", None)
    required_success_keys = {
        "minimum_height",
        "maximum_gravity_z",
        "maximum_linear_velocity",
        "maximum_angular_velocity",
        "maximum_body_pose_error",
        "maximum_body_joint_velocity",
    }
    if not required_success_keys.issubset(success_cfg):
        raise RuntimeError(
            f"recovery success configuration is incomplete: {sorted(success_cfg)}"
        )

    model_dir = amp_config_path.parent
    forward_actor = TorchNativeAmpActor(
        model_dir / amp_config["model_files"]["forward"],
        amp_config["model_sha256"]["forward"],
        device=device,
        dtype=dtype,
    )
    amp_policy = TorchAmpZeroCommandPolicy(
        amp_config,
        forward_actor,
        num_envs=task_env.num_envs,
        device=device,
        dtype=dtype,
    )

    recovery_stiffness = task_env.cfg.actions.joint_pos.stiffness
    recovery_damping = task_env.cfg.actions.joint_pos.damping
    recovery_torque_limit = task_env.cfg.actions.joint_pos.command_torque_limit
    locomotion_stiffness = amp_config["stiffness"]
    locomotion_damping = amp_config["damping"]
    locomotion_torque_limit = amp_config["maximum_torque"]
    validation_steps_required = max(1, round(validation_s / task_env.step_dt))
    commanded_steps_required = max(
        1,
        round(commanded_validation_s / task_env.step_dt),
    )
    ramp_steps_required = (
        0
        if engage_ramp_s == 0.0
        else math.ceil(engage_ramp_s / task_env.step_dt) + 1
    )
    recovery_timeout_steps = math.ceil(args_cli.recovery_timeout_s / task_env.step_dt)
    handoff_timeout_steps = (
        ramp_steps_required
        + validation_steps_required
        + commanded_steps_required
        + math.ceil(args_cli.handoff_grace_s / task_env.step_dt)
    )
    active_terminations = set(task_env.termination_manager.active_terms)
    all_env_ids = torch.arange(task_env.num_envs, device=device)
    results: dict[str, dict] = {}

    for mode in args_cli.modes:
        configure_mode(task_env, mode)
        episodes = 0
        handoff_starts = 0
        successes = 0
        joint_limits = 0
        joint_limit_terminations_by_joint = {name: 0 for name in joint_names}
        ankle_failures = 0
        nonfinite_actions = 0
        handoff_drift_failures = 0
        commanded_progress_failures = 0
        commanded_lateral_failures = 0
        commanded_posture_failures = 0
        timeouts = 0
        other_failures = 0
        parallel_violation_samples = 0
        parallel_total_samples = 0
        recovery_times: list[torch.Tensor] = []
        total_times: list[torch.Tensor] = []
        xy_drifts: list[torch.Tensor] = []
        final_pose_errors: list[torch.Tensor] = []
        final_heights: list[torch.Tensor] = []
        commanded_forward_progress: list[torch.Tensor] = []
        commanded_lateral_drift: list[torch.Tensor] = []
        commanded_minimum_height: list[torch.Tensor] = []
        commanded_maximum_gravity_z: list[torch.Tensor] = []
        handoff_start_linear_speeds: list[torch.Tensor] = []
        handoff_start_angular_speeds: list[torch.Tensor] = []
        handoff_start_pose_errors: list[torch.Tensor] = []
        handoff_start_body_joint_speeds: list[torch.Tensor] = []
        handoff_start_joint_abs_errors: list[torch.Tensor] = []
        drift_failure_start_linear_speeds: list[torch.Tensor] = []
        drift_failure_start_angular_speeds: list[torch.Tensor] = []
        drift_failure_start_pose_errors: list[torch.Tensor] = []
        drift_failure_start_body_joint_speeds: list[torch.Tensor] = []
        timeout_start_linear_speeds: list[torch.Tensor] = []
        timeout_start_angular_speeds: list[torch.Tensor] = []
        timeout_start_pose_errors: list[torch.Tensor] = []
        timeout_start_body_joint_speeds: list[torch.Tensor] = []

        while episodes < args_cli.episodes_per_mode and simulation_app.is_running():
            batch_size = min(
                task_env.num_envs,
                args_cli.episodes_per_mode - episodes,
            )
            tracked = all_env_ids < batch_size
            action_term.set_runtime_impedance(
                recovery_stiffness,
                recovery_damping,
                recovery_torque_limit,
            )
            amp_policy.reset(all_env_ids)
            env._reset_once = True
            with torch.inference_mode():
                observations, _ = env.reset()

            resolved = ~tracked.clone()
            handoff_active = torch.zeros_like(tracked)
            command_active = torch.zeros_like(tracked)
            recovery_stable_steps = torch.zeros(
                task_env.num_envs,
                dtype=torch.long,
                device=device,
            )
            handoff_steps = torch.zeros_like(recovery_stable_steps)
            handoff_stable_steps = torch.zeros_like(recovery_stable_steps)
            command_steps = torch.zeros_like(recovery_stable_steps)
            episode_steps = torch.zeros_like(recovery_stable_steps)
            recovery_time = torch.full(
                (task_env.num_envs,),
                torch.nan,
                dtype=dtype,
                device=device,
            )
            handoff_start_linear_speed = torch.full(
                (task_env.num_envs,),
                torch.nan,
                dtype=dtype,
                device=device,
            )
            handoff_start_angular_speed = torch.full_like(
                handoff_start_linear_speed,
                torch.nan,
            )
            handoff_start_pose_error = torch.full_like(
                handoff_start_linear_speed,
                torch.nan,
            )
            handoff_start_body_joint_speed = torch.full_like(
                handoff_start_linear_speed,
                torch.nan,
            )
            handoff_start_joint_abs_error = torch.full(
                (task_env.num_envs, len(body_joint_names)),
                torch.nan,
                dtype=dtype,
                device=device,
            )
            engage_start = torch.zeros(
                (task_env.num_envs, 22),
                dtype=dtype,
                device=device,
            )
            handoff_root_xy = torch.zeros(
                (task_env.num_envs, 2),
                dtype=dtype,
                device=device,
            )
            handoff_peak_xy_drift = torch.zeros(
                task_env.num_envs,
                dtype=dtype,
                device=device,
            )
            command_start_xy = torch.zeros(
                (task_env.num_envs, 2),
                dtype=dtype,
                device=device,
            )
            command_forward_w = torch.zeros_like(command_start_xy)
            command_lateral_w = torch.zeros_like(command_start_xy)
            command_min_height = torch.full(
                (task_env.num_envs,),
                torch.inf,
                dtype=dtype,
                device=device,
            )
            command_max_gravity_z = torch.full(
                (task_env.num_envs,),
                -torch.inf,
                dtype=dtype,
                device=device,
            )
            command_peak_abs_lateral = torch.zeros(
                task_env.num_envs,
                dtype=dtype,
                device=device,
            )

            if args_cli.start_handoff_at_reset:
                gate, metrics = recovery_gate(task_env, goal, success_cfg)
                pose_domain = (
                    metrics["pose_error"]
                    <= float(handoff["maximum_start_pose_error"])
                )
                eligible = tracked & gate & pose_domain
                rejected = tracked & ~eligible
                other_failures += count_true(rejected)
                resolved[rejected] = True
                start_ids = torch.nonzero(eligible, as_tuple=False).flatten()
                if start_ids.numel():
                    handoff_starts += start_ids.numel()
                    handoff_active[start_ids] = True
                    recovery_time[start_ids] = 0.0
                    handoff_start_linear_speed[start_ids] = metrics[
                        "linear_velocity"
                    ][start_ids]
                    handoff_start_angular_speed[start_ids] = metrics[
                        "angular_velocity"
                    ][start_ids]
                    handoff_start_pose_error[start_ids] = metrics["pose_error"][
                        start_ids
                    ]
                    handoff_start_body_joint_speed[start_ids] = metrics[
                        "body_joint_velocity"
                    ][start_ids]
                    handoff_start_joint_abs_error[start_ids] = torch.abs(
                        metrics["joint_position"][start_ids, 2:] - goal[2:]
                    )
                    engage_start[start_ids] = metrics["joint_position"][start_ids]
                    handoff_root_xy[start_ids] = (
                        robot.data.root_pos_w[start_ids, :2]
                        - task_env.scene.env_origins[start_ids, :2]
                    )
                    amp_policy.reset(start_ids)
                    action_term.set_runtime_impedance(
                        locomotion_stiffness,
                        locomotion_damping,
                        locomotion_torque_limit,
                        start_ids,
                    )

            while not bool(torch.all(resolved)) and simulation_app.is_running():
                with torch.inference_mode():
                    outputs = runner.agent.act(
                        observations,
                        None,
                        timestep=0,
                        timesteps=0,
                    )
                    actions = outputs[-1].get("mean_actions", outputs[0]).clone()
                    current_joint_position = robot.data.joint_pos[:, joint_ids]
                    hold_ids = torch.nonzero(resolved, as_tuple=False).flatten()
                    if hold_ids.numel():
                        actions[hold_ids] = normalized_target(
                            action_term,
                            current_joint_position[hold_ids],
                        ).to(actions.dtype)

                    commanded_handoff = handoff_active & ~resolved
                    handoff_ids = torch.nonzero(
                        commanded_handoff,
                        as_tuple=False,
                    ).flatten()
                    if handoff_ids.numel():
                        requested_command = torch.zeros(
                            (handoff_ids.numel(), 3),
                            dtype=dtype,
                            device=device,
                        )
                        requested_command[
                            command_active[handoff_ids],
                            0,
                        ] = commanded_forward_velocity
                        _, body_target = amp_policy.step(
                            handoff_ids,
                            angular_velocity=robot.data.root_ang_vel_b[handoff_ids],
                            projected_gravity=robot.data.projected_gravity_b[handoff_ids],
                            joint_position=current_joint_position[handoff_ids],
                            joint_velocity=robot.data.joint_vel[handoff_ids][
                                :,
                                joint_ids,
                            ],
                            requested_command=requested_command,
                            dt=task_env.step_dt,
                        )
                        if engage_ramp_s == 0.0:
                            alpha = torch.ones(
                                (handoff_ids.numel(), 1),
                                dtype=dtype,
                                device=device,
                            )
                        else:
                            alpha = torch.clamp(
                                handoff_steps[handoff_ids]
                                .to(dtype)
                                .unsqueeze(-1)
                                * task_env.step_dt
                                / engage_ramp_s,
                                max=1.0,
                            )
                        target = engage_start[handoff_ids].clone()
                        target[:, 2:] = (
                            engage_start[handoff_ids, 2:]
                            + alpha
                            * (
                                body_target
                                - engage_start[handoff_ids, 2:]
                            )
                        )
                        actions[handoff_ids] = normalized_target(
                            action_term,
                            target,
                        ).to(actions.dtype)

                    observations, _, terminated, truncated, _ = env.step(actions)
                    episode_steps[tracked & ~resolved] += 1
                    handoff_steps[commanded_handoff] += 1
                    command_steps[commanded_handoff & command_active] += 1
                    done = (terminated | truncated).reshape(-1)
                    unresolved_before_done = tracked & ~resolved
                    failed_done = done & unresolved_before_done
                    if torch.any(failed_done):
                        termination_masks = {
                            name: task_env.termination_manager.get_term(name).reshape(-1)
                            for name in active_terminations
                        }
                        joint_limit_failures = failed_done & termination_masks.get(
                            "joint_limit",
                            torch.zeros_like(done),
                        )
                        joint_limits += count_true(joint_limit_failures)
                        joint_limit_mask = getattr(
                            task_env,
                            "_hard_joint_limit_violation_mask",
                            None,
                        )
                        if (
                            torch.any(joint_limit_failures)
                            and isinstance(joint_limit_mask, torch.Tensor)
                            and joint_limit_mask.shape
                            == (task_env.num_envs, len(joint_names))
                        ):
                            counts = torch.count_nonzero(
                                joint_limit_mask[joint_limit_failures],
                                dim=0,
                            )
                            for index, name in enumerate(joint_names):
                                joint_limit_terminations_by_joint[name] += int(
                                    counts[index].item()
                                )
                        ankle_failures += count_true(
                            failed_done
                            & termination_masks.get(
                                "parallel_ankle",
                                torch.zeros_like(done),
                            )
                        )
                        nonfinite_actions += count_true(
                            failed_done
                            & termination_masks.get(
                                "nonfinite_action",
                                torch.zeros_like(done),
                            )
                        )
                        timed_out = failed_done & termination_masks.get(
                            "time_out",
                            torch.zeros_like(done),
                        )
                        timeouts += count_true(timed_out)
                        classified = timed_out.clone()
                        for name in (
                            "joint_limit",
                            "parallel_ankle",
                            "nonfinite_action",
                        ):
                            if name in termination_masks:
                                classified |= failed_done & termination_masks[name]
                        other_failures += count_true(failed_done & ~classified)
                        resolved[failed_done] = True
                        handoff_active[failed_done] = False
                        command_active[failed_done] = False

                    gate, metrics = recovery_gate(task_env, goal, success_cfg)
                    motor_margins = []
                    for foot, indexes in enumerate(((14, 15), (20, 21))):
                        motor_margins.append(
                            task_env._parallel.motor_margin(
                                metrics["joint_position"][:, indexes],
                                foot,
                            )
                        )
                    parallel_margin = torch.amin(
                        torch.cat(motor_margins, dim=-1),
                        dim=-1,
                    )
                    sampled = tracked & ~resolved
                    parallel_violation_samples += count_true(
                        sampled & (parallel_margin < 0.0)
                    )
                    parallel_total_samples += count_true(sampled)

                    active_handoff = handoff_active & ~resolved
                    zero_handoff = active_handoff & ~command_active
                    current_xy = (
                        robot.data.root_pos_w[:, :2]
                        - task_env.scene.env_origins[:, :2]
                    )
                    current_xy_drift = torch.linalg.vector_norm(
                        current_xy - handoff_root_xy,
                        dim=-1,
                    )
                    handoff_peak_xy_drift = torch.where(
                        zero_handoff,
                        torch.maximum(
                            handoff_peak_xy_drift,
                            current_xy_drift,
                        ),
                        handoff_peak_xy_drift,
                    )
                    excessive_drift = (
                        zero_handoff
                        & (
                            handoff_peak_xy_drift
                            > float(handoff["maximum_zero_command_xy_drift_m"])
                        )
                    )
                    if torch.any(excessive_drift):
                        drift_failure_start_linear_speeds.append(
                            handoff_start_linear_speed[excessive_drift]
                            .detach()
                            .cpu()
                        )
                        drift_failure_start_angular_speeds.append(
                            handoff_start_angular_speed[excessive_drift]
                            .detach()
                            .cpu()
                        )
                        drift_failure_start_pose_errors.append(
                            handoff_start_pose_error[excessive_drift]
                            .detach()
                            .cpu()
                        )
                        drift_failure_start_body_joint_speeds.append(
                            handoff_start_body_joint_speed[excessive_drift]
                            .detach()
                            .cpu()
                        )
                        handoff_drift_failures += count_true(excessive_drift)
                        resolved[excessive_drift] = True
                        handoff_active[excessive_drift] = False
                        command_active[excessive_drift] = False

                    active_command = active_handoff & command_active
                    command_min_height = torch.where(
                        active_command,
                        torch.minimum(
                            command_min_height,
                            metrics["height"],
                        ),
                        command_min_height,
                    )
                    command_max_gravity_z = torch.where(
                        active_command,
                        torch.maximum(
                            command_max_gravity_z,
                            metrics["gravity_z"],
                        ),
                        command_max_gravity_z,
                    )
                    command_displacement = current_xy - command_start_xy
                    forward_displacement = torch.sum(
                        command_displacement * command_forward_w,
                        dim=-1,
                    )
                    lateral_displacement = torch.sum(
                        command_displacement * command_lateral_w,
                        dim=-1,
                    )
                    command_peak_abs_lateral = torch.where(
                        active_command,
                        torch.maximum(
                            command_peak_abs_lateral,
                            torch.abs(lateral_displacement),
                        ),
                        command_peak_abs_lateral,
                    )

                    pending_recovery = tracked & ~resolved & ~handoff_active
                    recovery_stable_steps = torch.where(
                        pending_recovery & gate,
                        recovery_stable_steps + 1,
                        torch.where(
                            pending_recovery,
                            torch.zeros_like(recovery_stable_steps),
                            recovery_stable_steps,
                        ),
                    )
                    newly_handoff = (
                        pending_recovery
                        & (recovery_stable_steps >= hold_steps_required)
                    )
                    new_ids = torch.nonzero(
                        newly_handoff,
                        as_tuple=False,
                    ).flatten()
                    if new_ids.numel():
                        handoff_starts += new_ids.numel()
                        handoff_active[new_ids] = True
                        recovery_time[new_ids] = (
                            episode_steps[new_ids].to(dtype) * task_env.step_dt
                        )
                        handoff_start_linear_speed[new_ids] = metrics[
                            "linear_velocity"
                        ][new_ids]
                        handoff_start_angular_speed[new_ids] = metrics[
                            "angular_velocity"
                        ][new_ids]
                        handoff_start_pose_error[new_ids] = metrics["pose_error"][
                            new_ids
                        ]
                        handoff_start_body_joint_speed[new_ids] = metrics[
                            "body_joint_velocity"
                        ][new_ids]
                        handoff_start_joint_abs_error[new_ids] = torch.abs(
                            metrics["joint_position"][new_ids, 2:] - goal[2:]
                        )
                        engage_start[new_ids] = metrics["joint_position"][new_ids]
                        handoff_root_xy[new_ids] = (
                            robot.data.root_pos_w[new_ids, :2]
                            - task_env.scene.env_origins[new_ids, :2]
                        )
                        amp_policy.reset(new_ids)
                        action_term.set_runtime_impedance(
                            locomotion_stiffness,
                            locomotion_damping,
                            locomotion_torque_limit,
                            new_ids,
                        )

                    validating = (
                        handoff_active
                        & ~resolved
                        & ~command_active
                        & (handoff_steps >= ramp_steps_required)
                    )
                    locomotion_posture = (
                        metrics["height"]
                        >= float(handoff["minimum_commanded_height_m"])
                    ) & (
                        metrics["gravity_z"]
                        <= float(handoff["maximum_commanded_gravity_z"])
                    )
                    handoff_stable_steps = torch.where(
                        validating & locomotion_posture,
                        handoff_stable_steps + 1,
                        torch.where(
                            validating,
                            torch.zeros_like(handoff_stable_steps),
                            handoff_stable_steps,
                        ),
                    )
                    zero_command_passed = (
                        validating
                        & (
                            handoff_stable_steps
                            >= validation_steps_required
                        )
                    )
                    command_start_ids = torch.nonzero(
                        zero_command_passed,
                        as_tuple=False,
                    ).flatten()
                    if command_start_ids.numel():
                        command_active[command_start_ids] = True
                        command_steps[command_start_ids] = 0
                        command_start_xy[command_start_ids] = current_xy[
                            command_start_ids
                        ]
                        forward_axis = torch.zeros(
                            (command_start_ids.numel(), 3),
                            dtype=dtype,
                            device=device,
                        )
                        forward_axis[:, 0] = 1.0
                        heading = quat_rotate(
                            yaw_quaternion(
                                robot.data.root_quat_w[command_start_ids]
                            ),
                            forward_axis,
                        )
                        command_forward_w[command_start_ids] = heading[:, :2]
                        command_lateral_w[command_start_ids, 0] = -heading[:, 1]
                        command_lateral_w[command_start_ids, 1] = heading[:, 0]
                        command_min_height[command_start_ids] = metrics["height"][
                            command_start_ids
                        ]
                        command_max_gravity_z[command_start_ids] = metrics[
                            "gravity_z"
                        ][command_start_ids]

                    command_finished = (
                        handoff_active
                        & ~resolved
                        & command_active
                        & (command_steps >= commanded_steps_required)
                    )
                    finished_ids = torch.nonzero(
                        command_finished,
                        as_tuple=False,
                    ).flatten()
                    if finished_ids.numel():
                        progress_ok = (
                            forward_displacement[finished_ids]
                            >= float(
                                handoff[
                                    "minimum_commanded_forward_progress_m"
                                ]
                            )
                        )
                        lateral_ok = (
                            command_peak_abs_lateral[finished_ids]
                            <= float(
                                handoff[
                                    "maximum_commanded_lateral_drift_m"
                                ]
                            )
                        )
                        posture_ok = (
                            (
                                command_min_height[finished_ids]
                                >= float(handoff["minimum_commanded_height_m"])
                            )
                            & (
                                command_max_gravity_z[finished_ids]
                                <= float(handoff["maximum_commanded_gravity_z"])
                            )
                        )
                        command_passed = progress_ok & lateral_ok & posture_ok
                        passed_ids = finished_ids[command_passed]
                        successes += passed_ids.numel()
                        commanded_progress_failures += count_true(~progress_ok)
                        commanded_lateral_failures += count_true(~lateral_ok)
                        commanded_posture_failures += count_true(~posture_ok)
                        if passed_ids.numel():
                            recovery_times.append(
                                recovery_time[passed_ids].detach().cpu()
                            )
                            total_times.append(
                                (
                                    episode_steps[passed_ids].to(dtype)
                                    * task_env.step_dt
                                )
                                .detach()
                                .cpu()
                            )
                            xy_drifts.append(
                                handoff_peak_xy_drift[passed_ids].detach().cpu()
                            )
                            final_pose_errors.append(
                                metrics["pose_error"][passed_ids].detach().cpu()
                            )
                            final_heights.append(
                                metrics["height"][passed_ids].detach().cpu()
                            )
                            commanded_forward_progress.append(
                                forward_displacement[passed_ids].detach().cpu()
                            )
                            commanded_lateral_drift.append(
                                command_peak_abs_lateral[passed_ids].detach().cpu()
                            )
                            commanded_minimum_height.append(
                                command_min_height[passed_ids].detach().cpu()
                            )
                            commanded_maximum_gravity_z.append(
                                command_max_gravity_z[passed_ids].detach().cpu()
                            )
                        resolved[finished_ids] = True
                        handoff_active[finished_ids] = False
                        command_active[finished_ids] = False

                    recovery_timeout = (
                        pending_recovery
                        & ~newly_handoff
                        & (episode_steps >= recovery_timeout_steps)
                    )
                    handoff_timeout = (
                        handoff_active
                        & ~resolved
                        & (handoff_steps >= handoff_timeout_steps)
                    )
                    manual_timeout = recovery_timeout | handoff_timeout
                    if torch.any(manual_timeout):
                        timed_out_after_handoff = manual_timeout & handoff_active
                        if torch.any(timed_out_after_handoff):
                            timeout_start_linear_speeds.append(
                                handoff_start_linear_speed[
                                    timed_out_after_handoff
                                ]
                                .detach()
                                .cpu()
                            )
                            timeout_start_angular_speeds.append(
                                handoff_start_angular_speed[
                                    timed_out_after_handoff
                                ]
                                .detach()
                                .cpu()
                            )
                            timeout_start_pose_errors.append(
                                handoff_start_pose_error[
                                    timed_out_after_handoff
                                ]
                                .detach()
                                .cpu()
                            )
                            timeout_start_body_joint_speeds.append(
                                handoff_start_body_joint_speed[
                                    timed_out_after_handoff
                                ]
                                .detach()
                                .cpu()
                            )
                        timeouts += count_true(manual_timeout)
                        resolved[manual_timeout] = True
                        handoff_active[manual_timeout] = False
                        command_active[manual_timeout] = False

            valid_handoff_start = tracked & torch.isfinite(
                handoff_start_linear_speed
            )
            if torch.any(valid_handoff_start):
                handoff_start_linear_speeds.append(
                    handoff_start_linear_speed[valid_handoff_start]
                    .detach()
                    .cpu()
                )
                handoff_start_angular_speeds.append(
                    handoff_start_angular_speed[valid_handoff_start]
                    .detach()
                    .cpu()
                )
                handoff_start_pose_errors.append(
                    handoff_start_pose_error[valid_handoff_start]
                    .detach()
                    .cpu()
                )
                handoff_start_body_joint_speeds.append(
                    handoff_start_body_joint_speed[valid_handoff_start]
                    .detach()
                    .cpu()
                )
                handoff_start_joint_abs_errors.append(
                    handoff_start_joint_abs_error[valid_handoff_start]
                    .detach()
                    .cpu()
                )
            episodes += batch_size

        result = {
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / max(episodes, 1),
            "recovery_gate_count": handoff_starts,
            "recovery_gate_rate": handoff_starts / max(episodes, 1),
            "handoff_conditional_success_rate": successes
            / max(handoff_starts, 1),
            "joint_limit_terminations": joint_limits,
            "joint_limit_terminations_by_joint": {
                name: count
                for name, count in joint_limit_terminations_by_joint.items()
                if count > 0
            },
            "parallel_ankle_terminations": ankle_failures,
            "nonfinite_action_terminations": nonfinite_actions,
            "handoff_drift_failures": handoff_drift_failures,
            "commanded_progress_failures": commanded_progress_failures,
            "commanded_lateral_failures": commanded_lateral_failures,
            "commanded_posture_failures": commanded_posture_failures,
            "timeouts": timeouts,
            "other_failures": other_failures,
            "parallel_state_violation_fraction": parallel_violation_samples
            / max(parallel_total_samples, 1),
            "successful_recovery_time_s": tensor_stats(recovery_times),
            "handoff_start_linear_speed_mps": tensor_stats(
                handoff_start_linear_speeds
            ),
            "handoff_start_angular_speed_radps": tensor_stats(
                handoff_start_angular_speeds
            ),
            "handoff_start_pose_error_rad": tensor_stats(
                handoff_start_pose_errors
            ),
            "handoff_start_body_joint_speed_radps": tensor_stats(
                handoff_start_body_joint_speeds
            ),
            "handoff_start_joint_abs_error_rad": tensor_column_stats(
                handoff_start_joint_abs_errors,
                body_joint_names,
            ),
            "drift_failure_start_linear_speed_mps": tensor_stats(
                drift_failure_start_linear_speeds
            ),
            "drift_failure_start_angular_speed_radps": tensor_stats(
                drift_failure_start_angular_speeds
            ),
            "drift_failure_start_pose_error_rad": tensor_stats(
                drift_failure_start_pose_errors
            ),
            "drift_failure_start_body_joint_speed_radps": tensor_stats(
                drift_failure_start_body_joint_speeds
            ),
            "timeout_start_linear_speed_mps": tensor_stats(
                timeout_start_linear_speeds
            ),
            "timeout_start_angular_speed_radps": tensor_stats(
                timeout_start_angular_speeds
            ),
            "timeout_start_pose_error_rad": tensor_stats(
                timeout_start_pose_errors
            ),
            "timeout_start_body_joint_speed_radps": tensor_stats(
                timeout_start_body_joint_speeds
            ),
            "successful_recovery_and_handoff_time_s": tensor_stats(total_times),
            "zero_command_xy_drift_m": tensor_stats(xy_drifts),
            "handoff_final_body_pose_error_rad": tensor_stats(final_pose_errors),
            "handoff_final_height_m": tensor_stats(final_heights),
            "commanded_forward_progress_m": tensor_stats(
                commanded_forward_progress
            ),
            "commanded_peak_lateral_drift_m": tensor_stats(
                commanded_lateral_drift
            ),
            "commanded_minimum_height_m": tensor_stats(
                commanded_minimum_height
            ),
            "commanded_maximum_gravity_z": tensor_stats(
                commanded_maximum_gravity_z
            ),
            "handoff_contract": {
                "actor": "forward",
                "policy_rate_hz": amp_config["policy_rate_hz"],
                "engage_ramp_s": engage_ramp_s,
                "zero_command_validation_s": validation_s,
                "commanded_forward_velocity_mps": commanded_forward_velocity,
                "commanded_validation_s": commanded_validation_s,
                "minimum_commanded_forward_progress_m": handoff[
                    "minimum_commanded_forward_progress_m"
                ],
                "maximum_commanded_lateral_drift_m": handoff[
                    "maximum_commanded_lateral_drift_m"
                ],
                "minimum_commanded_height_m": handoff[
                    "minimum_commanded_height_m"
                ],
                "maximum_commanded_gravity_z": handoff[
                    "maximum_commanded_gravity_z"
                ],
                "maximum_start_pose_error": handoff["maximum_start_pose_error"],
                "maximum_start_linear_speed_mps": handoff[
                    "maximum_start_linear_speed_mps"
                ],
                "maximum_start_angular_speed_radps": handoff[
                    "maximum_start_angular_speed_radps"
                ],
                "maximum_start_body_joint_speed_radps": handoff[
                    "maximum_start_body_joint_speed_radps"
                ],
                "maximum_zero_command_xy_drift_m": handoff[
                    "maximum_zero_command_xy_drift_m"
                ],
                "model_sha256": amp_config["model_sha256"]["forward"],
            },
            "recovery_entry_contract": {
                **success_cfg,
                "hold_steps": hold_steps_required,
                "curriculum_progress": args_cli.recovery_handoff_progress,
            },
        }
        results[mode] = result
        print(f"[RESULT] {mode}: {json.dumps(result, sort_keys=True)}")

    output = (
        args_cli.output.expanduser().resolve()
        if args_cli.output is not None
        else Path(checkpoint).parent.parent / "handoff_evaluation.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[INFO] Wrote handoff evaluation: {output}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

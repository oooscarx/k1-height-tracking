# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate deterministic K1 AMP recovery from canonical and random falls."""

import argparse
import json
from pathlib import Path

import numpy as np
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Evaluate a K1 AMP recovery checkpoint.")
parser.add_argument("--task", default="Booster-K1-Fall-Recovery-AMP-v0")
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
        "failure",
        "standing",
        "reference",
        "terminal_reference",
    ],
)
parser.add_argument(
    "--observation_noise",
    action="store_true",
    help="Keep the training observation noise enabled during evaluation.",
)
parser.add_argument(
    "--sample_actions",
    action="store_true",
    help="Evaluate sampled policy actions instead of deterministic mean actions.",
)
parser.add_argument(
    "--hold_current_position",
    action="store_true",
    help="Hold the reset joint pose instead of evaluating the policy.",
)
parser.add_argument(
    "--policy_log_std",
    type=float,
    default=None,
    help="Clamp policy log standard deviation to this value during sampled evaluation.",
)
parser.add_argument("--canonical_joint_noise", type=float, default=None)
parser.add_argument("--canonical_ankle_joint_noise", type=float, default=None)
parser.add_argument("--canonical_root_xy_noise", type=float, default=None)
parser.add_argument("--canonical_height_noise", type=float, default=None)
parser.add_argument("--canonical_orientation_noise", type=float, default=None)
parser.add_argument("--canonical_velocity_scale", type=float, default=None)
parser.add_argument(
    "--canonical_initial_height",
    type=float,
    default=None,
    help="Override the canonical ground-pose root height for reset calibration",
)
parser.add_argument("--failure_state_blend", type=float, default=None)
parser.add_argument("--random_fall_difficulty", type=float, default=None)
parser.add_argument("--random_fall_height_minimum", type=float, default=None)
parser.add_argument("--random_fall_height_maximum", type=float, default=None)
parser.add_argument("--domain_randomization_scale", type=float, default=0.0)
parser.add_argument("--push_randomization_scale", type=float, default=0.0)
parser.add_argument("--observation_noise_scale", type=float, default=0.0)
parser.add_argument("--push_interval_min_s", type=float, default=0.8)
parser.add_argument("--push_interval_max_s", type=float, default=2.0)
parser.add_argument("--reference_phase_min", type=float, default=None)
parser.add_argument("--reference_phase_max", type=float, default=None)
parser.add_argument(
    "--reference_clip_index",
    type=int,
    default=None,
    help="Restrict reference resets to one motion clip",
)
parser.add_argument(
    "--reference_clip_indices",
    type=int,
    nargs="+",
    default=None,
    help="Evaluate every requested motion clip in one Isaac process; requires phase bins",
)
parser.add_argument(
    "--reference_phase_bins",
    nargs="+",
    type=float,
    default=None,
    help="Evaluate adjacent reference phase bins from the supplied boundaries",
)
parser.add_argument("--joint_state_tolerance", type=float, default=None)
parser.add_argument("--parallel_state_tolerance", type=float, default=None)
parser.add_argument(
    "--handoff_curriculum_progress",
    type=float,
    default=None,
    help="Evaluate recovery with the success thresholds at this training progress",
)
parser.add_argument("--episode_length_s", type=float, default=None)
parser.add_argument("--task_reward_scale", type=float, default=None)
parser.add_argument("--style_reward_scale", type=float, default=None)
parser.add_argument("--output", type=Path, default=None)
parser.add_argument("--failure_states_output", type=Path, default=None)
parser.add_argument(
    "--failure_states_terminal_only",
    action="store_true",
    help="Export only safe terminal timeout states, excluding failed episode starts",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import booster_train.tasks  # noqa: F401
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from booster_train.tasks.manager_based.fall_recovery.recovery_math import (
    recovery_success_config_at_progress,
    safe_terminal_replay_mask,
    strict_recovery_success_config,
)
from booster_train.tasks.manager_based.fall_recovery.robots.k1.env_cfg import (
    configure_training_randomization,
)
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from rollout_numeric_guard import configure_normalized_policy_action_bounds
from skrl.utils.runner.torch import Runner


def count_true(value: torch.Tensor) -> int:
    return int(torch.count_nonzero(value).item())


def tensor_stats(values: list[torch.Tensor]) -> dict[str, float]:
    value = torch.cat(values).to(torch.float32)
    return {
        "mean": float(torch.mean(value).item()),
        "minimum": float(torch.amin(value).item()),
        "p10": float(torch.quantile(value, 0.1).item()),
        "p50": float(torch.quantile(value, 0.5).item()),
        "p90": float(torch.quantile(value, 0.9).item()),
        "maximum": float(torch.amax(value).item()),
    }


def optional_tensor_stats(values: list[torch.Tensor]) -> dict[str, float] | None:
    if not values or sum(value.numel() for value in values) == 0:
        return None
    return tensor_stats(values)


def optional_vector_stats(values: list[torch.Tensor]) -> dict[str, list[float]] | None:
    if not values or sum(value.shape[0] for value in values) == 0:
        return None
    value = torch.cat(values).to(torch.float32)
    return {
        "mean": torch.mean(value, dim=0).tolist(),
        "minimum": torch.amin(value, dim=0).tolist(),
        "maximum": torch.amax(value, dim=0).tolist(),
    }


def amp_style_outputs(agent, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    normalized = agent._amp_observation_preprocessor(observations)
    logits, _ = agent.models["discriminator"].act(
        {"observations": normalized},
        role="discriminator",
    )
    logits = logits.reshape(-1).to(torch.float32)
    return logits, torch.nn.functional.softplus(logits)


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    observation_noise_scale = args_cli.observation_noise_scale
    if args_cli.observation_noise and observation_noise_scale == 0.0:
        observation_noise_scale = 1.0
    configure_training_randomization(
        env_cfg,
        domain_scale=args_cli.domain_randomization_scale,
        push_scale=args_cli.push_randomization_scale,
        observation_noise_scale=observation_noise_scale,
        push_interval_range_s=(
            args_cli.push_interval_min_s,
            args_cli.push_interval_max_s,
        ),
    )
    env_cfg.observations.policy.enable_corruption = observation_noise_scale > 0.0
    if args_cli.episode_length_s is not None:
        if args_cli.episode_length_s <= 0.0:
            raise ValueError("--episode_length_s must be positive")
        env_cfg.episode_length_s = args_cli.episode_length_s
    if args_cli.canonical_initial_height is not None:
        if args_cli.canonical_initial_height <= 0.0:
            raise ValueError("--canonical_initial_height must be positive")
        env_cfg.commands.recovery.initial_height = args_cli.canonical_initial_height
        env_cfg.amp_canonical_side_height = args_cli.canonical_initial_height
        env_cfg.amp_canonical_random_height = args_cli.canonical_initial_height
    for argument, attribute in (
        ("canonical_joint_noise", "amp_canonical_joint_noise"),
        ("canonical_ankle_joint_noise", "amp_canonical_ankle_joint_noise"),
        ("canonical_root_xy_noise", "amp_canonical_root_xy_noise"),
        ("canonical_height_noise", "amp_canonical_height_noise"),
        ("canonical_orientation_noise", "amp_canonical_orientation_noise"),
        ("canonical_velocity_scale", "amp_canonical_velocity_scale"),
    ):
        value = getattr(args_cli, argument)
        if value is not None:
            if value < 0.0:
                raise ValueError(f"--{argument} must be non-negative")
            setattr(env_cfg, attribute, value)
    if args_cli.failure_state_blend is not None:
        if not 0.0 <= args_cli.failure_state_blend <= 1.0:
            raise ValueError("--failure_state_blend must be in [0, 1]")
        env_cfg.amp_failure_state_blend_start = args_cli.failure_state_blend
        env_cfg.amp_failure_state_blend_end = args_cli.failure_state_blend
    if args_cli.random_fall_difficulty is not None:
        if not 0.0 <= args_cli.random_fall_difficulty <= 1.0:
            raise ValueError("--random_fall_difficulty must be in [0, 1]")
        env_cfg.amp_random_fall_difficulty_start = args_cli.random_fall_difficulty
        env_cfg.amp_random_fall_difficulty_end = args_cli.random_fall_difficulty
    if (
        args_cli.random_fall_height_minimum is None
    ) != (
        args_cli.random_fall_height_maximum is None
    ):
        raise ValueError(
            "random fall height minimum and maximum must be specified together"
        )
    if args_cli.random_fall_height_minimum is not None:
        if (
            args_cli.random_fall_height_minimum <= 0.0
            or args_cli.random_fall_height_maximum
            <= args_cli.random_fall_height_minimum
        ):
            raise ValueError(
                "random fall height range must be positive and increasing"
            )
        env_cfg.amp_random_fall_height_range = (
            args_cli.random_fall_height_minimum,
            args_cli.random_fall_height_maximum,
        )
    for argument, attribute in (
        ("reference_phase_min", "amp_reference_phase_minimum"),
        ("reference_phase_max", "amp_reference_phase_maximum"),
    ):
        value = getattr(args_cli, argument)
        if value is not None:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"--{argument} must be in [0, 1]")
            setattr(env_cfg, attribute, value)
    if env_cfg.amp_reference_phase_minimum > env_cfg.amp_reference_phase_maximum:
        raise ValueError("reference phase minimum must not exceed maximum")
    if args_cli.reference_clip_index is not None:
        if args_cli.reference_clip_index < 0:
            raise ValueError("--reference_clip_index must be non-negative")
        env_cfg.amp_reference_clip_indices = [args_cli.reference_clip_index]
    if args_cli.reference_clip_indices is not None:
        if args_cli.reference_clip_index is not None:
            raise ValueError(
                "--reference_clip_index and --reference_clip_indices are mutually exclusive"
            )
        if args_cli.reference_phase_bins is None:
            raise ValueError("--reference_clip_indices requires --reference_phase_bins")
        if len(set(args_cli.reference_clip_indices)) != len(args_cli.reference_clip_indices):
            raise ValueError("--reference_clip_indices must be unique")
        if any(index < 0 for index in args_cli.reference_clip_indices):
            raise ValueError("--reference_clip_indices must be non-negative")
    if args_cli.reference_phase_bins is not None:
        if len(args_cli.reference_phase_bins) < 2:
            raise ValueError("--reference_phase_bins requires at least two boundaries")
        if any(not 0.0 <= value <= 1.0 for value in args_cli.reference_phase_bins):
            raise ValueError("reference phase bin boundaries must be in [0, 1]")
        if any(
            first >= second
            for first, second in zip(
                args_cli.reference_phase_bins,
                args_cli.reference_phase_bins[1:],
            )
        ):
            raise ValueError("reference phase bin boundaries must be strictly increasing")
    if args_cli.joint_state_tolerance is not None:
        if args_cli.joint_state_tolerance < 0.0:
            raise ValueError("--joint_state_tolerance must be non-negative")
        params = env_cfg.terminations.joint_limit.params
        if "margin_start" in params:
            params["margin_start"] = args_cli.joint_state_tolerance
            params["margin_end"] = args_cli.joint_state_tolerance
        else:
            params["margin"] = args_cli.joint_state_tolerance
    if args_cli.parallel_state_tolerance is not None:
        if args_cli.parallel_state_tolerance < 0.0:
            raise ValueError("--parallel_state_tolerance must be non-negative")
        params = env_cfg.terminations.parallel_ankle.params
        if "motor_tolerance_start" in params:
            params["motor_tolerance_start"] = args_cli.parallel_state_tolerance
            params["motor_tolerance_end"] = args_cli.parallel_state_tolerance
        else:
            params["motor_tolerance"] = args_cli.parallel_state_tolerance
    if args_cli.handoff_curriculum_progress is not None:
        if not 0.0 <= args_cli.handoff_curriculum_progress <= 1.0:
            raise ValueError("--handoff_curriculum_progress must be in [0, 1]")
        params = env_cfg.terminations.recovered.params
        resolved = recovery_success_config_at_progress(
            params,
            args_cli.handoff_curriculum_progress,
        )
        for key in (
            "maximum_angular_velocity",
            "maximum_linear_velocity",
            "maximum_body_pose_error",
            "maximum_body_joint_velocity",
        ):
            if f"{key}_end" in params:
                params[f"{key}_start"] = resolved[key]
                params[f"{key}_end"] = resolved[key]
            else:
                params[key] = resolved[key]
    agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_amp_cfg_entry_point")
    for argument in ("task_reward_scale", "style_reward_scale"):
        value = getattr(args_cli, argument)
        if value is not None:
            if value < 0.0:
                raise ValueError(f"--{argument} must be non-negative")
            agent_cfg["agent"][argument] = value
    task_reward_scale = float(agent_cfg["agent"]["task_reward_scale"])
    style_reward_scale = float(agent_cfg["agent"]["style_reward_scale"])
    if args_cli.policy_log_std is not None:
        if args_cli.policy_log_std > 0.0:
            raise ValueError("--policy_log_std must be non-positive")
        policy_cfg = agent_cfg["models"]["policy"]
        policy_cfg["min_log_std"] = args_cli.policy_log_std
        policy_cfg["max_log_std"] = args_cli.policy_log_std
        policy_cfg["initial_log_std"] = args_cli.policy_log_std
    if args_cli.hold_current_position and args_cli.sample_actions:
        raise ValueError("--hold_current_position and --sample_actions are mutually exclusive")
    env_cfg.seed = int(agent_cfg["seed"])
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    env = SkrlVecEnvWrapper(raw_env, ml_framework="torch")
    runner = Runner(env, agent_cfg)
    configure_normalized_policy_action_bounds(runner.agent)
    checkpoint = str(Path(args_cli.checkpoint).expanduser().resolve())
    print(f"[INFO] Loading checkpoint: {checkpoint}")
    runner.agent.load(checkpoint)
    runner.agent.enable_training_mode(False, apply_to_models=True)

    task_env = raw_env.unwrapped
    robot = task_env.robot
    joint_ids = task_env._joint_ids
    joint_names = [robot.joint_names[index] for index in joint_ids]
    action_term = task_env.action_manager.get_term("joint_pos")
    joint_minimum = action_term._minimum.reshape(-1)
    joint_maximum = action_term._maximum.reshape(-1)
    with torch.inference_mode():
        reference_amp_observations = task_env.collect_reference_motions(8192)
        reference_style_logits, reference_style_rewards = amp_style_outputs(
            runner.agent,
            reference_amp_observations,
        )
    reference_style = {
        "amp_discriminator_logit": tensor_stats([reference_style_logits.cpu()]),
        "amp_style_reward": tensor_stats([reference_style_rewards.cpu()]),
    }
    goal = torch.tensor(
        task_env.cfg.commands.recovery.standing_joint_position,
        dtype=robot.data.joint_pos.dtype,
        device=robot.data.joint_pos.device,
    )
    success_cfg = strict_recovery_success_config(
        task_env.cfg.terminations.recovered.params
    )
    reported_success_thresholds = {
        key: success_cfg[key]
        for key in (
            "hold_steps",
            "minimum_height",
            "maximum_gravity_z",
            "maximum_angular_velocity",
            "maximum_linear_velocity",
            "maximum_body_pose_error",
            "maximum_body_joint_velocity",
        )
    }
    training_terminal_probability = task_env.cfg.amp_reference_terminal_reset_probability
    active_terminations = set(task_env.termination_manager.active_terms)
    results = {}
    failure_initial_joint_position = []
    failure_initial_joint_velocity = []
    failure_initial_root_pose = []
    failure_initial_root_velocity = []
    failure_termination_code = []
    failure_mode = []
    if args_cli.reference_phase_bins is None:
        mode_specs = [
            (
                mode,
                mode,
                env_cfg.amp_reference_phase_minimum,
                env_cfg.amp_reference_phase_maximum,
                args_cli.reference_clip_index if mode in {"reference", "terminal_reference"} else None,
            )
            for mode in args_cli.modes
        ]
    else:
        requested_clips = (
            args_cli.reference_clip_indices
            if args_cli.reference_clip_indices is not None
            else [args_cli.reference_clip_index]
        )
        mode_specs = []
        for clip_index in requested_clips:
            prefix = (
                "reference"
                if clip_index is None
                else f"reference_clip_{clip_index}"
            )
            mode_specs.extend(
                (
                    f"{prefix}_phase_{minimum:.3f}_{maximum:.3f}",
                    "reference",
                    minimum,
                    maximum,
                    clip_index,
                )
                for minimum, maximum in zip(
                    args_cli.reference_phase_bins,
                    args_cli.reference_phase_bins[1:],
                )
            )
    configured_reference_clips = list(env_cfg.amp_reference_clip_indices)
    for (
        result_name,
        mode,
        reference_phase_minimum,
        reference_phase_maximum,
        reference_clip_index,
    ) in mode_specs:
        task_env.cfg.amp_reference_phase_minimum = reference_phase_minimum
        task_env.cfg.amp_reference_phase_maximum = reference_phase_maximum
        task_env.cfg.amp_reference_clip_indices = (
            configured_reference_clips
            if reference_clip_index is None
            else [reference_clip_index]
        )
        if mode == "reference":
            task_env.cfg.amp_reset_mode = "reference"
            task_env.cfg.amp_reference_terminal_reset_probability = 0.0
        elif mode == "terminal_reference":
            task_env.cfg.amp_reset_mode = "reference"
            task_env.cfg.amp_reference_terminal_reset_probability = 1.0
        else:
            task_env.cfg.amp_reset_mode = mode
            task_env.cfg.amp_reference_terminal_reset_probability = training_terminal_probability
        # skrl intentionally resets Isaac Lab only once. Evaluation changes the
        # reset distribution between modes, so each mode needs a real reset.
        env._reset_once = True
        with torch.inference_mode():
            observations, _ = env.reset()
        episodes = 0
        successes = 0
        joint_limits = 0
        joint_limit_terminations_by_joint = {name: 0 for name in joint_names}
        ankle_failures = 0
        ankle_target_fallbacks = 0
        nonfinite_actions = 0
        parallel_violation_samples = 0
        parallel_total_samples = 0
        timeouts = 0
        initial_failure_states_saved = 0
        terminal_timeout_states_saved = 0
        steps = 0
        episode_step = torch.zeros(task_env.num_envs, dtype=torch.long, device=task_env.device)
        episode_initial_joint_position = robot.data.joint_pos[:, joint_ids].detach().clone()
        episode_initial_joint_velocity = robot.data.joint_vel[:, joint_ids].detach().clone()
        episode_initial_minimum_joint_margin = torch.amin(
            torch.minimum(
                episode_initial_joint_position - joint_minimum,
                joint_maximum - episode_initial_joint_position,
            ),
            dim=-1,
        )
        episode_initial_root_pose = torch.cat(
            (
                robot.data.root_pos_w - task_env.scene.env_origins,
                robot.data.root_quat_w,
            ),
            dim=-1,
        ).detach().clone()
        episode_initial_root_velocity = torch.cat(
            (
                robot.data.root_lin_vel_w,
                robot.data.root_ang_vel_w,
            ),
            dim=-1,
        ).detach().clone()
        episode_initial_root_height = (
            robot.data.root_pos_w[:, 2] - task_env.scene.env_origins[:, 2]
        ).detach().clone()
        episode_initial_minimum_body_height = torch.amin(
            robot.data.body_pos_w[:, :, 2]
            - task_env.scene.env_origins[:, 2].unsqueeze(-1),
            dim=-1,
        ).detach().clone()
        peak_height = torch.full((task_env.num_envs,), -torch.inf, device=task_env.device)
        best_gravity_z = torch.full((task_env.num_envs,), torch.inf, device=task_env.device)
        best_linear_velocity = torch.full((task_env.num_envs,), torch.inf, device=task_env.device)
        best_angular_velocity = torch.full((task_env.num_envs,), torch.inf, device=task_env.device)
        best_pose_error = torch.full((task_env.num_envs,), torch.inf, device=task_env.device)
        best_pose_joint_error = torch.full(
            (task_env.num_envs, len(joint_ids) - 2),
            torch.inf,
            device=task_env.device,
        )
        best_joint_velocity = torch.full((task_env.num_envs,), torch.inf, device=task_env.device)
        minimum_parallel_margin = torch.full((task_env.num_envs,), torch.inf, device=task_env.device)
        episode_amp_style_reward = torch.zeros(task_env.num_envs, device=task_env.device)
        episode_amp_discriminator_logit = torch.zeros(
            task_env.num_envs,
            device=task_env.device,
        )
        episode_task_reward = torch.zeros(task_env.num_envs, device=task_env.device)
        completed_peak_height = []
        completed_best_gravity_z = []
        completed_best_linear_velocity = []
        completed_best_angular_velocity = []
        completed_best_pose_error = []
        completed_best_pose_joint_error = []
        completed_best_joint_velocity = []
        completed_minimum_parallel_margin = []
        completed_initial_root_height = []
        completed_initial_minimum_body_height = []
        completed_initial_minimum_joint_margin = []
        completed_amp_style_reward = []
        completed_amp_discriminator_logit = []
        completed_task_reward = []
        completed_episode_step = []
        completed_success_episode_step = []
        completed_maximum_success_hold_step = []
        completed_parallel_ankle_step = []
        completed_parallel_failure_motor_margin = []
        completed_parallel_failure_target_motor_margin = []
        completed_parallel_failure_joint_velocity = []
        completed_parallel_failure_motor_velocity_ratio = []
        completed_joint_failure_limit_margin = []
        completed_joint_failure_joint_position = []
        completed_joint_failure_joint_velocity = []
        stage_names = (
            "height_and_upright",
            "height_upright_and_pose",
            "instant_success",
        )
        reached_stage = {
            name: torch.zeros(task_env.num_envs, dtype=torch.bool, device=task_env.device)
            for name in stage_names
        }
        completed_stage = {name: [] for name in stage_names}
        consecutive_success_step = torch.zeros(
            task_env.num_envs,
            dtype=torch.long,
            device=task_env.device,
        )
        maximum_success_hold_step = torch.zeros_like(consecutive_success_step)
        success_hold_break_count = 0
        success_hold_break_gate_counts = {
            "height": 0,
            "upright": 0,
            "body_pose": 0,
            "linear_velocity": 0,
            "angular_velocity": 0,
            "body_joint_velocity": 0,
        }
        success_hold_break_pose_joint_counts = torch.zeros(
            len(joint_ids) - 2,
            dtype=torch.long,
            device=task_env.device,
        )

        while episodes < args_cli.episodes_per_mode and simulation_app.is_running():
            with torch.inference_mode():
                height = robot.data.root_pos_w[:, 2] - task_env.scene.env_origins[:, 2]
                gravity_z = robot.data.projected_gravity_b[:, 2]
                linear_velocity = torch.linalg.vector_norm(robot.data.root_lin_vel_b, dim=-1)
                angular_velocity = torch.linalg.vector_norm(robot.data.root_ang_vel_b, dim=-1)
                joint_position = robot.data.joint_pos[:, joint_ids]
                joint_velocity = robot.data.joint_vel[:, joint_ids]
                motor_margins = []
                for foot, indexes in enumerate(((14, 15), (20, 21))):
                    motor_margins.append(task_env._parallel.motor_margin(joint_position[:, indexes], foot))
                parallel_margin = torch.amin(torch.cat(motor_margins, dim=-1), dim=-1)
                pose_joint_error = torch.abs(joint_position[:, 2:] - goal[2:])
                pose_error = torch.amax(pose_joint_error, dim=-1)
                body_joint_velocity = torch.amax(torch.abs(joint_velocity[:, 2:]), dim=-1)
                peak_height = torch.maximum(peak_height, height)
                best_gravity_z = torch.minimum(best_gravity_z, gravity_z)
                best_linear_velocity = torch.minimum(best_linear_velocity, linear_velocity)
                best_angular_velocity = torch.minimum(best_angular_velocity, angular_velocity)
                improved_pose = pose_error < best_pose_error
                best_pose_joint_error = torch.where(
                    improved_pose.unsqueeze(-1),
                    pose_joint_error,
                    best_pose_joint_error,
                )
                best_pose_error = torch.minimum(best_pose_error, pose_error)
                best_joint_velocity = torch.minimum(best_joint_velocity, body_joint_velocity)
                minimum_parallel_margin = torch.minimum(minimum_parallel_margin, parallel_margin)
                parallel_violation_samples += count_true(parallel_margin < 0.0)
                parallel_total_samples += parallel_margin.numel()
                height_gate = height >= success_cfg["minimum_height"]
                upright_gate = gravity_z <= success_cfg["maximum_gravity_z"]
                pose_gate = pose_error <= success_cfg["maximum_body_pose_error"]
                linear_velocity_gate = linear_velocity <= success_cfg["maximum_linear_velocity"]
                angular_velocity_gate = angular_velocity <= success_cfg["maximum_angular_velocity"]
                joint_velocity_gate = body_joint_velocity <= success_cfg["maximum_body_joint_velocity"]
                instant_success = (
                    height_gate
                    & upright_gate
                    & pose_gate
                    & linear_velocity_gate
                    & angular_velocity_gate
                    & joint_velocity_gate
                )
                success_hold_break = (consecutive_success_step > 0) & ~instant_success
                if torch.any(success_hold_break):
                    success_hold_break_count += count_true(success_hold_break)
                    break_gates = {
                        "height": height_gate,
                        "upright": upright_gate,
                        "body_pose": pose_gate,
                        "linear_velocity": linear_velocity_gate,
                        "angular_velocity": angular_velocity_gate,
                        "body_joint_velocity": joint_velocity_gate,
                    }
                    for name, gate in break_gates.items():
                        success_hold_break_gate_counts[name] += count_true(
                            success_hold_break & ~gate
                        )
                    pose_break = success_hold_break & ~pose_gate
                    if torch.any(pose_break):
                        limiting_joint = torch.argmax(
                            pose_joint_error[pose_break],
                            dim=-1,
                        )
                        success_hold_break_pose_joint_counts += torch.bincount(
                            limiting_joint,
                            minlength=len(joint_ids) - 2,
                        )
                current_stage = {
                    "height_and_upright": height_gate & upright_gate,
                    "height_upright_and_pose": height_gate & upright_gate & pose_gate,
                    "instant_success": instant_success,
                }
                for name in stage_names:
                    reached_stage[name] |= current_stage[name]
                consecutive_success_step = torch.where(
                    current_stage["instant_success"],
                    consecutive_success_step + 1,
                    0,
                )
                maximum_success_hold_step = torch.maximum(
                    maximum_success_hold_step,
                    consecutive_success_step,
                )

                episode_step += 1
                if args_cli.hold_current_position:
                    actions = torch.clamp(
                        (
                            action_term.deployment_joint_position
                            - action_term._center
                        )
                        / action_term._scale,
                        -1.0,
                        1.0,
                    )
                else:
                    outputs = runner.agent.act(observations, None, timestep=0, timesteps=0)
                    actions = (
                        outputs[0]
                        if args_cli.sample_actions
                        else outputs[-1].get("mean_actions", outputs[0])
                    )
                observations, rewards, terminated, truncated, extras = env.step(actions)
                episode_task_reward += rewards.reshape(-1).to(torch.float32)
                amp_observations = extras.get("amp_obs")
                if amp_observations is None:
                    raise RuntimeError("AMP evaluation step did not return amp_obs")
                amp_logits, amp_style_rewards = amp_style_outputs(
                    runner.agent,
                    amp_observations,
                )
                episode_amp_discriminator_logit += amp_logits
                episode_amp_style_reward += amp_style_rewards

                # skrl keeps termination flags as (num_envs, 1), while the
                # episode accumulators are flat per-environment tensors.
                done = (terminated | truncated).reshape(-1)
                done_count = count_true(done)
                if done_count:
                    termination_masks = {
                        name: task_env.termination_manager.get_term(name).reshape(-1)
                        for name in active_terminations
                    }
                    remaining = args_cli.episodes_per_mode - episodes
                    completed_indices = torch.nonzero(done, as_tuple=False).flatten()[:remaining]
                    completed = torch.zeros_like(done)
                    completed[completed_indices] = True
                    completed_count = completed_indices.numel()
                    recovered = termination_masks["recovered"]
                    failed_episode = completed & ~recovered
                    if torch.any(failed_episode):
                        if not args_cli.failure_states_terminal_only:
                            failure_initial_joint_position.append(
                                episode_initial_joint_position[failed_episode]
                                .detach()
                                .cpu()
                            )
                            failure_initial_joint_velocity.append(
                                episode_initial_joint_velocity[failed_episode]
                                .detach()
                                .cpu()
                            )
                            failure_initial_root_pose.append(
                                episode_initial_root_pose[failed_episode]
                                .detach()
                                .cpu()
                            )
                            failure_initial_root_velocity.append(
                                episode_initial_root_velocity[failed_episode]
                                .detach()
                                .cpu()
                            )
                            code = torch.zeros(
                                count_true(failed_episode),
                                dtype=torch.int64,
                                device=task_env.device,
                            )
                            for name, value in (
                                ("joint_limit", 1),
                                ("parallel_ankle", 2),
                                ("time_out", 4),
                                ("nonfinite_action", 8),
                            ):
                                if name in termination_masks:
                                    code |= (
                                        termination_masks[name][failed_episode].to(
                                            torch.int64
                                        )
                                        * value
                                    )
                            failure_termination_code.append(code.cpu())
                            failure_mode.extend([mode] * code.numel())
                            initial_failure_states_saved += code.numel()
                        safe_terminal_timeout = safe_terminal_replay_mask(
                            failed_episode,
                            termination_masks.get(
                                "time_out",
                                torch.zeros_like(failed_episode),
                            ),
                            task_env._terminal_state_valid,
                            joint_limit=termination_masks.get("joint_limit"),
                            parallel_ankle=termination_masks.get(
                                "parallel_ankle"
                            ),
                            nonfinite_action=termination_masks.get(
                                "nonfinite_action"
                            ),
                        )
                        if torch.any(safe_terminal_timeout):
                            failure_initial_joint_position.append(
                                task_env._terminal_joint_position[
                                    safe_terminal_timeout
                                ]
                                .detach()
                                .cpu()
                            )
                            failure_initial_joint_velocity.append(
                                task_env._terminal_joint_velocity[
                                    safe_terminal_timeout
                                ]
                                .detach()
                                .cpu()
                            )
                            failure_initial_root_pose.append(
                                task_env._terminal_root_pose[
                                    safe_terminal_timeout
                                ]
                                .detach()
                                .cpu()
                            )
                            failure_initial_root_velocity.append(
                                task_env._terminal_root_velocity[
                                    safe_terminal_timeout
                                ]
                                .detach()
                                .cpu()
                            )
                            terminal_count = count_true(safe_terminal_timeout)
                            failure_termination_code.append(
                                torch.full(
                                    (terminal_count,),
                                    4,
                                    dtype=torch.int64,
                                )
                            )
                            failure_mode.extend([mode] * terminal_count)
                            terminal_timeout_states_saved += terminal_count
                    completed_peak_height.append(peak_height[completed].detach().cpu())
                    completed_best_gravity_z.append(best_gravity_z[completed].detach().cpu())
                    completed_best_linear_velocity.append(best_linear_velocity[completed].detach().cpu())
                    completed_best_angular_velocity.append(best_angular_velocity[completed].detach().cpu())
                    completed_best_pose_error.append(best_pose_error[completed].detach().cpu())
                    completed_best_pose_joint_error.append(
                        best_pose_joint_error[completed].detach().cpu()
                    )
                    completed_best_joint_velocity.append(best_joint_velocity[completed].detach().cpu())
                    completed_minimum_parallel_margin.append(
                        minimum_parallel_margin[completed].detach().cpu()
                    )
                    completed_initial_root_height.append(
                        episode_initial_root_height[completed].detach().cpu()
                    )
                    completed_initial_minimum_body_height.append(
                        episode_initial_minimum_body_height[completed]
                        .detach()
                        .cpu()
                    )
                    completed_initial_minimum_joint_margin.append(
                        episode_initial_minimum_joint_margin[completed]
                        .detach()
                        .cpu()
                    )
                    completed_steps = episode_step[completed].clamp_min(1).to(torch.float32)
                    completed_amp_style_reward.append(
                        (episode_amp_style_reward[completed] / completed_steps)
                        .detach()
                        .cpu()
                    )
                    completed_amp_discriminator_logit.append(
                        (episode_amp_discriminator_logit[completed] / completed_steps)
                        .detach()
                        .cpu()
                    )
                    completed_task_reward.append(
                        (episode_task_reward[completed] / completed_steps)
                        .detach()
                        .cpu()
                    )
                    completed_episode_step.append(episode_step[completed].detach().cpu())
                    completed_success_episode_step.append(
                        episode_step[completed & recovered].detach().cpu()
                    )
                    observed_hold_step = maximum_success_hold_step[completed]
                    observed_hold_step = torch.where(
                        recovered[completed],
                        torch.maximum(
                            observed_hold_step,
                            torch.full_like(
                                observed_hold_step,
                                success_cfg["hold_steps"],
                            ),
                        ),
                        observed_hold_step,
                    )
                    completed_maximum_success_hold_step.append(
                        observed_hold_step.detach().cpu()
                    )
                    if "parallel_ankle" in active_terminations:
                        parallel_ankle = task_env.termination_manager.get_term("parallel_ankle").reshape(-1)
                        failed = completed & parallel_ankle
                        completed_parallel_ankle_step.append(episode_step[failed].detach().cpu())
                        if torch.any(failed):
                            completed_parallel_failure_motor_margin.append(
                                task_env._parallel_ankle_state_motor_margin[failed].detach().cpu()
                            )
                            completed_parallel_failure_target_motor_margin.append(
                                task_env._parallel_ankle_target_motor_margin[failed].detach().cpu()
                            )
                            completed_parallel_failure_joint_velocity.append(
                                task_env._parallel_ankle_joint_velocity[failed].detach().cpu()
                            )
                            completed_parallel_failure_motor_velocity_ratio.append(
                                task_env._parallel_ankle_motor_velocity_ratio[failed].detach().cpu()
                            )
                    if "joint_limit" in active_terminations:
                        joint_limit = task_env.termination_manager.get_term(
                            "joint_limit"
                        ).reshape(-1)
                        failed = completed & joint_limit
                        if torch.any(failed):
                            joint_limit_mask = getattr(
                                task_env,
                                "_hard_joint_limit_violation_mask",
                                None,
                            )
                            if (
                                isinstance(joint_limit_mask, torch.Tensor)
                                and joint_limit_mask.shape
                                == (task_env.num_envs, len(joint_names))
                            ):
                                counts = torch.count_nonzero(
                                    joint_limit_mask[failed],
                                    dim=0,
                                )
                                for index, name in enumerate(joint_names):
                                    joint_limit_terminations_by_joint[name] += int(
                                        counts[index].item()
                                    )
                            failed_position = task_env._terminal_joint_position[
                                failed
                            ]
                            failed_velocity = task_env._terminal_joint_velocity[
                                failed
                            ]
                            completed_joint_failure_limit_margin.append(
                                torch.minimum(
                                    failed_position - joint_minimum,
                                    joint_maximum - failed_position,
                                )
                                .detach()
                                .cpu()
                            )
                            completed_joint_failure_joint_position.append(
                                failed_position.detach().cpu()
                            )
                            completed_joint_failure_joint_velocity.append(
                                failed_velocity.detach().cpu()
                            )
                    for name in stage_names:
                        completed_stage[name].append(reached_stage[name][completed].detach().cpu())
                        reached_stage[name][done] = False
                    peak_height[done] = -torch.inf
                    best_gravity_z[done] = torch.inf
                    best_linear_velocity[done] = torch.inf
                    best_angular_velocity[done] = torch.inf
                    best_pose_error[done] = torch.inf
                    best_pose_joint_error[done] = torch.inf
                    best_joint_velocity[done] = torch.inf
                    minimum_parallel_margin[done] = torch.inf
                    episode_amp_style_reward[done] = 0.0
                    episode_amp_discriminator_logit[done] = 0.0
                    episode_task_reward[done] = 0.0
                    episode_step[done] = 0
                    consecutive_success_step[done] = 0
                    maximum_success_hold_step[done] = 0
                    episode_initial_joint_position[done] = robot.data.joint_pos[done][:, joint_ids]
                    episode_initial_joint_velocity[done] = robot.data.joint_vel[done][:, joint_ids]
                    episode_initial_root_pose[done] = torch.cat(
                        (
                            robot.data.root_pos_w[done] - task_env.scene.env_origins[done],
                            robot.data.root_quat_w[done],
                        ),
                        dim=-1,
                    )
                    episode_initial_root_velocity[done] = torch.cat(
                        (
                            robot.data.root_lin_vel_w[done],
                            robot.data.root_ang_vel_w[done],
                        ),
                        dim=-1,
                    )
                    episode_initial_root_height[done] = (
                        robot.data.root_pos_w[done, 2]
                        - task_env.scene.env_origins[done, 2]
                    )
                    episode_initial_minimum_body_height[done] = torch.amin(
                        robot.data.body_pos_w[done, :, 2]
                        - task_env.scene.env_origins[done, 2].unsqueeze(-1),
                        dim=-1,
                    )
                    reset_joint_position = robot.data.joint_pos[done][
                        :, joint_ids
                    ]
                    episode_initial_minimum_joint_margin[done] = torch.amin(
                        torch.minimum(
                            reset_joint_position - joint_minimum,
                            joint_maximum - reset_joint_position,
                        ),
                        dim=-1,
                    )

                    def completed_termination(name: str) -> int:
                        term = task_env.termination_manager.get_term(name).reshape(-1)
                        return count_true(term & completed)

                    episodes += completed_count
                    successes += completed_termination("recovered")
                    joint_limits += completed_termination("joint_limit")
                    if "parallel_ankle" in active_terminations:
                        ankle_failures += completed_termination("parallel_ankle")
                        ankle_target_fallbacks += count_true(
                            action_term.parallel_target_fallback & completed
                        )
                    if "nonfinite_action" in active_terminations:
                        nonfinite_actions += completed_termination("nonfinite_action")
                    timeouts += completed_termination("time_out")
            steps += 1

        task_reward_per_step = torch.cat(completed_task_reward).to(torch.float32)
        style_reward_per_step = torch.cat(completed_amp_style_reward).to(torch.float32)
        weighted_task_reward = task_reward_scale * task_reward_per_step
        weighted_style_reward = style_reward_scale * style_reward_per_step
        style_to_task_ratio = weighted_style_reward / torch.clamp(
            torch.abs(weighted_task_reward),
            min=1.0e-6,
        )
        style_fraction = weighted_style_reward / torch.clamp(
            torch.abs(weighted_task_reward) + weighted_style_reward,
            min=1.0e-6,
        )
        result = {
            "episodes": episodes,
            "successes": successes,
            "success_rate": successes / max(episodes, 1),
            "success_thresholds": reported_success_thresholds,
            "joint_limit_terminations": joint_limits,
            "joint_limit_terminations_by_joint": {
                name: count
                for name, count in joint_limit_terminations_by_joint.items()
                if count > 0
            },
            "parallel_ankle_terminations": ankle_failures,
            "parallel_target_fallback_terminations": ankle_target_fallbacks,
            "nonfinite_action_terminations": nonfinite_actions,
            "timeouts": timeouts,
            "initial_failure_states_saved": initial_failure_states_saved,
            "terminal_timeout_states_saved": terminal_timeout_states_saved,
            "vector_steps": steps,
            "peak_height_m": tensor_stats(completed_peak_height),
            "best_gravity_z": tensor_stats(completed_best_gravity_z),
            "best_linear_velocity_mps": tensor_stats(completed_best_linear_velocity),
            "best_angular_velocity_rad_s": tensor_stats(completed_best_angular_velocity),
            "minimum_body_pose_error_rad": tensor_stats(completed_best_pose_error),
            "minimum_body_pose_error_by_joint_rad": {
                name: tensor_stats(
                    [torch.cat(completed_best_pose_joint_error)[:, index]]
                )
                for index, name in enumerate(joint_names[2:])
            },
            "minimum_body_joint_velocity_rad_s": tensor_stats(completed_best_joint_velocity),
            "minimum_parallel_motor_margin_rad": tensor_stats(completed_minimum_parallel_margin),
            "initial_root_height_m": tensor_stats(completed_initial_root_height),
            "initial_minimum_body_height_m": tensor_stats(
                completed_initial_minimum_body_height
            ),
            "initial_minimum_joint_margin_rad": tensor_stats(
                completed_initial_minimum_joint_margin
            ),
            "amp_discriminator_logit": tensor_stats(
                completed_amp_discriminator_logit
            ),
            "amp_style_reward": tensor_stats(completed_amp_style_reward),
            "amp_reference": reference_style,
            "amp_reward_scales": {
                "task": task_reward_scale,
                "style": style_reward_scale,
            },
            "task_reward_per_step": tensor_stats([task_reward_per_step]),
            "weighted_task_reward_per_step": tensor_stats([weighted_task_reward]),
            "weighted_amp_style_reward_per_step": tensor_stats(
                [weighted_style_reward]
            ),
            "weighted_amp_style_to_task_ratio": tensor_stats(
                [style_to_task_ratio]
            ),
            "weighted_amp_style_fraction": tensor_stats([style_fraction]),
            "episode_steps": tensor_stats(completed_episode_step),
            "successful_episode_steps": optional_tensor_stats(completed_success_episode_step),
            "successful_recovery_time_s": optional_tensor_stats(
                [steps.to(torch.float32) * task_env.step_dt for steps in completed_success_episode_step]
            ),
            "maximum_observed_success_hold_steps": tensor_stats(
                completed_maximum_success_hold_step
            ),
            "maximum_observed_success_hold_time_s": tensor_stats(
                [
                    steps.to(torch.float32) * task_env.step_dt
                    for steps in completed_maximum_success_hold_step
                ]
            ),
            "success_hold_breaks": {
                "total": success_hold_break_count,
                "by_gate": success_hold_break_gate_counts,
                "pose_limiting_joint": {
                    name: int(success_hold_break_pose_joint_counts[index].item())
                    for index, name in enumerate(joint_names[2:])
                },
            },
            "parallel_ankle_termination_steps": optional_tensor_stats(completed_parallel_ankle_step),
            "parallel_ankle_failure_motor_margin_rad": optional_vector_stats(
                completed_parallel_failure_motor_margin
            ),
            "parallel_ankle_failure_target_motor_margin_rad": optional_vector_stats(
                completed_parallel_failure_target_motor_margin
            ),
            "parallel_ankle_failure_joint_velocity_rad_s": optional_vector_stats(
                completed_parallel_failure_joint_velocity
            ),
            "parallel_ankle_failure_motor_velocity_ratio": optional_vector_stats(
                completed_parallel_failure_motor_velocity_ratio
            ),
            "joint_limit_failure_margin_rad": optional_vector_stats(
                completed_joint_failure_limit_margin
            ),
            "joint_limit_failure_joint_position_rad": optional_vector_stats(
                completed_joint_failure_joint_position
            ),
            "joint_limit_failure_joint_velocity_rad_s": optional_vector_stats(
                completed_joint_failure_joint_velocity
            ),
            "parallel_state_violation_fraction": parallel_violation_samples / max(parallel_total_samples, 1),
            "gate_reach_rate": {
                "height": count_true(
                    torch.cat(completed_peak_height) >= success_cfg["minimum_height"]
                )
                / max(episodes, 1),
                "upright": count_true(
                    torch.cat(completed_best_gravity_z) <= success_cfg["maximum_gravity_z"]
                )
                / max(episodes, 1),
                "linear_velocity": count_true(
                    torch.cat(completed_best_linear_velocity)
                    <= success_cfg["maximum_linear_velocity"]
                )
                / max(episodes, 1),
                "angular_velocity": count_true(
                    torch.cat(completed_best_angular_velocity) <= success_cfg["maximum_angular_velocity"]
                )
                / max(episodes, 1),
                "body_pose": count_true(
                    torch.cat(completed_best_pose_error) <= success_cfg["maximum_body_pose_error"]
                )
                / max(episodes, 1),
                "body_joint_velocity": count_true(
                    torch.cat(completed_best_joint_velocity) <= success_cfg["maximum_body_joint_velocity"]
                )
                / max(episodes, 1),
            },
            "combined_gate_reach_rate": {
                name: count_true(torch.cat(completed_stage[name])) / max(episodes, 1)
                for name in stage_names
            },
        }
        results[result_name] = result
        print(f"[RESULT] {result_name}: {json.dumps(result, sort_keys=True)}")

    output = args_cli.output or Path(checkpoint).parent.parent / "evaluation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[INFO] Wrote evaluation: {output}")
    if args_cli.failure_states_output is not None:
        failure_output = args_cli.failure_states_output.expanduser().resolve()
        failure_output.parent.mkdir(parents=True, exist_ok=True)
        if failure_initial_joint_position:
            joint_position = torch.cat(failure_initial_joint_position).numpy()
            joint_velocity = torch.cat(failure_initial_joint_velocity).numpy()
            root_pose = torch.cat(failure_initial_root_pose).numpy()
            root_velocity = torch.cat(failure_initial_root_velocity).numpy()
            termination_code = torch.cat(failure_termination_code).numpy()
            modes = np.asarray(failure_mode)
        else:
            joint_position = np.empty((0, len(joint_ids)), dtype=np.float32)
            joint_velocity = np.empty_like(joint_position)
            root_pose = np.empty((0, 7), dtype=np.float32)
            root_velocity = np.empty((0, 6), dtype=np.float32)
            termination_code = np.empty((0,), dtype=np.int64)
            modes = np.empty((0,), dtype="<U1")
        np.savez_compressed(
            failure_output,
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            root_pose=root_pose,
            root_velocity=root_velocity,
            termination_code=termination_code,
            mode=modes,
        )
        print(f"[INFO] Wrote {joint_position.shape[0]} failure replay states: {failure_output}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .actions import K1ParallelJointPositionAction
from .commands import RecoveryReferenceCommand
from .recovery_math import (
    bounded_potential_progress,
    held_recovery_completion_bonus,
    quality_weighted_completion_bonus,
    stability_hold_progress_reward,
    strict_threshold_bottleneck_quality,
    training_curriculum_value,
)


def _robot(env, asset_cfg: SceneEntityCfg):
    return env.scene[asset_cfg.name]


def _command(env, command_name: str) -> RecoveryReferenceCommand:
    return env.command_manager.get_term(command_name)


def _action(env, action_name: str) -> K1ParallelJointPositionAction:
    return env.action_manager.get_term(action_name)


def _height(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    return robot.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]


def recovery_success_mask(
    env,
    asset_cfg: SceneEntityCfg,
    minimum_height: float,
    maximum_gravity_z: float,
    maximum_angular_velocity: float,
    maximum_linear_velocity: float,
    goal_position: list[float],
    maximum_body_pose_error: float,
    maximum_body_joint_velocity: float,
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    goal = torch.tensor(
        goal_position,
        dtype=joint_pos.dtype,
        device=joint_pos.device,
    )
    body_pose_error = torch.amax(torch.abs(joint_pos[:, 2:] - goal[2:]), dim=-1)
    body_joint_velocity = torch.amax(torch.abs(joint_vel[:, 2:]), dim=-1)
    return (
        (_height(env, asset_cfg) >= minimum_height)
        & (robot.data.projected_gravity_b[:, 2] <= maximum_gravity_z)
        & (torch.linalg.vector_norm(robot.data.root_ang_vel_b, dim=-1) <= maximum_angular_velocity)
        & (torch.linalg.vector_norm(robot.data.root_lin_vel_b, dim=-1) <= maximum_linear_velocity)
        & (body_pose_error <= maximum_body_pose_error)
        & (body_joint_velocity <= maximum_body_joint_velocity)
    )


def recovery_success_curriculum_values(
    env,
    *,
    maximum_angular_velocity_start: float,
    maximum_angular_velocity_end: float,
    maximum_linear_velocity_start: float,
    maximum_linear_velocity_end: float,
    maximum_body_pose_error_start: float,
    maximum_body_pose_error_end: float,
    maximum_body_joint_velocity_start: float,
    maximum_body_joint_velocity_end: float,
    curriculum_steps: int,
) -> tuple[float, float, float, float]:
    reset_mode = getattr(env.cfg, "amp_reset_mode", "evaluation")
    adaptive = bool(
        getattr(env.cfg, "amp_handoff_curriculum_adaptive", False)
    )
    adaptive_progress = getattr(env, "_handoff_curriculum_progress", None)
    if reset_mode == "train" and adaptive and adaptive_progress is not None:
        progress = min(max(float(adaptive_progress), 0.0), 1.0)
        return (
            maximum_angular_velocity_start
            + progress * (maximum_angular_velocity_end - maximum_angular_velocity_start),
            maximum_linear_velocity_start
            + progress * (maximum_linear_velocity_end - maximum_linear_velocity_start),
            maximum_body_pose_error_start
            + progress * (maximum_body_pose_error_end - maximum_body_pose_error_start),
            maximum_body_joint_velocity_start
            + progress
            * (maximum_body_joint_velocity_end - maximum_body_joint_velocity_start),
        )
    common = {
        "common_step_counter": env.common_step_counter,
        "num_envs": env.num_envs,
        "reset_mode": reset_mode,
        "curriculum_steps": curriculum_steps,
    }
    return (
        training_curriculum_value(
            start=maximum_angular_velocity_start,
            end=maximum_angular_velocity_end,
            **common,
        ),
        training_curriculum_value(
            start=maximum_linear_velocity_start,
            end=maximum_linear_velocity_end,
            **common,
        ),
        training_curriculum_value(
            start=maximum_body_pose_error_start,
            end=maximum_body_pose_error_end,
            **common,
        ),
        training_curriculum_value(
            start=maximum_body_joint_velocity_start,
            end=maximum_body_joint_velocity_end,
            **common,
        ),
    )


def strict_handoff_quality(
    env,
    minimum_height: float,
    maximum_gravity_z: float,
    maximum_angular_velocity_start: float,
    maximum_angular_velocity_end: float,
    maximum_linear_velocity_start: float,
    maximum_linear_velocity_end: float,
    goal_position: list[float],
    maximum_body_pose_error_start: float,
    maximum_body_pose_error_end: float,
    maximum_body_joint_velocity_start: float,
    maximum_body_joint_velocity_end: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Continuously improve the weakest handoff metric toward its strict limit."""
    robot = _robot(env, asset_cfg)
    joint_pos = robot.data.joint_pos[:, asset_cfg.joint_ids]
    joint_vel = robot.data.joint_vel[:, asset_cfg.joint_ids]
    goal = torch.tensor(goal_position, dtype=joint_pos.dtype, device=joint_pos.device)
    values = torch.stack(
        (
            torch.linalg.vector_norm(robot.data.root_ang_vel_b, dim=-1),
            torch.linalg.vector_norm(robot.data.root_lin_vel_b, dim=-1),
            torch.amax(torch.abs(joint_pos[:, 2:] - goal[2:]), dim=-1),
            torch.amax(torch.abs(joint_vel[:, 2:]), dim=-1),
        ),
        dim=-1,
    )
    loose = values.new_tensor(
        (
            maximum_angular_velocity_start,
            maximum_linear_velocity_start,
            maximum_body_pose_error_start,
            maximum_body_joint_velocity_start,
        )
    )
    strict = values.new_tensor(
        (
            maximum_angular_velocity_end,
            maximum_linear_velocity_end,
            maximum_body_pose_error_end,
            maximum_body_joint_velocity_end,
        )
    )
    standing = (
        (_height(env, asset_cfg) >= minimum_height)
        & (robot.data.projected_gravity_b[:, 2] <= maximum_gravity_z)
    ).to(values.dtype)
    return strict_threshold_bottleneck_quality(values, loose, strict) * standing


def upright_exp(env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    gravity = _robot(env, asset_cfg).data.projected_gravity_b
    target = torch.tensor((0.0, 0.0, -1.0), device=gravity.device)
    error = torch.sum(torch.square(gravity - target), dim=-1)
    return torch.exp(-error / (std * std))


def trunk_height_progress(
    env,
    minimum_height: float,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return torch.clamp(
        (_height(env, asset_cfg) - minimum_height) / (target_height - minimum_height),
        0.0,
        1.0,
    )


def trunk_height_exp(
    env,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    height = torch.clamp(_height(env, asset_cfg), min=0.0, max=target_height)
    return torch.exp(height) - 1.0


def trunk_target_height_exp(
    env,
    target_height: float,
    sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    error = _height(env, asset_cfg) - target_height
    return torch.exp(-torch.square(error) / sigma)


def _upright_scale(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    gravity_z = _robot(env, asset_cfg).data.projected_gravity_b[:, 2]
    return torch.clamp(-gravity_z, 0.0, 1.0)


def trunk_target_height_upright_exp(
    env,
    target_height: float,
    sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return trunk_target_height_exp(env, target_height, sigma, asset_cfg) * _upright_scale(
        env, asset_cfg
    )


def standing_height(
    env,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return (_height(env, asset_cfg) >= minimum_height).to(torch.float32)


def standing_height_upright(
    env,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return standing_height(env, minimum_height, asset_cfg) * _upright_scale(env, asset_cfg)


def body_height_exp(
    env,
    target_height: float,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    height = robot.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(-1)
    height = torch.clamp(torch.mean(height, dim=-1), min=0.0, max=target_height)
    return torch.exp(height) - 1.0


def trunk_height_increase(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    height = _height(env, asset_cfg)
    if not hasattr(env, "_previous_recovery_height"):
        env._previous_recovery_height = height.clone()
    previous = torch.where(env.episode_length_buf <= 1, height, env._previous_recovery_height)
    reward = (height > previous).to(torch.float32)
    env._previous_recovery_height = height.clone()
    return reward


def standing_pose_exp(
    env,
    goal_position: list[float],
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    joint_pos = _robot(env, asset_cfg).data.joint_pos[:, asset_cfg.joint_ids]
    goal = torch.tensor(goal_position, dtype=joint_pos.dtype, device=joint_pos.device)
    error = torch.mean(torch.square(joint_pos - goal), dim=-1)
    return torch.exp(-error / (std * std))


def standing_pose_max_error_exp(
    env,
    goal_position: list[float],
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward the least-aligned joint so one limb cannot hide in the mean."""
    joint_pos = _robot(env, asset_cfg).data.joint_pos[:, asset_cfg.joint_ids]
    goal = torch.tensor(goal_position, dtype=joint_pos.dtype, device=joint_pos.device)
    error = torch.amax(torch.square(joint_pos - goal), dim=-1)
    return torch.exp(-error / (std * std))


def body_up_exp(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    gravity_z = _robot(env, asset_cfg).data.projected_gravity_b[:, 2]
    return torch.exp(-gravity_z)


def body_orientation_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    gravity_xy = _robot(env, asset_cfg).data.projected_gravity_b[:, :2]
    return torch.sum(torch.square(gravity_xy), dim=-1)


def upright_progress(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    gravity_z = _robot(env, asset_cfg).data.projected_gravity_b[:, 2]
    return torch.clamp(-gravity_z, 0.0, 1.0)


def gravity_z_progress(
    env,
    initial_gravity_z: float,
    target_gravity_z: float = -1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    gravity_z = _robot(env, asset_cfg).data.projected_gravity_b[:, 2]
    return torch.clamp(
        (initial_gravity_z - gravity_z) / (initial_gravity_z - target_gravity_z),
        0.0,
        1.0,
    )


def feet_height_exp(env, scale: float, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    feet_height = robot.data.body_pos_w[:, asset_cfg.body_ids, 2] - env.scene.env_origins[:, 2].unsqueeze(-1)
    feet_height = torch.clamp(torch.mean(feet_height, dim=-1), min=0.0)
    return torch.exp(-scale * feet_height)


def bilateral_action_symmetry(
    env,
    action_name: str,
    left_indices: list[int],
    right_indices: list[int],
    mirror_signs: list[float],
) -> torch.Tensor:
    actions = _action(env, action_name).raw_actions
    left = actions[:, left_indices]
    right = actions[:, right_indices]
    signs = torch.tensor(mirror_signs, dtype=actions.dtype, device=actions.device)
    return torch.linalg.vector_norm(left * signs - right, dim=-1)


def standing_pose_after_height_exp(
    env,
    goal_position: list[float],
    std: float,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    reward = standing_pose_exp(env, goal_position, std, asset_cfg)
    return reward * (_height(env, asset_cfg) >= minimum_height).to(torch.float32)


def standing_pose_after_height_upright_exp(
    env,
    goal_position: list[float],
    std: float,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return standing_pose_after_height_exp(
        env,
        goal_position,
        std,
        minimum_height,
        asset_cfg,
    ) * _upright_scale(env, asset_cfg)


def standing_pose_max_error_after_height_upright_exp(
    env,
    goal_position: list[float],
    std: float,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    reward = standing_pose_max_error_exp(env, goal_position, std, asset_cfg)
    height_gate = (_height(env, asset_cfg) >= minimum_height).to(torch.float32)
    return reward * height_gate * _upright_scale(env, asset_cfg)


def _standing_pose_quality_progress(
    env,
    quality: torch.Tensor,
    state_name: str,
    maximum_rate: float,
) -> torch.Tensor:
    previous = getattr(env, state_name, quality)
    previous = torch.where(env.episode_length_buf <= 1, quality, previous)
    setattr(env, state_name, quality.clone())
    return bounded_potential_progress(
        quality,
        previous,
        env.step_dt,
        maximum_rate,
    )


def standing_pose_progress_after_height_upright_exp(
    env,
    goal_position: list[float],
    std: float,
    minimum_height: float,
    maximum_rate: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    quality = standing_pose_after_height_upright_exp(
        env,
        goal_position,
        std,
        minimum_height,
        asset_cfg,
    )
    return _standing_pose_quality_progress(
        env,
        quality,
        "_previous_standing_pose_quality",
        maximum_rate,
    )


def standing_pose_max_error_progress_after_height_upright_exp(
    env,
    goal_position: list[float],
    std: float,
    minimum_height: float,
    maximum_rate: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    quality = standing_pose_max_error_after_height_upright_exp(
        env,
        goal_position,
        std,
        minimum_height,
        asset_cfg,
    )
    return _standing_pose_quality_progress(
        env,
        quality,
        "_previous_standing_pose_max_error_quality",
        maximum_rate,
    )


def body_stillness_exp(env, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    robot = _robot(env, asset_cfg)
    error = torch.sum(torch.square(robot.data.root_lin_vel_b), dim=-1)
    error += torch.sum(torch.square(robot.data.root_ang_vel_b), dim=-1)
    error += 0.02 * torch.mean(torch.square(robot.data.joint_vel), dim=-1)
    return torch.exp(-error / (std * std))


def body_stillness_after_height_exp(
    env,
    std: float,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    reward = body_stillness_exp(env, std, asset_cfg)
    return reward * (_height(env, asset_cfg) >= minimum_height).to(torch.float32)


def body_stillness_after_height_upright_exp(
    env,
    std: float,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return body_stillness_after_height_exp(
        env,
        std,
        minimum_height,
        asset_cfg,
    ) * _upright_scale(env, asset_cfg)


def early_recovery_bonus(
    env,
    hold_steps: int,
    minimum_height: float,
    maximum_gravity_z: float,
    maximum_angular_velocity: float,
    maximum_linear_velocity: float,
    goal_position: list[float],
    maximum_body_pose_error: float,
    maximum_body_joint_velocity: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Pay the time-scaled completion bonus once, after the full stability hold."""
    if hold_steps < 1:
        raise ValueError("hold_steps must be positive")
    success = recovery_success_mask(
        env,
        asset_cfg,
        minimum_height,
        maximum_gravity_z,
        maximum_angular_velocity,
        maximum_linear_velocity,
        goal_position,
        maximum_body_pose_error,
        maximum_body_joint_velocity,
    )
    stable_steps = getattr(env, "_recovery_stable_steps", None)
    if stable_steps is None:
        stable_steps = success.to(torch.long)
    # RewardManager multiplies by dt. Scaling by hold_steps preserves the
    # intended total bonus of a clean hold without rewarding near-completions.
    return held_recovery_completion_bonus(
        success,
        stable_steps,
        hold_steps,
        env.episode_length_buf,
        env.max_episode_length,
    )


def recovery_stability_hold_progress_curriculum(
    env,
    hold_steps: int,
    minimum_height: float,
    maximum_gravity_z: float,
    maximum_angular_velocity_start: float,
    maximum_angular_velocity_end: float,
    maximum_linear_velocity_start: float,
    maximum_linear_velocity_end: float,
    goal_position: list[float],
    maximum_body_pose_error_start: float,
    maximum_body_pose_error_end: float,
    maximum_body_joint_velocity_start: float,
    maximum_body_joint_velocity_end: float,
    curriculum_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward consecutive handoff-ready steps while preserving the exact gate."""
    angular, linear, pose, joint = recovery_success_curriculum_values(
        env,
        maximum_angular_velocity_start=maximum_angular_velocity_start,
        maximum_angular_velocity_end=maximum_angular_velocity_end,
        maximum_linear_velocity_start=maximum_linear_velocity_start,
        maximum_linear_velocity_end=maximum_linear_velocity_end,
        maximum_body_pose_error_start=maximum_body_pose_error_start,
        maximum_body_pose_error_end=maximum_body_pose_error_end,
        maximum_body_joint_velocity_start=maximum_body_joint_velocity_start,
        maximum_body_joint_velocity_end=maximum_body_joint_velocity_end,
        curriculum_steps=curriculum_steps,
    )
    success = recovery_success_mask(
        env,
        asset_cfg,
        minimum_height,
        maximum_gravity_z,
        angular,
        linear,
        goal_position,
        pose,
        joint,
    )
    stable_steps = getattr(env, "_recovery_stable_steps", None)
    if stable_steps is None:
        stable_steps = success.to(torch.long)
    return stability_hold_progress_reward(success, stable_steps, hold_steps)


def early_recovery_bonus_curriculum(
    env,
    hold_steps: int,
    minimum_height: float,
    maximum_gravity_z: float,
    maximum_angular_velocity_start: float,
    maximum_angular_velocity_end: float,
    maximum_linear_velocity_start: float,
    maximum_linear_velocity_end: float,
    goal_position: list[float],
    maximum_body_pose_error_start: float,
    maximum_body_pose_error_end: float,
    maximum_body_joint_velocity_start: float,
    maximum_body_joint_velocity_end: float,
    curriculum_steps: int,
    minimum_strict_quality_scale: float = 1.0,
    maximum_strict_quality_scale: float = 1.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    angular, linear, pose, joint = recovery_success_curriculum_values(
        env,
        maximum_angular_velocity_start=maximum_angular_velocity_start,
        maximum_angular_velocity_end=maximum_angular_velocity_end,
        maximum_linear_velocity_start=maximum_linear_velocity_start,
        maximum_linear_velocity_end=maximum_linear_velocity_end,
        maximum_body_pose_error_start=maximum_body_pose_error_start,
        maximum_body_pose_error_end=maximum_body_pose_error_end,
        maximum_body_joint_velocity_start=maximum_body_joint_velocity_start,
        maximum_body_joint_velocity_end=maximum_body_joint_velocity_end,
        curriculum_steps=curriculum_steps,
    )
    completion = early_recovery_bonus(
        env,
        hold_steps,
        minimum_height,
        maximum_gravity_z,
        angular,
        linear,
        goal_position,
        pose,
        joint,
        asset_cfg,
    )
    strict_quality = strict_handoff_quality(
        env,
        minimum_height,
        maximum_gravity_z,
        maximum_angular_velocity_start,
        maximum_angular_velocity_end,
        maximum_linear_velocity_start,
        maximum_linear_velocity_end,
        goal_position,
        maximum_body_pose_error_start,
        maximum_body_pose_error_end,
        maximum_body_joint_velocity_start,
        maximum_body_joint_velocity_end,
        asset_cfg,
    )
    return quality_weighted_completion_bonus(
        completion,
        strict_quality,
        minimum_strict_quality_scale,
        maximum_strict_quality_scale,
    )


def recovery_potential_progress(
    env,
    minimum_height: float,
    target_height: float,
    height_fraction: float,
    maximum_rate: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    coupling_fraction: float = 0.0,
) -> torch.Tensor:
    """Reward signed progress toward upright standing without prescribing a path."""
    if target_height <= minimum_height:
        raise ValueError("target_height must be greater than minimum_height")
    if not 0.0 <= height_fraction <= 1.0:
        raise ValueError("height_fraction must be in [0, 1]")
    if not 0.0 <= coupling_fraction <= 1.0:
        raise ValueError("coupling_fraction must be in [0, 1]")
    if maximum_rate <= 0.0:
        raise ValueError("maximum_rate must be positive")

    robot = _robot(env, asset_cfg)
    height = torch.clamp(
        (_height(env, asset_cfg) - minimum_height) / (target_height - minimum_height),
        0.0,
        1.0,
    )
    upright = torch.clamp(-robot.data.projected_gravity_b[:, 2], 0.0, 1.0)
    linear = height_fraction * height + (1.0 - height_fraction) * upright
    coupled = height * upright
    potential = (1.0 - coupling_fraction) * linear + coupling_fraction * coupled
    if not hasattr(env, "_previous_recovery_potential"):
        env._previous_recovery_potential = potential.clone()
    previous = torch.where(
        env.episode_length_buf <= 1,
        potential,
        env._previous_recovery_potential,
    )
    progress_rate = (potential - previous) / env.step_dt
    env._previous_recovery_potential = potential.clone()
    return torch.clamp(progress_rate, -maximum_rate, maximum_rate)


def recovery_time_cost(
    env,
    minimum_height: float,
    maximum_gravity_z: float,
    maximum_angular_velocity: float,
    maximum_linear_velocity: float,
    goal_position: list[float],
    maximum_body_pose_error: float,
    maximum_body_joint_velocity: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Charge one unit per second until the handoff-ready state is reached."""
    success = recovery_success_mask(
        env,
        asset_cfg,
        minimum_height,
        maximum_gravity_z,
        maximum_angular_velocity,
        maximum_linear_velocity,
        goal_position,
        maximum_body_pose_error,
        maximum_body_joint_velocity,
    )
    return (~success).to(torch.float32)


def recovery_time_cost_curriculum(
    env,
    minimum_height: float,
    maximum_gravity_z: float,
    maximum_angular_velocity_start: float,
    maximum_angular_velocity_end: float,
    maximum_linear_velocity_start: float,
    maximum_linear_velocity_end: float,
    goal_position: list[float],
    maximum_body_pose_error_start: float,
    maximum_body_pose_error_end: float,
    maximum_body_joint_velocity_start: float,
    maximum_body_joint_velocity_end: float,
    curriculum_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    angular, linear, pose, joint = recovery_success_curriculum_values(
        env,
        maximum_angular_velocity_start=maximum_angular_velocity_start,
        maximum_angular_velocity_end=maximum_angular_velocity_end,
        maximum_linear_velocity_start=maximum_linear_velocity_start,
        maximum_linear_velocity_end=maximum_linear_velocity_end,
        maximum_body_pose_error_start=maximum_body_pose_error_start,
        maximum_body_pose_error_end=maximum_body_pose_error_end,
        maximum_body_joint_velocity_start=maximum_body_joint_velocity_start,
        maximum_body_joint_velocity_end=maximum_body_joint_velocity_end,
        curriculum_steps=curriculum_steps,
    )
    return recovery_time_cost(
        env,
        minimum_height,
        maximum_gravity_z,
        angular,
        linear,
        goal_position,
        pose,
        joint,
        asset_cfg,
    )


def _amp_tracking_frames(env) -> tuple[torch.Tensor, torch.Tensor, int]:
    current = env._compute_amp_tracking_frame()
    return current, env.reference_amp_frame, len(env.cfg.amp_joint_names)


def amp_reference_max_joint_position_exp(env, std: float) -> torch.Tensor:
    current, reference, joint_count = _amp_tracking_frames(env)
    error = torch.amax(torch.abs(current[:, :joint_count] - reference[:, :joint_count]), dim=-1)
    return torch.exp(-error / std)


def amp_reference_joint_position_exp(env, std: float) -> torch.Tensor:
    current, reference, joint_count = _amp_tracking_frames(env)
    error = torch.mean(torch.square(current[:, :joint_count] - reference[:, :joint_count]), dim=-1)
    return torch.exp(-error / (std * std))


def amp_reference_joint_velocity_exp(env, std: float) -> torch.Tensor:
    current, reference, joint_count = _amp_tracking_frames(env)
    joint_slice = slice(joint_count, 2 * joint_count)
    error = torch.mean(torch.square(current[:, joint_slice] - reference[:, joint_slice]), dim=-1)
    return torch.exp(-error / (std * std))


def amp_reference_root_height_exp(env, std: float) -> torch.Tensor:
    current, reference, joint_count = _amp_tracking_frames(env)
    index = 2 * joint_count
    error = torch.square(current[:, index] - reference[:, index])
    return torch.exp(-error / (std * std))


def _amp_reference_slice_exp(env, std: float, start_offset: int, size: int) -> torch.Tensor:
    current, reference, joint_count = _amp_tracking_frames(env)
    start = 2 * joint_count + start_offset
    error = torch.mean(torch.square(current[:, start : start + size] - reference[:, start : start + size]), dim=-1)
    return torch.exp(-error / (std * std))


def amp_reference_root_orientation_exp(env, std: float) -> torch.Tensor:
    return _amp_reference_slice_exp(env, std, start_offset=1, size=6)


def amp_reference_root_linear_velocity_exp(env, std: float) -> torch.Tensor:
    return _amp_reference_slice_exp(env, std, start_offset=7, size=3)


def amp_reference_root_angular_velocity_exp(env, std: float) -> torch.Tensor:
    return _amp_reference_slice_exp(env, std, start_offset=10, size=3)


def amp_reference_key_body_position_exp(env, std: float) -> torch.Tensor:
    current, reference, joint_count = _amp_tracking_frames(env)
    start = 2 * joint_count + 13
    error = torch.mean(torch.square(current[:, start:] - reference[:, start:]), dim=-1)
    return torch.exp(-error / (std * std))


def amp_reference_action_exp(env, action_name: str, std: float) -> torch.Tensor:
    """Track the deployed teacher's normalized action at the current reference frame."""

    action = _action(env, action_name).raw_actions
    error = torch.mean(torch.square(action - env.reference_teacher_action), dim=-1)
    return torch.exp(-error / (std * std))


def reference_joint_pose_exp(
    env,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    command = _command(env, command_name)
    joint_pos = _robot(env, asset_cfg).data.joint_pos[:, asset_cfg.joint_ids]
    error = torch.mean(
        torch.square(joint_pos - command.reference_joint_pos),
        dim=-1,
    )
    return command.reference_weight * torch.exp(-error / (std * std))


def reference_gravity_exp(
    env,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    command = _command(env, command_name)
    error = torch.sum(
        torch.square(_robot(env, asset_cfg).data.projected_gravity_b - command.reference_gravity),
        dim=-1,
    )
    return command.reference_weight * torch.exp(-error / (std * std))


def reference_height_exp(
    env,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    command = _command(env, command_name)
    error = torch.square(_height(env, asset_cfg) - command.reference_height[:, 0])
    return command.reference_weight * torch.exp(-error / (std * std))


def feet_support(env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    contacts = torch.max(torch.linalg.vector_norm(forces, dim=-1), dim=1).values > threshold
    return torch.mean(contacts.to(torch.float32), dim=-1)


def feet_contact_force_increase(env, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :2, sensor_cfg.body_ids, 2]
    increase = torch.abs(forces[:, 0]) > torch.abs(forces[:, 1])
    return torch.mean(increase.to(torch.float32), dim=-1)


def capped_contact_force_penalty(
    env,
    threshold: float,
    maximum_force: float,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Bound impact cost so an unavoidable fall contact cannot dominate a recovery episode."""
    if maximum_force <= threshold:
        raise ValueError("maximum_force must be greater than threshold")
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    peak_force = torch.max(torch.linalg.vector_norm(forces, dim=-1), dim=1).values
    normalized = torch.clamp((peak_force - threshold) / (maximum_force - threshold), min=0.0, max=1.0)
    return torch.sum(normalized, dim=-1)


def hardware_joint_limit_violation(
    env,
    position_minimum: list[float],
    position_maximum: list[float],
    soft_margin: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Squared violation of the K1 deployment limits, including a soft interior margin."""
    if soft_margin < 0.0:
        raise ValueError("soft_margin must be non-negative")
    joint_pos = _robot(env, asset_cfg).data.joint_pos[:, asset_cfg.joint_ids]
    minimum = torch.tensor(position_minimum, dtype=joint_pos.dtype, device=joint_pos.device)
    maximum = torch.tensor(position_maximum, dtype=joint_pos.dtype, device=joint_pos.device)
    lower_violation = torch.clamp(minimum + soft_margin - joint_pos, min=0.0)
    upper_violation = torch.clamp(joint_pos - maximum + soft_margin, min=0.0)
    return torch.sum(torch.square(lower_violation) + torch.square(upper_violation), dim=-1)


def joint_limit_target_reduction(env, action_name: str) -> torch.Tensor:
    """Penalize policy targets changed by deployment-limit braking."""
    return _action(env, action_name).joint_limit_target_reduction


def upright_feet_support(
    env,
    threshold: float,
    minimum_height: float,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    support = feet_support(env, threshold, sensor_cfg)
    gravity_z = _robot(env, asset_cfg).data.projected_gravity_b[:, 2]
    upright = torch.clamp(-gravity_z, 0.0, 1.0)
    height_scale = torch.clamp(
        (_height(env, asset_cfg) - minimum_height) / max(0.52 - minimum_height, 1.0e-6),
        0.0,
        1.0,
    )
    return support * upright * height_scale


def parallel_target_reduction(env, action_name: str) -> torch.Tensor:
    return _action(env, action_name).target_reduction


def parallel_motor_stress(env, action_name: str, margin: float, velocity_ratio: float) -> torch.Tensor:
    action = _action(env, action_name)
    low_margin = torch.clamp(margin - action.parallel_motor_margin, min=0.0) / margin
    high_speed = torch.clamp(action.parallel_motor_velocity_ratio - velocity_ratio, min=0.0)
    utilization = torch.clamp(action.parallel_motor_utilization - 0.85, min=0.0)
    return torch.sum(low_margin.square() + high_speed.square() + utilization.square(), dim=-1)


def parallel_state_limit_penalty(env, action_name: str, soft_margin: float) -> torch.Tensor:
    """Penalize physical motor-limit pressure in the serial-equivalent ankle model."""

    action = _action(env, action_name)
    joint_position = action.deployment_joint_position
    penalties = []
    for foot, indexes in enumerate(((14, 15), (20, 21))):
        motor_margin = action._parallel.motor_margin(joint_position[:, indexes], foot)
        low_margin = torch.clamp(soft_margin - motor_margin, min=0.0) / soft_margin
        penalties.append(torch.sum(low_margin.square(), dim=-1))
    return torch.stack(penalties, dim=-1).sum(dim=-1)

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor

from .actions import K1ParallelJointPositionAction
from .recovery_math import joint_limit_violation_mask
from .rewards import recovery_success_curriculum_values, recovery_success_mask


def _training_curriculum_value(env, start: float, end: float, curriculum_steps: int) -> float:
    if getattr(env.cfg, "amp_reset_mode", None) != "train":
        return end
    transition_count = float(env.common_step_counter * env.num_envs)
    progress = min(transition_count / float(curriculum_steps), 1.0)
    return start + progress * (end - start)


def recovered(
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
    if not hasattr(env, "_recovery_stable_steps"):
        env._recovery_stable_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._recovery_stable_steps = torch.where(success, env._recovery_stable_steps + 1, 0)
    return env._recovery_stable_steps >= hold_steps


def recovered_curriculum(
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
    return recovered(
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


def hard_joint_limit(
    env,
    position_minimum: list[float],
    position_maximum: list[float],
    margin: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    joint_pos = env.scene[asset_cfg.name].data.joint_pos[:, asset_cfg.joint_ids]
    minimum = torch.tensor(position_minimum, dtype=joint_pos.dtype, device=joint_pos.device)
    maximum = torch.tensor(position_maximum, dtype=joint_pos.dtype, device=joint_pos.device)
    violation = joint_limit_violation_mask(
        joint_pos,
        minimum,
        maximum,
        margin,
    )
    # The environment resets terminated rows before step() returns. Preserve the
    # pre-reset attribution for diagnostics and checkpoint evaluation.
    env._hard_joint_limit_violation_mask = violation.detach()
    return torch.any(violation, dim=-1)


def hard_joint_limit_curriculum(
    env,
    position_minimum: list[float],
    position_maximum: list[float],
    margin_start: float,
    margin_end: float,
    curriculum_steps: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    margin = _training_curriculum_value(env, margin_start, margin_end, curriculum_steps)
    return hard_joint_limit(env, position_minimum, position_maximum, margin, asset_cfg)


def parallel_ankle_infeasible(env, action_name: str, motor_tolerance: float = 0.0) -> torch.Tensor:
    action: K1ParallelJointPositionAction = env.action_manager.get_term(action_name)
    joint_pos = action.deployment_joint_position
    failed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    motor_margins = []
    for foot, indexes in enumerate(((14, 15), (20, 21))):
        margin = action._parallel.motor_margin(joint_pos[:, indexes], foot)
        motor_margins.append(margin)
        failed |= torch.any(margin < -motor_tolerance, dim=-1)
    env._parallel_ankle_state_motor_margin = torch.cat(motor_margins, dim=-1).detach().clone()
    env._parallel_ankle_target_motor_margin = action.parallel_motor_margin.detach().clone()
    env._parallel_ankle_joint_velocity = (
        action._asset.data.joint_vel[:, action._joint_ids][:, [14, 15, 20, 21]].detach().clone()
    )
    env._parallel_ankle_motor_velocity_ratio = action.parallel_motor_velocity_ratio.detach().clone()
    env._parallel_ankle_target_fallback = action.parallel_target_fallback.detach().clone()
    failed |= action.parallel_target_fallback
    return failed


def parallel_ankle_infeasible_curriculum(
    env,
    action_name: str,
    motor_tolerance_start: float,
    motor_tolerance_end: float,
    curriculum_steps: int,
) -> torch.Tensor:
    tolerance = _training_curriculum_value(
        env,
        motor_tolerance_start,
        motor_tolerance_end,
        curriculum_steps,
    )
    return parallel_ankle_infeasible(env, action_name, tolerance)


def nonfinite_action_or_state(env, action_name: str) -> torch.Tensor:
    action: K1ParallelJointPositionAction = env.action_manager.get_term(action_name)
    return action.nonfinite_action


def invalid_state(
    env,
    max_joint_vel: float = 100.0,
    max_root_height: float = 10.0,
    max_root_xy_distance: float = 200.0,
    max_lin_vel: float = 20.0,
    max_ang_vel: float = 50.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reset non-finite or physically exploded simulation states."""
    robot = env.scene[asset_cfg.name]
    root_pos = robot.data.root_pos_w
    root_lin_vel = robot.data.root_lin_vel_w
    root_ang_vel = robot.data.root_ang_vel_w
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel

    nonfinite = (
        ~torch.isfinite(root_pos).all(dim=-1)
        | ~torch.isfinite(root_lin_vel).all(dim=-1)
        | ~torch.isfinite(root_ang_vel).all(dim=-1)
        | ~torch.isfinite(joint_pos).all(dim=-1)
        | ~torch.isfinite(joint_vel).all(dim=-1)
    )
    root_pos_rel = root_pos - env.scene.env_origins
    return (
        nonfinite
        | (torch.abs(joint_vel).amax(dim=-1) > max_joint_vel)
        | (root_pos_rel[:, 2] > max_root_height)
        | (torch.linalg.vector_norm(root_pos_rel[:, :2], dim=-1) > max_root_xy_distance)
        | (torch.linalg.vector_norm(root_lin_vel, dim=-1) > max_lin_vel)
        | (torch.linalg.vector_norm(root_ang_vel, dim=-1) > max_ang_vel)
    )


def excessive_impact(env, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
    peak = torch.amax(torch.linalg.vector_norm(forces, dim=-1), dim=(1, 2))
    return peak > threshold

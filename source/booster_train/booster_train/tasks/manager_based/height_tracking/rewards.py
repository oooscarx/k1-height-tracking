# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import ContactSensor, RayCaster


def _height(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg | None) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if sensor_cfg is None:
        return asset.data.root_pos_w[:, 2]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    return asset.data.root_pos_w[:, 2] - torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)


def _body_quat(asset: Articulation, body_ids) -> torch.Tensor:
    quaternions = getattr(asset.data, "body_link_quat_w", asset.data.body_quat_w)
    return quaternions[:, body_ids]


def _projected_gravity(asset: Articulation, body_ids) -> torch.Tensor:
    quaternions = _body_quat(asset, body_ids)
    if quaternions.ndim == 2:
        quaternions = quaternions.unsqueeze(1)
    gravity = torch.zeros((*quaternions.shape[:-1], 3), device=quaternions.device)
    gravity[..., 2] = -1.0
    return math_utils.quat_apply_inverse(quaternions, gravity)


def track_height_command_exp(
    env,
    command_name: str,
    std: float,
    settle_gate: bool = False,
    stationary_gate: bool = False,
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    error = torch.square(command.measured_height - command.target_height)
    reward = torch.exp(-error / std**2)
    if settle_gate:
        reward *= (command.settled & (command.target_height >= 0.0)).float()
    if stationary_gate:
        reward *= (command.command_stationary & (command.target_height >= 0.0)).float()
    return reward


def track_height_command_l1(
    env,
    command_name: str,
    settle_gate: bool = False,
    stationary_gate: bool = False,
) -> torch.Tensor:
    """Keep a useful height gradient outside the narrow exponential kernel."""
    command = env.command_manager.get_term(command_name)
    penalty = torch.abs(command.measured_height - command.target_height)
    if settle_gate:
        penalty *= (command.settled & (command.target_height >= 0.0)).float()
    if stationary_gate:
        penalty *= (command.command_stationary & (command.target_height >= 0.0)).float()
    return penalty


def joint_pos_tracking_error_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    action_name: str = "joint_pos",
) -> torch.Tensor:
    robot: Articulation = env.scene[asset_cfg.name]
    action = env.action_manager._terms[action_name]
    return torch.sum(torch.square(action.processed_actions - robot.data.joint_pos[:, action._joint_ids]), dim=1)


class action_rate_rate_l2(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env) -> None:
        super().__init__(cfg, env)
        self._previous_rate = torch.zeros_like(env.action_manager.action)

    def __call__(self, env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
        del asset_cfg
        rate = env.action_manager.action - env.action_manager.prev_action
        penalty = torch.sum(torch.square(rate - self._previous_rate), dim=1)
        self._previous_rate.copy_(rate)
        return penalty

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._previous_rate.zero_()
        else:
            self._previous_rate[env_ids] = 0.0


def relaxation_penalty(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    pos_weight: float = 1.0,
    torque_weight: float = 0.001,
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    deviation = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    torques = asset.data.applied_torque[:, asset_cfg.joint_ids]
    return (
        command.relaxation_intensity * pos_weight * torch.sum(torch.square(deviation), dim=1)
        + (command.target_height < 0.0).float() * torque_weight * torch.sum(torch.square(torques), dim=1)
    )


def _smoothstep(value: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    if lower >= upper:
        raise ValueError(f"smoothstep lower bound must be below upper bound, got {lower} >= {upper}")
    phase = torch.clamp((value - lower) / (upper - lower), 0.0, 1.0)
    return phase * phase * (3.0 - 2.0 * phase)


def _height_command_regularization_gate(
    env,
    command_name: str,
    height_gate_range: tuple[float, float],
    age_gate_range: tuple[float, float],
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    height_gate = _smoothstep(command.target_height, *height_gate_range)
    age_gate = _smoothstep(command.command_age_s, *age_gate_range)
    return height_gate * age_gate


def joint_deviation_for_height_command(
    env,
    command_name: str,
    height_gate_range: tuple[float, float],
    age_gate_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    mode: Literal["l1", "l2"] = "l1",
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    deviation = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    penalty = torch.sum(torch.abs(deviation), dim=1) if mode == "l1" else torch.sum(torch.square(deviation), dim=1)
    return penalty * _height_command_regularization_gate(
        env,
        command_name=command_name,
        height_gate_range=height_gate_range,
        age_gate_range=age_gate_range,
    )


def joint_deviation_from_action_center(
    env,
    action_name: str,
    mode: Literal["l1", "l2"] = "l1",
) -> torch.Tensor:
    """Penalize deviation from the command-conditioned posture scaffold."""

    action = env.action_manager.get_term(action_name)
    deviation = action.deployment_joint_position - action.command_center
    if mode == "l1":
        return torch.sum(torch.abs(deviation), dim=1)
    if mode == "l2":
        return torch.sum(torch.square(deviation), dim=1)
    raise ValueError(f"unsupported joint-deviation mode: {mode}")


def joint_deviation_if_standing(
    env,
    standing_height_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    mode: Literal["l1", "l2"] = "l1",
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    deviation = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    penalty = torch.sum(torch.abs(deviation), dim=1) if mode == "l1" else torch.sum(torch.square(deviation), dim=1)
    return penalty * (_height(env, SceneEntityCfg(asset_cfg.name), sensor_cfg) > standing_height_threshold)


def moving(env, asset_cfg: SceneEntityCfg, weight_lin: float = 1.0, weight_ang: float = 1.0) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return (
        asset.data.body_lin_vel_w.norm(dim=-1).mean(dim=1) * weight_lin
        + asset.data.body_ang_vel_w.norm(dim=-1).mean(dim=1) * weight_ang
    )


def stationary_base_motion_l2(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    linear_weight: float = 1.0,
    angular_weight: float = 1.0,
) -> torch.Tensor:
    """Suppress sway after a command ramp without penalizing vertical height changes."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    planar_velocity = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    angular_velocity = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    active = command.command_stationary & command.settled & (command.target_height >= 0.0)
    return (linear_weight * planar_velocity + angular_weight * angular_velocity) * active.float()


def positive_height_base_motion_l2(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    linear_weight: float = 1.0,
    angular_weight: float = 1.0,
) -> torch.Tensor:
    """Suppress planar drift and body rotation without resisting vertical tracking."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_term(command_name)
    planar_velocity = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)
    angular_velocity = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    active = command.target_height >= 0.0
    return (linear_weight * planar_velocity + angular_weight * angular_velocity) * active.float()


def moving_if_tracking(
    env,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    error_threshold: float,
    full_penalty_error_threshold: float | None = None,
    settle_gate: bool = False,
    weight_lin: float = 1.0,
    weight_ang: float = 1.0,
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    error = torch.abs(command.measured_height - command.target_height)
    if full_penalty_error_threshold is None:
        tracking_weight = (error < error_threshold).float()
    else:
        if not 0.0 <= full_penalty_error_threshold < error_threshold:
            raise ValueError(
                "full_penalty_error_threshold must be non-negative and below error_threshold, "
                f"got {full_penalty_error_threshold} >= {error_threshold}"
            )
        tracking_weight = 1.0 - _smoothstep(
            error,
            full_penalty_error_threshold,
            error_threshold,
        )
    if settle_gate:
        tracking_weight *= (command.settled & (command.target_height >= 0.0)).float()
    return moving(env, asset_cfg, weight_lin, weight_ang) * tracking_weight


def _body_tilt_angles(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if asset_cfg.body_names is None:
        return torch.acos(torch.clamp(-asset.data.projected_gravity_b[:, 2], -1.0, 1.0)).unsqueeze(-1)
    gravity = _projected_gravity(asset, asset_cfg.body_ids)
    return torch.acos(torch.clamp(-gravity[..., 2], -1.0, 1.0))


class upright_orientation_after_standing(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env) -> None:
        super().__init__(cfg, env)
        self._standing_duration = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self,
        env,
        standing_height_threshold: float,
        min_standing_duration_s: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sensor_cfg: SceneEntityCfg | None = None,
        norm: Literal["l1", "l2"] = "l1",
    ) -> torch.Tensor:
        standing = _height(env, SceneEntityCfg(asset_cfg.name), sensor_cfg) > standing_height_threshold
        self._standing_duration = torch.where(
            standing,
            self._standing_duration + env.step_dt,
            torch.zeros_like(self._standing_duration),
        )
        angles = _body_tilt_angles(env, asset_cfg)
        penalty = torch.sum(torch.square(angles), dim=-1) if norm == "l2" else torch.sum(angles, dim=-1)
        return penalty * (self._standing_duration >= min_standing_duration_s)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._standing_duration.zero_()
        else:
            self._standing_duration[env_ids] = 0.0


def upright_orientation_for_height_command(
    env,
    command_name: str,
    height_gate_range: tuple[float, float],
    age_gate_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    norm: Literal["l1", "l2"] = "l1",
) -> torch.Tensor:
    angles = _body_tilt_angles(env, asset_cfg)
    penalty = torch.sum(torch.square(angles), dim=-1) if norm == "l2" else torch.sum(angles, dim=-1)
    return penalty * _height_command_regularization_gate(
        env,
        command_name=command_name,
        height_gate_range=height_gate_range,
        age_gate_range=age_gate_range,
    )


def severely_tilted_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold_rad: float = 1.5708,
) -> torch.Tensor:
    return (_body_tilt_angles(env, asset_cfg) > threshold_rad).any(dim=-1).float()


def body_orientation_penalty(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    axis: Literal["roll", "pitch"] = "roll",
    direction: Literal["both", "forward", "backward"] = "both",
    kernel: Literal["l1", "l2"] = "l1",
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    gravity = _projected_gravity(asset, asset_cfg.body_ids)
    component = gravity[..., 0] if axis == "pitch" else gravity[..., 1]
    if direction == "forward":
        component = torch.clamp(component, min=0.0)
    elif direction == "backward":
        component = torch.clamp(-component, min=0.0)
    else:
        component = component.abs()
    return torch.mean(torch.square(component) if kernel == "l2" else component, dim=-1)


def feet_distance_from_ref(
    env,
    asset_cfg: SceneEntityCfg,
    ref_distance: float = 0.2,
    norm: Literal["l1", "l2"] = "l1",
    error_threshold: float = 0.0,
    distance_mode: Literal["lateral", "absolute"] = "lateral",
    close_multiplier: float = 1.0,
    episode_delay_s: float = 0.0,
    episode_ramp_s: float = 0.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet = asset.data.body_pos_w[:, asset_cfg.body_ids]
    if len(asset_cfg.body_ids) != 2:
        raise ValueError("feet-distance reward requires exactly two feet")
    if distance_mode == "absolute":
        distance = torch.norm(feet[:, 0] - feet[:, 1], dim=-1)
    else:
        relative = feet - asset.data.root_pos_w.unsqueeze(1)
        root_quat = asset.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1)
        feet_b = math_utils.quat_apply_inverse(root_quat, relative)
        distance = torch.abs(feet_b[:, 0, 1] - feet_b[:, 1, 1])
    error = distance - ref_distance
    error = torch.where(error > error_threshold, error, torch.where(error < 0.0, error, torch.zeros_like(error)))
    error = torch.where(error < 0.0, error * close_multiplier, error)
    penalty = torch.abs(error) if norm == "l1" else torch.square(error)
    if episode_delay_s > 0.0 or episode_ramp_s > 0.0:
        time = env.episode_length_buf * env.step_dt
        if episode_ramp_s > 0.0:
            scale = torch.clamp((time - episode_delay_s) / episode_ramp_s, 0.0, 1.0)
        else:
            scale = (time >= episode_delay_s).float()
        penalty *= scale
    return penalty


def feet_distance_from_ref_if_standing(
    env,
    standing_height_threshold: float,
    asset_cfg: SceneEntityCfg,
    ref_distance: float = 0.2,
    norm: Literal["l1", "l2"] = "l1",
    error_threshold: float = 0.0,
    distance_mode: Literal["lateral", "absolute"] = "lateral",
    close_multiplier: float = 1.0,
    sensor_cfg: SceneEntityCfg | None = None,
    episode_delay_s: float = 0.0,
    episode_ramp_s: float = 0.0,
) -> torch.Tensor:
    penalty = feet_distance_from_ref(
        env,
        asset_cfg=asset_cfg,
        ref_distance=ref_distance,
        norm=norm,
        error_threshold=error_threshold,
        distance_mode=distance_mode,
        close_multiplier=close_multiplier,
        episode_delay_s=episode_delay_s,
        episode_ramp_s=episode_ramp_s,
    )
    return penalty * (_height(env, SceneEntityCfg(asset_cfg.name), sensor_cfg) > standing_height_threshold)


def feet_distance_from_ref_for_height_command(
    env,
    command_name: str,
    height_gate_range: tuple[float, float],
    age_gate_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    ref_distance: float = 0.2,
    norm: Literal["l1", "l2"] = "l1",
    error_threshold: float = 0.0,
    distance_mode: Literal["lateral", "absolute"] = "lateral",
    close_multiplier: float = 1.0,
    episode_delay_s: float = 0.0,
    episode_ramp_s: float = 0.0,
) -> torch.Tensor:
    penalty = feet_distance_from_ref(
        env,
        asset_cfg=asset_cfg,
        ref_distance=ref_distance,
        norm=norm,
        error_threshold=error_threshold,
        distance_mode=distance_mode,
        close_multiplier=close_multiplier,
        episode_delay_s=episode_delay_s,
        episode_ramp_s=episode_ramp_s,
    )
    return penalty * _height_command_regularization_gate(
        env,
        command_name=command_name,
        height_gate_range=height_gate_range,
        age_gate_range=age_gate_range,
    )


class ground_unloaded(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env) -> None:
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self._expected_weight = asset.data.default_mass.sum(dim=1).to(env.device) * abs(env.cfg.sim.gravity[2])

    def __call__(
        self,
        env,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        command_name: str | None = None,
    ) -> torch.Tensor:
        del asset_cfg
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        total_force = sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2].sum(dim=1)
        penalty = 1.0 - torch.clamp(total_force / (self._expected_weight + 1.0e-6), 0.0, 1.0)
        if command_name is not None:
            command = env.command_manager.get_term(command_name)
            penalty *= (command.target_height >= 0.0).float()
        return penalty


def foot_orientation_l1(
    env,
    asset_cfg: SceneEntityCfg,
    roll_weight: float = 1.0,
    pitch_weight: float = 1.0,
    yaw_weight: float = 1.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    feet_quat = asset.data.body_quat_w[:, asset_cfg.body_ids]
    root_yaw = math_utils.yaw_quat(asset.data.root_quat_w)
    feet_b = math_utils.quat_mul(math_utils.quat_inv(root_yaw).unsqueeze(1).expand_as(feet_quat), feet_quat)
    roll, pitch, yaw = math_utils.euler_xyz_from_quat(feet_b.reshape(-1, 4))
    count = len(asset_cfg.body_ids)
    return (
        roll.abs().reshape(env.num_envs, count).mean(dim=1) * roll_weight
        + pitch.abs().reshape(env.num_envs, count).mean(dim=1) * pitch_weight
        + yaw.abs().reshape(env.num_envs, count).mean(dim=1) * yaw_weight
    )


def feet_yaw_mean_vs_base(env, feet_asset_cfg: SceneEntityCfg, base_body_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[feet_asset_cfg.name]
    if len(feet_asset_cfg.body_ids) != 2 or len(base_body_cfg.body_ids) != 1:
        raise ValueError("feet yaw reward requires two feet and one base body")
    feet_quat = asset.data.body_quat_w[:, feet_asset_cfg.body_ids]
    base_quat = asset.data.body_quat_w[:, base_body_cfg.body_ids].squeeze(1)
    relative = math_utils.quat_mul(math_utils.quat_inv(base_quat).unsqueeze(1).expand(-1, 2, -1), feet_quat)
    _, _, yaw = math_utils.euler_xyz_from_quat(relative.reshape(-1, 4))
    return torch.square(yaw.reshape(env.num_envs, 2)).sum(dim=1)


def feet_slide(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    contact = torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids], dim=-1) > force_threshold
    tangential_speed = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    return torch.sum(torch.square(tangential_speed) * contact, dim=1)


def completely_airborne(env, sensor_cfg: SceneEntityCfg, threshold: float = 1.0) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force = sensor.data.net_forces_w[:, sensor_cfg.body_ids]
    return (~(torch.norm(force, dim=-1) > threshold).any(dim=-1)).float()


def bodies_lin_vel_l2(
    env,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 0.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    velocity = asset.data.body_lin_vel_w[:, asset_cfg.body_ids]
    excess = torch.clamp(torch.norm(velocity, dim=-1) - threshold, min=0.0)
    return torch.sum(torch.square(excess), dim=-1)


class impact_velocity(ManagerTermBase):
    """Isaac Lab 2.2-compatible approximation of WBC-AGILE impact history."""

    def __init__(self, cfg: RewardTermCfg, env) -> None:
        super().__init__(cfg, env)
        asset_cfg = cfg.params["asset_cfg"]
        asset: Articulation = env.scene[asset_cfg.name]
        self._asset_cfg = asset_cfg
        count = len(asset_cfg.body_ids)
        self._previous_speed = torch.zeros(env.num_envs, count, device=env.device)

    def __call__(
        self,
        env,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        force_threshold: float = 10.0,
        kernel: Literal["l1", "l2"] = "l1",
    ) -> torch.Tensor:
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        asset: Articulation = env.scene[asset_cfg.name]
        history_force = sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids]
        in_contact = torch.norm(history_force, dim=-1).amax(dim=1) > force_threshold
        speed = torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids], dim=-1)
        impact = torch.where(in_contact, torch.maximum(speed, self._previous_speed), torch.zeros_like(speed))
        self._previous_speed.copy_(speed)
        if kernel == "l2":
            impact = torch.square(impact)
        return impact.sum(dim=1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._previous_speed.zero_()
        else:
            self._previous_speed[env_ids] = 0.0

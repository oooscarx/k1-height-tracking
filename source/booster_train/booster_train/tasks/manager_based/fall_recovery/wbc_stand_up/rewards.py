# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from typing import Literal

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers.manager_term_cfg import RewardTermCfg
from isaaclab.sensors import RayCaster

from .terminations import stable_standing_mask


def _height(env, asset_cfg: SceneEntityCfg, sensor_cfg: SceneEntityCfg | None) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    if sensor_cfg is None:
        return asset.data.root_pos_w[:, 2]
    sensor: RayCaster = env.scene[sensor_cfg.name]
    return asset.data.root_pos_w[:, 2] - torch.mean(
        sensor.data.ray_hits_w[..., 2],
        dim=1,
    )


def base_height_exp(
    env,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    error = torch.square(_height(env, asset_cfg, sensor_cfg) - target_height)
    return torch.exp(-error / std**2)


def action_rate_l2_if_actor_active(
    env,
    rest_duration_s: float = 0.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Legacy WBC action-rate penalty, optionally gated during settling."""
    del asset_cfg
    penalty = torch.sum(
        torch.square(env.action_manager.action - env.action_manager.prev_action),
        dim=1,
    )
    if rest_duration_s > 0.0:
        penalty = penalty * (env.episode_length_buf >= int(rest_duration_s / env.step_dt))
    return penalty


def action_l2_if_actor_active(
    env,
    rest_duration_s: float = 0.0,
) -> torch.Tensor:
    """Legacy WBC action-magnitude penalty, optionally gated during settling."""
    penalty = torch.sum(torch.square(env.action_manager.action), dim=1)
    if rest_duration_s > 0.0:
        penalty = penalty * (env.episode_length_buf >= int(rest_duration_s / env.step_dt))
    return penalty


class action_rate_rate_l2_if_actor_is_active(ManagerTermBase):
    """Legacy WBC penalty on the second finite difference of policy actions."""

    def __init__(self, cfg: RewardTermCfg, env) -> None:
        super().__init__(cfg, env)
        self._previous_rate = torch.zeros_like(env.action_manager.action)

    def __call__(
        self,
        env,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        rest_duration_s: float = 0.0,
    ) -> torch.Tensor:
        del asset_cfg
        action_rate = env.action_manager.action - env.action_manager.prev_action
        penalty = torch.sum(
            torch.square(action_rate - self._previous_rate),
            dim=1,
        )
        self._previous_rate.copy_(action_rate)
        if rest_duration_s > 0.0:
            penalty = penalty * (env.episode_length_buf >= int(rest_duration_s / env.step_dt))
        return penalty

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._previous_rate.zero_()
        else:
            self._previous_rate[env_ids] = 0.0


def max_incoming_forces_penalty(
    env,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    threshold: float = 100.0,
) -> torch.Tensor:
    """Legacy WBC maximum incoming joint-force penalty."""
    robot: Articulation = env.scene[robot_cfg.name]
    incoming_wrench = robot.data.body_incoming_joint_wrench_b[:, robot_cfg.body_ids]
    forces = torch.linalg.vector_norm(incoming_wrench[..., :3], dim=2)
    over_limit = torch.clamp(forces - threshold, min=0.0)
    return torch.square(torch.max(over_limit, dim=1).values)


def joint_deviation_exp_if_standing(
    env,
    standing_height_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    std: float = 0.25,
) -> torch.Tensor:
    """Legacy positive reward for default-pose joints after standing."""
    asset: Articulation = env.scene[asset_cfg.name]
    deviation = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    standing = _height(env, asset_cfg, sensor_cfg) > standing_height_threshold
    return torch.sum(torch.exp(-torch.square(deviation) / std**2), dim=1) * standing


class joint_deviation_strict_success_potential(ManagerTermBase):
    """Bound strict hold shaping and pay pose quality only on success."""

    def __init__(self, cfg: RewardTermCfg, env) -> None:
        super().__init__(cfg, env)
        self._stable_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self._previous_potential = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self,
        env,
        standing_height_threshold: float,
        duration_s: float,
        gamma: float,
        successful_termination_term: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sensor_cfg: SceneEntityCfg | None = None,
        std: float = 0.25,
        projected_gravity_z_max: float | None = None,
        projected_gravity_xy_norm_max: float | None = None,
        root_linear_speed_max: float | None = None,
        root_angular_speed_max: float | None = None,
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        deviation = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        quality = torch.sum(torch.exp(-torch.square(deviation) / std**2), dim=1)
        stable = stable_standing_mask(
            _height(env, asset_cfg, sensor_cfg),
            asset.data.projected_gravity_b,
            asset.data.root_lin_vel_b,
            asset.data.root_ang_vel_b,
            standing_height_threshold,
            projected_gravity_z_max,
            projected_gravity_xy_norm_max,
            root_linear_speed_max,
            root_angular_speed_max,
        )
        self._stable_steps = torch.where(
            stable,
            self._stable_steps + 1,
            torch.zeros_like(self._stable_steps),
        )
        required_steps = max(1, math.ceil(duration_s / env.step_dt))
        hold_progress = torch.clamp(
            self._stable_steps.to(torch.float32) / required_steps,
            max=1.0,
        )
        potential = quality * hold_progress
        next_potential = torch.where(env.reset_buf, 0.0, potential)
        reward = (gamma * next_potential - self._previous_potential) / env.step_dt
        success = env.termination_manager.get_term(successful_termination_term)
        reward += success.to(reward.dtype) * quality / env.step_dt
        self._previous_potential.copy_(next_potential)
        return reward

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._stable_steps.zero_()
            self._previous_potential.zero_()
        else:
            self._stable_steps[env_ids] = 0
            self._previous_potential[env_ids] = 0.0


def joint_deviation_if_standing(
    env,
    standing_height_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    mode: Literal["l1", "l2"] = "l1",
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    deviation = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    if mode == "l1":
        penalty = torch.sum(torch.abs(deviation), dim=1)
    elif mode == "l2":
        penalty = torch.sum(torch.square(deviation), dim=1)
    else:
        raise ValueError(f"unsupported joint deviation mode: {mode}")
    standing = _height(env, asset_cfg, sensor_cfg) > standing_height_threshold
    return penalty * standing


def equal_foot_force_if_standing(
    env,
    sensor_cfg: SceneEntityCfg,
    standing_height_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    height_measurement_sensor: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Legacy reward for sharing vertical load equally between both feet."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    feet_z_forces = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    mean_force = feet_z_forces.mean(dim=1)
    reward = 1.0 - torch.abs(mean_force.unsqueeze(1) - feet_z_forces).mean(dim=1) / (mean_force + 1.0e-6)
    standing = _height(env, asset_cfg, height_measurement_sensor) > standing_height_threshold
    return reward * standing


def moving_if_standing(
    env,
    asset_cfg: SceneEntityCfg,
    standing_height_threshold: float,
    weight_lin: float = 1.0,
    weight_ang: float = 1.0,
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    penalty = (
        asset.data.body_lin_vel_w.norm(dim=-1).mean(dim=1) * weight_lin
        + asset.data.body_ang_vel_w.norm(dim=-1).mean(dim=1) * weight_ang
    )
    standing = _height(env, asset_cfg, sensor_cfg) > standing_height_threshold
    return penalty * standing


def feet_distance_from_ref_if_standing(
    env,
    standing_height_threshold: float,
    asset_cfg: SceneEntityCfg,
    ref_distance: float = 0.2,
    sensor_cfg: SceneEntityCfg | None = None,
    norm: Literal["l1", "l2"] = "l1",
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    if len(asset_cfg.body_ids) != 2:
        raise ValueError("feet distance reward requires exactly two feet")
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    feet_pos_b = math_utils.quat_apply_inverse(
        asset.data.root_quat_w.unsqueeze(1).expand(-1, 2, -1),
        feet_pos_w - asset.data.root_pos_w.unsqueeze(1),
    )
    distance_error = torch.abs(feet_pos_b[:, 0, 1] - feet_pos_b[:, 1, 1]) - ref_distance
    penalty = torch.abs(distance_error) if norm == "l1" else torch.square(distance_error)
    standing = _height(env, SceneEntityCfg(asset_cfg.name), sensor_cfg) > standing_height_threshold
    return penalty * standing


def feet_yaw_mean_vs_base_if_standing(
    env,
    standing_height_threshold: float,
    feet_asset_cfg: SceneEntityCfg,
    base_body_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    asset: Articulation = env.scene[feet_asset_cfg.name]
    if len(feet_asset_cfg.body_ids) != 2 or len(base_body_cfg.body_ids) != 1:
        raise ValueError("feet yaw reward requires two feet and one base body")
    feet_quat = asset.data.body_quat_w[:, feet_asset_cfg.body_ids]
    base_quat = asset.data.body_quat_w[:, base_body_cfg.body_ids].squeeze(1)
    relative = math_utils.quat_mul(
        math_utils.quat_inv(base_quat).unsqueeze(1).expand(-1, 2, -1),
        feet_quat,
    )
    _, _, yaw = math_utils.euler_xyz_from_quat(relative.reshape(-1, 4))
    penalty = torch.square(yaw.reshape(env.num_envs, 2)).sum(dim=1)
    standing = _height(env, SceneEntityCfg(feet_asset_cfg.name), sensor_cfg) > standing_height_threshold
    return penalty * standing


class body_acc_l2(ManagerTermBase):
    """WBC body acceleration penalty computed from velocity history."""

    def __init__(self, cfg: RewardTermCfg, env) -> None:
        super().__init__(cfg, env)
        asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._body_idx: int | None = None
        if asset_cfg.body_names is not None:
            asset: Articulation = env.scene[asset_cfg.name]
            self._body_idx = asset.find_bodies(asset_cfg.body_names)[0][0]
        self._previous = torch.zeros(env.num_envs, 6, device=env.device)
        self._initialized = False

    def __call__(
        self,
        env,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        if self._body_idx is None:
            linear = asset.data.root_lin_vel_w
            angular = asset.data.root_ang_vel_w
        else:
            linear = asset.data.body_lin_vel_w[:, self._body_idx]
            angular = asset.data.body_ang_vel_w[:, self._body_idx]
        velocity = torch.cat((linear, angular), dim=-1)
        if not self._initialized:
            self._previous.copy_(velocity)
            self._initialized = True
            return torch.zeros(env.num_envs, device=env.device)
        acceleration = (velocity - self._previous) / env.step_dt
        self._previous.copy_(velocity)
        return torch.sum(torch.square(acceleration), dim=-1)

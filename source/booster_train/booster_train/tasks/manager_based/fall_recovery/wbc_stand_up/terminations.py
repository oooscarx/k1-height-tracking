# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math

import torch
from isaaclab.assets import RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import RayCaster


def stable_standing_mask(
    height: torch.Tensor,
    projected_gravity_b: torch.Tensor,
    root_lin_vel_b: torch.Tensor,
    root_ang_vel_b: torch.Tensor,
    min_height: float,
    projected_gravity_z_max: float | None = None,
    projected_gravity_xy_norm_max: float | None = None,
    root_linear_speed_max: float | None = None,
    root_angular_speed_max: float | None = None,
) -> torch.Tensor:
    """Return whether each robot satisfies the requested standing criteria."""
    stable = height >= min_height
    if projected_gravity_z_max is not None:
        stable &= projected_gravity_b[:, 2] <= projected_gravity_z_max
    if projected_gravity_xy_norm_max is not None:
        stable &= (
            torch.linalg.vector_norm(projected_gravity_b[:, :2], dim=-1)
            <= projected_gravity_xy_norm_max
        )
    if root_linear_speed_max is not None:
        stable &= (
            torch.linalg.vector_norm(root_lin_vel_b, dim=-1)
            <= root_linear_speed_max
        )
    if root_angular_speed_max is not None:
        stable &= (
            torch.linalg.vector_norm(root_ang_vel_b, dim=-1)
            <= root_angular_speed_max
        )
    return stable


class standing(ManagerTermBase):
    """Legacy success termination after maintaining standing height."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self.standing_timer = torch.zeros(env.num_envs, device=env.device)

    def __call__(
        self,
        env,
        asset_cfg: SceneEntityCfg,
        min_height: float,
        duration_s: float,
        sensor_cfg: SceneEntityCfg | None = None,
        projected_gravity_z_max: float | None = None,
        projected_gravity_xy_norm_max: float | None = None,
        root_linear_speed_max: float | None = None,
        root_angular_speed_max: float | None = None,
    ) -> torch.Tensor:
        asset: RigidObject = env.scene[asset_cfg.name]
        current_height = asset.data.root_pos_w[:, 2]
        if sensor_cfg is not None:
            sensor: RayCaster = env.scene[sensor_cfg.name]
            current_height = current_height - torch.mean(
                sensor.data.ray_hits_w[..., 2],
                dim=1,
            )
        stable = stable_standing_mask(
            current_height,
            asset.data.projected_gravity_b,
            asset.data.root_lin_vel_b,
            asset.data.root_ang_vel_b,
            min_height,
            projected_gravity_z_max,
            projected_gravity_xy_norm_max,
            root_linear_speed_max,
            root_angular_speed_max,
        )
        self.standing_timer = torch.where(
            stable,
            self.standing_timer + 1,
            torch.zeros_like(self.standing_timer),
        )
        uses_strict_stability = any(
            threshold is not None
            for threshold in (
                projected_gravity_z_max,
                projected_gravity_xy_norm_max,
                root_linear_speed_max,
                root_angular_speed_max,
            )
        )
        if uses_strict_stability:
            return self.standing_timer >= math.ceil(duration_s / env.step_dt)
        return self.standing_timer > int(duration_s / env.step_dt)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.standing_timer.zero_()
        else:
            self.standing_timer[env_ids] = 0.0


class no_height_progress(ManagerTermBase):
    """End attempts that gain less than the configured height within the time window."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self.initial_height = torch.zeros(env.num_envs, device=env.device)
        self.made_progress = torch.zeros(
            env.num_envs,
            dtype=torch.bool,
            device=env.device,
        )

    def _height(
        self,
        env,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        robot = env.scene[asset_cfg.name]
        sensor: RayCaster = env.scene[sensor_cfg.name]
        return robot.data.root_pos_w[:, 2] - torch.mean(
            sensor.data.ray_hits_w[..., 2],
            dim=1,
        )

    def __call__(
        self,
        env,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
        height_increase_threshold: float,
        time_limit_s: float,
    ) -> torch.Tensor:
        current_height = self._height(env, asset_cfg, sensor_cfg)
        self.made_progress |= (
            current_height > self.initial_height + height_increase_threshold
        )
        return (
            env.episode_length_buf * env.step_dt > time_limit_s
        ) & ~self.made_progress

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self._env.num_envs, device=self._env.device)
        current_height = self._height(
            self._env,
            self.cfg.params["asset_cfg"],
            self.cfg.params["sensor_cfg"],
        )
        self.initial_height[env_ids] = current_height[env_ids]
        self.made_progress[env_ids] = False

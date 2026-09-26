# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster


def is_env_inactive(env, rest_duration_s: float) -> torch.Tensor:
    """Return one while an environment is in its unactuated settling phase."""
    if hasattr(env, "episode_length_buf"):
        return (env.episode_length_buf < int(rest_duration_s / env.step_dt)).float().unsqueeze(1)
    return torch.ones(env.num_envs, 1, device=env.device)


def base_height_from_sensor(
    env,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    robot = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene[sensor_cfg.name]
    ground_height = torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    return (robot.data.root_pos_w[:, 2] - ground_height).unsqueeze(1)


def contact_force_norm(
    env,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    sensor = env.scene.sensors[sensor_cfg.name]
    return sensor.data.net_forces_w[:, sensor_cfg.body_ids].norm(dim=2)

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class HeightLiftAction(ActionTerm):
    """Apply the WBC-AGILE height-command lift assistance using Isaac Lab 2.2 APIs."""

    cfg: "HeightLiftActionCfg"

    def __init__(self, cfg: "HeightLiftActionCfg", env: ManagerBasedEnv) -> None:
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]
        self._height_sensor: RayCaster = env.scene.sensors[cfg.height_sensor]
        body_ids, _ = self._asset.find_bodies(cfg.link_to_lift)
        if len(body_ids) != 1:
            raise ValueError(f"lift link must resolve exactly once: {cfg.link_to_lift}")
        self._body_ids = body_ids
        self._force_offset = torch.tensor(cfg.force_offset, device=env.device, dtype=torch.float32)
        force_limit = cfg.force_limit
        if cfg.force_limit_weight_fraction is not None:
            total_mass = self._asset.data.default_mass.sum(dim=1).mean().item()
            gravity = abs(env.cfg.sim.gravity[2])
            force_limit = cfg.force_limit_weight_fraction * total_mass * gravity
        self._base_force_limit = float(force_limit)
        self._height_command = env.command_manager.get_term(cfg.height_command) if cfg.height_command else None
        self._force_scale = 1.0
        self._is_disabled = False
        self._max_heights = torch.zeros(env.num_envs, device=env.device)
        self._raw_actions = torch.empty(env.num_envs, 0, device=env.device)
        self._processed_actions = torch.empty_like(self._raw_actions)

    @property
    def action_dim(self) -> int:
        return 0

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def force_scale(self) -> float:
        return self._force_scale

    @property
    def max_heights(self) -> torch.Tensor:
        return self._max_heights

    def scale_forces(self, scale: float) -> None:
        self._force_scale = min(max(float(scale), 0.0), 1.0)
        self._is_disabled = self._force_scale <= 0.0

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions = actions
        self._processed_actions = actions

    def _measure_height(self) -> torch.Tensor:
        if self._height_command is not None:
            return self._height_command.measured_height
        ground = torch.mean(self._height_sensor.data.ray_hits_w[..., 2], dim=-1)
        return self._asset.data.root_pos_w[:, 2] - ground

    def apply_actions(self) -> None:
        height = self._measure_height()
        self._max_heights = torch.maximum(self._max_heights, height)
        forces_w = torch.zeros(self._env.num_envs, 1, 3, device=self.device)
        torques_w = torch.zeros_like(forces_w)
        if not self._is_disabled:
            if self._height_command is not None:
                target = self._height_command.target_height
            else:
                elapsed = self._env.episode_length_buf * self._env.step_dt
                ratio = torch.clamp(
                    (elapsed - self.cfg.start_lifting_time_s) / self.cfg.lifting_duration_s,
                    0.0,
                    1.0,
                )
                target = ratio * self.cfg.target_height
            vertical_force = self.cfg.stiffness_forces * self._force_scale * (target - height)
            if self._height_command is not None:
                vertical_force = torch.where(target >= 0.0, vertical_force, torch.zeros_like(vertical_force))
            limit = self._base_force_limit * self._force_scale
            lower = -limit if self.cfg.allow_push_down else 0.0
            forces_w[:, 0, 2] = torch.clamp(vertical_force, lower, limit)
            torque_limit = self.cfg.torque_limit * self._force_scale
            torques_w[:, 0, 2] = torch.clamp(
                -self.cfg.damping_torques * self._force_scale * self._asset.data.root_ang_vel_w[:, 2],
                -torque_limit,
                torque_limit,
            )

        link_quat = self._asset.data.body_quat_w[:, self._body_ids]
        forces_b = math_utils.quat_apply_inverse(link_quat, forces_w)
        torques_b = math_utils.quat_apply_inverse(link_quat, torques_w)
        positions = self._force_offset.view(1, 1, 3).expand_as(forces_b)
        self._asset.set_external_force_and_torque(
            forces_b,
            torques_b,
            positions=positions,
            body_ids=self._body_ids,
            is_global=False,
        )
        self._asset.has_external_wrench = True

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._max_heights.zero_()
        else:
            self._max_heights[env_ids] = 0.0


@configclass
class HeightLiftActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = HeightLiftAction
    link_to_lift: str = MISSING
    stiffness_forces: float = 0.0
    damping_forces: float = 0.0
    force_limit: float = 0.0
    force_limit_weight_fraction: float | None = None
    damping_torques: float = 0.0
    torque_limit: float = 0.0
    height_sensor: str = "height_measurement_sensor"
    target_height: float = 0.52
    height_command: str | None = None
    allow_push_down: bool = False
    start_lifting_time_s: float = 0.0
    lifting_duration_s: float = 10.0
    force_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)

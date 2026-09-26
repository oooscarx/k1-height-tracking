# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import RayCaster


class terrain_levels_successful_termination(ManagerTermBase):
    """Legacy terrain curriculum driven by a successful termination term."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self.num_failures = torch.zeros(
            env.num_envs,
            dtype=torch.int32,
            device=env.device,
        )
        self.num_successes = torch.zeros_like(self.num_failures)

    def __call__(
        self,
        env,
        env_ids: torch.Tensor,
        successful_termination_term: str,
        n_failures: int = 3,
        n_successes: int = 3,
    ) -> torch.Tensor:
        terrain = env.scene.terrain
        if len(env_ids) == 0:
            return torch.mean(terrain.terrain_levels.float())
        succeeded = env.termination_manager.get_term(
            successful_termination_term
        )[env_ids]
        self.num_successes[env_ids] += succeeded
        self.num_failures[env_ids] += ~succeeded
        move_up = self.num_successes[env_ids] >= n_successes
        move_down = self.num_failures[env_ids] >= n_failures
        changed = env_ids[move_up | move_down]
        self.num_successes[changed] = 0
        self.num_failures[changed] = 0
        terrain.update_env_origins(env_ids, move_up, move_down)
        return torch.mean(terrain.terrain_levels.float())


class action_limit_successful_termination(ManagerTermBase):
    """Legacy action-range curriculum driven by successful recoveries."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self.ema_success_ratio = 0.0
        self._action = env.action_manager._terms[cfg.params["action_name"]]

    def __call__(
        self,
        env,
        env_ids: torch.Tensor,
        action_name: str,
        successful_termination_term: str,
        activate_after_steps: int = 0,
        move_up_ratio: float = 0.95,
        move_down_ratio: float = 0.8,
        update_rate: float = 0.0001,
        ema_decay: float = 0.99,
        max_action_limit: float = 1.0,
        min_action_limit: float = 0.0,
    ) -> float:
        del action_name
        if env.common_step_counter < activate_after_steps or len(env_ids) == 0:
            return float(torch.max(torch.abs(self._action._clip)).item())
        succeeded = env.termination_manager.get_term(
            successful_termination_term
        )[env_ids]
        ratio = float(succeeded.float().mean().item())
        self.ema_success_ratio = (
            ema_decay * self.ema_success_ratio + (1.0 - ema_decay) * ratio
        )
        if self.ema_success_ratio > move_up_ratio:
            factor = 1.0 + (
                move_up_ratio - self.ema_success_ratio
            ) * update_rate
            self._action._clip *= factor
        elif self.ema_success_ratio < move_down_ratio:
            factor = 1.0 + (
                move_down_ratio - self.ema_success_ratio
            ) * update_rate
            self._action._clip *= factor
        signs = torch.sign(self._action._clip)
        limits = torch.clamp(
            torch.abs(self._action._clip),
            min=min_action_limit,
            max=max_action_limit,
        )
        self._action._clip.copy_(signs * limits)
        return float(torch.max(limits).item())


def remove_harness(
    env,
    env_ids: Sequence[int],
    harness_action_name: str,
    start: int,
    num_steps: int,
    linear: bool = True,
) -> float:
    """Legacy fixed schedule for removing lift assistance."""
    del env_ids
    lift = env.action_manager._terms[harness_action_name]
    if env.common_step_counter <= start:
        scale = 1.0
    elif env.common_step_counter >= start + num_steps:
        scale = 0.0
    elif linear:
        scale = 1.0 - (env.common_step_counter - start) / num_steps
    else:
        progress = (env.common_step_counter - start) / num_steps
        scale = math.exp(progress * math.log(0.01))
    lift.scale_forces(scale)
    return float(scale)


class terrain_levels_standing_at_timeout(ManagerTermBase):
    """Advance terrain after repeated standing timeouts and regress after failures."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self.num_failures = torch.zeros(
            env.num_envs,
            dtype=torch.int32,
            device=env.device,
        )
        self.num_successes = torch.zeros_like(self.num_failures)

    def __call__(
        self,
        env,
        env_ids: torch.Tensor,
        min_height: float,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sensor_cfg: SceneEntityCfg | None = None,
        n_failures: int = 5,
        n_successes: int = 5,
    ) -> torch.Tensor:
        terrain = env.scene.terrain
        if len(env_ids) == 0:
            return torch.mean(terrain.terrain_levels.float())
        asset = env.scene[asset_cfg.name]
        current_height = asset.data.root_pos_w[env_ids, 2]
        if sensor_cfg is not None:
            sensor: RayCaster = env.scene[sensor_cfg.name]
            current_height -= torch.mean(
                sensor.data.ray_hits_w[env_ids, ..., 2],
                dim=1,
            )
        succeeded = (
            env.termination_manager.time_outs[env_ids]
            & (current_height > min_height)
        )
        self.num_successes[env_ids] += succeeded
        self.num_failures[env_ids] += ~succeeded
        move_up = self.num_successes[env_ids] >= n_successes
        move_down = self.num_failures[env_ids] >= n_failures
        changed = env_ids[move_up | move_down]
        self.num_successes[changed] = 0
        self.num_failures[changed] = 0
        terrain.update_env_origins(env_ids, move_up, move_down)
        return torch.mean(terrain.terrain_levels.float())


class adaptive_force_decay(ManagerTermBase):
    """WBC standing-ratio EMA curriculum for removing lift assistance."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._force_scale = 1.0
        self._ema = 0.0
        self._action = env.action_manager._terms[cfg.params["action_name"]]

    def __call__(
        self,
        env,
        env_ids: torch.Tensor,
        action_name: str,
        standing_height_threshold: float,
        threshold: float = 0.7,
        ema_alpha: float = 0.01,
        decay: float = 0.999,
        disable_threshold: float = 0.01,
    ) -> float:
        del env, action_name
        if self._force_scale <= 0.0 or len(env_ids) == 0:
            return self._force_scale
        standing_ratio = float(
            (
                self._action.max_heights[env_ids]
                > standing_height_threshold
            )
            .float()
            .mean()
            .item()
        )
        self._ema = ema_alpha * standing_ratio + (1.0 - ema_alpha) * self._ema
        if self._ema > threshold:
            self._force_scale *= decay
            if self._force_scale < disable_threshold:
                self._force_scale = 0.0
        self._action.scale_forces(self._force_scale)
        return self._force_scale


class update_reward_weight_step(ManagerTermBase):
    """Linearly or logarithmically ramp a reward weight by environment steps."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        reward_name = cfg.params["reward_name"]
        self.start_weight = float(env.reward_manager.get_term_cfg(reward_name).weight)
        terminal_weight = float(cfg.params["terminal_weight"])
        if cfg.params.get("use_log_space", False):
            if (
                self.start_weight == 0.0
                or terminal_weight == 0.0
                or (self.start_weight > 0.0) != (terminal_weight > 0.0)
            ):
                raise ValueError("log-space reward curriculum requires equal non-zero signs")

    def __call__(
        self,
        env,
        env_ids: Sequence[int],
        reward_name: str,
        start_step: int,
        num_steps: int,
        terminal_weight: float,
        use_log_space: bool = False,
    ) -> float:
        del env_ids
        if env.common_step_counter <= start_step:
            new_weight = self.start_weight
        elif env.common_step_counter >= start_step + num_steps:
            new_weight = terminal_weight
        else:
            scale = (env.common_step_counter - start_step) / num_steps
            if use_log_space:
                magnitude = math.exp(
                    math.log(abs(self.start_weight))
                    + scale
                    * (
                        math.log(abs(terminal_weight))
                        - math.log(abs(self.start_weight))
                    )
                )
                new_weight = magnitude if self.start_weight > 0.0 else -magnitude
            else:
                new_weight = self.start_weight + scale * (
                    terminal_weight - self.start_weight
                )
        env.reward_manager.get_term_cfg(reward_name).weight = new_weight
        return float(new_weight)

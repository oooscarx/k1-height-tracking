# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.sensors import RayCaster
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

from .governor import HeightCommandGovernor

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class SmoothHeightCommand(CommandTerm):
    """Sample terrain-relative body-height targets with optional smooth ramps."""

    cfg: "SmoothHeightCommandCfg"

    def __init__(self, cfg: "SmoothHeightCommandCfg", env: ManagerBasedRLEnv) -> None:
        super().__init__(cfg, env)
        if not 0.0 <= cfg.focus_ratio <= 1.0:
            raise ValueError("focus_ratio must be within [0, 1]")
        focus_minimum, focus_maximum = cfg.focus_height_range
        command_minimum, command_maximum = cfg.ranges.height
        if cfg.focus_ratio > 0.0 and not command_minimum <= focus_minimum < focus_maximum <= command_maximum:
            raise ValueError("focus_height_range must be ordered and inside the command height range")
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._height_sensor: RayCaster = env.scene.sensors[cfg.height_sensor]
        body_ids, _ = self.robot.find_bodies(cfg.body_name)
        if len(body_ids) != 1:
            raise ValueError(f"height command body must resolve exactly once: {cfg.body_name}")
        self._body_id = body_ids[0]
        self._offset_local = torch.tensor(cfg.offset.pos, device=self.device, dtype=torch.float32)
        self._target_height = torch.zeros(self.num_envs, device=self.device)
        self._current_height_cmd = torch.zeros(self.num_envs, device=self.device)
        self._velocity = torch.zeros(self.num_envs, device=self.device)
        self._manual_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._manual_height = torch.zeros(self.num_envs, device=self.device)
        self.metrics["height_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["high_height_error"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["high_height_success"] = torch.zeros(self.num_envs, device=self.device)
        self._episode_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_step_count = torch.zeros(self.num_envs, device=self.device)
        self._episode_high_error_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_high_success_sum = torch.zeros(self.num_envs, device=self.device)
        self._episode_high_step_count = torch.zeros(self.num_envs, device=self.device)
        self._steps_since_resample = torch.zeros(self.num_envs, device=self.device)
        self._settle_steps = int(cfg.settle_time_s / env.step_dt)
        self.governor = HeightCommandGovernor(self.num_envs, self.device, cfg.governor_step_m,
                                             cfg.governor_hold_s) if cfg.governed_commands else None
        self.uses_reference_cohort = self.governor is not None
        self._reference = torch.arange(self.num_envs, device=self.device) % 5 == 0
        self._reference_last = {name: 0.0 for name in self.metrics}
        self._goal_active = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._final_count = torch.zeros(self.num_envs, device=self.device)
        self._final_good = torch.zeros_like(self._final_count)
        self._first_success_s = torch.full_like(self._final_count, -1.0)
        # Per-process completed-goal counters, indexed by governed/jump group.
        self._goal_totals = torch.zeros((2, 7), device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._current_height_cmd.unsqueeze(-1)

    @property
    def target_height(self) -> torch.Tensor:
        return self._current_height_cmd

    @property
    def command_age_s(self) -> torch.Tensor:
        return self._steps_since_resample * self._env.step_dt

    @property
    def measured_height(self) -> torch.Tensor:
        return self._measure_height()

    @property
    def settled(self) -> torch.Tensor:
        return self._steps_since_resample > self._settle_steps

    @property
    def relaxation_intensity(self) -> torch.Tensor:
        minimum = self.cfg.ranges.height[0]
        if minimum >= 0.0:
            return torch.zeros(self.num_envs, device=self.device)
        return torch.clamp(-self._current_height_cmd / -minimum, 0.0, 1.0)

    @torch.inference_mode()
    def set_manual_height(
        self,
        height: float | torch.Tensor,
        env_ids: Sequence[int] | None = None,
    ) -> None:
        """Hold selected environments at an explicit height command."""
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        values = torch.as_tensor(height, device=self.device, dtype=torch.float32).reshape(-1)
        if values.numel() == 1:
            values = values.expand(ids.numel())
        if values.numel() != ids.numel():
            raise ValueError(f"manual height count {values.numel()} does not match environment count {ids.numel()}")
        if not torch.isfinite(values).all():
            raise ValueError("manual height must be finite")
        self._manual_height[ids] = values.clamp(*self.cfg.ranges.height)
        self._manual_mask[ids] = True
        self._apply_manual_height(ids)

    @torch.inference_mode()
    def clear_manual_height(self, env_ids: Sequence[int] | None = None) -> None:
        """Return selected environments to normal command sampling."""
        if env_ids is None:
            ids = torch.arange(self.num_envs, device=self.device)
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._manual_mask[ids] = False
        self._resample_command(ids)

    def _apply_manual_height(self, env_ids: Sequence[int] | torch.Tensor) -> None:
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        ids = ids[self._manual_mask[ids]]
        if ids.numel() == 0:
            return
        values = self._manual_height[ids]
        self._target_height[ids] = values
        self._current_height_cmd[ids] = values
        self._velocity[ids] = 0.0
        self.time_left[ids] = 1.0e9
        self._goal_active[ids] = False
        if self.governor is not None:
            self.governor.managed[ids] = False
            self.governor.final[ids] = values
            self.governor.first[ids] = values
            self.governor.elapsed[ids] = 0.0
            self.governor.advances[ids] = 0.0

    def _tracked_point_pos_w(self) -> torch.Tensor:
        body_pos = self.robot.data.body_pos_w[:, self._body_id]
        body_quat = self.robot.data.body_quat_w[:, self._body_id]
        offset = quat_apply(body_quat, self._offset_local.expand(self.num_envs, -1))
        return body_pos + offset

    def _measure_height(self) -> torch.Tensor:
        ground_height = torch.mean(self._height_sensor.data.ray_hits_w[..., 2], dim=-1)
        return self._tracked_point_pos_w()[:, 2] - ground_height

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        if self.governor is not None:
            self._finish_goals(env_ids)
        self._steps_since_resample[env_ids] = 0
        count = len(env_ids)
        roll = torch.rand(count, device=self.device)
        standing = roll < self.cfg.standing_ratio
        flat = (~standing) & (roll < self.cfg.standing_ratio + self.cfg.flat_ratio)
        uniform = ~standing & ~flat
        heights = torch.empty(count, device=self.device)
        heights[standing] = self.cfg.ranges.height[1]
        heights[flat] = self.cfg.ranges.height[0]
        if torch.any(uniform):
            uniform_ids = torch.where(uniform)[0]
            heights[uniform_ids] = torch.empty(uniform_ids.numel(), device=self.device).uniform_(
                *self.cfg.ranges.height
            )
            if self.cfg.focus_ratio > 0.0:
                focused = torch.rand(uniform_ids.numel(), device=self.device) < self.cfg.focus_ratio
                focused_ids = uniform_ids[focused]
                heights[focused_ids] = torch.empty(focused_ids.numel(), device=self.device).uniform_(
                    *self.cfg.focus_height_range
                )
        self._target_height[env_ids] = heights
        self._velocity[env_ids] = torch.empty(count, device=self.device).uniform_(*self.cfg.velocity_range)
        if self.governor is not None:
            self.time_left[env_ids] = self.governor.request(
                env_ids, heights, self.measured_height[env_ids], ~self._reference[env_ids],
                self.time_left[env_ids])
            self._target_height[env_ids] = self.governor.command[env_ids]
            self._goal_active[env_ids] = heights >= 0
            self._final_count[env_ids] = 0
            self._final_good[env_ids] = 0
            self._first_success_s[env_ids] = -1
        self._apply_manual_height(env_ids)

    def _update_command(self) -> None:
        if self.governor is not None:
            self._target_height = self.governor.step(self._env.step_dt)
            # Reward age measures the whole request; a scheduled advance cannot
            # remove the fine reward by restarting its settling gate.
        if torch.any(self._manual_mask):
            self._target_height[self._manual_mask] = self._manual_height[self._manual_mask]
        difference = self._target_height - self._current_height_cmd
        maximum_step = self._velocity * self._env.step_dt
        self._current_height_cmd += torch.clamp(difference, -maximum_step, maximum_step)

    def _update_metrics(self) -> None:
        error = torch.abs(self.measured_height - self._current_height_cmd)
        self._steps_since_resample += 1
        finite_error = torch.isfinite(error)
        valid = self.settled & (self._current_height_cmd >= 0.0) & finite_error
        if self.governor is not None:
            # Unlike reward age, diagnostic settling is per intermediate target.
            valid = (
                (self.governor.command_age > self.cfg.settle_time_s)
                & (self._current_height_cmd >= 0)
                & finite_error
            )
            final_valid = valid & self.governor.at_final & self._goal_active
            self._final_count += final_valid
            self._final_good += final_valid & (error < self.cfg.success_error_threshold)
            successful = (self._final_count * self._env.step_dt >= 1.0) & (self._final_good >= .9 * self._final_count)
            first = successful & (self._first_success_s < 0)
            self._first_success_s[first] = self.governor.elapsed[first]
        self._episode_error_sum += torch.where(valid, error, 0.0)
        self._episode_step_count += valid.float()
        self.metrics["height_error"] = self._episode_error_sum / self._episode_step_count.clamp(min=1.0)
        high = valid & (self._current_height_cmd >= self.cfg.high_height_threshold)
        self._episode_high_error_sum += torch.where(high, error, 0.0)
        self._episode_high_success_sum += (error < self.cfg.success_error_threshold) * high
        self._episode_high_step_count += high.float()
        high_count = self._episode_high_step_count.clamp(min=1.0)
        self.metrics["high_height_error"] = self._episode_high_error_sum / high_count
        self.metrics["high_height_success"] = self._episode_high_success_sum / high_count

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        group_values = {}
        if self.governor is not None:
            for name, values in self.metrics.items():
                reference_ids = env_ids[self._reference[env_ids]]
                if reference_ids.numel():
                    self._reference_last[name] = float(values[reference_ids].mean())
                managed_ids = env_ids[~self._reference[env_ids]]
                group_values[name] = self._reference_last[name]
                if managed_ids.numel():
                    group_values["governed_" + name] = float(values[managed_ids].mean())
            self._finish_goals(env_ids, timeout=True,
                               terminated=self._env.termination_manager.terminated[env_ids])
        self._episode_error_sum[env_ids] = 0.0
        self._episode_step_count[env_ids] = 0.0
        self._episode_high_error_sum[env_ids] = 0.0
        self._episode_high_success_sum[env_ids] = 0.0
        self._episode_high_step_count[env_ids] = 0.0
        self._current_height_cmd[env_ids] = self._measure_height()[env_ids].clamp(*self.cfg.ranges.height)
        logged = super().reset(env_ids)
        if self.governor is not None:
            for name in self.metrics:
                logged["mixed_" + name] = logged[name]
            logged.update(group_values)
            logged["reference_reset_count"] = float(self._reference[env_ids].sum())
            logged.update(getattr(self, "lift_gate_metrics", {}))
            logged.update(getattr(self, "disturbance_gate_metrics", {}))
            for group, row in zip(("governed", "jump"), self._goal_totals.cpu().tolist()):
                for name, value in zip(("ended", "observed", "success", "timeout", "terminated",
                                         "success_time_sum", "censored"), row):
                    logged[f"governor_{group}_{name}"] = value
        return logged

    def reference_curriculum_metric(self, env_ids, name):
        selected = env_ids[self._reference[env_ids]]
        self._reference_curriculum_fraction = selected.numel() / max(len(env_ids), 1)
        return self.metrics[name][selected].mean() if selected.numel() else None

    def cohort_curriculum_metrics(self, env_ids, name):
        valid_ids = env_ids[self._episode_step_count[env_ids] > 0]
        result = {}
        for group, mask in (("jump", self._reference), ("governed", ~self._reference)):
            selected = valid_ids[mask[valid_ids]]
            if selected.numel():
                result[group] = (float(self.metrics[name][selected].mean()),
                                 selected.numel() / max(len(env_ids), 1))
        return result

    def _finish_goals(self, env_ids, timeout=False, terminated=None):
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        active = self._goal_active[ids]
        observed = self._final_count[ids] * self._env.step_dt >= 1.0
        failed = torch.zeros_like(active) if terminated is None else terminated
        success = observed & (self._final_good[ids] >= .9 * self._final_count[ids]) & ~failed
        values = torch.stack((active.float(), (active & observed).float(), (active & success).float(),
                              (active & ~failed).float() * timeout, (active & failed).float(),
                              torch.where(active & success, self._first_success_s[ids], 0),
                              (active & ~observed).float()), dim=1)
        self._goal_totals.index_add_(0, self._reference[ids].long(), values)
        self._goal_active[ids] = False

    def _set_debug_vis_impl(self, debug_vis: bool) -> None:
        if debug_vis and not hasattr(self, "_goal_marker"):
            self._goal_marker = VisualizationMarkers(self.cfg.goal_visualizer_cfg)
            self._measured_marker = VisualizationMarkers(self.cfg.measured_visualizer_cfg)
        if hasattr(self, "_goal_marker"):
            self._goal_marker.set_visibility(debug_vis)
            self._measured_marker.set_visibility(debug_vis)

    def _debug_vis_callback(self, event: object) -> None:
        del event
        if not self.robot.is_initialized:
            return
        tracked = self._tracked_point_pos_w()
        ground_height = torch.mean(self._height_sensor.data.ray_hits_w[..., 2], dim=-1)
        goal = tracked.clone()
        goal[:, 0] += 0.25
        goal[:, 2] = ground_height + self._current_height_cmd
        measured = tracked.clone()
        measured[:, 0] += 0.25
        self._goal_marker.visualize(goal)
        self._measured_marker.visualize(measured)


@configclass
class SmoothHeightCommandCfg(CommandTermCfg):
    class_type: type = SmoothHeightCommand
    asset_name: str = "robot"
    body_name: str = "Trunk"
    height_sensor: str = "height_measurement_sensor"
    velocity_range: tuple[float, float] = (0.05, 0.5)
    settle_time_s: float = 1.0
    standing_ratio: float = 0.0
    flat_ratio: float = 0.0
    focus_ratio: float = 0.0
    focus_height_range: tuple[float, float] = (0.0, 0.72)
    high_height_threshold: float = 0.6
    success_error_threshold: float = 0.08
    governed_commands: bool = False
    governor_step_m: float = 0.12
    governor_hold_s: float = 4.0

    @configclass
    class Ranges:
        height: tuple[float, float] = (0.0, 0.72)

    @configclass
    class OffsetCfg:
        pos: tuple[float, float, float] = (0.0, 0.0, 0.0)

    ranges: Ranges = Ranges()
    offset: OffsetCfg = OffsetCfg()
    goal_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/height_goal",
        markers={
            "cube": sim_utils.CuboidCfg(
                size=(0.05, 0.05, 0.05),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0), opacity=0.7),
            )
        },
    )
    measured_visualizer_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Command/height_measured",
        markers={
            "cube": sim_utils.CuboidCfg(
                size=(0.05, 0.05, 0.05),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.5, 1.0), opacity=0.7),
            )
        },
    )

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.standing_ratio + self.flat_ratio > 1.0:
            raise ValueError("standing_ratio + flat_ratio must not exceed 1")
        if self.success_error_threshold <= 0.0:
            raise ValueError("success_error_threshold must be positive")

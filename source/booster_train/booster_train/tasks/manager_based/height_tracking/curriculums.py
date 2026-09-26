# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch
from isaaclab.managers import ManagerTermBase

Direction = Literal["above", "below"]


def _state_value(env, name: str):
    state = getattr(env.curriculum_manager, "_curriculum_state", {})
    value = state.get(name)
    if isinstance(value, torch.Tensor) and value.numel() == 1:
        return float(value.item())
    return value


def _check_prerequisite(env, name: str | None, threshold: float, direction: Direction) -> bool:
    if name is None:
        return True
    value = _state_value(env, name)
    if value is None:
        return False
    return bool(value <= threshold if direction == "below" else value >= threshold)


class adaptive_force_decay(ManagerTermBase):
    """Decay lift assistance after the command tracking metric crosses its threshold."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._force_scale = float(cfg.params.get("initial_force_scale", 1.0))
        if not 0.0 <= self._force_scale <= 1.0:
            raise ValueError("initial_force_scale must be within [0, 1]")
        self._action = env.action_manager._terms[cfg.params["action_name"]]
        self._scale_method = cfg.params.get("scale_method", "scale_forces")
        default_ema = 0.0 if cfg.params.get("decay_when", "above") == "above" else 1.0
        self._ema = float(cfg.params.get("initial_ema", default_ema))
        if not math.isfinite(self._ema) or self._ema < 0.0:
            raise ValueError("initial_ema must be finite and non-negative")
        self._command = env.command_manager.get_term(cfg.params["command_name"])
        self._governed_ema: float | None = None
        self._jump_seen = False
        self._resume_warmup_until_step: int | None = None
        getattr(self._action, self._scale_method)(self._force_scale)

    def arm_resume_warmup(self, current_step: int, warmup_steps: int) -> None:
        """Freeze lift decay while resumed environments produce valid episode metrics."""
        if warmup_steps < 0:
            raise ValueError("resume warmup steps must be non-negative")
        self._resume_warmup_until_step = int(current_step) + int(warmup_steps)

    def _dual_cohort_gate(self, env_ids, metric_name, ema_alpha, threshold, governed_threshold):
        values = self._command.cohort_curriculum_metrics(env_ids, metric_name)
        if any(not math.isfinite(value) or value < 0 for value, _ in values.values()):
            return False
        for group, (value, fraction) in values.items():
            alpha = 1.0 - (1.0 - ema_alpha) ** fraction
            if group == "jump":
                self._ema = alpha * value + (1.0 - alpha) * self._ema
                self._jump_seen = True
            else:
                self._governed_ema = value if self._governed_ema is None else (
                    alpha * value + (1.0 - alpha) * self._governed_ema)
        ready = self._jump_seen and self._governed_ema is not None
        passed = ready and self._ema < threshold and self._governed_ema < governed_threshold
        self._command.lift_gate_metrics = {
            "lift_jump_error_ema": self._ema,
            "lift_governed_error_ema": self._governed_ema if self._governed_ema is not None else 0.0,
            "lift_gate_ready": float(ready),
            "lift_jump_threshold": threshold,
            "lift_governed_threshold": governed_threshold,
            "lift_gate_passed": float(passed),
        }
        # Preserve the reference-cohort update cadence; managed-only resets
        # cannot repeatedly decay lift against an old reference measurement.
        return passed and "jump" in values

    def __call__(
        self,
        env,
        env_ids: torch.Tensor,
        action_name: str,
        metric_name: str = "height_error",
        decay_when: Direction = "below",
        threshold: float = 0.1,
        ema_alpha: float = 0.05,
        decay: float = 0.9999,
        disable_threshold: float = 0.01,
        command_name: str | None = None,
        scale_method: str = "scale_forces",
        prerequisite_curriculum: str | None = None,
        prerequisite_threshold: float = 0.0,
        prerequisite_direction: Direction = "below",
        initial_force_scale: float = 1.0,
        initial_ema: float | None = None,
        governed_threshold: float | None = None,
    ) -> float:
        del action_name, command_name, scale_method, initial_force_scale, initial_ema
        if self._force_scale <= 0.0:
            return 0.0
        if self._resume_warmup_until_step is not None:
            if env.common_step_counter < self._resume_warmup_until_step:
                getattr(self._action, self._scale_method)(self._force_scale)
                return self._force_scale
            self._resume_warmup_until_step = None
        uses_instantaneous_height = (
            metric_name == "height_error"
            and hasattr(self._command, "base_height")
            and not hasattr(self._command, "settled")
        )
        if not uses_instantaneous_height and len(env_ids) == 0:
            return self._force_scale
        if not _check_prerequisite(
            env,
            prerequisite_curriculum,
            prerequisite_threshold,
            prerequisite_direction,
        ):
            return self._force_scale
        if governed_threshold is not None:
            if decay_when != "below" or not getattr(self._command, "uses_reference_cohort", False):
                raise ValueError("dual-cohort lift requires governed commands and below thresholds")
            should_decay = self._dual_cohort_gate(
                env_ids, metric_name, ema_alpha, threshold, governed_threshold)
        elif uses_instantaneous_height:
            metric = float(torch.abs(self._command.base_height - self._command.target_height).mean().item())
        else:
            if getattr(self._command, "uses_reference_cohort", False):
                value = self._command.reference_curriculum_metric(env_ids, metric_name)
                if value is None:
                    return self._force_scale
                metric = float(value.item())
                ema_alpha = 1.0 - (1.0 - ema_alpha) ** self._command._reference_curriculum_fraction
            else:
                metric = float(self._command.metrics[metric_name][env_ids].mean().item())
        if governed_threshold is None:
            self._ema = ema_alpha * metric + (1.0 - ema_alpha) * self._ema
            should_decay = self._ema > threshold if decay_when == "above" else self._ema < threshold
        if should_decay:
            self._force_scale *= decay
            if self._force_scale < disable_threshold:
                self._force_scale = 0.0
        getattr(self._action, self._scale_method)(self._force_scale)
        return self._force_scale


class terrain_levels_tracking_at_timeout(ManagerTermBase):
    """Advance terrain after repeated low-error timeouts and regress after failures."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        self._failures = torch.zeros(env.num_envs, device=env.device, dtype=torch.int32)
        self._successes = torch.zeros_like(self._failures)
        self._activated = False
        self._command = env.command_manager.get_term(cfg.params["command_name"])

    def __call__(
        self,
        env,
        env_ids: torch.Tensor,
        command_name: str,
        error_threshold: float = 0.1,
        n_failures: int = 5,
        n_successes: int = 5,
        prerequisite_curriculum: str | None = None,
        prerequisite_threshold: float = 0.0,
        prerequisite_direction: Direction = "below",
        error_metric_name: str = "height_error",
    ) -> torch.Tensor:
        del command_name
        terrain = env.scene.terrain
        if len(env_ids) == 0:
            return torch.mean(terrain.terrain_levels.float())
        if not self._activated and prerequisite_curriculum is not None:
            if not _check_prerequisite(
                env,
                prerequisite_curriculum,
                prerequisite_threshold,
                prerequisite_direction,
            ):
                return torch.mean(terrain.terrain_levels.float())
            self._activated = True
        error = self._command.metrics[error_metric_name][env_ids]
        succeeded = env.termination_manager.time_outs[env_ids] & (error < error_threshold)
        self._successes[env_ids] += succeeded
        self._failures[env_ids] += ~succeeded
        move_up = self._successes[env_ids] >= n_successes
        move_down = self._failures[env_ids] >= n_failures
        changed = env_ids[move_up | move_down]
        self._successes[changed] = 0
        self._failures[changed] = 0
        terrain.update_env_origins(env_ids, move_up, move_down)
        return torch.mean(terrain.terrain_levels.float())


class adaptive_external_force_growth(ManagerTermBase):
    """Increase interval-force randomization only after tracking is mastered."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        initial_scale = float(cfg.params.get("initial_scale", 1.0))
        maximum_scale = float(cfg.params.get("maximum_scale", 1.5))
        growth_per_step = float(cfg.params.get("growth_per_step", 2.5e-5))
        if not 0.0 <= initial_scale <= maximum_scale:
            raise ValueError("external-force scales must satisfy 0 <= initial <= maximum")
        if growth_per_step <= 0.0:
            raise ValueError("growth_per_step must be positive")

        self._scale = initial_scale
        self._last_step = int(env.common_step_counter)
        self._jump_ema: float | None = None
        self._governed_ema: float | None = None
        self._command = env.command_manager.get_term(cfg.params["command_name"])
        self._event_manager = env.event_manager
        self._event_ranges: dict[str, dict[str, tuple[float, float]]] = {}
        for event_name in cfg.params["event_names"]:
            event_cfg = self._event_manager.get_term_cfg(event_name)
            ranges = {}
            for range_name in ("force_range", "torque_range"):
                lower, upper = event_cfg.params[range_name]
                ranges[range_name] = (float(lower), float(upper))
            self._event_ranges[event_name] = ranges
        self._apply_scale()

    def _apply_scale(self) -> None:
        for event_name, ranges in self._event_ranges.items():
            event_cfg = self._event_manager.get_term_cfg(event_name)
            for range_name, (lower, upper) in ranges.items():
                event_cfg.params[range_name] = (lower * self._scale, upper * self._scale)
            self._event_manager.set_term_cfg(event_name, event_cfg)

    @staticmethod
    def _updated_ema(current: float | None, value: float, fraction: float, alpha: float) -> float:
        effective_alpha = 1.0 - (1.0 - alpha) ** fraction
        return value if current is None else effective_alpha * value + (1.0 - effective_alpha) * current

    def __call__(
        self,
        env,
        env_ids: torch.Tensor,
        event_names: Sequence[str],
        command_name: str,
        error_metric_name: str = "high_height_error",
        jump_error_threshold: float = 0.14,
        governed_error_threshold: float = 0.12,
        ema_alpha: float = 0.05,
        minimum_terrain_level: float = 3.5,
        initial_scale: float = 1.0,
        maximum_scale: float = 1.5,
        growth_per_step: float = 2.5e-5,
    ) -> float:
        del event_names, command_name, initial_scale
        current_step = int(env.common_step_counter)
        elapsed_steps = max(0, current_step - self._last_step)
        self._last_step = current_step

        values = self._command.cohort_curriculum_metrics(env_ids, error_metric_name)
        for group, (value, fraction) in values.items():
            if not math.isfinite(value) or value < 0.0:
                continue
            if group == "jump":
                self._jump_ema = self._updated_ema(self._jump_ema, value, fraction, ema_alpha)
            elif group == "governed":
                self._governed_ema = self._updated_ema(self._governed_ema, value, fraction, ema_alpha)

        terrain_level = float(env.scene.terrain.terrain_levels.float().mean().item())
        ready = self._jump_ema is not None and self._governed_ema is not None
        passed = bool(
            ready
            and self._jump_ema < jump_error_threshold
            and self._governed_ema < governed_error_threshold
            and terrain_level >= minimum_terrain_level
        )
        if passed and elapsed_steps > 0:
            self._scale = min(maximum_scale, self._scale + growth_per_step * elapsed_steps)
            self._apply_scale()

        self._command.disturbance_gate_metrics = {
            "disturbance_scale": self._scale,
            "disturbance_jump_error_ema": self._jump_ema or 0.0,
            "disturbance_governed_error_ema": self._governed_ema or 0.0,
            "disturbance_terrain_level": terrain_level,
            "disturbance_gate_ready": float(ready),
            "disturbance_gate_passed": float(passed),
        }
        return self._scale


class update_reward_weight_after_curriculum(ManagerTermBase):
    """Ramp a reward weight after another curriculum reaches a threshold."""

    def __init__(self, cfg, env) -> None:
        super().__init__(cfg, env)
        reward_name = cfg.params["reward_name"]
        self._start_weight = float(env.reward_manager.get_term_cfg(reward_name).weight)
        self._trigger_step: int | None = None
        terminal = float(cfg.params["terminal_weight"])
        if cfg.params.get("use_log_space", False) and (
            self._start_weight == 0.0
            or terminal == 0.0
            or (self._start_weight > 0.0) != (terminal > 0.0)
        ):
            raise ValueError("log-space reward curriculum requires equal non-zero signs")

    def __call__(
        self,
        env,
        env_ids: Sequence[int],
        reward_name: str,
        prerequisite_curriculum: str,
        prerequisite_threshold: float,
        delay_steps: int,
        num_steps: int,
        terminal_weight: float,
        use_log_space: bool = False,
    ) -> float:
        del env_ids
        if self._trigger_step is None:
            value = _state_value(env, prerequisite_curriculum)
            if value is None or value < prerequisite_threshold:
                return self._start_weight
            self._trigger_step = env.common_step_counter
        ramp_start = self._trigger_step + delay_steps
        if env.common_step_counter <= ramp_start:
            weight = self._start_weight
        elif env.common_step_counter >= ramp_start + num_steps:
            weight = terminal_weight
        else:
            scale = (env.common_step_counter - ramp_start) / num_steps
            if use_log_space:
                magnitude = math.exp(
                    math.log(abs(self._start_weight))
                    + scale * (math.log(abs(terminal_weight)) - math.log(abs(self._start_weight)))
                )
                weight = magnitude if self._start_weight > 0.0 else -magnitude
            else:
                weight = self._start_weight + scale * (terminal_weight - self._start_weight)
        env.reward_manager.get_term_cfg(reward_name).weight = weight
        return float(weight)

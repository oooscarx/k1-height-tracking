from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

_RECOVERY_MAXIMUM_KEYS = (
    "maximum_angular_velocity",
    "maximum_linear_velocity",
    "maximum_body_pose_error",
    "maximum_body_joint_velocity",
)


def sanitize_nonfinite_action_inputs(
    actions: torch.Tensor,
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    position_center: torch.Tensor,
    fallback_center: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Isolate non-finite environments while preserving finite command inputs."""
    if actions.shape != joint_position.shape or actions.shape != joint_velocity.shape:
        raise ValueError("action and joint-state tensors must have the same shape")
    if actions.ndim != 2:
        raise ValueError("action and joint-state tensors must be matrices")
    for name, value in (
        ("position center", position_center),
        ("fallback center", fallback_center),
    ):
        if value.ndim != 2 or value.shape[-1] != actions.shape[-1]:
            raise ValueError(f"{name} must have the action dimension")
        if value.shape[0] not in (1, actions.shape[0]):
            raise ValueError(f"{name} must contain one row or one row per environment")
    expanded_center = position_center.expand(actions.shape[0], -1)
    expanded_fallback = fallback_center.expand(actions.shape[0], -1)
    finite_action = torch.all(torch.isfinite(actions), dim=-1)
    finite_position = torch.all(torch.isfinite(joint_position), dim=-1)
    finite_velocity = torch.all(torch.isfinite(joint_velocity), dim=-1)
    finite_center = torch.all(torch.isfinite(expanded_center), dim=-1)
    nonfinite = ~(finite_action & finite_position & finite_velocity & finite_center)

    safe_center = torch.where(
        torch.isfinite(expanded_center),
        expanded_center,
        expanded_fallback,
    )
    safe_actions = torch.where(
        nonfinite.unsqueeze(-1),
        torch.zeros_like(actions),
        actions,
    )
    safe_position = torch.where(
        torch.isfinite(joint_position),
        joint_position,
        safe_center,
    )
    safe_velocity = torch.where(
        torch.isfinite(joint_velocity),
        joint_velocity,
        torch.zeros_like(joint_velocity),
    )
    return safe_actions, safe_position, safe_velocity, safe_center, nonfinite


def scaled_joint_position_delta(
    actions: torch.Tensor,
    scale: torch.Tensor,
    *,
    normalize_input: bool,
    delta_minimum: torch.Tensor | float | None = None,
    delta_maximum: torch.Tensor | float | None = None,
) -> torch.Tensor:
    """Scale policy actions and optionally clip the resulting joint delta."""
    if actions.ndim != 2:
        raise ValueError("joint position actions must be a matrix")
    if scale.ndim != 2 or scale.shape[-1] != actions.shape[-1]:
        raise ValueError("joint position action scale must have the action dimension")
    if scale.shape[0] not in (1, actions.shape[0]):
        raise ValueError("joint position action scale must have one row or one row per environment")
    if (delta_minimum is None) != (delta_maximum is None):
        raise ValueError("joint position delta limits must be configured together")

    action_input = torch.clamp(actions, -1.0, 1.0) if normalize_input else actions
    delta = action_input * scale
    if delta_minimum is not None and delta_maximum is not None:
        delta = torch.clamp(delta, min=delta_minimum, max=delta_maximum)
    return delta


def strict_recovery_success_config(params: Mapping[str, Any]) -> dict[str, Any]:
    """Resolve curriculum parameters to the strict deployment thresholds."""
    return recovery_success_config_at_progress(params, 1.0)


def recovery_success_config_at_progress(
    params: Mapping[str, Any],
    progress: float,
) -> dict[str, Any]:
    """Resolve recovery thresholds at one explicit mastery-curriculum progress."""
    if not 0.0 <= progress <= 1.0:
        raise ValueError("recovery success curriculum progress must be in [0, 1]")
    resolved = dict(params)
    for key in _RECOVERY_MAXIMUM_KEYS:
        start_key = f"{key}_start"
        end_key = f"{key}_end"
        if start_key in resolved and end_key in resolved:
            resolved[key] = resolved[start_key] + progress * (
                resolved[end_key] - resolved[start_key]
            )
        elif progress == 1.0 and end_key in resolved:
            resolved[key] = resolved[end_key]
        elif key not in resolved:
            raise KeyError(f"recovery success configuration is missing {key}")
        resolved.pop(start_key, None)
        resolved.pop(end_key, None)
    resolved.pop("curriculum_steps", None)
    return resolved


def training_curriculum_value(
    *,
    common_step_counter: int,
    num_envs: int,
    reset_mode: str,
    start: float,
    end: float,
    curriculum_steps: int,
) -> float:
    """Interpolate by simulated transitions and use the strict value outside training."""
    if num_envs < 1 or curriculum_steps < 1:
        raise ValueError("curriculum environment and transition counts must be positive")
    if common_step_counter < 0:
        raise ValueError("curriculum step counter must be non-negative")
    if reset_mode != "train":
        return end
    transition_count = float(common_step_counter * num_envs)
    progress = min(transition_count / float(curriculum_steps), 1.0)
    return start + progress * (end - start)


def mastery_curriculum_update(
    *,
    progress: float,
    promotion_streak: int,
    demotion_streak: int,
    success_rate: float,
    promotion_threshold: float,
    demotion_threshold: float,
    promotion_windows: int,
    demotion_windows: int,
    promotion_step: float,
    demotion_step: float,
    promotion_allowed: bool = True,
) -> tuple[float, int, int]:
    """Advance a curriculum only after repeated mastery and back off on collapse."""
    if not 0.0 <= progress <= 1.0 or not 0.0 <= success_rate <= 1.0:
        raise ValueError("curriculum progress and success rate must be in [0, 1]")
    if not 0.0 <= demotion_threshold < promotion_threshold <= 1.0:
        raise ValueError("curriculum thresholds must be ordered inside [0, 1]")
    if (
        promotion_windows < 1
        or demotion_windows < 1
        or promotion_streak < 0
        or demotion_streak < 0
    ):
        raise ValueError("curriculum window counts must be non-negative and non-zero")
    if promotion_step <= 0.0 or demotion_step <= 0.0:
        raise ValueError("curriculum progress steps must be positive")

    if success_rate <= demotion_threshold:
        demotion_streak += 1
        if demotion_streak < demotion_windows:
            return progress, 0, demotion_streak
        return max(0.0, progress - demotion_step), 0, 0
    if success_rate < promotion_threshold or not promotion_allowed:
        return progress, 0, 0
    promotion_streak += 1
    if promotion_streak < promotion_windows:
        return progress, promotion_streak, 0
    return min(1.0, progress + promotion_step), 0, 0


def strict_threshold_bottleneck_quality(
    values: torch.Tensor,
    loose_thresholds: torch.Tensor,
    strict_thresholds: torch.Tensor,
) -> torch.Tensor:
    """Score the worst normalized metric between loose and strict upper limits."""
    if values.ndim < 1:
        raise ValueError("handoff values must have a metric dimension")
    if loose_thresholds.ndim != 1 or strict_thresholds.ndim != 1:
        raise ValueError("handoff thresholds must be vectors")
    if values.shape[-1] != loose_thresholds.numel():
        raise ValueError("handoff values and threshold vectors must have the same metric count")
    if loose_thresholds.shape != strict_thresholds.shape:
        raise ValueError("loose and strict handoff thresholds must have the same shape")
    if torch.any(loose_thresholds <= strict_thresholds):
        raise ValueError("loose handoff thresholds must exceed strict thresholds")
    quality = (loose_thresholds - values) / (loose_thresholds - strict_thresholds)
    return torch.amin(torch.clamp(quality, 0.0, 1.0), dim=-1)


def quality_weighted_completion_bonus(
    completion_bonus: torch.Tensor,
    strict_quality: torch.Tensor,
    minimum_scale: float,
    maximum_scale: float,
) -> torch.Tensor:
    """Preserve completion value while adding a deployment-quality premium."""
    if completion_bonus.shape != strict_quality.shape:
        raise ValueError("completion bonus and strict quality must have matching shapes")
    if minimum_scale < 0.0 or maximum_scale < minimum_scale:
        raise ValueError("completion scales must be non-negative and ordered")
    scale = minimum_scale + (maximum_scale - minimum_scale) * torch.clamp(
        strict_quality,
        0.0,
        1.0,
    )
    return completion_bonus * scale


def bounded_potential_progress(
    potential: torch.Tensor,
    previous_potential: torch.Tensor,
    step_dt: float,
    maximum_rate: float,
) -> torch.Tensor:
    """Turn a bounded state quality into non-harvestable progress shaping."""
    if potential.shape != previous_potential.shape:
        raise ValueError("potential tensors must have matching shapes")
    if step_dt <= 0.0 or maximum_rate <= 0.0:
        raise ValueError("potential progress rates must be positive")
    rate = (potential - previous_potential) / step_dt
    return torch.clamp(rate, -maximum_rate, maximum_rate)


def optional_termination_mask(
    termination_manager,
    name: str,
    env_ids: torch.Tensor,
) -> torch.Tensor:
    """Read an active termination term or return an all-false mask."""
    if name not in termination_manager.active_terms:
        return torch.zeros(
            env_ids.numel(),
            dtype=torch.bool,
            device=env_ids.device,
        )
    value = termination_manager.get_term(name)[env_ids].reshape(-1)
    if value.dtype != torch.bool:
        raise ValueError(f"termination term {name} must be boolean")
    return value


def mirror_bilateral_joint_position(
    joint_position: torch.Tensor,
    left_indices: list[int] | tuple[int, ...],
    right_indices: list[int] | tuple[int, ...],
    mirror_signs: list[float] | tuple[float, ...],
) -> torch.Tensor:
    """Swap bilateral joints and apply the configured reflection signs."""
    if not len(left_indices) == len(right_indices) == len(mirror_signs):
        raise ValueError("bilateral index and sign lists must have the same length")
    if len(set(left_indices) | set(right_indices)) != 2 * len(left_indices):
        raise ValueError("bilateral joint indexes must be unique")
    if any(index < 0 or index >= joint_position.shape[-1] for index in (*left_indices, *right_indices)):
        raise ValueError("bilateral joint index is out of range")
    signs = torch.as_tensor(
        mirror_signs,
        dtype=joint_position.dtype,
        device=joint_position.device,
    )
    mirrored = joint_position.clone()
    mirrored[..., left_indices] = joint_position[..., right_indices] * signs
    mirrored[..., right_indices] = joint_position[..., left_indices] * signs
    return mirrored


def reference_phase_bin_indices(phases: torch.Tensor, bin_count: int = 10) -> torch.Tensor:
    """Map normalized reference phases to fixed-width curriculum bins."""
    if bin_count < 1:
        raise ValueError("bin_count must be positive")
    if torch.any(~torch.isfinite(phases)):
        raise ValueError("phases must be finite")
    if torch.any((phases < 0.0) | (phases > 1.0)):
        raise ValueError("phases must be in [0, 1]")
    return torch.clamp(
        torch.floor(phases * bin_count).to(torch.long),
        max=bin_count - 1,
    )


def joint_limit_violation_mask(
    joint_position: torch.Tensor,
    position_minimum: torch.Tensor,
    position_maximum: torch.Tensor,
    margin: float,
    numerical_tolerance: float = 1.0e-6,
) -> torch.Tensor:
    """Return the joints outside deployment limits plus the permitted margin."""
    if joint_position.ndim != 2:
        raise ValueError("joint positions must have shape (environments, joints)")
    if position_minimum.shape != (joint_position.shape[1],):
        raise ValueError("minimum joint limits do not match the joint position width")
    if position_maximum.shape != position_minimum.shape:
        raise ValueError("maximum joint limits do not match minimum joint limits")
    if margin < 0.0 or numerical_tolerance < 0.0:
        raise ValueError("joint-limit margin and numerical tolerance must be non-negative")
    return (joint_position < position_minimum - margin - numerical_tolerance) | (
        joint_position > position_maximum + margin + numerical_tolerance
    )


def limit_joint_position_target(
    target: torch.Tensor,
    current: torch.Tensor,
    velocity: torch.Tensor,
    safe_minimum: torch.Tensor,
    safe_maximum: torch.Tensor,
    braking_horizon_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep targets inside deployment limits and brake outward motion near them."""
    if target.shape != current.shape or target.shape != velocity.shape:
        raise ValueError("joint target, position, and velocity shapes must match")
    if target.ndim != 2:
        raise ValueError("joint targets must have shape (environments, joints)")
    if safe_minimum.shape not in {(target.shape[1],), (1, target.shape[1])}:
        raise ValueError("minimum joint limits do not match the target width")
    if safe_maximum.shape != safe_minimum.shape:
        raise ValueError("maximum joint limits do not match minimum joint limits")
    if braking_horizon_s < 0.0:
        raise ValueError("braking horizon must be non-negative")

    limited = torch.clamp(target, safe_minimum, safe_maximum)
    if braking_horizon_s == 0.0:
        return limited, torch.zeros_like(target, dtype=torch.bool)
    predicted = current + velocity * braking_horizon_s
    braking = ((velocity > 0.0) & (predicted > safe_maximum)) | (
        (velocity < 0.0) & (predicted < safe_minimum)
    )
    braking_target = torch.clamp(
        current - velocity * braking_horizon_s,
        safe_minimum,
        safe_maximum,
    )
    return torch.where(braking, braking_target, limited), braking


def limit_reset_joint_velocity(
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    position_minimum: torch.Tensor,
    position_maximum: torch.Tensor,
    safety_horizon_s: float,
) -> torch.Tensor:
    """Keep a reset velocity inside its safe position envelope over a short horizon."""
    if safety_horizon_s <= 0.0:
        raise ValueError("reset velocity safety horizon must be positive")
    if joint_position.shape != joint_velocity.shape:
        raise ValueError("reset joint position and velocity must have the same shape")
    if torch.any(position_minimum > position_maximum):
        raise ValueError("reset joint position minimum exceeds maximum")
    minimum_velocity = (position_minimum - joint_position) / safety_horizon_s
    maximum_velocity = (position_maximum - joint_position) / safety_horizon_s
    return torch.maximum(
        torch.minimum(joint_velocity, maximum_velocity),
        minimum_velocity,
    )


def held_recovery_completion_bonus(
    success: torch.Tensor,
    stable_steps: torch.Tensor,
    hold_steps: int,
    episode_length: torch.Tensor,
    max_episode_length: int,
) -> torch.Tensor:
    """Return a one-shot, time-scaled bonus after the full recovery hold."""
    if hold_steps < 1:
        raise ValueError("hold_steps must be positive")
    if max_episode_length < 1:
        raise ValueError("max_episode_length must be positive")
    completed = success & (stable_steps >= hold_steps)
    remaining_fraction = 1.0 - episode_length.to(torch.float32) / float(max_episode_length)
    return completed.to(torch.float32) * hold_steps * (1.0 + remaining_fraction)


def stability_hold_progress_reward(
    success: torch.Tensor,
    stable_steps: torch.Tensor,
    hold_steps: int,
) -> torch.Tensor:
    """Densely reward consecutive in-gate steps without rewarding near misses."""
    if success.dtype != torch.bool:
        raise ValueError("stability-hold success mask must be boolean")
    if success.shape != stable_steps.shape:
        raise ValueError("stability-hold tensors must have matching shapes")
    if hold_steps < 1:
        raise ValueError("hold_steps must be positive")
    progress = torch.clamp(
        stable_steps.to(torch.float32) / float(hold_steps),
        0.0,
        1.0,
    )
    return progress * success.to(torch.float32)


def safe_terminal_replay_mask(
    failed_episode: torch.Tensor,
    time_out: torch.Tensor,
    terminal_state_valid: torch.Tensor,
    *,
    joint_limit: torch.Tensor | None = None,
    parallel_ankle: torch.Tensor | None = None,
    nonfinite_action: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select failed timeout states that did not trip a hardware-safety termination."""
    tensors = [failed_episode, time_out, terminal_state_valid]
    tensors.extend(
        value
        for value in (joint_limit, parallel_ankle, nonfinite_action)
        if value is not None
    )
    if any(value.dtype != torch.bool for value in tensors):
        raise ValueError("terminal replay masks must be boolean")
    if any(value.shape != failed_episode.shape for value in tensors[1:]):
        raise ValueError("terminal replay masks must have the same shape")
    selected = failed_episode & time_out & terminal_state_valid
    if joint_limit is not None:
        selected &= ~joint_limit
    if parallel_ankle is not None:
        selected &= ~parallel_ankle
    if nonfinite_action is not None:
        selected &= ~nonfinite_action
    return selected

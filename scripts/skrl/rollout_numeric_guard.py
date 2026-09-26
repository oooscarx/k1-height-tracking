"""Numeric safety helpers for vectorized skrl rollouts."""

from __future__ import annotations

from collections.abc import Mapping

import torch


def configure_normalized_policy_action_bounds(agent) -> None:
    """Give skrl finite bounds for K1's logically normalized action space."""
    policy = agent.models["policy"]
    action_shape = tuple(policy.action_space.shape)
    if len(action_shape) != 1 or action_shape[0] < 1:
        raise ValueError("K1 AMP policy action space must be one-dimensional")
    policy._g_min_actions = torch.full(  # noqa: SLF001
        action_shape,
        -1.0,
        dtype=torch.float32,
        device=agent.device,
    )
    policy._g_max_actions = torch.full(  # noqa: SLF001
        action_shape,
        1.0,
        dtype=torch.float32,
        device=agent.device,
    )
    policy._g_clip_actions = True  # noqa: SLF001
    policy._g_clip_mean_actions = True  # noqa: SLF001
    print("[INFO] Configured normalized AMP policy action bounds: [-1, 1]")


def finite_batch_rows(
    value,
    num_envs: int,
    device: torch.device,
    *,
    maximum_absolute_value: float | None = None,
) -> torch.Tensor:
    """Return the rows containing only finite, plausibly-sized floating values."""
    valid = torch.ones(num_envs, dtype=torch.bool, device=device)
    if value is None:
        return valid
    if isinstance(value, Mapping):
        for nested in value.values():
            valid &= finite_batch_rows(
                nested,
                num_envs,
                device,
                maximum_absolute_value=maximum_absolute_value,
            )
        return valid
    if not isinstance(value, torch.Tensor) or not (
        value.is_floating_point() or value.is_complex()
    ):
        return valid
    if value.ndim < 1 or value.shape[0] != num_envs:
        raise ValueError("rollout tensor must contain one row per environment")
    flattened = value.reshape(num_envs, -1)
    valid &= torch.all(torch.isfinite(flattened), dim=-1)
    if maximum_absolute_value is not None:
        if maximum_absolute_value <= 0.0:
            raise ValueError("maximum absolute rollout value must be positive")
        valid &= torch.all(torch.abs(flattened) <= maximum_absolute_value, dim=-1)
    return valid


def sanitize_batch_rows(value, failed: torch.Tensor | None = None):
    """Replace non-finite values and, when requested, zero complete failed rows."""
    if isinstance(value, Mapping):
        return {
            key: sanitize_batch_rows(nested, failed)
            for key, nested in value.items()
        }
    if not isinstance(value, torch.Tensor) or not (
        value.is_floating_point() or value.is_complex()
    ):
        return value
    safe = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    if failed is None:
        return safe
    if failed.dtype != torch.bool or failed.ndim != 1:
        raise ValueError("failed rollout mask must be one-dimensional and boolean")
    if value.ndim < 1 or value.shape[0] != failed.numel():
        raise ValueError("failed rollout mask must match the tensor batch")
    mask = failed.reshape((failed.numel(),) + (1,) * (value.ndim - 1))
    return torch.where(mask, torch.zeros_like(safe), safe)


def tensors_are_finite(value) -> bool:
    """Return whether every floating tensor in a nested value is finite."""
    if isinstance(value, Mapping):
        return all(tensors_are_finite(nested) for nested in value.values())
    if isinstance(value, (list, tuple)):
        return all(tensors_are_finite(nested) for nested in value)
    if isinstance(value, torch.Tensor) and (
        value.is_floating_point() or value.is_complex()
    ):
        return bool(torch.all(torch.isfinite(value)).item())
    return True

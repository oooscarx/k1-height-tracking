"""Observation terms for deployment-consistent height-control adaptation."""

from __future__ import annotations


def scaled_last_action(
    env,
    joint_indices: tuple[int, ...],
    scale: float,
):
    """Return the last action with selected residual coordinates rescaled."""
    actions = env.action_manager.action.clone()
    actions[:, list(joint_indices)] *= scale
    return actions

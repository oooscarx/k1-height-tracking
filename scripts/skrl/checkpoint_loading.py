"""Selective AMP checkpoint restoration shared by training and unit tests."""

from __future__ import annotations

from typing import Any

import torch


def clamp_gaussian_log_std_parameter(policy: Any) -> dict[str, float] | None:
    """Project a loaded Gaussian log-std parameter into the active model bounds."""
    parameter = getattr(policy, "log_std_parameter", None)
    minimum = getattr(policy, "_g_min_log_std", None)
    maximum = getattr(policy, "_g_max_log_std", None)
    if parameter is None or minimum is None or maximum is None:
        return None
    minimum = float(minimum)
    maximum = float(maximum)
    if minimum > maximum:
        raise ValueError("Gaussian log-std bounds are reversed")
    before_minimum = float(torch.amin(parameter.detach()).item())
    before_maximum = float(torch.amax(parameter.detach()).item())
    with torch.no_grad():
        parameter.clamp_(minimum, maximum)
    return {
        "before_minimum": before_minimum,
        "before_maximum": before_maximum,
        "after_minimum": float(torch.amin(parameter.detach()).item()),
        "after_maximum": float(torch.amax(parameter.detach()).item()),
    }


def restore_amp_checkpoint_components(
    agent: Any,
    data: dict[str, Any],
    *,
    reset_value: bool,
    reset_discriminator: bool,
) -> list[str]:
    """Restore shared normalization state while optionally resetting trainable heads."""
    agent.models["policy"].load_state_dict(data["policy"])
    agent._state_preprocessor.load_state_dict(data["state_preprocessor"])
    loaded = ["policy", "state_preprocessor"]

    if not reset_value:
        agent.models["value"].load_state_dict(data["value"])
        agent._value_preprocessor.load_state_dict(data["value_preprocessor"])
        loaded.append("value")

    if not reset_discriminator:
        agent.models["discriminator"].load_state_dict(data["discriminator"])
        agent._amp_observation_preprocessor.load_state_dict(
            data["amp_observation_preprocessor"]
        )
        loaded.append("discriminator")

    return loaded

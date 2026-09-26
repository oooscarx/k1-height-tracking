"""TensorBoard diagnostics for the reward mixture used by skrl AMP."""

from __future__ import annotations

import types

import torch
import torch.nn.functional as F


def amp_reward_metrics(
    task_rewards: torch.Tensor,
    discriminator_logits: torch.Tensor,
    *,
    task_reward_scale: float,
    style_reward_scale: float,
) -> dict[str, float]:
    """Return rollout metrics for the exact reward mixture optimized by skrl AMP."""
    style_rewards = F.softplus(discriminator_logits).view_as(task_rewards)
    weighted_task = task_reward_scale * task_rewards
    weighted_style = style_reward_scale * style_rewards
    mean_task_magnitude = torch.abs(weighted_task).mean()
    mean_style = weighted_style.mean()
    style_to_task = mean_style / torch.clamp(mean_task_magnitude, min=1.0e-6)
    style_fraction = mean_style / torch.clamp(
        mean_task_magnitude + mean_style,
        min=1.0e-6,
    )
    return {
        "Reward / AMP task contribution (mean)": weighted_task.mean().item(),
        "Reward / AMP style contribution (mean)": weighted_style.mean().item(),
        "Reward / AMP combined contribution (mean)": (weighted_task + weighted_style).mean().item(),
        "Reward / AMP style to task ratio": style_to_task.item(),
        "Reward / AMP style fraction": style_fraction.item(),
        "Reward / AMP discriminator logit (mean)": discriminator_logits.mean().item(),
    }


def capped_style_reward_scale(
    task_rewards: torch.Tensor,
    discriminator_logits: torch.Tensor,
    *,
    task_reward_scale: float,
    configured_style_reward_scale: float,
    maximum_style_fraction: float | None,
    minimum_style_reward_scale: float,
) -> float:
    """Limit style dominance while retaining a minimum AMP contribution."""
    if maximum_style_fraction is None:
        return configured_style_reward_scale
    if not 0.0 < maximum_style_fraction < 1.0:
        raise ValueError("maximum_style_fraction must be inside (0, 1)")
    if not 0.0 <= minimum_style_reward_scale <= configured_style_reward_scale:
        raise ValueError(
            "minimum_style_reward_scale must be between zero and the configured scale"
        )

    weighted_task_magnitude = torch.abs(task_reward_scale * task_rewards).mean()
    mean_unweighted_style = F.softplus(discriminator_logits).mean()
    style_to_task_limit = maximum_style_fraction / (1.0 - maximum_style_fraction)
    capped_scale = style_to_task_limit * weighted_task_magnitude / torch.clamp(
        mean_unweighted_style,
        min=1.0e-6,
    )
    return max(
        minimum_style_reward_scale,
        min(configured_style_reward_scale, float(capped_scale.item())),
    )


def install_amp_reward_diagnostics(
    agent,
    *,
    maximum_style_fraction: float | None = None,
    minimum_style_reward_scale: float = 0.0,
) -> None:
    """Record and optionally bound the reward mixture before each AMP update."""
    if getattr(agent, "_amp_reward_diagnostics_installed", False):
        return

    configured_style_reward_scale = float(agent.cfg.style_reward_scale)
    if maximum_style_fraction is not None and not 0.0 < maximum_style_fraction < 1.0:
        raise ValueError("maximum_style_fraction must be inside (0, 1)")
    if not 0.0 <= minimum_style_reward_scale <= configured_style_reward_scale:
        raise ValueError(
            "minimum_style_reward_scale must be between zero and the configured scale"
        )

    original_update = agent.update

    def update(self, *, timestep: int, timesteps: int) -> None:
        task_rewards = self.memory.get_tensor_by_name("rewards")
        amp_observations = self.memory.get_tensor_by_name("amp_observations")
        with torch.no_grad(), torch.autocast(
            device_type=self._device_type,
            enabled=self.cfg.mixed_precision,
        ):
            discriminator_logits, _ = self.discriminator.act(
                {
                    "observations": self._amp_observation_preprocessor(
                        amp_observations
                    )
                },
                role="discriminator",
            )
            effective_style_reward_scale = capped_style_reward_scale(
                task_rewards,
                discriminator_logits,
                task_reward_scale=self.cfg.task_reward_scale,
                configured_style_reward_scale=configured_style_reward_scale,
                maximum_style_fraction=maximum_style_fraction,
                minimum_style_reward_scale=minimum_style_reward_scale,
            )
            metrics = amp_reward_metrics(
                task_rewards,
                discriminator_logits,
                task_reward_scale=self.cfg.task_reward_scale,
                style_reward_scale=effective_style_reward_scale,
            )
        metrics["Reward / AMP effective style scale"] = effective_style_reward_scale
        for name, value in metrics.items():
            self.track_data(name, value)
        self.cfg.style_reward_scale = effective_style_reward_scale
        try:
            original_update(timestep=timestep, timesteps=timesteps)
        finally:
            self.cfg.style_reward_scale = configured_style_reward_scale

    agent.update = types.MethodType(update, agent)
    agent._amp_reward_diagnostics_installed = True

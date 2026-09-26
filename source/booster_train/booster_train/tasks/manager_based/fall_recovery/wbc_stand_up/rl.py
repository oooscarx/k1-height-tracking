# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Literal

import torch
from isaaclab.managers import TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg, RslRlVecEnvWrapper


@configclass
class DoneTermCfg(TerminationTermCfg):
    termination_type: Literal["neutral", "good", "bad"] = "neutral"
    sigma: float = 5.0

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.time_out and self.termination_type != "neutral":
            raise ValueError("timeout termination handling must remain neutral")


@configclass
class RslRlRewardNormalizationCfg:
    decay: float = 0.999
    epsilon: float = 1.0e-2
    return_scale_decay: float | None = 0.999
    outlier_threshold: float | None = 10.0


@configclass
class RslRlL2C2Cfg:
    lambda_actor: float = 1.0
    lambda_critic: float = 0.1
    max_actor_observation_delta: float | None = None
    max_critic_observation_delta: float | None = None


@configclass
class WbcRslRlPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    max_learning_rate: float = 1.0e-2
    value_loss_huber_delta: float | None = None
    l2c2_cfg: RslRlL2C2Cfg | None = None
    reward_normalization_cfg: RslRlRewardNormalizationCfg | None = None


class WbcStandUpVecEnvWrapper(RslRlVecEnvWrapper):
    """Isaac Lab wrapper with WBC good/bad termination metadata."""

    def __init__(self, env, clip_actions: float | None = None) -> None:
        self._termination_handling: dict[str, tuple[str, float]] = {}
        super().__init__(env, clip_actions=clip_actions)
        cfg = self.unwrapped.cfg.terminations
        for name in self.unwrapped.termination_manager._term_names:
            term_cfg = getattr(cfg, name, None)
            term_type = getattr(term_cfg, "termination_type", "neutral")
            if term_type in ("good", "bad"):
                self._termination_handling[name] = (
                    term_type,
                    float(getattr(term_cfg, "sigma", 5.0)),
                )

    def step(self, actions):
        obs, rewards, dones, extras = super().step(actions)
        if not self._termination_handling:
            return obs, rewards, dones, extras
        bad_sigma = torch.zeros(self.num_envs, device=self.device)
        good_sigma = torch.zeros_like(bad_sigma)
        for name, (term_type, sigma) in self._termination_handling.items():
            mask = self.unwrapped.termination_manager.get_term(name)
            target = bad_sigma if term_type == "bad" else good_sigma
            target[mask] = torch.maximum(
                target[mask],
                torch.full_like(target[mask], sigma),
            )
        if torch.any(bad_sigma > 0.0):
            extras["bad_termination_sigma"] = bad_sigma
        if torch.any(good_sigma > 0.0):
            extras["good_termination_sigma"] = good_sigma
        return obs, rewards, dones, extras

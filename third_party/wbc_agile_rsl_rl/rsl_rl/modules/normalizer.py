# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

#  Copyright (c) 2020 Preferred Networks, Inc.

from __future__ import annotations

import math

import torch
from torch import nn


class EmpiricalNormalization(nn.Module):
    """Normalize mean and variance of values based on empirical values."""

    def __init__(self, shape, eps=1e-2, until=None):
        """Initialize EmpiricalNormalization module.

        Args:
            shape (int or tuple of int): Shape of input values except batch axis.
            eps (float): Small value for stability.
            until (int or None): If this arg is specified, the link learns input values until the sum of batch sizes
            exceeds it.
        """
        super().__init__()
        self.eps = eps
        self.until = until
        self.register_buffer("_mean", torch.zeros(shape).unsqueeze(0))
        self.register_buffer("_var", torch.ones(shape).unsqueeze(0))
        self.register_buffer("_std", torch.ones(shape).unsqueeze(0))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @property
    def mean(self):
        return self._mean.squeeze(0).clone()

    @property
    def std(self):
        return self._std.squeeze(0).clone()

    def forward(self, x):
        """Normalize mean and variance of values based on empirical values.

        Args:
            x (ndarray or Variable): Input values

        Returns:
            ndarray or Variable: Normalized output values
        """

        if self.training:
            self.update(x)
        return (x - self._mean) / (self._std + self.eps)

    @torch.jit.unused
    def update(self, x):
        """Learn input values without computing the output values of them"""

        if self.until is not None and self.count >= self.until:
            return

        count_x = x.shape[0]
        self.count += count_x
        rate = count_x / self.count

        var_x = torch.var(x, dim=0, unbiased=False, keepdim=True)
        mean_x = torch.mean(x, dim=0, keepdim=True)
        delta_mean = mean_x - self._mean
        self._mean += rate * delta_mean
        self._var += rate * (var_x - self._var + delta_mean * (mean_x - self._mean))
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def inverse(self, y):
        return y * (self._std + self.eps) + self._mean


class EmpiricalDiscountedVariationNormalization(nn.Module):
    """Reward normalization from Pathak's large scale study on PPO.

    Reward normalization. Since the reward function is non-stationary, it is useful to normalize
    the scale of the rewards so that the value function can learn quickly. We did this by dividing
    the rewards by a running estimate of the standard deviation of the sum of discounted rewards.
    """

    def __init__(self, shape, eps=1e-2, gamma=0.99, until=None):
        super().__init__()

        self.emp_norm = EmpiricalNormalization(shape, eps, until)
        self.disc_avg = DiscountedAverage(gamma)

    def forward(self, rew):
        if self.training:
            # update discounected rewards
            avg = self.disc_avg.update(rew)

            # update moments from discounted rewards
            self.emp_norm.update(avg)

        if self.emp_norm._std > 0:
            return rew / self.emp_norm._std
        else:
            return rew


class DiscountedAverage:
    r"""Discounted average of rewards.

    The discounted average is defined as:

    .. math::

        \bar{R}_t = \gamma \bar{R}_{t-1} + r_t

    Args:
        gamma (float): Discount factor.
    """

    def __init__(self, gamma):
        self.avg = None
        self.gamma = gamma

    def update(self, rew: torch.Tensor) -> torch.Tensor:
        if self.avg is None:
            self.avg = rew
        else:
            self.avg = self.avg * self.gamma + rew
        return self.avg


class ReturnVarianceNormalization(nn.Module):
    """WBC-AGILE reward normalization targeting unit-variance returns."""

    def __init__(
        self,
        shape,
        eps=1e-2,
        gamma=0.99,
        decay=0.999,
        return_scale_decay: float | None = 0.999,
        outlier_threshold: float | None = 10.0,
    ):
        super().__init__()
        self.eps = eps
        self.gamma = gamma
        self.decay = decay
        self.gamma_factor = 1.0 / math.sqrt(1.0 - gamma**2)
        self.return_scale_decay = return_scale_decay
        self.outlier_threshold = outlier_threshold
        self.register_buffer("_var", torch.ones(1))
        self.register_buffer("_std", torch.ones(1))
        self.register_buffer("_mean", torch.zeros(1))
        self.register_buffer("_return_correction", torch.ones(1))

    def forward(self, rewards):
        if self.training:
            self.update(rewards)
        scale = self._std * self.gamma_factor * self._return_correction + self.eps
        normalized = rewards / scale
        if self.outlier_threshold is not None:
            normalized_mean = self._mean / scale
            normalized = torch.clamp(
                normalized,
                normalized_mean - self.outlier_threshold,
                normalized_mean + self.outlier_threshold,
            )
        return normalized

    @torch.jit.unused
    def update(self, rewards):
        if self.outlier_threshold is not None:
            lower = self._mean - self.outlier_threshold * (self._std + self.eps)
            upper = self._mean + self.outlier_threshold * (self._std + self.eps)
            rewards = torch.clamp(rewards, lower, upper)
        variance = torch.var(rewards, unbiased=False)
        mean = torch.mean(rewards)
        alpha = 1.0 - self.decay
        self._mean = self.decay * self._mean + alpha * mean
        self._var = self.decay * self._var + alpha * variance
        self._std = torch.sqrt(self._var)

    @torch.jit.unused
    def update_return_scale(self, returns: torch.Tensor):
        if self.return_scale_decay is None:
            return
        return_std = returns.std().clamp(min=1e-6)
        ideal_correction = return_std * self._return_correction
        alpha = 1.0 - self.return_scale_decay
        self._return_correction = (
            self.return_scale_decay * self._return_correction
            + alpha * ideal_correction
        )

    def load_state_dict(self, state_dict, strict=True):
        for key in ("_var", "_std", "_mean"):
            if key in state_dict:
                state_dict[key] = state_dict[key].flatten()[:1]
        if "_return_correction" not in state_dict:
            state_dict["_return_correction"] = torch.ones(1)
        return super().load_state_dict(state_dict, strict=strict)

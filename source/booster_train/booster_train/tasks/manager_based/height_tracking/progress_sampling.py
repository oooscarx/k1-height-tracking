"""Learning-progress sampling with an independent, uniform command cohort."""

from __future__ import annotations

import torch


class HeightProgressSampler:
    def __init__(self, maximum, device, bins=12, uniform_fraction=0.5,
                 update_steps=240, warmup_steps=6000, success_threshold=0.08):
        if maximum <= 0 or bins < 2 or not 0.5 <= uniform_fraction <= 1.0:
            raise ValueError("invalid height sampling support or uniform fraction")
        if update_steps < 1 or warmup_steps < 0 or success_threshold <= 0:
            raise ValueError("invalid height sampling timing or threshold")
        self.maximum = float(maximum)
        self.bins = int(bins)
        self.uniform_fraction = float(uniform_fraction)
        self.update_steps = int(update_steps)
        self.warmup_steps = int(warmup_steps)
        self.success_threshold = float(success_threshold)
        self.steps = 0
        self.centers = (torch.arange(bins, device=device) + 0.5) * maximum / bins
        for name in ("counts", "error_sum", "success_sum", "fast", "slow", "success", "seen"):
            setattr(self, name, torch.zeros(2 * bins, device=device))

    def observe(self, heights, origins, errors, valid):
        bins = (heights / self.maximum * self.bins).long().clamp(0, self.bins - 1)
        keys = bins + (heights >= origins).long() * self.bins
        finite = valid & torch.isfinite(errors) & torch.isfinite(heights) & torch.isfinite(origins)
        weight = finite.float()
        clean = torch.where(finite, errors, 0.0)
        self.counts.scatter_add_(0, keys, weight)
        self.error_sum.scatter_add_(0, keys, clean)
        self.success_sum.scatter_add_(0, keys, (clean < self.success_threshold).float() * weight)
        self.steps += 1
        if self.steps % self.update_steps == 0:
            enough = self.counts >= 128
            mean = self.error_sum / self.counts.clamp(min=1)
            rate = self.success_sum / self.counts.clamp(min=1)
            first = self.seen == 0
            self.fast.copy_(torch.where(enough, torch.where(first, mean, 0.5 * mean + 0.5 * self.fast), self.fast))
            self.slow.copy_(torch.where(enough, torch.where(first, mean, 0.1 * mean + 0.9 * self.slow), self.slow))
            self.success.copy_(torch.where(enough, torch.where(first, rate, 0.1 * rate + 0.9 * self.success), self.success))
            self.seen.add_(enough.float())
            self.counts.zero_()
            self.error_sum.zero_()
            self.success_sum.zero_()

    def probabilities(self, origins):
        # Score progress, with a small preference for partly mastered regions.
        progress = ((self.slow - self.fast) / self.success_threshold).clamp(min=0)
        score = (progress * (1 - self.success) + 0.05 * self.success * (1 - self.success))
        score = torch.where(self.seen >= 2, score, 0.0).view(2, self.bins)
        if self.steps < self.warmup_steps:
            score = torch.zeros_like(score)
        up = self.centers.unsqueeze(0) >= origins.unsqueeze(1)
        weights = torch.where(up, score[1], score[0])
        total = weights.sum(dim=-1, keepdim=True)
        adaptive = torch.where(total > 0, weights / total.clamp(min=1e-12), 1.0 / self.bins)
        return self.uniform_fraction / self.bins + (1 - self.uniform_fraction) * adaptive

    def sample(self, origins):
        if origins.numel() == 0:
            return origins.clone()
        bins = torch.multinomial(self.probabilities(origins), 1).squeeze(-1)
        return (bins + torch.rand_like(origins)) * (self.maximum / self.bins)

    def state_dict(self):
        return {"version": 1, "settings": self.settings(), "steps": self.steps,
                **{name: getattr(self, name).detach().cpu().clone() for name in
                   ("counts", "error_sum", "success_sum", "fast", "slow", "success", "seen")}}

    def settings(self):
        return (self.maximum, self.bins, self.uniform_fraction, self.update_steps,
                self.warmup_steps, self.success_threshold)

    def load_state_dict(self, state):
        if state["version"] != 1 or tuple(state["settings"]) != self.settings():
            raise ValueError("incompatible height sampler checkpoint")
        if not isinstance(state["steps"], int) or state["steps"] < 0:
            raise ValueError("invalid sampler step count")
        for name in ("counts", "error_sum", "success_sum", "fast", "slow", "success", "seen"):
            value = state[name]
            if value.shape != self.fast.shape or not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError(f"invalid sampler state: {name}")
        if (state["success"] > 1).any():
            raise ValueError("invalid sampler success probability")
        self.steps = state["steps"]
        for name in ("counts", "error_sum", "success_sum", "fast", "slow", "success", "seen"):
            getattr(self, name).copy_(state[name])


def attach_sampler_checkpoint(runner, sampler, resume_infos=None, state_key="height_progress_sampler"):
    if sampler is None:
        return
    if isinstance(resume_infos, dict) and state_key in resume_infos:
        sampler.load_state_dict(resume_infos[state_key])
        print(f"[INFO] Restored height sampler at step {sampler.steps}")
    original_save = runner.save

    def save(path, infos=None):
        payload = dict(infos or {})
        payload[state_key] = sampler.state_dict()
        return original_save(path, infos=payload)

    runner.save = save

"""Torch-only height-command governor shared by simulation and deployment.

Load this file directly in deployment; it has no Isaac dependencies. Timing is
independent of tracking success. A new request starts relative to measured height;
an existing request never chases a falling body downward.
"""

import math
import torch


class HeightCommandGovernor:
    def __init__(self, num_envs, device="cpu", upward_step=0.12, hold_s=4.0):
        if not math.isfinite(upward_step) or upward_step <= 0:
            raise ValueError("upward_step must be positive and finite")
        if not math.isfinite(hold_s) or hold_s < 4.0:
            raise ValueError("hold_s must allow 2s settling and >=1s observation")
        self.upward_step = upward_step
        self.hold_s = hold_s
        self.final = torch.zeros(num_envs, device=device)
        self.first = torch.zeros_like(self.final)
        self.elapsed = torch.zeros_like(self.final)
        self.advances = torch.zeros_like(self.final)
        self.managed = torch.zeros(num_envs, device=device, dtype=torch.bool)

    def request(self, ids, goals, measured, managed, durations):
        if not torch.isfinite(goals).all() or not torch.isfinite(measured).all():
            raise ValueError("nonfinite requested or measured height")
        if not torch.isfinite(durations).all() or (durations <= 0).any():
            raise ValueError("duration must be positive and finite")
        active = managed & (goals >= 0)
        first = torch.where(active, torch.minimum(goals, measured.clamp(min=0) + self.upward_step), goals)
        advances = torch.ceil(((goals - first) / self.upward_step - 1e-5).clamp(min=0))
        self.final[ids] = goals
        self.first[ids] = first
        self.elapsed[ids] = 0
        self.advances[ids] = advances
        self.managed[ids] = active
        return torch.where(active, advances * self.hold_s + durations.clamp(min=self.hold_s), durations)

    @property
    def stage(self):
        return torch.minimum(torch.floor((self.elapsed + 1e-5) / self.hold_s), self.advances)

    @property
    def command(self):
        return torch.minimum(self.first + self.stage * self.upward_step, self.final)

    @property
    def command_age(self):
        return (self.elapsed - self.stage * self.hold_s).clamp(min=0)

    @property
    def at_final(self):
        return self.stage >= self.advances

    def step(self, dt):
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError("dt must be positive and finite")
        self.elapsed += dt
        return self.command

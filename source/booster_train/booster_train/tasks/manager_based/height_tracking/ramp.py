"""Clock-driven height transitions; progress cannot be withheld by the policy."""

import torch


class HeightRamp:
    def __init__(self, num_envs, device, step_duration=0.5):
        if step_duration <= 0:
            raise ValueError("ramp step duration must be positive")
        self.step_duration = step_duration
        self.steps = 0
        self.active = torch.zeros(num_envs, device=device, dtype=torch.bool)
        self.origin = torch.zeros(num_envs, device=device)
        self.final = self.origin.clone()
        self.elapsed = self.origin.clone()
        self.speed = self.origin.clone()
        self.at_final = self.active.clone()
        # Process-lifetime totals; in-flight commands reset with the physics.
        self.totals = torch.zeros(9, device=device)

    def sample(self, ids, final, active, measured, delta):
        self.active[ids] = active
        self.origin[ids] = measured.clamp(min=0)
        self.final[ids] = final
        self.elapsed[ids] = 0
        self.speed[ids] = delta / self.step_duration
        self.at_final[ids] = False

    def step(self, dt):
        self.steps += 1
        self.elapsed += self.active * dt
        command = torch.minimum(self.origin + self.speed * self.elapsed, self.final)
        self.at_final = self.active & (command >= self.final - 1e-6)
        return command

    def finish(self, ids, counts, error_sum, success_sum, failures, dt, timeout=False):
        active = self.active[ids]
        failed = active & failures
        observed = active & (counts * dt >= 1.0)
        success = observed & ~failed & (success_sum / counts.clamp(min=1) >= .9)
        timed_out = active & ~failed if timeout else torch.zeros_like(active)
        self.totals += torch.stack((active.sum(), (active & self.at_final[ids]).sum(),
            observed.sum(), success.sum(), (active & ~observed & ~failed).sum(),
            failed.sum(), timed_out.sum(), torch.where(active, error_sum, 0).sum(),
            torch.where(active, counts, 0).sum()))
        self.active[ids] = False
        return observed, failed

    def diagnostics(self):
        names = ("ended", "target_arrived", "observed", "success", "censored",
                 "terminated", "timeout", "error_sum", "valid_count")
        return {f"ramp_run_{name}": value for name, value in zip(names, self.totals.tolist())}

    def state_dict(self):
        return dict(version=1, steps=self.steps, step_duration=self.step_duration)

    def load_state_dict(self, state):
        if (state["version"] != 1 or state["step_duration"] != self.step_duration
                or not isinstance(state["steps"], int) or state["steps"] < 0):
            raise ValueError("incompatible height ramp state")
        self.steps = state["steps"]

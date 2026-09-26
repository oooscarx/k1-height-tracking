"""Advance intermediate targets without discarding the sampled final height."""

import torch


class HeightStaircase:
    def __init__(self, num_envs, device, tolerance=0.04, stable_time=0.5, settle_time=2.0, threshold=.08):
        if tolerance <= 0 or stable_time <= 0:
            raise ValueError("staircase tolerances must be positive")
        self.tolerance = tolerance
        self.stable_time = stable_time
        self.settle_time = settle_time
        self.threshold = threshold
        self.final = torch.zeros(num_envs, device=device)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.stable = torch.zeros(num_envs, device=device)
        self.reached = torch.zeros(num_envs, dtype=torch.bool, device=device)
        # Diagnostic counters describe this process lifetime, not resumed history.
        self.totals = torch.zeros(3, device=device)
        self.steps = 0
        self.just_completed = self.active.clone()
        self.step_closed = self.active.clone()
        self.completed_error = self.stable.clone()
        self.stable_error = self.stable.clone()
        self.step_age = self.stable.clone()
        self.step_error = self.stable.clone()
        self.goal_age = self.stable.clone()
        self.start_height = self.stable.clone()
        self.advances = self.stable.clone()
        self.final_time = self.stable.clone()
        self.final_valid_time = self.stable.clone()
        self.final_good_time = self.stable.clone()
        self.final_error = self.stable.clone()
        self.final_signed_error = self.stable.clone()
        self.last_measured = self.stable.clone()
        self.end_totals = torch.zeros(11, device=device)
        # Rows: final target <.3, .3-.5, .5-.6, >=.6m. Columns are summed exposure/outcomes.
        self.height_bins = torch.zeros((4, 7), device=device)
        self.delta_bins = torch.zeros((4, 3), device=device)
        self.step_outcomes = torch.zeros(3, device=device)

    def sample(self, env_ids, final, frontier, measured=None):
        self.final[env_ids] = final
        self.active[env_ids] = frontier
        self.stable[env_ids] = 0
        self.reached[env_ids] = False
        self.step_closed[env_ids] = False
        self.just_completed[env_ids] = False
        for value in (self.stable_error, self.step_age, self.step_error, self.goal_age,
                      self.advances, self.final_time, self.final_valid_time, self.final_good_time,
                      self.final_error, self.final_signed_error):
            value[env_ids] = 0
        self.start_height[env_ids] = final if measured is None else measured
        self.totals[0] += frontier.sum()

    def advance(self, targets, measured, step_size, dt):
        self.steps += 1
        self.last_measured.copy_(measured)
        self.goal_age += self.active * dt
        self.step_age += self.active * dt
        error = (measured - targets).abs()
        self.step_error += torch.where(self.active & torch.isfinite(error), error * dt, 0)
        at_final = self.active & (targets >= self.final - 1e-6)
        valid = at_final & (self.step_age > self.settle_time) & torch.isfinite(measured)
        self.final_time += at_final * dt
        self.final_valid_time += valid * dt
        self.final_good_time += (valid & (error < self.threshold)) * dt
        self.final_error += torch.where(valid, error * dt, 0)
        self.final_signed_error += torch.where(valid, (measured - self.final) * dt, 0)
        stable = self.active & torch.isfinite(measured) & ((measured - targets).abs() < self.tolerance)
        self.stable = torch.where(stable, self.stable + dt, 0.0)
        self.stable_error = torch.where(stable, self.stable_error + error * dt, 0.0)
        ready = self.stable + 1e-6 >= self.stable_time
        self.just_completed = ready & ~self.step_closed
        self.completed_error = self.stable_error / self.stable.clamp(min=dt)
        self.step_closed |= self.just_completed
        self.step_outcomes[0] += self.just_completed.sum()
        advance = ready & (targets < self.final - 1e-6)
        reached = ready & ~advance & ~self.reached
        self.reached |= reached
        self.totals[1] += advance.sum()
        self.totals[2] += reached.sum()
        self.advances += advance
        next_targets = torch.where(advance, torch.minimum(targets + step_size, self.final), targets)
        self.stable[advance] = 0
        self.stable_error[advance] = 0
        self.step_age[advance] = 0
        self.step_error[advance] = 0
        self.step_closed[advance] = False
        return next_targets, advance

    def finish(self, env_ids, failures=None, timeout=False):
        active = self.active[env_ids]
        failure = torch.zeros_like(active) if failures is None else failures & active
        incomplete = active & ~self.step_closed[env_ids]
        # An unfinished step needs a full 2s transition + 1s observation budget to be failed.
        failed_step = incomplete & ((self.step_age[env_ids] >= self.settle_time + 1.0) | failure)
        censored_step = incomplete & ~failed_step
        error = self.step_error[env_ids] / self.step_age[env_ids].clamp(min=.02)
        self.step_outcomes[1] += failed_step.sum()
        self.step_outcomes[2] += censored_step.sum()
        final_observed = self.final_valid_time[env_ids] >= 1.0
        final_success = active & final_observed & ~failure & (
            self.final_good_time[env_ids] / self.final_valid_time[env_ids].clamp(min=.02) >= .9)
        reached = self.reached[env_ids] & active
        ended_timeout = active & ~failure if timeout else torch.zeros_like(active)
        resampled = active & ~failure & ~ended_timeout
        no_advance = active & (self.advances[env_ids] == 0)
        phase_mid = active & ~no_advance & (self.final_time[env_ids] == 0)
        phase_final_unsettled = active & ~no_advance & ~phase_mid & ~final_success
        self.end_totals += torch.stack((active.sum(), resampled.sum(), ended_timeout.sum(),
            failure.sum(), no_advance.sum(), phase_mid.sum(), phase_final_unsettled.sum(),
            final_success.sum(), reached.sum(), (active & final_observed).sum(),
            torch.where(active, self.goal_age[env_ids], 0).sum()))
        values = torch.stack((active.float(), reached.float(), final_success.float(),
            self.final_valid_time[env_ids], self.final_error[env_ids],
            self.final_signed_error[env_ids], self.final_good_time[env_ids]), dim=1)
        values = torch.where(active[:, None], values, 0)
        height_bucket = torch.bucketize(self.final[env_ids], self.final.new_tensor([.3, .5, .6]), right=True)
        self.height_bins.index_add_(0, height_bucket, values)
        delta_bucket = torch.bucketize(self.final[env_ids] - self.start_height[env_ids],
                                      self.final.new_tensor([.12, .24, .4]), right=True)
        self.delta_bins.index_add_(0, delta_bucket, values[:, :3])
        self.active[env_ids] = False
        return failed_step, error

    def diagnostics(self):
        result = {}
        groups = (("step", ("success", "failed", "censored"), self.step_outcomes),
                  ("end", ("total", "resample", "timeout", "terminated", "no_advance",
                           "midway", "final_unsettled", "final_success", "reached", "observed", "age_sum"), self.end_totals))
        for group, names, tensor in groups:
            for name, value in zip(names, tensor.tolist()):
                result[f"stair_run_{group}_{name}"] = value
        for group, tensor, names in (("height", self.height_bins, ("count", "reached", "success", "time", "abs_error", "signed_error", "good_time")),
                                     ("delta", self.delta_bins, ("count", "reached", "success"))):
            for i, row in enumerate(tensor.tolist()):
                for name, value in zip(names, row):
                    result[f"stair_run_{group}_{i}_{name}"] = value
        return result

    def state_dict(self):
        return dict(version=2, steps=self.steps, tolerance=self.tolerance, stable_time=self.stable_time,
                    settle_time=self.settle_time, threshold=self.threshold)

    def load_state_dict(self, state):
        if (state["version"] != 2 or state["tolerance"] != self.tolerance or state["stable_time"] != self.stable_time
                or state["settle_time"] != self.settle_time or state["threshold"] != self.threshold
                or not isinstance(state["steps"], int) or state["steps"] < 0):
            raise ValueError("incompatible staircase accounting state")
        self.steps = int(state["steps"])


def classify_failures(counts, error_sums, success_sums, eligible, failures, threshold=0.08):
    """Disjoint completed-command categories; high mean does not prove constant bias."""
    selected = eligible | failures
    mean_error = error_sums / counts.clamp(min=1)
    success = (success_sums / counts.clamp(min=1) >= 0.9) & ~failures
    persistent = selected & ~success & ~failures & (mean_error >= threshold)
    intermittent = selected & ~success & ~failures & ~persistent
    return torch.stack((selected.sum(), persistent.sum(), intermittent.sum(), failures.sum()))

"""Increase upward transition difficulty using completed frontier commands."""

import math
import torch


class HeightDeltaCurriculum:
    def __init__(self, device, initial=0.12, increment=0.04, maximum=0.72,
                 window_steps=6000, minimum_segments=1024, success_rate=0.8):
        if not 0 < initial <= maximum or increment <= 0 or window_steps < 1 or minimum_segments < 1:
            raise ValueError("invalid height delta curriculum settings")
        if not 0 < success_rate <= 1:
            raise ValueError("invalid success threshold")
        self.settings = (initial, increment, maximum, window_steps, minimum_segments, success_rate)
        self.limit = float(initial)
        self.steps = 0
        self.stage = 0
        self.totals = torch.zeros(3, device=device)
        self.last_rate = self.last_error = 0.0
        self.last_count = 0

    def constrain(self, targets, measured, reference):
        finite = torch.isfinite(measured)
        frontier = (~reference) & finite & (targets >= 0) & (targets - measured > self.limit)
        if self.limit >= self.settings[2]:
            frontier = torch.zeros_like(frontier)
        upper = (measured + self.limit).clamp(min=0)
        return torch.where(frontier, torch.minimum(targets, upper), targets), frontier

    def finish(self, counts, error_sums, success_sums, eligible, failures):
        selected = eligible | failures
        errors = error_sums / counts.clamp(min=1)
        errors = torch.where(failures, errors.clamp(min=0.16), errors)
        good = (success_sums / counts.clamp(min=1) >= 0.9) & ~failures
        self.totals += torch.stack((selected.sum(), (good & selected).sum(),
                                   torch.where(selected, errors, 0).sum()))

    def step(self):
        self.steps += 1
        if self.steps % self.settings[3]:
            return
        count, successes, error_sum = self.totals.tolist()
        self.last_count = int(count)
        self.last_rate = successes / max(count, 1)
        self.last_error = error_sum / max(count, 1)
        if (count >= self.settings[4] and self.last_rate >= self.settings[5]
                and self.last_error <= 0.08 and self.limit < self.settings[2]):
            self.limit = min(self.settings[2], round(self.limit + self.settings[1], 8))
            self.stage += 1
        self.totals.zero_()

    def state_dict(self):
        return dict(version=1, settings=self.settings, limit=self.limit, steps=self.steps,
                    stage=self.stage, totals=self.totals.detach().cpu().clone(),
                    last_rate=self.last_rate, last_error=self.last_error, last_count=self.last_count)

    def load_state_dict(self, state):
        if state["version"] != 1 or tuple(state["settings"]) != self.settings:
            raise ValueError("incompatible delta curriculum state")
        if not isinstance(state["steps"], int) or state["steps"] < 0:
            raise ValueError("invalid delta curriculum steps")
        if not isinstance(state["stage"], int) or state["stage"] < 0:
            raise ValueError("invalid delta curriculum stage")
        expected = min(self.settings[2], self.settings[0] + state["stage"] * self.settings[1])
        if not math.isfinite(state["limit"]) or abs(state["limit"] - expected) > 1e-6:
            raise ValueError("invalid delta curriculum limit")
        totals = state["totals"]
        if totals.shape != (3,) or not torch.isfinite(totals).all() or (totals < 0).any() or totals[1] > totals[0]:
            raise ValueError("invalid delta curriculum counters")
        if not 0 <= state["last_rate"] <= 1 or not math.isfinite(state["last_error"]) or state["last_error"] < 0:
            raise ValueError("invalid delta curriculum metrics")
        if not isinstance(state["last_count"], int) or state["last_count"] < 0:
            raise ValueError("invalid delta curriculum sample count")
        for name in ("limit", "steps", "stage", "last_rate", "last_error", "last_count"):
            setattr(self, name, state[name])
        self.totals.copy_(totals)

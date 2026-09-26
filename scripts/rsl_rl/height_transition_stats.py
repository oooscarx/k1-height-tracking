"""Diagnostic-only command segment collection; never imported by training."""

import json
import math
from collections import deque
from pathlib import Path


def contact_label(forces, foot_ids, threshold=5.0):
    active = [sum(v * v for v in force) ** 0.5 > threshold for force in forces]
    feet = sum(active[i] for i in foot_ids)
    other = any(value for i, value in enumerate(active) if i not in foot_ids)
    if other:
        return "nonfoot_contact"
    return ("airborne", "one_foot", "two_feet")[feet]


def summarize_segment(record, dt, settle_time, threshold):
    samples = record.pop("samples")
    settled = [(age, h) for age, h in samples if age > settle_time + 1e-6]
    errors = [abs(h - record["target_height"]) for _, h in settled]
    signed = [h - record["target_height"] for _, h in settled]
    streak = longest = 0
    for error in errors:
        streak = streak + 1 if error < threshold else 0
        longest = max(longest, streak)
    record.update(
        observed_s=len(samples) * dt,
        settled_s=len(errors) * dt,
        mean_abs_error=sum(errors) / len(errors) if errors else None,
        mean_signed_error=sum(signed) / len(signed) if signed else None,
        within_threshold_fraction=sum(e < threshold for e in errors) / len(errors) if errors else None,
        longest_success_s=longest * dt,
        sustained_success=longest * dt >= 1.0 - 1e-6,
        final_height=samples[-1][1] if samples else None,
        # Keep a compact time series so grouping and definitions can be audited.
        trajectory=[[round(a, 4), round(h, 6)] for a, h in samples[::5]],
    )
    return record


class TransitionCollector:
    def __init__(self, env, output):
        self.env = env
        self.command = env.command_manager.get_term("height")
        self.sensor = env.scene.sensors["contact_forces"]
        self.foot_ids, foot_names = self.sensor.find_bodies(["left_foot_link", "right_foot_link"])
        assert len(foot_names) == 2, foot_names
        self.dt = env.step_dt
        self.tick = 0
        self.active = [None] * env.num_envs
        self.history = [deque(maxlen=round(0.2 / self.dt)) for _ in self.active]
        self.in_reset = False
        self.count = 0
        self.output = Path(output)
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.stream = self.output.open("x")
        original_resample = self.command._resample_command
        original_reset = self.command.reset
        original_reward = env.reward_manager.compute

        def resample(ids):
            ids_list = list(map(int, ids))
            old_target = self.command.target_height.detach().cpu().tolist()
            for i in ids_list:
                self.finish(i, "resample")
            original_resample(ids)
            heights = self.command.measured_height.detach().cpu().tolist()
            targets = self.command._target_height.detach().cpu().tolist()
            durations = self.command.time_left.detach().cpu().tolist()
            labels = self.contacts()
            speeds = env.scene["robot"].data.root_lin_vel_w.norm(dim=-1).detach().cpu().tolist()
            for i in ids_list:
                hist = self.history[i]
                self.active[i] = dict(
                    env_id=i, start_tick=self.tick, start_kind="reset" if self.in_reset else "natural",
                    start_height=heights[i], target_height=targets[i], delta_height=targets[i] - heights[i],
                    previous_target=old_target[i], planned_hold_s=durations[i],
                    start_contact="unknown_reset" if self.in_reset else labels[i],
                    stable_two_feet=(not self.in_reset and len(hist) == hist.maxlen
                                     and all(x == "two_feet" for x in hist)),
                    start_speed=speeds[i], samples=[],
                )

        def reset(env_ids=None):
            ids = range(env.num_envs) if env_ids is None else list(map(int, env_ids))
            for i in ids:
                timeouts = getattr(env, "reset_time_outs", None)
                reason = "timeout" if timeouts is not None and bool(timeouts[i]) else "termination_or_reset"
                self.finish(i, reason)
                self.history[i].clear()
            self.in_reset = True
            try:
                return original_reset(env_ids)
            finally:
                self.in_reset = False

        def reward(*args, **kwargs):
            self.tick += 1
            heights = self.command.measured_height.detach().cpu().tolist()
            labels = self.contacts()
            for i, record in enumerate(self.active):
                self.history[i].append(labels[i])
                if record is not None:
                    if not math.isfinite(heights[i]):
                        raise RuntimeError(f"nonfinite diagnostic height in env {i}")
                    record["samples"].append(((self.tick - record["start_tick"]) * self.dt, heights[i]))
            return original_reward(*args, **kwargs)

        self.command._resample_command = resample
        self.command.reset = reset
        env.reward_manager.compute = reward

    def contacts(self):
        return [contact_label(f, self.foot_ids) for f in self.sensor.data.net_forces_w.detach().cpu().tolist()]

    def finish(self, i, reason):
        record = self.active[i]
        if record is None:
            return
        record["end_reason"] = reason
        record = summarize_segment(record, self.dt, self.command.cfg.settle_time_s,
                                   self.command.cfg.success_error_threshold)
        self.stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.active[i] = None
        self.count += 1

    def close(self):
        for i in range(len(self.active)):
            self.finish(i, "recording_end")
        self.stream.close()

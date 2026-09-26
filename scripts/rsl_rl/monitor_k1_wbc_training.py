#!/usr/bin/env python3
"""Record WBC stand-up training health without controlling the trainer."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import time
from pathlib import Path

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")
ITERATION = re.compile(r"Learning iteration (\d+)/(\d+)")
CHECKPOINT = re.compile(r"model_(\d+)\.pt$")

METRICS = {
    "steps_per_second": r"Computation:\s+([\d.]+) steps/s",
    "policy_noise_std": r"Mean action noise std:\s+([-+\deE.]+)",
    "value_loss": r"Mean value_function loss:\s+([-+\deE.]+)",
    "entropy_loss": r"Mean entropy loss:\s+([-+\deE.]+)",
    "symmetry_loss": r"Mean symmetry loss:\s+([-+\deE.]+)",
    "l2c2_actor_loss": r"Mean l2c2_actor loss:\s+([-+\deE.]+)",
    "l2c2_critic_loss": r"Mean l2c2_critic loss:\s+([-+\deE.]+)",
    "mean_reward": r"Mean reward:\s+([-+\deE.]+)",
    "mean_episode_length": r"Mean episode length:\s+([-+\deE.]+)",
    "lift_force_scale": r"Curriculum/remove_lift:\s+([-+\deE.]+)",
    "terrain_level": r"Curriculum/terrain_levels:\s+([-+\deE.]+)",
    "timeout_termination_count": r"Episode_Termination/time_out:\s+([-+\deE.]+)",
    "invalid_state_termination_count": r"Episode_Termination/invalid_state:\s+([-+\deE.]+)",
    "standing_termination_count": r"Episode_Termination/standing:\s+([-+\deE.]+)",
}


def read_tail(path: Path, maximum_bytes: int = 512 * 1024) -> str:
    with path.open("rb") as source:
        source.seek(0, os.SEEK_END)
        size = source.tell()
        source.seek(max(0, size - maximum_bytes))
        return source.read().decode("utf-8", errors="ignore")


def parse_complete_iterations(log_tail: str) -> list[dict[str, float | int]]:
    text = ANSI_ESCAPE.sub("", log_tail)
    matches = list(ITERATION.finditer(text))
    results = []
    for index in range(len(matches)):
        start = matches[index].start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[start:end]
        if "Iteration time:" not in block:
            continue
        result: dict[str, float | int] = {
            "iteration": int(matches[index].group(1)),
            "maximum_iteration": int(matches[index].group(2)),
        }
        for name, pattern in METRICS.items():
            match = re.search(pattern, block)
            if match:
                result[name] = float(match.group(1))
        termination_keys = (
            "timeout_termination_count",
            "invalid_state_termination_count",
            "standing_termination_count",
        )
        if all(key in result for key in termination_keys):
            termination_count = sum(float(result[key]) for key in termination_keys)
            result["termination_count"] = termination_count
            if termination_count > 0.0:
                result["standing_success_rate"] = (
                    float(result["standing_termination_count"]) / termination_count
                )
                result["timeout_rate"] = (
                    float(result["timeout_termination_count"]) / termination_count
                )
                result["invalid_state_rate"] = (
                    float(result["invalid_state_termination_count"]) / termination_count
                )
        results.append(result)
    return results


def parse_latest_iteration(log_tail: str) -> dict[str, float | int] | None:
    results = parse_complete_iterations(log_tail)
    return results[-1] if results else None


def latest_recorded_iteration(path: Path) -> int | None:
    if not path.exists():
        return None
    for line in reversed(read_tail(path, maximum_bytes=64 * 1024).splitlines()):
        try:
            iteration = json.loads(line).get("iteration")
        except (AttributeError, json.JSONDecodeError):
            continue
        if isinstance(iteration, int):
            return iteration
    return None


def latest_checkpoint(run_dir: Path) -> Path | None:
    candidates = []
    for path in run_dir.glob("model_*.pt"):
        match = CHECKPOINT.fullmatch(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(0, None))[1]


def expected_lift_force_scale(
    iteration: int,
    rollout_steps: int = 24,
    start_step: int = 5_000,
    num_steps: int = 200_000,
    linear: bool = False,
) -> float:
    """Return the legacy remove_harness scale expected at an iteration."""
    common_step = iteration * rollout_steps
    if common_step <= start_step:
        return 1.0
    if common_step >= start_step + num_steps:
        return 0.0
    progress = (common_step - start_step) / num_steps
    if linear:
        return 1.0 - progress
    return math.exp(progress * math.log(0.01))


def nonfinite_tensor_paths(value: object, path: str) -> list[str]:
    import torch

    if torch.is_tensor(value):
        return [] if bool(torch.isfinite(value).all()) else [path]
    if isinstance(value, dict):
        return [
            tensor_path
            for name, child in value.items()
            for tensor_path in nonfinite_tensor_paths(child, f"{path}.{name}")
        ]
    if isinstance(value, (list, tuple)):
        return [
            tensor_path
            for index, child in enumerate(value)
            for tensor_path in nonfinite_tensor_paths(child, f"{path}[{index}]")
        ]
    return []


def inspect_checkpoint(path: Path) -> dict[str, object]:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    nonfinite: list[str] = []
    for section_name in (
        "model_state_dict",
        "optimizer_state_dict",
        "reward_norm_state_dict",
    ):
        section = checkpoint.get(section_name)
        if not isinstance(section, dict):
            nonfinite.append(f"missing:{section_name}")
            continue
        nonfinite.extend(nonfinite_tensor_paths(section, section_name))

    normalizer = checkpoint.get("reward_norm_state_dict", {})
    policy_std = checkpoint.get("model_state_dict", {}).get("std")
    optimizer = checkpoint.get("optimizer_state_dict", {})
    optimizer_states = (
        optimizer.get("state", {}) if isinstance(optimizer, dict) else {}
    )
    optimizer_steps = []
    if isinstance(optimizer_states, dict):
        for state in optimizer_states.values():
            if not isinstance(state, dict) or "step" not in state:
                continue
            step = state["step"]
            if torch.is_tensor(step):
                step = step.item()
            optimizer_steps.append(int(step))
    optimizer_learning_rates = []
    if isinstance(optimizer, dict):
        for param_group in optimizer.get("param_groups", []):
            if isinstance(param_group, dict) and "lr" in param_group:
                optimizer_learning_rates.append(float(param_group["lr"]))

    summary = {
        "checkpoint": str(path),
        "checkpoint_iteration": int(checkpoint.get("iter", -1)),
        "checkpoint_nonfinite": nonfinite,
        "reward_normalizer_mean": float(normalizer["_mean"].item()),
        "reward_normalizer_std": float(normalizer["_std"].item()),
        "reward_normalizer_return_correction": float(normalizer["_return_correction"].item()),
        "policy_std_minimum": float(policy_std.min().item()),
        "policy_std_mean": float(policy_std.mean().item()),
        "policy_std_maximum": float(policy_std.max().item()),
        "optimizer_state_count": len(optimizer_states),
    }
    if optimizer_steps:
        summary["optimizer_step_minimum"] = min(optimizer_steps)
        summary["optimizer_step_maximum"] = max(optimizer_steps)
    if optimizer_learning_rates:
        summary["optimizer_learning_rate_minimum"] = min(optimizer_learning_rates)
        summary["optimizer_learning_rate_maximum"] = max(optimizer_learning_rates)
    return summary


def health_alerts(record: dict[str, object]) -> list[str]:
    alerts = []
    numeric_limits = {
        "value_loss": 50.0,
        "symmetry_loss": 5.0,
        "l2c2_critic_loss": 0.5,
        "policy_noise_std": 5.0,
        "reward_normalizer_std": 5.0,
        "policy_std_maximum": 5.0,
        "optimizer_learning_rate_maximum": 0.01,
        "lift_force_schedule_error": 0.002,
    }
    for name, limit in numeric_limits.items():
        value = record.get(name)
        if isinstance(value, (float, int)) and (not math.isfinite(value) or abs(value) > limit):
            alerts.append(f"{name}={value}")
    lower_limits = {"mean_reward": -500.0}
    upper_limits = {"invalid_state_rate": 0.01}
    for name, limit in lower_limits.items():
        value = record.get(name)
        if isinstance(value, (float, int)) and (
            not math.isfinite(value) or value < limit
        ):
            alerts.append(f"{name}={value}")
    for name, limit in upper_limits.items():
        value = record.get(name)
        if isinstance(value, (float, int)) and (
            not math.isfinite(value) or value > limit
        ):
            alerts.append(f"{name}={value}")
    for name in (
        "policy_noise_std",
        "reward_normalizer_std",
        "policy_std_minimum",
        "optimizer_learning_rate_minimum",
    ):
        value = record.get(name)
        if isinstance(value, (float, int)) and value <= 0.0:
            alerts.append(f"{name}={value}")
    optimizer_step_minimum = record.get("optimizer_step_minimum")
    optimizer_step_maximum = record.get("optimizer_step_maximum")
    if (
        isinstance(optimizer_step_minimum, int)
        and isinstance(optimizer_step_maximum, int)
        and optimizer_step_minimum != optimizer_step_maximum
    ):
        alerts.append(
            "optimizer_step_range="
            f"{optimizer_step_minimum}..{optimizer_step_maximum}"
        )
    optimizer_state_count = record.get("optimizer_state_count")
    if isinstance(optimizer_state_count, int) and optimizer_state_count <= 0:
        alerts.append(f"optimizer_state_count={optimizer_state_count}")
    nonfinite = record.get("checkpoint_nonfinite")
    if nonfinite:
        alerts.append(f"checkpoint_nonfinite={nonfinite}")
    checkpoint_error = record.get("checkpoint_error")
    if checkpoint_error:
        alerts.append(f"checkpoint_error={checkpoint_error}")
    mean_reward = record.get("mean_reward")
    mean_episode_length = record.get("mean_episode_length")
    standing_success_rate = record.get("standing_success_rate")
    if (
        isinstance(mean_reward, (float, int))
        and isinstance(mean_episode_length, (float, int))
        and isinstance(standing_success_rate, (float, int))
        and mean_reward > 150.0
        and mean_episode_length > 750.0
        and standing_success_rate < 0.3
    ):
        alerts.append("reward_proxy_mismatch")
    return alerts


def process_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--console-log", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--training-pid", required=True, type=int)
    parser.add_argument("--poll-seconds", default=60.0, type=float)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--rollout-steps", default=24, type=int)
    parser.add_argument("--lift-start-step", default=5_000, type=int)
    parser.add_argument("--lift-num-steps", default=200_000, type=int)
    parser.add_argument("--lift-linear", action="store_true")
    args = parser.parse_args()

    if args.rollout_steps <= 0:
        parser.error("--rollout-steps must be positive")
    if args.lift_num_steps <= 0:
        parser.error("--lift-num-steps must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path: Path | None = None
    checkpoint_summary: dict[str, object] = {}
    last_recorded_iteration = latest_recorded_iteration(args.output)
    while process_alive(args.training_pid):
        metrics_batch: list[dict[str, float | int]] = []
        if args.console_log.exists():
            metrics_batch = parse_complete_iterations(read_tail(args.console_log))
            if last_recorded_iteration is not None:
                metrics_batch = [
                    metrics
                    for metrics in metrics_batch
                    if int(metrics["iteration"]) > last_recorded_iteration
                ]

        candidate = latest_checkpoint(args.run_dir)
        if candidate is not None and candidate != checkpoint_path:
            if time.time() - candidate.stat().st_mtime >= 10.0:
                try:
                    checkpoint_summary = inspect_checkpoint(candidate)
                except Exception as error:  # keep monitoring after a damaged checkpoint
                    checkpoint_summary = {
                        "checkpoint": str(candidate),
                        "checkpoint_error": f"{type(error).__name__}: {error}",
                    }
                else:
                    checkpoint_path = candidate
        if not metrics_batch:
            time.sleep(args.poll_seconds)
            continue

        for metrics in metrics_batch:
            record: dict[str, object] = {"timestamp": time.time(), **metrics}
            actual_lift = metrics.get("lift_force_scale")
            if isinstance(actual_lift, (float, int)):
                expected_lift = expected_lift_force_scale(
                    int(metrics["iteration"]),
                    rollout_steps=args.rollout_steps,
                    start_step=args.lift_start_step,
                    num_steps=args.lift_num_steps,
                    linear=args.lift_linear,
                )
                record["lift_force_schedule_expected"] = expected_lift
                record["lift_force_schedule_error"] = abs(
                    float(actual_lift) - expected_lift
                )
            checkpoint_iteration = checkpoint_summary.get("checkpoint_iteration")
            if not isinstance(checkpoint_iteration, int) or checkpoint_iteration <= int(
                metrics["iteration"]
            ):
                record.update(checkpoint_summary)
            record["alerts"] = health_alerts(record)
            line = json.dumps(record, sort_keys=True)
            with args.output.open("a", encoding="utf-8") as output:
                output.write(line + "\n")
            print(line, flush=True)
            last_recorded_iteration = int(metrics["iteration"])
        time.sleep(args.poll_seconds)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

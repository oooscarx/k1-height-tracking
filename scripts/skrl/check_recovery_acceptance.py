#!/usr/bin/env python3

"""Check a deterministic recovery evaluation against deployment gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_MODES = ("faceup", "facedown", "side_left", "side_right", "random")


def evaluate_acceptance(
    results: dict,
    required_modes: list[str] | tuple[str, ...],
    *,
    minimum_success_rate: float,
    maximum_joint_limit_rate: float,
    maximum_parallel_ankle_rate: float,
    maximum_nonfinite_action_rate: float,
    maximum_parallel_state_violation_fraction: float,
    maximum_p90_recovery_time_s: float,
    minimum_episodes_per_mode: int = 1,
    minimum_amp_style_reward_ratio: float | None = None,
    minimum_weighted_amp_style_fraction: float | None = None,
) -> dict:
    failures: list[str] = []
    modes: dict[str, dict] = {}
    for mode in required_modes:
        if mode not in results:
            failures.append(f"{mode}: missing evaluation result")
            continue
        result = results[mode]
        episodes = int(result.get("episodes", 0))
        if episodes < 1:
            failures.append(f"{mode}: no completed episodes")
            continue
        if episodes < minimum_episodes_per_mode:
            failures.append(
                f"{mode}: completed episodes {episodes} "
                f"< required {minimum_episodes_per_mode}"
            )
        success_rate = float(result.get("success_rate", 0.0))
        joint_limit_rate = int(result.get("joint_limit_terminations", 0)) / episodes
        parallel_ankle_rate = int(result.get("parallel_ankle_terminations", 0)) / episodes
        nonfinite_action_rate = int(result.get("nonfinite_action_terminations", 0)) / episodes
        parallel_violation = float(result.get("parallel_state_violation_fraction", 1.0))
        recovery_time = result.get("successful_recovery_time_s")
        p90_recovery_time = (
            None
            if recovery_time is None
            else float(recovery_time.get("p90", float("inf")))
        )
        style_reward = result.get("amp_style_reward")
        reference_style_reward = result.get("amp_reference", {}).get(
            "amp_style_reward"
        )
        mean_style_reward = (
            None if style_reward is None else float(style_reward.get("mean", 0.0))
        )
        mean_reference_style_reward = (
            None
            if reference_style_reward is None
            else float(reference_style_reward.get("mean", 0.0))
        )
        style_reward_ratio = (
            None
            if mean_style_reward is None
            or mean_reference_style_reward is None
            or mean_reference_style_reward <= 0.0
            else mean_style_reward / mean_reference_style_reward
        )
        weighted_style_fraction = result.get("weighted_amp_style_fraction")
        mean_weighted_style_fraction = (
            None
            if weighted_style_fraction is None
            else float(weighted_style_fraction.get("mean", 0.0))
        )
        modes[mode] = {
            "episodes": episodes,
            "success_rate": success_rate,
            "joint_limit_rate": joint_limit_rate,
            "parallel_ankle_rate": parallel_ankle_rate,
            "nonfinite_action_rate": nonfinite_action_rate,
            "parallel_state_violation_fraction": parallel_violation,
            "p90_recovery_time_s": p90_recovery_time,
            "mean_amp_style_reward": mean_style_reward,
            "mean_amp_reference_style_reward": mean_reference_style_reward,
            "amp_style_reward_ratio": style_reward_ratio,
            "mean_weighted_amp_style_fraction": mean_weighted_style_fraction,
        }
        checks = [
            (
                success_rate >= minimum_success_rate,
                f"success rate {success_rate:.4f} < {minimum_success_rate:.4f}",
            ),
            (
                joint_limit_rate <= maximum_joint_limit_rate,
                f"joint-limit rate {joint_limit_rate:.4f} > {maximum_joint_limit_rate:.4f}",
            ),
            (
                parallel_ankle_rate <= maximum_parallel_ankle_rate,
                f"parallel-ankle rate {parallel_ankle_rate:.4f} > {maximum_parallel_ankle_rate:.4f}",
            ),
            (
                nonfinite_action_rate <= maximum_nonfinite_action_rate,
                "non-finite action rate "
                f"{nonfinite_action_rate:.4f} > {maximum_nonfinite_action_rate:.4f}",
            ),
            (
                parallel_violation <= maximum_parallel_state_violation_fraction,
                "parallel-state violation fraction "
                f"{parallel_violation:.6f} > {maximum_parallel_state_violation_fraction:.6f}",
            ),
            (
                p90_recovery_time is not None
                and p90_recovery_time <= maximum_p90_recovery_time_s,
                "P90 recovery time "
                + (
                    "is unavailable"
                    if p90_recovery_time is None
                    else f"{p90_recovery_time:.3f}s > {maximum_p90_recovery_time_s:.3f}s"
                ),
            ),
        ]
        if minimum_amp_style_reward_ratio is not None:
            checks.append(
                (
                    style_reward_ratio is not None
                    and style_reward_ratio >= minimum_amp_style_reward_ratio,
                    "AMP style/reference reward ratio "
                    + (
                        "is unavailable"
                        if style_reward_ratio is None
                        else (
                            f"{style_reward_ratio:.4f} "
                            f"< {minimum_amp_style_reward_ratio:.4f}"
                        )
                    ),
                )
            )
        if minimum_weighted_amp_style_fraction is not None:
            checks.append(
                (
                    mean_weighted_style_fraction is not None
                    and mean_weighted_style_fraction
                    >= minimum_weighted_amp_style_fraction,
                    "weighted AMP style fraction "
                    + (
                        "is unavailable"
                        if mean_weighted_style_fraction is None
                        else (
                            f"{mean_weighted_style_fraction:.4f} "
                            f"< {minimum_weighted_amp_style_fraction:.4f}"
                        )
                    ),
                )
            )
        failures.extend(f"{mode}: {message}" for passed, message in checks if not passed)
    return {
        "accepted": not failures,
        "required_modes": list(required_modes),
        "thresholds": {
            "minimum_success_rate": minimum_success_rate,
            "maximum_joint_limit_rate": maximum_joint_limit_rate,
            "maximum_parallel_ankle_rate": maximum_parallel_ankle_rate,
            "maximum_nonfinite_action_rate": maximum_nonfinite_action_rate,
            "maximum_parallel_state_violation_fraction": maximum_parallel_state_violation_fraction,
            "maximum_p90_recovery_time_s": maximum_p90_recovery_time_s,
            "minimum_episodes_per_mode": minimum_episodes_per_mode,
            "minimum_amp_style_reward_ratio": minimum_amp_style_reward_ratio,
            "minimum_weighted_amp_style_fraction": minimum_weighted_amp_style_fraction,
        },
        "modes": modes,
        "failures": failures,
    }


def unit_interval(value: str) -> float:
    result = float(value)
    if not 0.0 <= result <= 1.0:
        raise argparse.ArgumentTypeError("value must be in [0, 1]")
    return result


def positive_float(value: str) -> float:
    result = float(value)
    if result <= 0.0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--required-modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--minimum-success-rate", type=unit_interval, default=0.90)
    parser.add_argument("--maximum-joint-limit-rate", type=unit_interval, default=0.001)
    parser.add_argument("--maximum-parallel-ankle-rate", type=unit_interval, default=0.001)
    parser.add_argument("--maximum-nonfinite-action-rate", type=unit_interval, default=0.0)
    parser.add_argument(
        "--maximum-parallel-state-violation-fraction",
        type=unit_interval,
        default=0.001,
    )
    parser.add_argument("--maximum-p90-recovery-time-s", type=positive_float, default=4.0)
    parser.add_argument("--minimum-episodes-per-mode", type=int, default=1)
    parser.add_argument("--minimum-amp-style-reward-ratio", type=unit_interval, default=None)
    parser.add_argument(
        "--minimum-weighted-amp-style-fraction",
        type=unit_interval,
        default=None,
    )
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.minimum_episodes_per_mode < 1:
        raise ValueError("--minimum-episodes-per-mode must be positive")
    results = json.loads(args.evaluation.read_text(encoding="utf-8"))
    summary = evaluate_acceptance(
        results,
        args.required_modes,
        minimum_success_rate=args.minimum_success_rate,
        maximum_joint_limit_rate=args.maximum_joint_limit_rate,
        maximum_parallel_ankle_rate=args.maximum_parallel_ankle_rate,
        maximum_nonfinite_action_rate=args.maximum_nonfinite_action_rate,
        maximum_parallel_state_violation_fraction=(
            args.maximum_parallel_state_violation_fraction
        ),
        maximum_p90_recovery_time_s=args.maximum_p90_recovery_time_s,
        minimum_episodes_per_mode=args.minimum_episodes_per_mode,
        minimum_amp_style_reward_ratio=args.minimum_amp_style_reward_ratio,
        minimum_weighted_amp_style_fraction=(
            args.minimum_weighted_amp_style_fraction
        ),
    )
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if summary["accepted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

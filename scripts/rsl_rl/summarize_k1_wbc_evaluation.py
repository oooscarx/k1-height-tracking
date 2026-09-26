#!/usr/bin/env python3
"""Aggregate per-orientation WBC evaluation files into one comparison artifact."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

COUNT_FIELDS = (
    "success_count",
    "incomplete_count",
    "no_progress_failure_count",
    "timeout_failure_count",
    "joint_limit_braking_count",
    "parallel_fallback_count",
    "nonfinite_action_count",
    "nonfinite_critic_value_count",
    "critic_value_abs_above_50_count",
    "motor_velocity_limit_exceeded_count",
    "raw_action_clip_exceeded_count",
)
MAXIMUM_FIELDS = (
    "maximum_contact_force_norm",
    "maximum_critic_value_abs",
    "p95_maximum_critic_value_abs",
    "p99_maximum_critic_value_abs",
    "maximum_joint_velocity_abs",
    "maximum_motor_utilization",
    "maximum_motor_velocity_ratio",
    "maximum_policy_observation_abs",
    "maximum_raw_action_abs",
)


def summarize(results: dict[str, dict]) -> dict:
    if not results:
        raise ValueError("at least one evaluation result is required")
    total_envs = sum(int(result["num_envs"]) for result in results.values())
    success_count = sum(int(result["success_count"]) for result in results.values())
    recovery_time_sum = sum(
        float(result["mean_recovery_time"]) * int(result["success_count"])
        for result in results.values()
        if result.get("mean_recovery_time") is not None
    )
    aggregate = {
        "num_envs": total_envs,
        "success_rate": success_count / total_envs,
        "mean_recovery_time": recovery_time_sum / success_count if success_count else None,
    }
    for field in COUNT_FIELDS:
        aggregate[field] = sum(int(result.get(field, 0)) for result in results.values())
    for field in MAXIMUM_FIELDS:
        aggregate[field] = max(float(result.get(field, 0.0)) for result in results.values())
    mirror_results = [
        result
        for result in results.values()
        if result.get("policy_mirror_sample_count", 0) > 0
    ]
    if mirror_results:
        joint_names = mirror_results[0]["policy_mirror_joint_names"]
        if any(result["policy_mirror_joint_names"] != joint_names for result in mirror_results):
            raise ValueError("policy mirror joint names differ between evaluation modes")
        mirror_sample_count = sum(
            int(result["policy_mirror_sample_count"])
            for result in mirror_results
        )
        mean_by_joint = [
            sum(
                float(result["mean_policy_mirror_error_abs_by_joint"][index])
                * int(result["policy_mirror_sample_count"])
                for result in mirror_results
            )
            / mirror_sample_count
            for index in range(len(joint_names))
        ]
        maximum_by_joint = [
            max(
                float(result["maximum_policy_mirror_error_abs_by_joint"][index])
                for result in mirror_results
            )
            for index in range(len(joint_names))
        ]
        aggregate.update(
            {
                "policy_mirror_joint_names": joint_names,
                "policy_mirror_sample_count": mirror_sample_count,
                "mean_policy_mirror_error_abs": sum(mean_by_joint) / len(mean_by_joint),
                "maximum_policy_mirror_error_abs": max(maximum_by_joint),
                "mean_policy_mirror_error_abs_by_joint": mean_by_joint,
                "maximum_policy_mirror_error_abs_by_joint": maximum_by_joint,
            }
        )
    return {"aggregate": aggregate, "modes": results}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_dir", type=Path)
    parser.add_argument("--lift-scale", required=True)
    parser.add_argument("--terrain-level", required=True)
    parser.add_argument("--modes", nargs="+", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    results = {}
    for mode in args.modes:
        path = args.evaluation_dir / (
            f"{mode}_lift_{args.lift_scale}_terrain_{args.terrain_level}.json"
        )
        with path.open(encoding="utf-8") as result_file:
            result = json.load(result_file)
        if result["orientation_mode"] != mode:
            raise RuntimeError(f"orientation mismatch in {path}: {result['orientation_mode']}")
        if not math.isclose(
            float(result["lift_force_scale"]),
            float(args.lift_scale),
            abs_tol=1.0e-8,
        ):
            raise RuntimeError(f"lift scale mismatch in {path}")
        results[mode] = result

    summary = summarize(results)
    summary["lift_force_scale"] = float(args.lift_scale)
    summary["terrain_level"] = int(args.terrain_level)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    print(json.dumps(summary["aggregate"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

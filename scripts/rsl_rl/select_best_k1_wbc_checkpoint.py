#!/usr/bin/env python3
"""Select the strongest safe K1 WBC checkpoint from zero-lift evaluations."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

MODEL_DIR = re.compile(r"model_(\d+)$")
DEFAULT_MODES = ("face_up", "face_down", "left_side", "right_side", "other")
DISQUALIFYING_COUNTS = (
    "incomplete_count",
    "nonfinite_action_count",
    "nonfinite_critic_value_count",
    "parallel_fallback_count",
)
DIAGNOSTIC_COUNT_FIELDS = (
    "critic_value_abs_above_50_count",
    "joint_limit_braking_count",
    "motor_velocity_limit_exceeded_count",
    "raw_action_clip_exceeded_count",
)
DIAGNOSTIC_MAXIMUM_FIELDS = (
    "maximum_contact_force_norm",
    "maximum_joint_velocity_abs",
    "maximum_motor_utilization",
    "maximum_motor_velocity_ratio",
    "maximum_raw_action_abs",
)


def wilson_lower_bound(successes: int, total: int, z: float = 1.959963984540054) -> float:
    if total <= 0:
        return 0.0
    proportion = successes / total
    z_squared = z * z
    denominator = 1.0 + z_squared / total
    centre = proportion + z_squared / (2.0 * total)
    margin = z * math.sqrt(
        proportion * (1.0 - proportion) / total
        + z_squared / (4.0 * total * total)
    )
    return (centre - margin) / denominator


def _finite_number(mapping: dict[str, Any], name: str, reasons: list[str]) -> float:
    value = mapping.get(name)
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        reasons.append(f"invalid {name}: {value!r}")
        return 0.0
    return float(value)


def _optional_finite_number(
    mapping: dict[str, Any],
    name: str,
    reasons: list[str],
    default: float = 0.0,
) -> float:
    if name not in mapping:
        return default
    return _finite_number(mapping, name, reasons)


def assess_candidate(
    summary_path: Path,
    run_dir: Path,
    expected_modes: tuple[str, ...] = DEFAULT_MODES,
    lift_scale: float = 0.0,
    checkpoint_override: Path | None = None,
) -> dict[str, Any]:
    reasons: list[str] = []
    if checkpoint_override is None:
        match = MODEL_DIR.fullmatch(summary_path.parent.name)
        if match is None:
            raise ValueError(
                f"evaluation directory is not a model directory: {summary_path}"
            )
        iteration = int(match.group(1))
        checkpoint = run_dir / f"model_{iteration}.pt"
    else:
        checkpoint = checkpoint_override
        match = MODEL_DIR.fullmatch(checkpoint.stem)
        if match is None:
            raise ValueError(
                f"external checkpoint must be named model_<iteration>.pt: {checkpoint}"
            )
        iteration = int(match.group(1))
    if not checkpoint.is_file() or checkpoint.stat().st_size <= 0:
        reasons.append(f"checkpoint is missing or empty: {checkpoint}")

    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "checkpoint": str(checkpoint),
            "eligible": False,
            "iteration": iteration,
            "reasons": [f"cannot read summary: {type(error).__name__}: {error}"],
            "summary": str(summary_path),
        }

    actual_lift = _finite_number(summary, "lift_force_scale", reasons)
    if not math.isclose(actual_lift, lift_scale, abs_tol=1.0e-8):
        reasons.append(f"lift scale is {actual_lift}, expected {lift_scale}")

    aggregate = summary.get("aggregate")
    if not isinstance(aggregate, dict):
        reasons.append("aggregate metrics are missing")
        aggregate = {}
    modes = summary.get("modes")
    if not isinstance(modes, dict):
        reasons.append("mode metrics are missing")
        modes = {}
    if set(modes) != set(expected_modes):
        reasons.append(
            f"mode set differs: actual={sorted(modes)} expected={sorted(expected_modes)}"
        )

    aggregate_total = int(_finite_number(aggregate, "num_envs", reasons))
    aggregate_successes = int(_finite_number(aggregate, "success_count", reasons))
    aggregate_rate = _finite_number(aggregate, "success_rate", reasons)
    recovery_time = _finite_number(aggregate, "mean_recovery_time", reasons)
    if aggregate_total <= 0 or not 0 <= aggregate_successes <= aggregate_total:
        reasons.append(
            f"invalid aggregate counts: success={aggregate_successes} total={aggregate_total}"
        )
    elif not math.isclose(
        aggregate_rate,
        aggregate_successes / aggregate_total,
        abs_tol=1.0e-8,
    ):
        reasons.append("aggregate success rate does not match its counts")

    for name in DISQUALIFYING_COUNTS:
        count = aggregate.get(name, 0)
        if not isinstance(count, (int, float)) or not math.isfinite(float(count)):
            reasons.append(f"invalid {name}: {count!r}")
        elif int(count) != 0:
            reasons.append(f"{name}={int(count)}")

    evaluation_diagnostics: dict[str, float | int] = {}
    for name in DIAGNOSTIC_COUNT_FIELDS:
        count = int(_optional_finite_number(aggregate, name, reasons))
        evaluation_diagnostics[name] = count
        evaluation_diagnostics[f"{name.removesuffix('_count')}_rate"] = (
            count / aggregate_total if aggregate_total > 0 else 0.0
        )
    for name in DIAGNOSTIC_MAXIMUM_FIELDS:
        evaluation_diagnostics[name] = _optional_finite_number(
            aggregate,
            name,
            reasons,
        )

    mode_wilson: dict[str, float] = {}
    mode_rates: dict[str, float] = {}
    mode_total = 0
    mode_successes = 0
    for mode in expected_modes:
        metrics = modes.get(mode)
        if not isinstance(metrics, dict):
            continue
        total = int(_finite_number(metrics, "num_envs", reasons))
        successes = int(_finite_number(metrics, "success_count", reasons))
        rate = _finite_number(metrics, "success_rate", reasons)
        match_rate = _finite_number(metrics, "initial_mode_match_rate", reasons)
        mode_lift = _finite_number(metrics, "lift_force_scale", reasons)
        if total <= 0 or not 0 <= successes <= total:
            reasons.append(f"{mode} has invalid counts: success={successes} total={total}")
            continue
        if not math.isclose(rate, successes / total, abs_tol=1.0e-8):
            reasons.append(f"{mode} success rate does not match its counts")
        if not math.isclose(match_rate, 1.0, abs_tol=1.0e-6):
            reasons.append(f"{mode} initial mode match rate is {match_rate}")
        if not math.isclose(mode_lift, lift_scale, abs_tol=1.0e-8):
            reasons.append(f"{mode} lift scale is {mode_lift}, expected {lift_scale}")
        mode_total += total
        mode_successes += successes
        mode_rates[mode] = rate
        mode_wilson[mode] = wilson_lower_bound(successes, total)

    if mode_total != aggregate_total or mode_successes != aggregate_successes:
        reasons.append(
            "aggregate counts differ from mode totals: "
            f"aggregate={aggregate_successes}/{aggregate_total} "
            f"modes={mode_successes}/{mode_total}"
        )

    aggregate_wilson = wilson_lower_bound(aggregate_successes, aggregate_total)
    weakest_mode_wilson = min(mode_wilson.values(), default=0.0)
    weakest_mode_rate = min(mode_rates.values(), default=0.0)
    robust_score = 0.7 * aggregate_wilson + 0.3 * weakest_mode_wilson
    return {
        "aggregate_success_rate": aggregate_rate,
        "aggregate_wilson_lower_95": aggregate_wilson,
        "checkpoint": str(checkpoint.resolve()),
        "eligible": not reasons,
        "evaluation_diagnostics": evaluation_diagnostics,
        "iteration": iteration,
        "mean_recovery_time": recovery_time,
        "mode_success_rates": mode_rates,
        "mode_wilson_lower_95": mode_wilson,
        "num_envs": aggregate_total,
        "reasons": reasons,
        "robust_score": robust_score,
        "summary": str(summary_path.resolve()),
        "weakest_mode_success_rate": weakest_mode_rate,
        "weakest_mode_wilson_lower_95": weakest_mode_wilson,
    }


def select_best(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [candidate for candidate in candidates if candidate["eligible"]]
    if not eligible:
        raise RuntimeError("no eligible zero-lift checkpoint evaluation")
    return max(
        eligible,
        key=lambda candidate: (
            candidate["robust_score"],
            candidate["aggregate_wilson_lower_95"],
            candidate["aggregate_success_rate"],
            -candidate["mean_recovery_time"],
            candidate["iteration"],
        ),
    )


def find_evaluation_summaries(run_dir: Path, summary_name: str) -> list[Path]:
    summaries = []
    for path in run_dir.glob(f"evaluations/model_*/{summary_name}"):
        match = MODEL_DIR.fullmatch(path.parent.name)
        if match is not None:
            summaries.append((int(match.group(1)), path))
    return [path for _, path in sorted(summaries)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--lift-scale", default=0.0, type=float)
    parser.add_argument("--terrain-level", default=0, type=int)
    parser.add_argument("--modes", nargs="+", default=list(DEFAULT_MODES))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--baseline-summary", type=Path)
    args = parser.parse_args()
    if (args.baseline_checkpoint is None) != (args.baseline_summary is None):
        parser.error(
            "--baseline-checkpoint and --baseline-summary must be provided together"
        )

    summary_name = (
        f"summary_lift_{args.lift_scale:g}_terrain_{args.terrain_level}.json"
    )
    summary_paths = find_evaluation_summaries(args.run_dir, summary_name)
    if not summary_paths:
        parser.error(f"no evaluation summaries named {summary_name} in {args.run_dir}")
    candidates = [
        assess_candidate(
            path,
            args.run_dir,
            tuple(args.modes),
            lift_scale=args.lift_scale,
        )
        for path in summary_paths
    ]
    if args.baseline_checkpoint is not None:
        candidates.append(
            assess_candidate(
                args.baseline_summary,
                args.run_dir,
                tuple(args.modes),
                lift_scale=args.lift_scale,
                checkpoint_override=args.baseline_checkpoint,
            )
        )
    try:
        selected = select_best(candidates)
    except RuntimeError as error:
        parser.error(str(error))
    report = {
        "candidates": candidates,
        "lift_force_scale": args.lift_scale,
        "ranking_method": (
            "0.7 * aggregate Wilson lower 95% + "
            "0.3 * weakest-mode Wilson lower 95%; safety gates first"
        ),
        "selected": selected,
        "terrain_level": args.terrain_level,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(selected["checkpoint"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

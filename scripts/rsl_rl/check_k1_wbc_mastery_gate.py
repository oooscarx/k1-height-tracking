#!/usr/bin/env python3
"""Check whether a selected WBC checkpoint clears a mastery confidence gate."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


def assess_gate(
    selection_report: dict[str, Any],
    minimum_aggregate_wilson: float = 0.9,
    minimum_weakest_mode_wilson: float = 0.9,
    terrain_level: int = 0,
) -> dict[str, Any]:
    selected = selection_report.get("selected")
    reasons: list[str] = []
    if not isinstance(selected, dict):
        return {"passed": False, "reasons": ["selected candidate is missing"]}
    if not selected.get("eligible", False):
        reasons.append("selected candidate is not safety eligible")

    def finite_metric(name: str) -> float:
        value = selected.get(name)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            reasons.append(f"invalid {name}: {value!r}")
            return 0.0
        return float(value)

    aggregate_wilson = finite_metric("aggregate_wilson_lower_95")
    weakest_mode_wilson = finite_metric("weakest_mode_wilson_lower_95")
    if aggregate_wilson < minimum_aggregate_wilson:
        reasons.append(
            "aggregate Wilson lower bound is below gate: "
            f"{aggregate_wilson:.6f} < {minimum_aggregate_wilson:.6f}"
        )
    if weakest_mode_wilson < minimum_weakest_mode_wilson:
        reasons.append(
            "weakest-mode Wilson lower bound is below gate: "
            f"{weakest_mode_wilson:.6f} < {minimum_weakest_mode_wilson:.6f}"
        )
    if float(selection_report.get("lift_force_scale", math.nan)) != 0.0:
        reasons.append("mastery gate requires zero lift")
    actual_terrain_level = int(selection_report.get("terrain_level", -2))
    if actual_terrain_level != terrain_level:
        reasons.append(
            "mastery gate terrain level differs: "
            f"{actual_terrain_level} != {terrain_level}"
        )

    return {
        "aggregate_wilson_lower_95": aggregate_wilson,
        "checkpoint": selected.get("checkpoint"),
        "minimum_aggregate_wilson": minimum_aggregate_wilson,
        "minimum_weakest_mode_wilson": minimum_weakest_mode_wilson,
        "num_envs": selected.get("num_envs"),
        "passed": not reasons,
        "reasons": reasons,
        "terrain_level": actual_terrain_level,
        "weakest_mode_success_rate": selected.get("weakest_mode_success_rate"),
        "weakest_mode_wilson_lower_95": weakest_mode_wilson,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("selection_report", type=Path)
    parser.add_argument("--minimum-aggregate-wilson", type=float, default=0.9)
    parser.add_argument("--minimum-weakest-mode-wilson", type=float, default=0.9)
    parser.add_argument("--terrain-level", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    for name in ("minimum_aggregate_wilson", "minimum_weakest_mode_wilson"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be within [0, 1]")

    report = json.loads(args.selection_report.read_text(encoding="utf-8"))
    result = assess_gate(
        report,
        minimum_aggregate_wilson=args.minimum_aggregate_wilson,
        minimum_weakest_mode_wilson=args.minimum_weakest_mode_wilson,
        terrain_level=args.terrain_level,
    )
    output = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output, encoding="utf-8")
    print(output, end="")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

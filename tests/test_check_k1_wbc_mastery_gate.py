from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "rsl_rl"
    / "check_k1_wbc_mastery_gate.py"
)
SPEC = importlib.util.spec_from_file_location("check_k1_wbc_mastery_gate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
GATE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GATE)


def _selection(aggregate: float, weakest: float) -> dict:
    return {
        "lift_force_scale": 0.0,
        "terrain_level": 0,
        "selected": {
            "aggregate_wilson_lower_95": aggregate,
            "checkpoint": "/tmp/model_500.pt",
            "eligible": True,
            "num_envs": 640,
            "weakest_mode_success_rate": 0.96,
            "weakest_mode_wilson_lower_95": weakest,
        },
    }


def test_mastery_gate_requires_both_confidence_bounds() -> None:
    assert GATE.assess_gate(_selection(0.95, 0.91))["passed"]
    result = GATE.assess_gate(_selection(0.95, 0.89))
    assert not result["passed"]
    assert any("weakest-mode" in reason for reason in result["reasons"])


def test_mastery_gate_requires_zero_lift_and_terrain_zero() -> None:
    report = _selection(0.95, 0.91)
    report["lift_force_scale"] = 0.1
    report["terrain_level"] = 1
    result = GATE.assess_gate(report)
    assert not result["passed"]
    assert "mastery gate requires zero lift" in result["reasons"]
    assert "mastery gate terrain level differs: 1 != 0" in result["reasons"]


def test_mastery_gate_accepts_balanced_all_terrain_report() -> None:
    report = _selection(0.95, 0.91)
    report["terrain_level"] = -1
    result = GATE.assess_gate(report, terrain_level=-1)
    assert result["passed"]
    assert result["terrain_level"] == -1

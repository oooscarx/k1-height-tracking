from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "rsl_rl"
    / "monitor_k1_height_tracking.py"
)
SPEC = importlib.util.spec_from_file_location("monitor_k1_height_tracking", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MONITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MONITOR)


def test_completion_does_not_override_an_existing_critical_issue() -> None:
    result = {"status": "critical", "reasons": ["non-finite scalar"]}

    MONITOR._mark_complete_if_healthy(result)

    assert result == {"status": "critical", "reasons": ["non-finite scalar"]}


def test_issue_overrides_a_previously_complete_status() -> None:
    result = {"status": "complete"}

    MONITOR._add_issue(result, "critical", "disk failure")

    assert result == {"status": "critical", "reasons": ["disk failure"]}


def test_training_quality_warning_requires_three_bad_samples() -> None:
    result = {"status": "ok"}
    history = [
        {"mean_reward": -700.0, "height_error": 0.30},
        {"mean_reward": -800.0, "height_error": 0.35},
    ]

    MONITOR._add_training_quality_issues(result, history)

    assert result == {"status": "ok"}


def test_training_quality_warning_reports_sustained_policy_degradation() -> None:
    result = {"status": "ok"}
    history = [
        {"mean_reward": -700.0, "height_error": 0.30},
        {"mean_reward": -800.0, "height_error": 0.35},
        {"mean_reward": -900.0, "height_error": 0.40},
    ]

    MONITOR._add_training_quality_issues(result, history)

    assert result["status"] == "warning"
    assert result["reasons"] == [
        "mean reward stayed below -500 for three samples",
        "height error stayed above 0.25 for three samples",
    ]


def test_training_quality_warning_clears_after_one_recovered_sample() -> None:
    result = {"status": "ok"}
    history = [
        {"mean_reward": -700.0, "height_error": 0.30},
        {"mean_reward": -800.0, "height_error": 0.35},
        {"mean_reward": -120.0, "height_error": 0.12},
    ]

    MONITOR._add_training_quality_issues(result, history)

    assert result == {"status": "ok"}


def test_serial_training_monitor_has_no_parallel_ankle_tags() -> None:
    assert not any("parallel" in name for name in MONITOR.DEFAULT_TAGS)


def test_progressive_robustness_metrics_are_monitored() -> None:
    expected = {
        "disturbance_scale": "Curriculum/disturbance_scale",
        "disturbance_jump_error_ema": "Metrics/height/disturbance_jump_error_ema",
        "disturbance_governed_error_ema": "Metrics/height/disturbance_governed_error_ema",
        "disturbance_gate_ready": "Metrics/height/disturbance_gate_ready",
        "disturbance_gate_passed": "Metrics/height/disturbance_gate_passed",
    }

    assert {name: MONITOR.DEFAULT_TAGS[name] for name in expected} == expected


def test_recent_nonfinite_scalar_scan_catches_recovered_single_step() -> None:
    series = {
        "Metrics/height/height_error": [
            SimpleNamespace(step=100, value=0.1),
            SimpleNamespace(step=150, value=float("inf")),
            SimpleNamespace(step=200, value=0.12),
        ]
    }

    assert MONITOR._recent_nonfinite_scalars(series, last_step=200) == [
        {"step": 150, "tag": "Metrics/height/height_error"}
    ]
    assert MONITOR._recent_nonfinite_scalars(series, last_step=400) == []


def test_training_quality_normalizes_invalid_state_for_larger_batches() -> None:
    result = {"status": "ok", "num_envs": 16384}
    history = [
        {"invalid_state": 0.08},
        {"invalid_state": 0.12},
        {"invalid_state": 0.16},
    ]

    MONITOR._add_training_quality_issues(result, history)

    assert result == {"status": "ok", "num_envs": 16384}


def test_training_quality_warns_after_normalized_invalid_state_exceeds_threshold() -> None:
    result = {"status": "ok", "num_envs": 16384}
    history = [
        {"invalid_state": 0.24},
        {"invalid_state": 0.28},
        {"invalid_state": 0.32},
    ]

    MONITOR._add_training_quality_issues(result, history)

    assert result["status"] == "warning"
    assert result["reasons"] == [
        "invalid-state term exceeded 0.05 per 4096 envs for three samples"
    ]


def test_training_quality_does_not_infer_curriculum_stall_from_episode_height_error() -> None:
    result = {"status": "ok"}
    history = [
        {"height_error": 0.05, "lift_scale": 1.0, "terrain_level": 0.0}
        for _ in range(8)
    ]

    MONITOR._add_training_quality_issues(result, history)

    assert result == {"status": "ok"}

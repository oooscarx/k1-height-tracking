from __future__ import annotations

import importlib.util
import math
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "rsl_rl"
    / "summarize_k1_wbc_evaluation.py"
)
SPEC = importlib.util.spec_from_file_location("summarize_k1_wbc_evaluation", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUMMARIZER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARIZER)


def test_summarize_weights_recovery_time_and_combines_safety_counts() -> None:
    results = {
        "face_up": {
            "num_envs": 10,
            "success_count": 4,
            "mean_recovery_time": 12.0,
            "motor_velocity_limit_exceeded_count": 2,
            "critic_value_abs_above_50_count": 1,
            "nonfinite_critic_value_count": 0,
            "maximum_raw_action_abs": 8.0,
            "p95_maximum_critic_value_abs": 12.0,
            "p99_maximum_critic_value_abs": 18.0,
        },
        "face_down": {
            "num_envs": 10,
            "success_count": 6,
            "mean_recovery_time": 14.0,
            "motor_velocity_limit_exceeded_count": 3,
            "critic_value_abs_above_50_count": 2,
            "nonfinite_critic_value_count": 1,
            "maximum_raw_action_abs": 20.0,
            "p95_maximum_critic_value_abs": 16.0,
            "p99_maximum_critic_value_abs": 24.0,
        },
    }
    summary = SUMMARIZER.summarize(results)
    aggregate = summary["aggregate"]
    assert aggregate["num_envs"] == 20
    assert aggregate["success_count"] == 10
    assert aggregate["success_rate"] == 0.5
    assert math.isclose(aggregate["mean_recovery_time"], 13.2)
    assert aggregate["motor_velocity_limit_exceeded_count"] == 5
    assert aggregate["critic_value_abs_above_50_count"] == 3
    assert aggregate["nonfinite_critic_value_count"] == 1
    assert aggregate["maximum_raw_action_abs"] == 20.0
    assert aggregate["p95_maximum_critic_value_abs"] == 16.0
    assert aggregate["p99_maximum_critic_value_abs"] == 24.0


def test_summarize_handles_no_successes() -> None:
    summary = SUMMARIZER.summarize(
        {"random": {"num_envs": 8, "success_count": 0, "mean_recovery_time": None}}
    )
    assert summary["aggregate"]["success_rate"] == 0.0
    assert summary["aggregate"]["mean_recovery_time"] is None


def test_summarize_weights_per_joint_policy_mirror_error() -> None:
    results = {
        "face_up": {
            "num_envs": 1,
            "success_count": 1,
            "mean_recovery_time": 4.0,
            "policy_mirror_joint_names": ["left_shoulder", "right_shoulder"],
            "policy_mirror_sample_count": 10,
            "mean_policy_mirror_error_abs_by_joint": [0.1, 0.2],
            "maximum_policy_mirror_error_abs_by_joint": [0.4, 0.3],
        },
        "face_down": {
            "num_envs": 1,
            "success_count": 1,
            "mean_recovery_time": 6.0,
            "policy_mirror_joint_names": ["left_shoulder", "right_shoulder"],
            "policy_mirror_sample_count": 30,
            "mean_policy_mirror_error_abs_by_joint": [0.3, 0.4],
            "maximum_policy_mirror_error_abs_by_joint": [0.2, 0.8],
        },
    }
    aggregate = SUMMARIZER.summarize(results)["aggregate"]
    assert aggregate["policy_mirror_sample_count"] == 40
    assert aggregate["mean_policy_mirror_error_abs_by_joint"] == [0.25, 0.35]
    assert math.isclose(aggregate["mean_policy_mirror_error_abs"], 0.3)
    assert aggregate["maximum_policy_mirror_error_abs_by_joint"] == [0.4, 0.8]
    assert aggregate["maximum_policy_mirror_error_abs"] == 0.8

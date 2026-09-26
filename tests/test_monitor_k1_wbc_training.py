from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest
import torch

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rsl_rl" / "monitor_k1_wbc_training.py"
SPEC = importlib.util.spec_from_file_location("monitor_k1_wbc_training", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MONITOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MONITOR)


def test_parse_latest_complete_iteration_skips_partial_tail() -> None:
    log = """
Learning iteration 41/100000
Computation: 12000 steps/s
Mean action noise std: 1.7
Mean value_function loss: 2.5
Mean symmetry loss: 0.003
Mean l2c2_critic loss: 0.04
Mean reward: -8.0
Mean episode length: 321.5
Curriculum/remove_lift: 0.4
Episode_Termination/time_out: 6.75
Episode_Termination/invalid_state: 0.0
Episode_Termination/standing: 2.25
Iteration time: 8.0s
Learning iteration 42/100000
Mean action noise std: 1.6
"""
    result = MONITOR.parse_latest_iteration(log)
    assert result == {
        "iteration": 41,
        "maximum_iteration": 100000,
        "steps_per_second": 12000.0,
        "policy_noise_std": 1.7,
        "value_loss": 2.5,
        "symmetry_loss": 0.003,
        "l2c2_critic_loss": 0.04,
        "mean_reward": -8.0,
        "mean_episode_length": 321.5,
        "lift_force_scale": 0.4,
        "timeout_termination_count": 6.75,
        "invalid_state_termination_count": 0.0,
        "standing_termination_count": 2.25,
        "termination_count": 9.0,
        "standing_success_rate": 0.25,
        "timeout_rate": 0.75,
        "invalid_state_rate": 0.0,
    }


def test_parse_complete_iterations_preserves_every_flushed_iteration() -> None:
    complete = """
Learning iteration {iteration}/100000
Mean value_function loss: {value_loss}
Mean l2c2_critic loss: {critic_loss}
Iteration time: 8.0s
"""
    log = (
        complete.format(iteration=41, value_loss=2.5, critic_loss=0.04)
        + complete.format(iteration=42, value_loss=7.1, critic_loss=7.0)
        + "Learning iteration 43/100000\nMean value_function loss: 2.0\n"
    )
    results = MONITOR.parse_complete_iterations(log)
    assert [result["iteration"] for result in results] == [41, 42]
    assert results[1]["l2c2_critic_loss"] == 7.0


def test_latest_recorded_iteration_ignores_invalid_trailing_lines(tmp_path: Path) -> None:
    output = tmp_path / "health.jsonl"
    output.write_text(
        json.dumps({"iteration": 41}) + "\n" + "partial json\n",
        encoding="utf-8",
    )
    assert MONITOR.latest_recorded_iteration(output) == 41
    assert MONITOR.latest_recorded_iteration(tmp_path / "missing.jsonl") is None


def test_inspect_checkpoint_detects_nonfinite_optimizer_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_42.pt"
    torch.save(
        {
            "model_state_dict": {"std": torch.ones(2)},
            "optimizer_state_dict": {
                "state": {0: {"exp_avg": torch.tensor([float("nan")])}}
            },
            "reward_norm_state_dict": {
                "_mean": torch.tensor(0.0),
                "_std": torch.tensor(1.0),
                "_return_correction": torch.tensor(1.0),
            },
            "iter": 42,
        },
        checkpoint,
    )

    summary = MONITOR.inspect_checkpoint(checkpoint)

    assert summary["checkpoint_iteration"] == 42
    assert summary["checkpoint_nonfinite"] == [
        "optimizer_state_dict.state.0.exp_avg"
    ]


def test_inspect_checkpoint_reports_optimizer_progress(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_42.pt"
    torch.save(
        {
            "model_state_dict": {"std": torch.ones(2)},
            "optimizer_state_dict": {
                "state": {
                    0: {"step": torch.tensor(123), "exp_avg": torch.zeros(2)},
                    1: {"step": torch.tensor(123), "exp_avg": torch.zeros(2)},
                },
                "param_groups": [{"lr": 3.0e-4}],
            },
            "reward_norm_state_dict": {
                "_mean": torch.tensor(0.0),
                "_std": torch.tensor(1.0),
                "_return_correction": torch.tensor(1.0),
            },
            "iter": 42,
        },
        checkpoint,
    )

    summary = MONITOR.inspect_checkpoint(checkpoint)

    assert summary["optimizer_state_count"] == 2
    assert summary["optimizer_step_minimum"] == 123
    assert summary["optimizer_step_maximum"] == 123
    assert summary["optimizer_learning_rate_minimum"] == pytest.approx(3.0e-4)
    assert summary["optimizer_learning_rate_maximum"] == pytest.approx(3.0e-4)


def test_inspect_checkpoint_reports_missing_optimizer_state(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model_42.pt"
    torch.save(
        {
            "model_state_dict": {"std": torch.ones(2)},
            "reward_norm_state_dict": {
                "_mean": torch.tensor(0.0),
                "_std": torch.tensor(1.0),
                "_return_correction": torch.tensor(1.0),
            },
            "iter": 42,
        },
        checkpoint,
    )

    summary = MONITOR.inspect_checkpoint(checkpoint)

    assert summary["checkpoint_nonfinite"] == ["missing:optimizer_state_dict"]


def test_health_alerts_report_only_numerical_failures() -> None:
    assert MONITOR.health_alerts({"value_loss": 2.0, "reward_normalizer_std": 0.18}) == []
    alerts = MONITOR.health_alerts(
        {
            "value_loss": 20_000.0,
            "reward_normalizer_std": 140.0,
            "checkpoint_nonfinite": ["model_state_dict.std"],
            "checkpoint_error": "EOFError: truncated",
        }
    )
    assert "value_loss=20000.0" in alerts
    assert "reward_normalizer_std=140.0" in alerts
    assert "checkpoint_nonfinite=['model_state_dict.std']" in alerts
    assert "checkpoint_error=EOFError: truncated" in alerts


def test_health_alerts_capture_transient_symmetry_and_l2c2_spikes() -> None:
    alerts = MONITOR.health_alerts(
        {
            "value_loss": 3.0,
            "symmetry_loss": 8.5,
            "l2c2_critic_loss": 0.77,
            "policy_noise_std": 1.6,
        }
    )
    assert "symmetry_loss=8.5" in alerts
    assert "l2c2_critic_loss=0.77" in alerts


def test_health_alerts_capture_invalid_optimizer_progress() -> None:
    alerts = MONITOR.health_alerts(
        {
            "optimizer_learning_rate_minimum": 0.0,
            "optimizer_learning_rate_maximum": 0.02,
            "optimizer_state_count": 0,
            "optimizer_step_minimum": 100,
            "optimizer_step_maximum": 101,
        }
    )
    assert "optimizer_learning_rate_minimum=0.0" in alerts
    assert "optimizer_learning_rate_maximum=0.02" in alerts
    assert "optimizer_state_count=0" in alerts
    assert "optimizer_step_range=100..101" in alerts


def test_expected_lift_force_scale_matches_legacy_exponential_schedule() -> None:
    assert MONITOR.expected_lift_force_scale(0) == 1.0
    assert MONITOR.expected_lift_force_scale(1540) == pytest.approx(0.4789, abs=0.0003)
    assert MONITOR.expected_lift_force_scale(8540) == pytest.approx(0.01, abs=0.0001)
    assert MONITOR.expected_lift_force_scale(8548) == 0.0


def test_health_alerts_capture_lift_schedule_drift() -> None:
    alerts = MONITOR.health_alerts({"lift_force_schedule_error": 0.1})
    assert "lift_force_schedule_error=0.1" in alerts


def test_health_alerts_capture_severe_reward_and_invalid_state_spikes() -> None:
    alerts = MONITOR.health_alerts(
        {
            "mean_reward": -750.0,
            "invalid_state_rate": 0.0125,
        }
    )
    assert "mean_reward=-750.0" in alerts
    assert "invalid_state_rate=0.0125" in alerts
    assert MONITOR.health_alerts(
        {
            "mean_reward": -60.0,
            "invalid_state_rate": 0.00125,
        }
    ) == []


def test_health_alerts_capture_reward_proxy_mismatch() -> None:
    assert MONITOR.health_alerts(
        {
            "mean_reward": 263.66,
            "mean_episode_length": 850.67,
            "standing_success_rate": 0.17083,
        }
    ) == ["reward_proxy_mismatch"]

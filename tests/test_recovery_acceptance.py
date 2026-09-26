from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "skrl"
    / "check_recovery_acceptance.py"
)
SPEC = importlib.util.spec_from_file_location("check_recovery_acceptance", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
ACCEPTANCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ACCEPTANCE)


def result(
    *,
    success_rate: float = 0.95,
    joint_limits: int = 0,
    ankle_failures: int = 0,
    nonfinite_actions: int = 0,
    parallel_violation: float = 0.0005,
    recovery_time: float | None = 3.5,
) -> dict:
    return {
        "episodes": 1000,
        "success_rate": success_rate,
        "joint_limit_terminations": joint_limits,
        "parallel_ankle_terminations": ankle_failures,
        "nonfinite_action_terminations": nonfinite_actions,
        "parallel_state_violation_fraction": parallel_violation,
        "successful_recovery_time_s": (
            None if recovery_time is None else {"p90": recovery_time}
        ),
    }


class RecoveryAcceptanceTest(unittest.TestCase):
    def evaluate(self, results: dict) -> dict:
        return ACCEPTANCE.evaluate_acceptance(
            results,
            ["faceup", "facedown"],
            minimum_success_rate=0.9,
            maximum_joint_limit_rate=0.001,
            maximum_parallel_ankle_rate=0.001,
            maximum_nonfinite_action_rate=0.0,
            maximum_parallel_state_violation_fraction=0.001,
            maximum_p90_recovery_time_s=4.0,
        )

    def test_accepts_only_when_every_mode_passes(self) -> None:
        summary = self.evaluate({"faceup": result(), "facedown": result()})
        self.assertTrue(summary["accepted"])
        self.assertEqual(summary["failures"], [])

    def test_rejects_missing_unsafe_slow_or_unsuccessful_modes(self) -> None:
        summary = self.evaluate(
            {
                "faceup": result(
                    success_rate=0.8,
                    joint_limits=2,
                    ankle_failures=2,
                    nonfinite_actions=1,
                    parallel_violation=0.002,
                    recovery_time=4.5,
                )
            }
        )
        self.assertFalse(summary["accepted"])
        self.assertEqual(len(summary["failures"]), 7)
        self.assertTrue(any("facedown: missing" in value for value in summary["failures"]))

    def test_rejects_missing_recovery_time(self) -> None:
        summary = self.evaluate(
            {
                "faceup": result(recovery_time=None),
                "facedown": result(),
            }
        )
        self.assertFalse(summary["accepted"])
        self.assertTrue(any("unavailable" in value for value in summary["failures"]))

    def test_rejects_partial_episode_count(self) -> None:
        summary = ACCEPTANCE.evaluate_acceptance(
            {
                "faceup": result(),
                "facedown": result(),
            },
            ["faceup", "facedown"],
            minimum_success_rate=0.9,
            maximum_joint_limit_rate=0.001,
            maximum_parallel_ankle_rate=0.001,
            maximum_nonfinite_action_rate=0.0,
            maximum_parallel_state_violation_fraction=0.001,
            maximum_p90_recovery_time_s=4.0,
            minimum_episodes_per_mode=1001,
        )
        self.assertFalse(summary["accepted"])
        self.assertEqual(
            sum("completed episodes" in failure for failure in summary["failures"]),
            2,
        )

    def test_reports_amp_style_diagnostics_without_gating_them(self) -> None:
        styled = result()
        styled["amp_style_reward"] = {"mean": 0.5}
        styled["amp_reference"] = {"amp_style_reward": {"mean": 2.0}}
        styled["weighted_amp_style_fraction"] = {"mean": 0.1}
        summary = self.evaluate({"faceup": styled, "facedown": result()})
        self.assertTrue(summary["accepted"])
        self.assertEqual(
            summary["modes"]["faceup"]["amp_style_reward_ratio"],
            0.25,
        )
        self.assertEqual(
            summary["modes"]["faceup"]["mean_weighted_amp_style_fraction"],
            0.1,
        )
        self.assertIsNone(
            summary["modes"]["facedown"]["amp_style_reward_ratio"],
        )

    def test_optional_amp_gate_rejects_nominal_amp_with_negligible_style(self) -> None:
        weak_style = result()
        weak_style["amp_style_reward"] = {"mean": 0.1}
        weak_style["amp_reference"] = {"amp_style_reward": {"mean": 2.0}}
        weak_style["weighted_amp_style_fraction"] = {"mean": 0.005}
        summary = ACCEPTANCE.evaluate_acceptance(
            {"faceup": weak_style},
            ["faceup"],
            minimum_success_rate=0.9,
            maximum_joint_limit_rate=0.001,
            maximum_parallel_ankle_rate=0.001,
            maximum_nonfinite_action_rate=0.0,
            maximum_parallel_state_violation_fraction=0.001,
            maximum_p90_recovery_time_s=4.0,
            minimum_amp_style_reward_ratio=0.10,
            minimum_weighted_amp_style_fraction=0.04,
        )

        self.assertFalse(summary["accepted"])
        self.assertEqual(len(summary["failures"]), 2)
        self.assertTrue(any("style/reference" in value for value in summary["failures"]))
        self.assertTrue(any("weighted AMP" in value for value in summary["failures"]))


if __name__ == "__main__":
    unittest.main()

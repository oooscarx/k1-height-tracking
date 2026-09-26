from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "skrl"
    / "rollout_numeric_guard.py"
)
SPEC = importlib.util.spec_from_file_location("rollout_numeric_guard", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
GUARD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(GUARD)


class RolloutNumericGuardTest(unittest.TestCase):
    def test_configure_normalized_policy_action_bounds(self) -> None:
        policy = SimpleNamespace(
            action_space=SimpleNamespace(shape=(3,)),
            _g_min_actions=None,
            _g_max_actions=None,
            _g_clip_actions=False,
            _g_clip_mean_actions=False,
        )
        agent = SimpleNamespace(
            models={"policy": policy},
            device=torch.device("cpu"),
        )
        GUARD.configure_normalized_policy_action_bounds(agent)
        torch.testing.assert_close(policy._g_min_actions, -torch.ones(3))
        torch.testing.assert_close(policy._g_max_actions, torch.ones(3))
        self.assertTrue(policy._g_clip_actions)
        self.assertTrue(policy._g_clip_mean_actions)

    def test_finite_batch_rows_rejects_nonfinite_and_implausible_rows(self) -> None:
        value = {
            "policy": torch.tensor(
                [[1.0, 2.0], [float("inf"), 0.0], [1.0e7, 0.0]]
            )
        }
        valid = GUARD.finite_batch_rows(
            value,
            3,
            torch.device("cpu"),
            maximum_absolute_value=1.0e6,
        )
        torch.testing.assert_close(valid, torch.tensor([True, False, False]))

    def test_sanitize_batch_rows_zeros_complete_failed_rows(self) -> None:
        value = torch.tensor(
            [[1.0, 2.0], [float("inf"), 4.0], [5.0, float("nan")]]
        )
        safe = GUARD.sanitize_batch_rows(
            value,
            torch.tensor([False, True, True]),
        )
        torch.testing.assert_close(
            safe,
            torch.tensor([[1.0, 2.0], [0.0, 0.0], [0.0, 0.0]]),
        )
        self.assertTrue(GUARD.tensors_are_finite({"safe": safe}))

    def test_sanitize_never_maps_infinity_to_float_maximum(self) -> None:
        safe = GUARD.sanitize_batch_rows(
            torch.tensor([float("inf"), -float("inf"), float("nan")])
        )
        torch.testing.assert_close(safe, torch.zeros(3))


if __name__ == "__main__":
    unittest.main()

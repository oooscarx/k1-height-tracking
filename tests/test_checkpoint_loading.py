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
    / "checkpoint_loading.py"
)
SPEC = importlib.util.spec_from_file_location("checkpoint_loading", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
CHECKPOINT_LOADING = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKPOINT_LOADING)


class Recorder:
    def __init__(self) -> None:
        self.loaded = []

    def load_state_dict(self, value) -> None:
        self.loaded.append(value)


def make_agent() -> SimpleNamespace:
    return SimpleNamespace(
        models={
            "policy": Recorder(),
            "value": Recorder(),
            "discriminator": Recorder(),
        },
        _state_preprocessor=Recorder(),
        _value_preprocessor=Recorder(),
        _amp_observation_preprocessor=Recorder(),
    )


CHECKPOINT = {
    "policy": "policy",
    "value": "value",
    "discriminator": "discriminator",
    "state_preprocessor": "state_preprocessor",
    "value_preprocessor": "value_preprocessor",
    "amp_observation_preprocessor": "amp_observation_preprocessor",
}


class CheckpointLoadingTest(unittest.TestCase):
    def test_loaded_log_std_is_projected_into_trainable_bounds(self) -> None:
        policy = SimpleNamespace(
            log_std_parameter=torch.nn.Parameter(
                torch.tensor([-5.0, -4.2, -2.5], dtype=torch.float32)
            ),
            _g_min_log_std=-4.6,
            _g_max_log_std=-4.0,
        )

        result = CHECKPOINT_LOADING.clamp_gaussian_log_std_parameter(policy)

        self.assertIsNotNone(result)
        torch.testing.assert_close(
            policy.log_std_parameter,
            torch.tensor([-4.6, -4.2, -4.0]),
        )
        self.assertEqual(result["before_minimum"], -5.0)
        self.assertEqual(result["before_maximum"], -2.5)
        self.assertAlmostEqual(result["after_minimum"], -4.6, places=6)
        self.assertEqual(result["after_maximum"], -4.0)

    def test_reset_value_preserves_state_normalization_statistics(self) -> None:
        agent = make_agent()
        loaded = CHECKPOINT_LOADING.restore_amp_checkpoint_components(
            agent,
            CHECKPOINT,
            reset_value=True,
            reset_discriminator=False,
        )

        self.assertEqual(loaded, ["policy", "state_preprocessor", "discriminator"])
        self.assertEqual(agent.models["policy"].loaded, ["policy"])
        self.assertEqual(agent._state_preprocessor.loaded, ["state_preprocessor"])
        self.assertEqual(agent.models["value"].loaded, [])
        self.assertEqual(agent._value_preprocessor.loaded, [])
        self.assertEqual(agent.models["discriminator"].loaded, ["discriminator"])
        self.assertEqual(
            agent._amp_observation_preprocessor.loaded,
            ["amp_observation_preprocessor"],
        )

    def test_reset_discriminator_keeps_value_and_actor_state(self) -> None:
        agent = make_agent()
        loaded = CHECKPOINT_LOADING.restore_amp_checkpoint_components(
            agent,
            CHECKPOINT,
            reset_value=False,
            reset_discriminator=True,
        )

        self.assertEqual(loaded, ["policy", "state_preprocessor", "value"])
        self.assertEqual(agent.models["value"].loaded, ["value"])
        self.assertEqual(agent._value_preprocessor.loaded, ["value_preprocessor"])
        self.assertEqual(agent.models["discriminator"].loaded, [])
        self.assertEqual(agent._amp_observation_preprocessor.loaded, [])


if __name__ == "__main__":
    unittest.main()

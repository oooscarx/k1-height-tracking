from __future__ import annotations

import importlib.util
import math
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "skrl"
    / "amp_reward_diagnostics.py"
)
SPEC = importlib.util.spec_from_file_location("amp_reward_diagnostics", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
AMP_REWARD_DIAGNOSTICS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AMP_REWARD_DIAGNOSTICS)


class AmpRewardDiagnosticsTest(unittest.TestCase):
    def test_metrics_match_skrl_amp_reward_mixture(self) -> None:
        metrics = AMP_REWARD_DIAGNOSTICS.amp_reward_metrics(
            torch.tensor([[-1.0], [2.0]]),
            torch.tensor([[0.0], [math.log(3.0)]]),
            task_reward_scale=2.0,
            style_reward_scale=3.0,
        )

        weighted_style = torch.tensor([[3.0 * math.log(2.0)], [3.0 * math.log(4.0)]])
        weighted_task = torch.tensor([[-2.0], [4.0]])
        expected = {
            "Reward / AMP task contribution (mean)": weighted_task.mean(),
            "Reward / AMP style contribution (mean)": weighted_style.mean(),
            "Reward / AMP combined contribution (mean)": (weighted_task + weighted_style).mean(),
            "Reward / AMP style to task ratio": weighted_style.mean() / weighted_task.abs().mean(),
            "Reward / AMP style fraction": weighted_style.mean()
            / (weighted_task.abs().mean() + weighted_style.mean()),
            "Reward / AMP discriminator logit (mean)": torch.tensor(math.log(3.0) / 2.0),
        }
        for name, value in expected.items():
            self.assertAlmostEqual(metrics[name], value.item(), places=6)

    def test_style_scale_is_capped_to_the_requested_weighted_fraction(self) -> None:
        task_rewards = torch.tensor([[1.0], [1.0]])
        logits = torch.zeros((2, 1))
        scale = AMP_REWARD_DIAGNOSTICS.capped_style_reward_scale(
            task_rewards,
            logits,
            task_reward_scale=2.0,
            configured_style_reward_scale=5.0,
            maximum_style_fraction=0.4,
            minimum_style_reward_scale=0.0,
        )
        weighted_style = scale * math.log(2.0)
        self.assertAlmostEqual(weighted_style / (2.0 + weighted_style), 0.4, places=6)

    def test_style_scale_floor_retains_amp_when_task_reward_is_sparse(self) -> None:
        scale = AMP_REWARD_DIAGNOSTICS.capped_style_reward_scale(
            torch.zeros((2, 1)),
            torch.zeros((2, 1)),
            task_reward_scale=5.0,
            configured_style_reward_scale=5.0,
            maximum_style_fraction=0.45,
            minimum_style_reward_scale=1.0,
        )
        self.assertEqual(scale, 1.0)

    def test_install_records_metrics_and_preserves_original_update(self) -> None:
        class Memory:
            tensors = {
                "rewards": torch.tensor([[1.0], [2.0]]),
                "amp_observations": torch.tensor([[0.0], [1.0]]),
            }

            def get_tensor_by_name(self, name: str) -> torch.Tensor:
                return self.tensors[name]

        class Discriminator:
            def act(self, inputs, role: str):
                self.inputs = inputs
                self.role = role
                return torch.zeros((2, 1)), {}

        class Agent:
            def __init__(self) -> None:
                self.memory = Memory()
                self.discriminator = Discriminator()
                self.cfg = SimpleNamespace(
                    mixed_precision=False,
                    task_reward_scale=2.0,
                    style_reward_scale=3.0,
                )
                self._device_type = "cpu"
                self._amp_observation_preprocessor = lambda value: value
                self.tracked = {}
                self.updates = []

            def track_data(self, name: str, value: float) -> None:
                self.tracked[name] = value

            def update(self, *, timestep: int, timesteps: int) -> None:
                self.updates.append((timestep, timesteps))

        agent = Agent()
        AMP_REWARD_DIAGNOSTICS.install_amp_reward_diagnostics(
            agent,
            maximum_style_fraction=0.4,
            minimum_style_reward_scale=0.5,
        )
        AMP_REWARD_DIAGNOSTICS.install_amp_reward_diagnostics(agent)
        agent.update(timestep=7, timesteps=11)

        self.assertEqual(agent.updates, [(7, 11)])
        self.assertEqual(agent.discriminator.role, "discriminator")
        self.assertIn("Reward / AMP style fraction", agent.tracked)
        self.assertIn("Reward / AMP effective style scale", agent.tracked)
        self.assertEqual(agent.cfg.style_reward_scale, 3.0)
        self.assertAlmostEqual(
            agent.tracked["Reward / AMP style contribution (mean)"],
            agent.tracked["Reward / AMP effective style scale"] * math.log(2.0),
            places=6,
        )


if __name__ == "__main__":
    unittest.main()

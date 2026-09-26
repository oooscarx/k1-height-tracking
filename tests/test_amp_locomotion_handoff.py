from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import torch

MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "booster_train"
    / "booster_train"
    / "tasks"
    / "manager_based"
    / "fall_recovery"
    / "amp_locomotion.py"
)
SPEC = importlib.util.spec_from_file_location("amp_locomotion", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
AMP_LOCOMOTION = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AMP_LOCOMOTION)


class AmpLocomotionHandoffTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config, cls.config_path = AMP_LOCOMOTION.load_amp_locomotion_config()
        cls.model_path = (
            cls.config_path.parent
            / cls.config["model_files"]["forward"]
        )

    def test_native_model_tensor_contract(self) -> None:
        tensors = AMP_LOCOMOTION.read_native_model(
            self.model_path,
            self.config["model_sha256"]["forward"],
        )
        self.assertEqual(
            {name: value.shape for name, value in tensors.items()},
            AMP_LOCOMOTION.EXPECTED_TENSORS,
        )

    def test_runtime_arrays_follow_explicit_joint_order(self) -> None:
        self.assertEqual(len(self.config["joint_names"]), 22)
        self.assertEqual(self.config["body_joint_indexes"], list(range(2, 22)))
        for name in (
            "default_dof_position",
            "stiffness",
            "damping",
            "maximum_torque",
        ):
            with self.subTest(array=name):
                self.assertEqual(len(self.config[name]), len(self.config["joint_names"]))

    def test_torch_forward_matches_recovered_zero_state_golden(self) -> None:
        actor = AMP_LOCOMOTION.TorchNativeAmpActor(
            self.model_path,
            self.config["model_sha256"]["forward"],
            device="cpu",
            dtype=torch.float64,
        )
        policy = AMP_LOCOMOTION.TorchAmpZeroCommandPolicy(
            self.config,
            actor,
            num_envs=1,
            device="cpu",
            dtype=torch.float64,
        )
        env_ids = torch.tensor([0], dtype=torch.long)
        position = torch.tensor(
            [self.config["default_dof_position"]],
            dtype=torch.float64,
        )
        action, target = policy.step(
            env_ids,
            angular_velocity=torch.zeros((1, 3), dtype=torch.float64),
            projected_gravity=torch.tensor(
                [[0.0, 0.0, -1.0]],
                dtype=torch.float64,
            ),
            joint_position=position,
            joint_velocity=torch.zeros((1, 22), dtype=torch.float64),
        )
        expected_action = torch.tensor(
            [
                [
                    -0.541291440733,
                    -0.707671920851,
                    -0.020978292349,
                    -0.003128423207,
                    0.013956573048,
                    0.350129519083,
                    0.112367565636,
                    -0.101945062496,
                    0.443195458731,
                    0.480252083983,
                    -0.009348885315,
                    -0.075716502950,
                    0.050785077163,
                    -0.312588720018,
                    0.056535132931,
                    0.020701734185,
                    0.299935292595,
                    -0.082492212300,
                    0.081247865009,
                    0.104234587463,
                ]
            ],
            dtype=torch.float64,
        )
        expected_target = torch.tensor(
            [
                [
                    0.091741711853,
                    -1.247208685390,
                    0.088639091746,
                    -0.489842984567,
                    0.058465615830,
                    1.320025903817,
                    0.056050416797,
                    0.437482255996,
                    -0.154195658470,
                    0.022473513127,
                    -0.001869777063,
                    0.311307026586,
                    -0.090012941481,
                    0.016249573002,
                    -0.150625684641,
                    -0.020389012499,
                    -0.015143300590,
                    0.304140346837,
                    -0.166498442460,
                    0.020846917493,
                ]
            ],
            dtype=torch.float64,
        )
        torch.testing.assert_close(action, expected_action, atol=1.0e-9, rtol=0.0)
        torch.testing.assert_close(target, expected_target, atol=1.0e-9, rtol=0.0)
        self.assertTrue(bool(torch.all(policy.history[0] == policy.history[0, 0])))

    def test_reset_restores_native_process_state(self) -> None:
        actor = AMP_LOCOMOTION.TorchNativeAmpActor(
            self.model_path,
            self.config["model_sha256"]["forward"],
            device="cpu",
            dtype=torch.float32,
        )
        policy = AMP_LOCOMOTION.TorchAmpZeroCommandPolicy(
            self.config,
            actor,
            num_envs=2,
            device="cpu",
        )
        policy.previous_action[:] = 1.0
        policy.filtered_target[:] = 2.0
        policy.update_count[:] = 3
        policy.reset(torch.tensor([1]))
        self.assertTrue(bool(torch.all(policy.previous_action[1] == 0.0)))
        self.assertTrue(bool(torch.all(policy.command_value[1] == 0.0)))
        torch.testing.assert_close(
            policy.filtered_target[1],
            policy.default_position[2:],
        )
        self.assertEqual(int(policy.update_count[1]), 0)
        self.assertEqual(int(policy.update_count[0]), 3)

    def test_forward_command_filter_matches_recovered_slew_and_map(self) -> None:
        class ZeroActor:
            def __call__(self, observation: torch.Tensor) -> torch.Tensor:
                return torch.zeros(
                    (observation.shape[0], 20),
                    dtype=observation.dtype,
                    device=observation.device,
                )

        policy = AMP_LOCOMOTION.TorchAmpZeroCommandPolicy(
            self.config,
            ZeroActor(),
            num_envs=1,
            device="cpu",
            dtype=torch.float64,
        )
        env_ids = torch.tensor([0], dtype=torch.long)
        position = torch.tensor(
            [self.config["default_dof_position"]],
            dtype=torch.float64,
        )
        command = torch.tensor([[0.2, 0.0, 0.0]], dtype=torch.float64)
        for _ in range(8):
            policy.step(
                env_ids,
                angular_velocity=torch.zeros((1, 3), dtype=torch.float64),
                projected_gravity=torch.tensor(
                    [[0.0, 0.0, -1.0]],
                    dtype=torch.float64,
                ),
                joint_position=position,
                joint_velocity=torch.zeros((1, 22), dtype=torch.float64),
                requested_command=command,
                dt=0.02,
            )
        torch.testing.assert_close(
            policy.command_value,
            torch.tensor([[0.208, 0.0, 0.0]], dtype=torch.float64),
            atol=1.0e-12,
            rtol=0.0,
        )
        torch.testing.assert_close(
            policy.history[0, -1, 6:9],
            torch.tensor([0.208, 0.0, 0.0], dtype=torch.float64),
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_turn_isolation_and_forward_mix_match_recovered_filter(self) -> None:
        class ZeroActor:
            def __call__(self, observation: torch.Tensor) -> torch.Tensor:
                return torch.zeros(
                    (observation.shape[0], 20),
                    dtype=observation.dtype,
                    device=observation.device,
                )

        policy = AMP_LOCOMOTION.TorchAmpZeroCommandPolicy(
            self.config,
            ZeroActor(),
            num_envs=2,
            device="cpu",
            dtype=torch.float64,
        )
        env_ids = torch.tensor([0, 1], dtype=torch.long)
        processed = policy._processed_command(
            env_ids,
            torch.tensor(
                [
                    [0.0, 0.0, 0.9],
                    [0.5, 0.9, 0.9],
                ],
                dtype=torch.float64,
            ),
            1.0,
        )
        torch.testing.assert_close(
            processed,
            torch.tensor(
                [
                    [0.0, 0.0, 1.8],
                    [0.825, 0.15, 1.0],
                ],
                dtype=torch.float64,
            ),
            atol=1.0e-12,
            rtol=0.0,
        )


if __name__ == "__main__":
    unittest.main()

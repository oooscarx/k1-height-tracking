from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import yaml

TASK_DIR = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "booster_train"
    / "booster_train"
    / "tasks"
    / "manager_based"
    / "fall_recovery"
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK_DIR))

from amp import (  # noqa: E402
    K1AmpMotionLoader,
    compute_amp_frame,
    quat_multiply,
    quat_rotate,
    quat_slerp_batch,
    sample_reference_clip_ids,
    sample_reference_phase_offsets,
)
from hardware_config import K1_HARDWARE_CONFIG  # noqa: E402
from joint_order import deployment_joint_ids  # noqa: E402
from parallel_ankle import (  # noqa: E402
    K1ParallelAnkleKinematics,
    select_safe_projected_target,
)
from recovery_math import (  # noqa: E402
    bounded_potential_progress,
    held_recovery_completion_bonus,
    joint_limit_violation_mask,
    limit_joint_position_target,
    limit_reset_joint_velocity,
    mastery_curriculum_update,
    mirror_bilateral_joint_position,
    optional_termination_mask,
    quality_weighted_completion_bonus,
    recovery_success_config_at_progress,
    reference_phase_bin_indices,
    safe_terminal_replay_mask,
    sanitize_nonfinite_action_inputs,
    scaled_joint_position_delta,
    stability_hold_progress_reward,
    strict_recovery_success_config,
    strict_threshold_bottleneck_quality,
    training_curriculum_value,
)

WATCHER_PATH = REPOSITORY_ROOT / "scripts" / "skrl" / "evaluate_checkpoint_when_ready.py"
WATCHER_SPEC = importlib.util.spec_from_file_location("evaluate_checkpoint_when_ready", WATCHER_PATH)
assert WATCHER_SPEC is not None and WATCHER_SPEC.loader is not None
WATCHER = importlib.util.module_from_spec(WATCHER_SPEC)
WATCHER_SPEC.loader.exec_module(WATCHER)


class K1HardwareConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = K1_HARDWARE_CONFIG
        cls.kinematics = K1ParallelAnkleKinematics(cls.config["parallel_ankle"])

    def test_raw_inverse_matches_recovered_vendor_golden_values(self) -> None:
        golden = (
            ((0.0, 0.0), (1.3925099507353333, 1.9502069910956918)),
            ((0.13956, 0.0), (1.2674172760234614, 1.832454991209189)),
            ((0.2, 0.0), (1.2136315321116877, 1.7812945839515333)),
            ((0.0, 0.1), (1.3314120381120635, 2.0029989451994448)),
            ((0.0, -0.1), (1.4567809010695811, 1.9018539795098708)),
            ((0.3, 0.2), (0.9842168831478523, 1.8271672750892707)),
            ((-0.2, -0.15), (1.6533085868261228, 2.0562594381544663)),
        )
        for orientation, expected in golden:
            actual, reachable = self.kinematics.raw_inverse(torch.tensor(orientation, dtype=torch.float64))
            self.assertTrue(bool(reachable))
            torch.testing.assert_close(
                actual,
                torch.tensor(expected, dtype=torch.float64),
                atol=2e-14,
                rtol=0.0,
            )

    def test_checkpoint_watcher_can_evaluate_reference_clips_separately(self) -> None:
        self.assertEqual(
            WATCHER.reference_clip_arguments([0, 1]),
            ["--reference_clip_indices", "0", "1"],
        )
        self.assertEqual(WATCHER.reference_clip_arguments(None), [])
        with self.assertRaisesRegex(ValueError, "non-negative"):
            WATCHER.reference_clip_arguments([-1])
        with self.assertRaisesRegex(ValueError, "unique"):
            WATCHER.reference_clip_arguments([0, 0])

    def test_right_ankle_roll_is_mirrored(self) -> None:
        serial = torch.tensor([[-0.15, 0.08]], dtype=torch.float32)
        left, left_ok = self.kinematics.serial_to_motor(serial, 0)
        right_mirrored, right_ok = self.kinematics.serial_to_motor(serial * torch.tensor([[1.0, -1.0]]), 1)
        self.assertTrue(bool(left_ok.item() and right_ok.item()))
        torch.testing.assert_close(left, right_mirrored)

    def test_projection_obeys_all_four_physical_crank_limits(self) -> None:
        pitch = torch.linspace(-0.87, 0.345, 51)
        roll = torch.linspace(-0.345, 0.345, 51)
        desired = torch.cartesian_prod(pitch, roll)
        for foot in (0, 1):
            neutral = self.kinematics.serial_zero[2 * foot : 2 * foot + 2].to(torch.float32).expand_as(desired)
            projected, _, feasible = self.kinematics.project(desired, neutral, foot)
            self.assertTrue(bool(torch.all(feasible)))
            motor, reachable = self.kinematics.serial_to_motor(projected, foot)
            self.assertTrue(bool(torch.all(reachable)))
            minimum = self.kinematics.motor_minimum[2 * foot : 2 * foot + 2]
            maximum = self.kinematics.motor_maximum[2 * foot : 2 * foot + 2]
            self.assertTrue(bool(torch.all(motor >= minimum - 1e-7)))
            self.assertTrue(bool(torch.all(motor <= maximum + 1e-7)))

    def test_reset_projection_preserves_configured_motor_interior(self) -> None:
        amp = self.config["training"]["amp"]
        self.assertGreaterEqual(amp["random_reset_joint_margin"], 0.1)
        self.assertGreaterEqual(
            amp["random_joint_velocity_safety_horizon_s"],
            0.1,
        )
        self.assertGreaterEqual(amp["random_fall_height_range"][0], 0.65)
        margin = float(amp["random_reset_parallel_motor_margin"])
        desired = torch.cartesian_prod(
            torch.linspace(-0.87, 0.345, 51),
            torch.linspace(-0.345, 0.345, 51),
        )
        for foot in (0, 1):
            neutral = self.kinematics.serial_zero[
                2 * foot : 2 * foot + 2
            ].to(torch.float32).expand_as(desired)
            projected, _, feasible = self.kinematics.project(
                desired,
                neutral,
                foot,
                minimum_motor_margin=margin,
            )
            self.assertTrue(bool(torch.all(feasible)))
            self.assertTrue(
                bool(
                    torch.all(
                        self.kinematics.motor_margin(projected, foot)
                        >= margin - 1e-6
                    )
                )
            )

    def test_random_reset_velocity_stays_inside_safe_horizon(self) -> None:
        minimum = torch.tensor([[-1.0, -0.2, -0.5]])
        maximum = torch.tensor([[1.0, 0.8, 0.5]])
        position = torch.tensor([[-0.99, 0.79, 0.0]])
        requested = torch.tensor([[-2.0, 2.0, -1.5]])
        horizon = 0.04
        limited = limit_reset_joint_velocity(
            position,
            requested,
            minimum,
            maximum,
            horizon,
        )
        predicted = position + horizon * limited

        self.assertTrue(bool(torch.all(predicted >= minimum)))
        self.assertTrue(bool(torch.all(predicted <= maximum)))
        self.assertAlmostEqual(float(limited[0, 0]), -0.25, places=5)
        self.assertAlmostEqual(float(limited[0, 1]), 0.25, places=5)
        self.assertAlmostEqual(float(limited[0, 2]), -1.5, places=5)

    def test_command_projection_preserves_configured_motor_margin(self) -> None:
        pitch = torch.linspace(-0.87, 0.345, 31)
        roll = torch.linspace(-0.345, 0.345, 31)
        desired = torch.cartesian_prod(pitch, roll)
        margin = float(self.config["parallel_ankle"]["motor_target_margin"])
        for foot in (0, 1):
            neutral = self.kinematics.serial_zero[2 * foot : 2 * foot + 2].to(torch.float32).expand_as(desired)
            projected, _, feasible = self.kinematics.project(
                desired,
                neutral,
                foot,
                minimum_motor_margin=margin,
            )
            self.assertTrue(bool(torch.all(feasible)))
            motor_margin = self.kinematics.motor_margin(projected, foot)
            self.assertTrue(bool(torch.all(motor_margin >= margin - 1.0e-6)))

    def test_invalid_parallel_projection_uses_known_feasible_target(self) -> None:
        desired = torch.tensor([[float("nan"), 0.2], [0.1, -0.1]])
        projected = torch.tensor([[float("nan"), float("nan")], [0.09, -0.08]])
        reduction = torch.tensor([float("nan"), 0.03])
        feasible = torch.tensor([False, True])
        fallback = torch.tensor([[0.13956, 0.0], [0.13956, 0.0]])

        safe, safe_reduction, accepted = select_safe_projected_target(
            desired,
            projected,
            reduction,
            feasible,
            fallback,
        )

        torch.testing.assert_close(safe[0], fallback[0])
        torch.testing.assert_close(safe[1], projected[1])
        self.assertTrue(bool(torch.all(torch.isfinite(safe_reduction))))
        self.assertEqual(accepted.tolist(), [False, True])

    def test_reference_ankles_can_be_projected_without_fallback_failure(self) -> None:
        for path in sorted((TASK_DIR / "motions").glob("*_reference.npz")):
            with self.subTest(path=path.name), np.load(path) as data:
                joint = torch.tensor(data["joint"], dtype=torch.float32)
                self.assertEqual(joint.shape[1], 22)
                for foot, indexes in enumerate(((14, 15), (20, 21))):
                    desired = joint[:, indexes]
                    self.assertTrue(bool(torch.all(self.kinematics.is_feasible(desired, foot))))
                    neutral = self.kinematics.serial_zero[2 * foot : 2 * foot + 2].to(torch.float32).expand_as(desired)
                    projected, reduction, feasible = self.kinematics.project(desired, neutral, foot)
                    self.assertTrue(bool(torch.all(feasible)))
                    torch.testing.assert_close(projected, desired)
                    torch.testing.assert_close(reduction, torch.zeros_like(reduction))

    def test_deepmimic_amp_assets_are_hardware_feasible(self) -> None:
        minimum = torch.tensor(self.config["position_minimum"], dtype=torch.float32)
        maximum = torch.tensor(self.config["position_maximum"], dtype=torch.float32)
        for path in sorted((TASK_DIR / "motions" / "deepmimic").glob("*.npz")):
            with self.subTest(path=path.name), np.load(path) as data:
                source_names = data["joint_names"].tolist()
                indexes = [source_names.index(name) for name in self.config["joint_names"]]
                joint = torch.tensor(data["joint_pos"][:, indexes], dtype=torch.float32)
                self.assertTrue(bool(torch.all(joint >= minimum - 1.0e-6)))
                self.assertTrue(bool(torch.all(joint <= maximum + 1.0e-6)))
                for foot, pair in enumerate(((14, 15), (20, 21))):
                    self.assertTrue(bool(torch.all(self.kinematics.is_feasible(joint[:, pair], foot))))

    def test_amp_motion_loader_excludes_retargeted_arms(self) -> None:
        amp = self.config["training"]["amp"]
        self.assertEqual(amp["key_body_names"], ["left_foot_link", "right_foot_link"])
        self.assertTrue(
            all("Arm" not in name and "Shoulder" not in name and "Elbow" not in name for name in amp["joint_names"])
        )
        paths = sorted((TASK_DIR / "motions" / "deepmimic").glob("*.npz"))
        loader = K1AmpMotionLoader(
            [str(path) for path in paths],
            self.config["joint_names"],
            amp["joint_names"],
            amp["root_body_name"],
            tuple(amp["key_body_names"]),
            history_length=amp["history_length"],
            expected_fps=50.0,
            device="cpu",
        )
        self.assertEqual(loader.amp_frame_size, 43)
        self.assertEqual(loader.sample_reference_observations(32).shape, (32, 86))

    def test_unified_amp_pool_uses_native_faceup_and_deepmimic_facedown(self) -> None:
        amp = self.config["training"]["amp"]
        native = self.config["native_teacher"]
        directional = amp["directional"]
        faceup = TASK_DIR / "motions" / native["faceup_teacher_rollout_file"]
        facedown = (
            TASK_DIR
            / "motions"
            / "deepmimic"
            / directional["facedown_motion_file"]
        )
        loader = K1AmpMotionLoader(
            [str(faceup), str(facedown)],
            self.config["joint_names"],
            amp["joint_names"],
            amp["root_body_name"],
            tuple(amp["key_body_names"]),
            history_length=amp["history_length"],
            expected_fps=50.0,
            device="cpu",
        )

        self.assertEqual(loader.clip_start_indexes.numel(), 2)
        self.assertIsNone(loader.teacher_action)
        faceup_frames = int(loader.clip_end_indexes[0] - loader.clip_start_indexes[0] + 1)
        self.assertTrue(bool(torch.all(loader.teacher_target_mask[:faceup_frames])))
        self.assertFalse(bool(torch.any(loader.teacher_target_mask[faceup_frames:])))
        with np.load(faceup) as data:
            source_names = data["joint_names"].tolist()
            indexes = [source_names.index(name) for name in self.config["joint_names"]]
            expected_target = torch.tensor(
                data["teacher_target"][:, indexes], dtype=torch.float32
            )
        torch.testing.assert_close(
            loader.teacher_target[:faceup_frames].cpu(), expected_target
        )
        self.assertEqual(loader.amp_frame_size, 43)
        self.assertEqual(loader.sample_reference_observations(32).shape, (32, 86))

        faceup_style = TASK_DIR / "motions" / native["task_driven_training"][
            "amp_style_motion_file"
        ]
        style_loader = K1AmpMotionLoader(
            [str(faceup_style), str(facedown)],
            self.config["joint_names"],
            amp["joint_names"],
            amp["root_body_name"],
            tuple(amp["key_body_names"]),
            history_length=amp["history_length"],
            expected_fps=50.0,
            device="cpu",
        )
        self.assertEqual(style_loader.clip_start_indexes.numel(), 2)
        self.assertIsNone(style_loader.teacher_action)
        self.assertEqual(style_loader.sample_reference_observations(32).shape, (32, 86))

        env_cfg = (TASK_DIR / "robots" / "k1" / "env_cfg.py").read_text(
            encoding="utf-8"
        )
        unified_block = env_cfg.split(
            "class K1FallRecoveryAmpTaskDrivenEnvCfg",
            maxsplit=1,
        )[1]
        self.assertIn('NATIVE_TEACHER["faceup_teacher_rollout_file"]', unified_block)
        self.assertIn('NATIVE_TASK["amp_style_motion_file"]', unified_block)
        self.assertGreaterEqual(
            unified_block.count('AMP_DIRECTIONAL["facedown_motion_file"]'),
            2,
        )

    def test_directional_amp_uses_single_reviewed_reference(self) -> None:
        amp = self.config["training"]["amp"]
        directional = amp["directional"]
        self.assertEqual(directional["joint_names"], amp["joint_names"])
        self.assertEqual(directional["key_body_names"], ["left_foot_link", "right_foot_link"])
        self.assertEqual(directional["reference_reset_height_offset"], 0.0)
        self.assertEqual(directional["reference_reset_joint_velocity_scale"], 0.0)
        self.assertGreater(directional["terminal_reset_probability"], 0.0)
        self.assertLessEqual(directional["terminal_reset_probability"], 1.0)
        self.assertGreater(directional["terminal_reset_window_s"], 0.0)
        self.assertTrue(directional["canonical_from_reference"])
        self.assertGreater(directional["canonical_joint_noise"], 0.0)
        self.assertGreater(directional["canonical_orientation_noise"], 0.0)
        self.assertEqual(directional["canonical_velocity_scale"], 0.0)
        self.assertAlmostEqual(
            directional["reference_reset_probability_start"]
            + directional["standing_reset_probability_start"],
            0.8,
        )
        self.assertAlmostEqual(
            directional["reference_reset_probability_end"]
            + directional["standing_reset_probability_end"],
            0.5,
        )
        self.assertGreater(directional["unsafe_termination_weight"], 0.0)
        self.assertGreater(directional["parallel_stress_weight"], 0.2)
        for direction in ("faceup", "facedown"):
            path = TASK_DIR / "motions" / "deepmimic" / directional[f"{direction}_motion_file"]
            loader = K1AmpMotionLoader(
                [str(path)],
                self.config["joint_names"],
                directional["joint_names"],
                amp["root_body_name"],
                tuple(directional["key_body_names"]),
                history_length=amp["history_length"],
                expected_fps=50.0,
                device="cpu",
            )
            self.assertEqual(loader.amp_frame_size, 43)
            self.assertEqual(loader.sample_reference_observations(32).shape, (32, 86))
            start = loader.clip_start_indexes
            end = loader.clip_end_indexes
            self.assertLessEqual(
                round(directional["terminal_reset_window_s"] * 50.0),
                int((end - start + 1).item()),
            )
            torch.testing.assert_close(loader.phase_at(start), torch.zeros_like(start, dtype=torch.float32))
            torch.testing.assert_close(loader.phase_at(end), torch.ones_like(end, dtype=torch.float32))
            torch.testing.assert_close(loader.advance_indexes(end), end)
            torch.testing.assert_close(loader.advance_indexes(start), start + 1)
            future = loader.advance_indexes(start)
            self.assertEqual(loader.amp_frame[future].shape, (1, 43))
            self.assertTrue(bool(torch.all(loader.phase_at(future) > loader.phase_at(start))))

    def test_task_driven_actor_cannot_observe_reference_or_phase(self) -> None:
        env_cfg = (
            TASK_DIR / "robots" / "k1" / "env_cfg.py"
        ).read_text(encoding="utf-8")
        observations = (TASK_DIR / "observations.py").read_text(encoding="utf-8")
        task_driven_block = env_cfg.split(
            "class AmpTaskDrivenObservationsCfg",
            maxsplit=1,
        )[1].split(
            "class K1FallRecoveryAmpEnvCfg",
            maxsplit=1,
        )[0]

        self.assertEqual(task_driven_block.count("func=mdp.zero_amp_future_reference,"), 2)
        self.assertEqual(task_driven_block.count("func=mdp.zero_amp_future_reference_phase"), 2)
        self.assertNotIn("func=mdp.amp_future_reference)", task_driven_block)
        self.assertNotIn("func=mdp.amp_future_reference_phase)", task_driven_block)
        self.assertIn("return torch.zeros((env.num_envs, frame_size)", observations)
        self.assertIn("return torch.zeros_like(amp_future_reference_phase(env))", observations)

    def test_random_fall_curriculum_expands_to_hard_joint_states(self) -> None:
        amp = self.config["training"]["amp"]
        self.assertEqual(amp["random_fall_difficulty_start"], 0.0)
        self.assertEqual(amp["random_fall_difficulty_end"], 1.0)
        for prefix in (
            "random_fall_linear_velocity",
            "random_fall_angular_velocity",
            "random_joint_noise",
            "random_ankle_joint_noise",
            "random_joint_velocity",
            "random_ankle_joint_velocity",
        ):
            self.assertGreaterEqual(amp[f"{prefix}_start"], 0.0)
            self.assertGreater(amp[f"{prefix}_end"], amp[f"{prefix}_start"])
        self.assertEqual(amp["random_joint_uniform_blend_start"], 0.0)
        self.assertGreaterEqual(amp["random_joint_uniform_blend_end"], 0.6)
        self.assertGreaterEqual(amp["random_joint_noise_end"], 0.5)
        self.assertGreaterEqual(amp["random_joint_velocity_end"], 2.0)

    def test_native_robust_failure_states_are_hardware_feasible(self) -> None:
        robust = self.config["native_teacher"]["robust_training"]
        for stage in ("start", "end"):
            self.assertLessEqual(
                robust["reference_reset_probability"]
                + robust[f"failure_reset_probability_{stage}"],
                1.0,
            )
            self.assertGreaterEqual(robust[f"failure_reset_probability_{stage}"], 0.0)
            self.assertGreaterEqual(robust[f"failure_state_blend_{stage}"], 0.0)
            self.assertLessEqual(robust[f"failure_state_blend_{stage}"], 1.0)
            self.assertGreaterEqual(robust[f"canonical_noise_scale_{stage}"], 0.0)
            self.assertLessEqual(robust[f"canonical_noise_scale_{stage}"], 1.0)
        self.assertGreater(robust["failure_reset_probability_end"], 0.0)
        self.assertGreater(robust["reset_curriculum_steps"], 0)
        self.assertGreaterEqual(
            robust["joint_state_tolerance_start"],
            robust["joint_state_tolerance_end"],
        )
        self.assertGreaterEqual(
            robust["parallel_state_tolerance_start"],
            robust["parallel_state_tolerance_end"],
        )
        self.assertGreaterEqual(robust["position_target_margin"], 0.05)
        self.assertGreater(robust["position_braking_horizon_s"], 0.0)
        path = TASK_DIR / "motions" / robust["failure_state_file"]
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), robust["failure_state_sha256"])
        with np.load(path, allow_pickle=False) as data:
            self.assertEqual(
                set(data.files),
                {
                    "joint_position",
                    "joint_velocity",
                    "root_pose",
                    "root_velocity",
                    "termination_code",
                    "mode",
                },
            )
            sample_count = data["joint_position"].shape[0]
            self.assertGreater(sample_count, 0)
            self.assertEqual(data["joint_position"].shape, (sample_count, 22))
            self.assertEqual(data["joint_velocity"].shape, (sample_count, 22))
            self.assertEqual(data["root_pose"].shape, (sample_count, 7))
            self.assertEqual(data["root_velocity"].shape, (sample_count, 6))
            self.assertTrue(bool(np.all(data["mode"] == "faceup")))
            self.assertTrue(bool(np.all(np.isin(data["termination_code"], (1, 2, 3)))))
            np.testing.assert_allclose(
                np.linalg.norm(data["root_pose"][:, 3:], axis=-1),
                np.ones(sample_count),
                atol=1.0e-4,
                rtol=0.0,
            )
            joint = torch.tensor(data["joint_position"], dtype=torch.float32)
            minimum = torch.tensor(self.config["position_minimum"], dtype=torch.float32)
            maximum = torch.tensor(self.config["position_maximum"], dtype=torch.float32)
            self.assertTrue(bool(torch.all(joint >= minimum)))
            self.assertTrue(bool(torch.all(joint <= maximum)))
            for foot, pair in enumerate(((14, 15), (20, 21))):
                self.assertTrue(bool(torch.all(self.kinematics.is_feasible(joint[:, pair], foot))))

    def test_task_driven_curriculum_waits_for_repeated_mastery(self) -> None:
        task = self.config["native_teacher"]["task_driven_training"]
        self.assertLessEqual(
            task["reference_reset_probability"] + task["failure_reset_probability"],
            1.0,
        )
        self.assertGreaterEqual(task["adaptive_min_trials"], 512)
        self.assertGreaterEqual(task["adaptive_promotion_windows"], 5)
        self.assertLessEqual(task["adaptive_promotion_step"], 0.025)
        self.assertGreater(task["early_success_weight"], task["progress_weight"])
        self.assertGreaterEqual(task["standing_pose_max_error_std"], 0.30)
        self.assertGreater(task["standing_pose_mean_error_weight"], 0.0)
        self.assertGreaterEqual(task["standing_pose_mean_error_std"], 0.30)
        self.assertGreater(task["standing_pose_progress_maximum_rate"], 0.0)
        self.assertGreater(task["standing_pose_max_error_hold_weight"], 0.0)
        self.assertGreater(task["stability_hold_progress_weight"], 0.0)
        env_source = (TASK_DIR / "robots" / "k1" / "env_cfg.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            "func=mdp.recovery_stability_hold_progress_curriculum",
            env_source,
        )
        self.assertIn(
            "func=mdp.standing_pose_max_error_after_height_upright_exp",
            env_source,
        )
        self.assertGreaterEqual(task["joint_limit_weight"], 20.0)
        self.assertGreaterEqual(task["joint_limit_soft_margin"], 0.02)
        self.assertGreaterEqual(task["joint_limit_target_reduction_weight"], 20.0)
        self.assertGreaterEqual(
            task["early_success_strict_quality_minimum_scale"],
            1.0,
        )
        self.assertGreater(
            task["early_success_strict_quality_maximum_scale"],
            task["early_success_strict_quality_minimum_scale"],
        )
        self.assertGreaterEqual(task["coupling_fraction"], 0.5)
        self.assertGreater(task["episode_length_s"], 4.0)
        self.assertLessEqual(task["failure_state_blend_start"], 0.25)
        self.assertLessEqual(task["failure_state_blend_start"], task["failure_state_blend_end"])
        self.assertGreater(task["handoff_curriculum_steps"], 0)
        self.assertGreaterEqual(task["handoff_curriculum_window_steps"], 1000)
        self.assertGreaterEqual(task["handoff_curriculum_minimum_trials"], 4096)
        self.assertGreaterEqual(task["handoff_curriculum_promotion_windows"], 2)
        self.assertGreaterEqual(task["handoff_curriculum_demotion_windows"], 3)
        self.assertLessEqual(task["handoff_curriculum_promotion_step"], 0.025)
        self.assertLess(
            task["handoff_curriculum_demotion_threshold"],
            task["handoff_curriculum_success_threshold"],
        )
        self.assertLessEqual(
            task["handoff_curriculum_maximum_parallel_state_violation_fraction"],
            0.001,
        )
        self.assertEqual(
            task["handoff_curriculum_maximum_hard_joint_limit_fraction"],
            0.0,
        )
        self.assertLessEqual(
            task["handoff_curriculum_maximum_joint_limit_termination_rate"],
            0.001,
        )
        self.assertLessEqual(
            task["handoff_curriculum_maximum_parallel_ankle_termination_rate"],
            0.001,
        )
        env_cfg_source = (
            TASK_DIR / "robots" / "k1" / "env_cfg.py"
        ).read_text(encoding="utf-8")
        task_rewards = env_cfg_source.split(
            "class AmpNativeTaskDrivenRewardsCfg",
            maxsplit=1,
        )[1].split("class AmpNativeRobustTerminationsCfg", maxsplit=1)[0]
        self.assertIn("hardware_joint_limit_violation", task_rewards)
        self.assertIn('NATIVE_TASK["joint_limit_soft_margin"]', task_rewards)
        recovery_time = task_rewards.split("recovery_time =", maxsplit=1)[1].split(
            "early_success =",
            maxsplit=1,
        )[0]
        early_success = task_rewards.split("early_success =", maxsplit=1)[1]
        self.assertNotIn("minimum_strict_quality_scale", recovery_time)
        self.assertIn("minimum_strict_quality_scale", early_success)
        self.assertIn("maximum_strict_quality_scale", early_success)
        standing_pose_rewards = task_rewards.split(
            "standing_pose =",
            maxsplit=1,
        )[1].split("stillness =", maxsplit=1)[0]
        self.assertEqual(
            standing_pose_rewards.count(
                '"minimum_height": TRAINING["success_height"]'
            ),
            3,
        )
        self.assertIn(
            "standing_pose_max_error_progress_after_height_upright_exp",
            standing_pose_rewards,
        )
        self.assertIn(
            "standing_pose_progress_after_height_upright_exp",
            standing_pose_rewards,
        )
        for start_key, strict_key in (
            ("success_angular_velocity_start", "success_angular_velocity"),
            ("success_linear_velocity_start", "success_linear_velocity"),
            ("success_body_pose_error_start", "success_body_pose_error"),
            ("success_body_joint_velocity_start", "success_body_joint_velocity"),
        ):
            self.assertGreaterEqual(task[start_key], self.config["training"][strict_key])
        style_path = TASK_DIR / "motions" / task["amp_style_motion_file"]
        with np.load(style_path, allow_pickle=False) as style:
            self.assertLess(style["joint_pos"].shape[0], 250)
            self.assertEqual(style["trajectory_index"][0], 0)
            self.assertGreater(style["trajectory_index"][-1], 0)

    def test_standing_evaluation_uses_an_explicit_amp_reset_mode(self) -> None:
        train_source = (REPOSITORY_ROOT / "scripts" / "skrl" / "train.py").read_text(
            encoding="utf-8"
        )
        amp_env_source = (TASK_DIR / "robots" / "k1" / "amp_env.py").read_text(
            encoding="utf-8"
        )
        evaluate_source = (REPOSITORY_ROOT / "scripts" / "skrl" / "evaluate.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("runner.agent._observation_preprocessor", train_source)
        self.assertIn('elif reset_mode == "standing":', amp_env_source)
        self.assertIn('"standing",', evaluate_source)

    def test_handoff_evaluator_separates_entry_and_locomotion_posture(self) -> None:
        source = (
            REPOSITORY_ROOT / "scripts" / "skrl" / "evaluate_handoff.py"
        ).read_text(encoding="utf-8")
        self.assertIn("validating & locomotion_posture", source)
        self.assertNotIn("validating & gate", source)
        self.assertIn("recovery_success_config_at_progress", source)

    def test_parallel_projection_fallback_is_reported_separately(self) -> None:
        amp_env_source = (
            TASK_DIR / "robots" / "k1" / "amp_env.py"
        ).read_text(encoding="utf-8")
        evaluator_source = (
            REPOSITORY_ROOT / "scripts" / "skrl" / "evaluate.py"
        ).read_text(encoding="utf-8")
        self.assertIn("Metrics/parallel_target_fallback_fraction", amp_env_source)
        self.assertIn("parallel_target_fallback_terminations", evaluator_source)

    def test_nonfinite_policy_environment_is_isolated_and_reported(self) -> None:
        actions = torch.tensor([[0.2, -0.1], [float("nan"), 0.4]])
        position = torch.tensor([[0.3, -0.2], [0.1, 0.2]])
        velocity = torch.tensor([[0.5, -0.5], [0.3, 0.4]])
        center = torch.zeros(1, 2)
        safe_actions, safe_position, safe_velocity, safe_center, failed = (
            sanitize_nonfinite_action_inputs(
                actions,
                position,
                velocity,
                center,
                center,
            )
        )
        torch.testing.assert_close(safe_actions[0], actions[0])
        torch.testing.assert_close(safe_actions[1], torch.zeros(2))
        torch.testing.assert_close(safe_position, position)
        torch.testing.assert_close(safe_velocity, velocity)
        torch.testing.assert_close(safe_center, center.expand(2, -1))
        torch.testing.assert_close(failed, torch.tensor([False, True]))

        env_source = (TASK_DIR / "robots" / "k1" / "env_cfg.py").read_text(
            encoding="utf-8"
        )
        amp_env_source = (TASK_DIR / "robots" / "k1" / "amp_env.py").read_text(
            encoding="utf-8"
        )
        evaluator_source = (
            REPOSITORY_ROOT / "scripts" / "skrl" / "evaluate.py"
        ).read_text(encoding="utf-8")
        train_source = (
            REPOSITORY_ROOT / "scripts" / "skrl" / "train.py"
        ).read_text(encoding="utf-8")
        self.assertIn("nonfinite_action = DoneTerm", env_source)
        self.assertIn("Metrics/nonfinite_action_fraction", amp_env_source)
        self.assertIn("nonfinite_action_terminations", evaluator_source)
        self.assertIn("--failure_states_terminal_only", evaluator_source)
        self.assertIn("failure_mode.extend([mode]", evaluator_source)
        self.assertIn("install_nonfinite_rollout_isolation(runner.agent, raw_env)", train_source)
        self.assertIn("configure_normalized_policy_action_bounds(runner.agent)", train_source)
        self.assertIn("configure_normalized_policy_action_bounds(runner.agent)", evaluator_source)

    def test_handoff_curriculum_is_strict_during_evaluation(self) -> None:
        common = {
            "num_envs": 100,
            "start": 1.2,
            "end": 0.5,
            "curriculum_steps": 1000,
        }
        self.assertEqual(
            training_curriculum_value(
                common_step_counter=0,
                reset_mode="train",
                **common,
            ),
            1.2,
        )
        self.assertAlmostEqual(
            training_curriculum_value(
                common_step_counter=5,
                reset_mode="train",
                **common,
            ),
            0.85,
        )
        self.assertEqual(
            training_curriculum_value(
                common_step_counter=20,
                reset_mode="train",
                **common,
            ),
            0.5,
        )
        self.assertEqual(
            training_curriculum_value(
                common_step_counter=0,
                reset_mode="reference",
                **common,
            ),
            0.5,
        )

    def test_task_driven_policy_keeps_recovery_action_noise_bounded(self) -> None:
        path = TASK_DIR / "robots" / "k1" / "agents" / "skrl_amp_native_task_driven_cfg.yaml"
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        policy = config["models"]["policy"]
        self.assertTrue(policy["clip_log_std"])
        self.assertEqual(policy["min_log_std"], -4.6)
        self.assertEqual(policy["max_log_std"], -4.0)
        self.assertEqual(policy["initial_log_std"], -4.0)

    def test_mastery_curriculum_waits_promotes_and_demotes(self) -> None:
        common = {
            "promotion_threshold": 0.85,
            "demotion_threshold": 0.50,
            "promotion_windows": 2,
            "demotion_windows": 3,
            "promotion_step": 0.05,
            "demotion_step": 0.025,
        }
        progress, promotion_streak, demotion_streak = mastery_curriculum_update(
            progress=0.0,
            promotion_streak=0,
            demotion_streak=0,
            success_rate=0.90,
            **common,
        )
        self.assertEqual((progress, promotion_streak, demotion_streak), (0.0, 1, 0))
        progress, promotion_streak, demotion_streak = mastery_curriculum_update(
            progress=progress,
            promotion_streak=promotion_streak,
            demotion_streak=demotion_streak,
            success_rate=0.90,
            **common,
        )
        self.assertEqual((progress, promotion_streak, demotion_streak), (0.05, 0, 0))
        progress, promotion_streak, demotion_streak = mastery_curriculum_update(
            progress=progress,
            promotion_streak=1,
            demotion_streak=0,
            success_rate=0.70,
            **common,
        )
        self.assertEqual((progress, promotion_streak, demotion_streak), (0.05, 0, 0))
        for expected_demotion_streak in (1, 2, 0):
            progress, promotion_streak, demotion_streak = mastery_curriculum_update(
                progress=progress,
                promotion_streak=1,
                demotion_streak=demotion_streak,
                success_rate=0.40,
                **common,
            )
            expected_progress = 0.025 if expected_demotion_streak == 0 else 0.05
            self.assertEqual(
                (progress, promotion_streak, demotion_streak),
                (expected_progress, 0, expected_demotion_streak),
            )

    def test_mastery_curriculum_blocks_promotion_for_an_unsafe_window(self) -> None:
        common = {
            "promotion_threshold": 0.85,
            "demotion_threshold": 0.50,
            "promotion_windows": 2,
            "demotion_windows": 3,
            "promotion_step": 0.05,
            "demotion_step": 0.025,
        }
        progress, promotion_streak, demotion_streak = mastery_curriculum_update(
            progress=0.1,
            promotion_streak=1,
            demotion_streak=0,
            success_rate=0.95,
            promotion_allowed=False,
            **common,
        )
        self.assertEqual((progress, promotion_streak, demotion_streak), (0.1, 0, 0))
        progress, promotion_streak, demotion_streak = mastery_curriculum_update(
            progress=progress,
            promotion_streak=promotion_streak,
            demotion_streak=demotion_streak,
            success_rate=0.95,
            promotion_allowed=True,
            **common,
        )
        self.assertEqual((progress, promotion_streak, demotion_streak), (0.1, 1, 0))

    def test_strict_handoff_quality_tracks_the_worst_metric(self) -> None:
        loose = torch.tensor([0.75, 0.15, 0.32, 1.20])
        strict = torch.tensor([0.50, 0.10, 0.15, 0.50])
        midpoint = (loose + strict) / 2.0
        values = torch.stack(
            (
                loose,
                midpoint,
                strict,
                torch.tensor([0.50, 0.10, 0.32, 0.50]),
            )
        )

        quality = strict_threshold_bottleneck_quality(values, loose, strict)

        torch.testing.assert_close(
            quality,
            torch.tensor([0.0, 0.5, 1.0, 0.0]),
        )

    def test_completion_bonus_increases_with_strict_handoff_quality(self) -> None:
        completion = torch.tensor([20.0, 20.0, 20.0])
        quality = torch.tensor([0.0, 0.5, 1.0])

        weighted = quality_weighted_completion_bonus(
            completion,
            quality,
            minimum_scale=1.0,
            maximum_scale=2.0,
        )

        torch.testing.assert_close(weighted, torch.tensor([20.0, 30.0, 40.0]))
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            quality_weighted_completion_bonus(
                torch.ones(2),
                torch.ones(3),
                minimum_scale=1.0,
                maximum_scale=2.0,
            )
        with self.assertRaisesRegex(ValueError, "ordered"):
            quality_weighted_completion_bonus(
                torch.ones(2),
                torch.ones(2),
                minimum_scale=2.0,
                maximum_scale=1.0,
            )

    def test_pose_potential_only_rewards_progress(self) -> None:
        progress = bounded_potential_progress(
            potential=torch.tensor([0.8, 0.5, 0.1]),
            previous_potential=torch.tensor([0.6, 0.5, 0.4]),
            step_dt=0.02,
            maximum_rate=4.0,
        )
        torch.testing.assert_close(progress, torch.tensor([4.0, 0.0, -4.0]))
        with self.assertRaisesRegex(ValueError, "matching shapes"):
            bounded_potential_progress(
                torch.ones(2),
                torch.ones(3),
                step_dt=0.02,
                maximum_rate=4.0,
            )

    def test_evaluators_resolve_handoff_curriculum_to_strict_gates(self) -> None:
        config = strict_recovery_success_config(
            {
                "minimum_height": 0.48,
                "maximum_angular_velocity_start": 0.75,
                "maximum_angular_velocity_end": 0.5,
                "maximum_linear_velocity_start": 0.15,
                "maximum_linear_velocity_end": 0.1,
                "maximum_body_pose_error_start": 0.32,
                "maximum_body_pose_error_end": 0.15,
                "maximum_body_joint_velocity_start": 1.2,
                "maximum_body_joint_velocity_end": 0.5,
                "curriculum_steps": 150_000_000,
            }
        )
        self.assertEqual(config["minimum_height"], 0.48)
        self.assertEqual(config["maximum_angular_velocity"], 0.5)
        self.assertEqual(config["maximum_linear_velocity"], 0.1)
        self.assertEqual(config["maximum_body_pose_error"], 0.15)
        self.assertEqual(config["maximum_body_joint_velocity"], 0.5)
        self.assertNotIn("maximum_body_pose_error_start", config)
        self.assertNotIn("maximum_body_pose_error_end", config)
        self.assertNotIn("curriculum_steps", config)

    def test_evaluator_can_resolve_one_handoff_curriculum_progress(self) -> None:
        config = recovery_success_config_at_progress(
            {
                "minimum_height": 0.48,
                "maximum_angular_velocity_start": 0.75,
                "maximum_angular_velocity_end": 0.5,
                "maximum_linear_velocity_start": 0.15,
                "maximum_linear_velocity_end": 0.1,
                "maximum_body_pose_error_start": 0.32,
                "maximum_body_pose_error_end": 0.15,
                "maximum_body_joint_velocity_start": 1.2,
                "maximum_body_joint_velocity_end": 0.5,
                "curriculum_steps": 150_000_000,
            },
            0.35,
        )
        self.assertEqual(config["minimum_height"], 0.48)
        self.assertAlmostEqual(config["maximum_angular_velocity"], 0.6625)
        self.assertAlmostEqual(config["maximum_linear_velocity"], 0.1325)
        self.assertAlmostEqual(config["maximum_body_pose_error"], 0.2605)
        self.assertAlmostEqual(config["maximum_body_joint_velocity"], 0.955)
        with self.assertRaisesRegex(ValueError, "progress must be in"):
            recovery_success_config_at_progress(config, 1.01)

    def test_completion_bonus_requires_the_full_stability_hold(self) -> None:
        value = held_recovery_completion_bonus(
            success=torch.tensor([True, True, True, False]),
            stable_steps=torch.tensor([14, 15, 20, 20]),
            hold_steps=15,
            episode_length=torch.tensor([20, 20, 20, 20]),
            max_episode_length=300,
        )
        torch.testing.assert_close(value, torch.tensor([0.0, 29.0, 29.0, 0.0]))

    def test_stability_hold_progress_rewards_only_consecutive_in_gate_steps(self) -> None:
        value = stability_hold_progress_reward(
            success=torch.tensor([True, True, True, False]),
            stable_steps=torch.tensor([1, 7, 20, 10]),
            hold_steps=15,
        )
        torch.testing.assert_close(
            value,
            torch.tensor([1.0 / 15.0, 7.0 / 15.0, 1.0, 0.0]),
        )
        with self.assertRaisesRegex(ValueError, "mask must be boolean"):
            stability_hold_progress_reward(torch.ones(1), torch.ones(1), 15)

    def test_recovery_scene_uses_repository_local_collision_ground(self) -> None:
        env_source = (TASK_DIR / "robots" / "k1" / "env_cfg.py").read_text(
            encoding="utf-8"
        )
        ground_path = TASK_DIR / "assets" / "k1_flat_ground.usda"
        ground_source = ground_path.read_text(encoding="utf-8")
        self.assertIn('terrain_type="usd"', env_source)
        self.assertIn('usd_path=str(ASSET_DIR / "k1_flat_ground.usda")', env_source)
        self.assertNotIn("NVIDIA_NUCLEUS_DIR", env_source)
        self.assertIn('prepend apiSchemas = ["PhysicsCollisionAPI"]', ground_source)
        self.assertIn("float physics:staticFriction = 0.8", ground_source)
        self.assertIn("float physics:dynamicFriction = 0.7", ground_source)
        self.assertIn("float physics:restitution = 0.02", ground_source)

    def test_reference_phase_bins_include_both_endpoints(self) -> None:
        phases = torch.tensor([0.0, 0.0999, 0.1, 0.4999, 0.5, 0.9999, 1.0])
        torch.testing.assert_close(
            reference_phase_bin_indices(phases),
            torch.tensor([0, 0, 1, 4, 5, 9, 9]),
        )
        with self.assertRaisesRegex(ValueError, "in \\[0, 1\\]"):
            reference_phase_bin_indices(torch.tensor([-0.01, 0.5]))

    def test_joint_limit_violation_mask_attributes_each_joint(self) -> None:
        position = torch.tensor(
            [
                [-1.051, 0.0, 1.051],
                [-1.049, 0.5, 1.049],
            ]
        )
        violation = joint_limit_violation_mask(
            position,
            torch.tensor([-1.0, -0.5, -1.0]),
            torch.tensor([1.0, 0.5, 1.0]),
            margin=0.05,
        )
        torch.testing.assert_close(
            violation,
            torch.tensor(
                [
                    [True, False, True],
                    [False, False, False],
                ]
            ),
        )
        with self.assertRaisesRegex(ValueError, "shape"):
            joint_limit_violation_mask(
                position[0],
                torch.tensor([-1.0, -0.5, -1.0]),
                torch.tensor([1.0, 0.5, 1.0]),
                margin=0.05,
            )

    def test_joint_target_limit_brakes_outward_motion(self) -> None:
        target = torch.tensor([[0.95, -0.95, 0.0]])
        current = torch.tensor([[0.85, -0.85, 0.0]])
        velocity = torch.tensor([[2.0, -2.0, 0.0]])
        limited, braking = limit_joint_position_target(
            target,
            current,
            velocity,
            torch.tensor([[-0.92, -0.92, -0.92]]),
            torch.tensor([[0.92, 0.92, 0.92]]),
            braking_horizon_s=0.10,
        )
        torch.testing.assert_close(limited, torch.tensor([[0.65, -0.65, 0.0]]))
        torch.testing.assert_close(
            braking,
            torch.tensor([[True, True, False]]),
        )

    def test_wbc_action_scales_before_clipping_joint_delta(self) -> None:
        actions = torch.tensor([[15.0, -15.0, 0.5]])
        scale = torch.full((1, 3), 0.1)

        wbc_delta = scaled_joint_position_delta(
            actions,
            scale,
            normalize_input=False,
            delta_minimum=-1.0,
            delta_maximum=1.0,
        )
        normalized_delta = scaled_joint_position_delta(
            actions,
            scale,
            normalize_input=True,
        )

        torch.testing.assert_close(
            wbc_delta,
            torch.tensor([[1.0, -1.0, 0.05]]),
        )
        torch.testing.assert_close(
            normalized_delta,
            torch.tensor([[0.1, -0.1, 0.05]]),
        )

    def test_weighted_reference_sampler_targets_hard_stratum(self) -> None:
        clip_ids = sample_reference_clip_ids(
            clip_count=2,
            sample_count=64,
            device="cpu",
            clip_weights=[0.0, 1.0],
        )
        torch.testing.assert_close(clip_ids, torch.ones(64, dtype=torch.long))

        early_offsets = sample_reference_phase_offsets(
            torch.tensor([100] * 64),
            0.7,
            0.9,
            phase_bin_edges=[0.7, 0.8, 0.9],
            phase_bin_weights=[1.0, 0.0],
        )
        self.assertTrue(
            bool(torch.all((early_offsets >= 70) & (early_offsets < 80)))
        )
        late_offsets = sample_reference_phase_offsets(
            torch.tensor([100] * 64),
            0.7,
            0.9,
            phase_bin_edges=[0.7, 0.8, 0.9],
            phase_bin_weights=[0.0, 1.0],
        )
        self.assertTrue(
            bool(torch.all((late_offsets >= 80) & (late_offsets <= 90)))
        )

    def test_weighted_reference_sampler_rejects_incomplete_configuration(self) -> None:
        with self.assertRaisesRegex(ValueError, "configured together"):
            sample_reference_phase_offsets(
                torch.tensor([100]),
                0.7,
                0.9,
                phase_bin_edges=[0.7, 0.8, 0.9],
            )
        with self.assertRaisesRegex(ValueError, "complete motion pool"):
            sample_reference_clip_ids(
                clip_count=2,
                sample_count=1,
                device="cpu",
                clip_weights=[1.0],
            )

    def test_terminal_replay_excludes_recovery_and_hardware_failures(self) -> None:
        selected = safe_terminal_replay_mask(
            torch.tensor([True, True, True, True, True, False]),
            torch.tensor([True, True, True, True, False, True]),
            torch.tensor([True, True, True, True, True, True]),
            joint_limit=torch.tensor([False, True, False, False, False, False]),
            parallel_ankle=torch.tensor([False, False, True, False, False, False]),
            nonfinite_action=torch.tensor([False, False, False, True, False, False]),
        )
        torch.testing.assert_close(
            selected,
            torch.tensor([True, False, False, False, False, False]),
        )
        with self.assertRaises(ValueError):
            safe_terminal_replay_mask(
                torch.ones(2, dtype=torch.bool),
                torch.ones(3, dtype=torch.bool),
                torch.ones(2, dtype=torch.bool),
            )

    def test_optional_termination_mask_handles_removed_recovery_term(self) -> None:
        class FakeTerminationManager:
            active_terms = ["time_out"]

            @staticmethod
            def get_term(name: str) -> torch.Tensor:
                if name != "time_out":
                    raise KeyError(name)
                return torch.tensor([[False], [True], [False]])

        manager = FakeTerminationManager()
        env_ids = torch.tensor([2, 1], dtype=torch.long)
        torch.testing.assert_close(
            optional_termination_mask(manager, "recovered", env_ids),
            torch.tensor([False, False]),
        )
        torch.testing.assert_close(
            optional_termination_mask(manager, "time_out", env_ids),
            torch.tensor([False, True]),
        )

    def test_bilateral_joint_mirror_is_an_involution(self) -> None:
        discovery = self.config["training"]["discovery"]
        joint_position = torch.tensor(
            discovery["fallen_joint_position"],
            dtype=torch.float32,
        )
        mirrored = mirror_bilateral_joint_position(
            joint_position,
            discovery["symmetry_left_indices"],
            discovery["symmetry_right_indices"],
            discovery["symmetry_mirror_signs"],
        )
        restored = mirror_bilateral_joint_position(
            mirrored,
            discovery["symmetry_left_indices"],
            discovery["symmetry_right_indices"],
            discovery["symmetry_mirror_signs"],
        )
        torch.testing.assert_close(restored, joint_position)
        self.assertFalse(bool(torch.allclose(mirrored, joint_position)))

    def test_checkpoint_watcher_matches_only_the_python_trainer(self) -> None:
        token = "terminal_bonus_reverse_phase"
        self.assertTrue(
            WATCHER.is_training_command(
                "/usr/bin/python3.11",
                [
                    ".venv/bin/python",
                    "scripts/skrl/train.py",
                    "--experiment_name",
                    token,
                ],
                token,
            )
        )
        self.assertFalse(
            WATCHER.is_training_command(
                "/usr/bin/tmux",
                [
                    ".venv/bin/python",
                    "scripts/skrl/train.py",
                    "--experiment_name",
                    token,
                ],
                token,
            )
        )
        self.assertFalse(
            WATCHER.is_training_command(
                "/usr/bin/bash",
                [
                    ".venv/bin/python",
                    "scripts/skrl/train.py",
                    "--experiment_name",
                    token,
                ],
                token,
            )
        )
        self.assertFalse(
            WATCHER.is_training_command(
                "/usr/bin/tmux",
                [
                    "tmux",
                    "new-session",
                    ".venv/bin/python scripts/skrl/train.py --experiment_name "
                    + token,
                ],
                token,
            )
        )
        self.assertFalse(
            WATCHER.is_training_command(
                "/usr/bin/bash",
                [
                    "bash",
                    "-c",
                    ".venv/bin/python scripts/skrl/train.py --experiment_name "
                    + token,
                ],
                token,
            )
        )

    def test_batched_quaternion_slerp_uses_shortest_normalized_path(self) -> None:
        start = torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
        end = torch.tensor(
            [
                [0.0, 1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
            ]
        )
        midpoint = quat_slerp_batch(start, end, 0.5)
        np.testing.assert_allclose(
            torch.linalg.vector_norm(midpoint, dim=-1).numpy(),
            np.ones(2),
            atol=1.0e-6,
            rtol=0.0,
        )
        torch.testing.assert_close(midpoint[0], torch.tensor([2.0**-0.5, 2.0**-0.5, 0.0, 0.0]))
        torch.testing.assert_close(midpoint[1], torch.tensor([2.0**-0.5, -2.0**-0.5, 0.0, 0.0]))
        per_sample = quat_slerp_batch(start, end, torch.tensor([[0.25], [0.75]]))
        expected = torch.tensor(
            [
                [torch.cos(torch.tensor(torch.pi / 8)), torch.sin(torch.tensor(torch.pi / 8)), 0.0, 0.0],
                [
                    torch.cos(torch.tensor(3 * torch.pi / 8)),
                    -torch.sin(torch.tensor(3 * torch.pi / 8)),
                    0.0,
                    0.0,
                ],
            ]
        )
        torch.testing.assert_close(per_sample, expected)

    def test_amp_frame_is_heading_invariant(self) -> None:
        amp_joint_count = len(self.config["training"]["amp"]["joint_names"])
        joint_position = torch.zeros((1, amp_joint_count))
        joint_velocity = torch.arange(amp_joint_count, dtype=torch.float32).unsqueeze(0) * 0.01
        root_position = torch.tensor([[0.2, -0.3, 0.5]])
        root_rotation = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        root_linear_velocity = torch.tensor([[0.4, -0.2, 0.1]])
        root_angular_velocity = torch.tensor([[0.2, 0.1, -0.3]])
        key_position = torch.tensor([[[0.1, 0.2, 0.0], [0.3, -0.2, 0.0]]]) + root_position.unsqueeze(1)
        original = compute_amp_frame(
            joint_position,
            joint_velocity,
            root_position,
            root_rotation,
            root_linear_velocity,
            root_angular_velocity,
            key_position,
        )

        yaw = torch.tensor([[2.0**-0.5, 0.0, 0.0, 2.0**-0.5]])
        rotated_root = torch.tensor([[1.4, 0.7, 0.5]])
        rotated_key = quat_rotate(yaw.unsqueeze(1).expand(-1, 2, -1), key_position - root_position.unsqueeze(1))
        rotated_key += rotated_root.unsqueeze(1)
        rotated = compute_amp_frame(
            joint_position,
            joint_velocity,
            rotated_root,
            quat_multiply(yaw, root_rotation),
            quat_rotate(yaw, root_linear_velocity),
            quat_rotate(yaw, root_angular_velocity),
            rotated_key,
        )
        torch.testing.assert_close(rotated, original, atol=1.0e-5, rtol=1.0e-5)

    def test_goal_is_inside_logical_and_parallel_limits(self) -> None:
        goal = torch.tensor(self.config["goal_position"], dtype=torch.float32)
        minimum = torch.tensor(self.config["position_minimum"], dtype=torch.float32)
        maximum = torch.tensor(self.config["position_maximum"], dtype=torch.float32)
        self.assertTrue(bool(torch.all(goal >= minimum)))
        self.assertTrue(bool(torch.all(goal <= maximum)))
        for foot, indexes in enumerate(((14, 15), (20, 21))):
            self.assertTrue(bool(self.kinematics.is_feasible(goal[list(indexes)], foot)))

    def test_discovery_actions_are_centered_on_goal_and_stay_inside_limits(self) -> None:
        discovery = self.config["training"]["discovery"]
        goal = torch.tensor(self.config["goal_position"], dtype=torch.float32)
        scale = torch.tensor(discovery["action_scale"], dtype=torch.float32)
        minimum = torch.tensor(self.config["position_minimum"], dtype=torch.float32)
        maximum = torch.tensor(self.config["position_maximum"], dtype=torch.float32)
        action_std = torch.tensor(discovery["action_std"], dtype=torch.float32)
        self.assertEqual(scale.shape, goal.shape)
        self.assertEqual(action_std.shape, goal.shape)
        self.assertTrue(bool(torch.all(goal - scale >= minimum)))
        self.assertTrue(bool(torch.all(goal + scale <= maximum)))
        self.assertTrue(bool(torch.all(action_std > 0.0)))
        self.assertTrue(discovery["freeze_action_std"])
        self.assertEqual(discovery["entropy_coef"], 0.0)
        left = discovery["symmetry_left_indices"]
        right = discovery["symmetry_right_indices"]
        signs = discovery["symmetry_mirror_signs"]
        self.assertEqual(len(left), len(right))
        self.assertEqual(len(left), len(signs))
        self.assertEqual(sorted(left + right), list(range(2, 22)))
        expected_signs = {
            "Shoulder_Pitch": 1.0,
            "Shoulder_Roll": -1.0,
            "Elbow_Pitch": 1.0,
            "Elbow_Yaw": -1.0,
            "Hip_Pitch": 1.0,
            "Hip_Roll": -1.0,
            "Hip_Yaw": -1.0,
            "Knee_Pitch": 1.0,
            "Ankle_Pitch": 1.0,
            "Ankle_Roll": -1.0,
        }
        joint_names = self.config["joint_names"]
        for left_index, right_index, sign in zip(left, right, signs, strict=True):
            left_name = joint_names[left_index].removeprefix("ALeft_").removeprefix("Left_")
            right_name = joint_names[right_index].removeprefix("ARight_").removeprefix("Right_")
            self.assertEqual(left_name, right_name)
            self.assertEqual(sign, expected_signs[left_name])

    def test_discovery_reset_curriculum_and_fallen_pose_are_hardware_feasible(self) -> None:
        discovery = self.config["training"]["discovery"]
        fallen = torch.tensor(discovery["fallen_joint_position"], dtype=torch.float32)
        minimum = torch.tensor(self.config["position_minimum"], dtype=torch.float32)
        maximum = torch.tensor(self.config["position_maximum"], dtype=torch.float32)
        self.assertEqual(fallen.shape, minimum.shape)
        self.assertTrue(bool(torch.all(fallen >= minimum)))
        self.assertTrue(bool(torch.all(fallen <= maximum)))
        self.assertTrue(bool(torch.all(fallen - minimum >= 0.1)))
        self.assertTrue(bool(torch.all(maximum - fallen >= 0.1)))
        self.assertGreater(discovery["standing_reset_fraction_start"], 0.0)
        self.assertGreaterEqual(
            discovery["standing_reset_fraction_start"],
            discovery["standing_reset_fraction_end"],
        )
        self.assertGreater(discovery["standing_reset_curriculum_steps"], 0)
        for foot, indexes in enumerate(((14, 15), (20, 21))):
            desired = fallen[list(indexes)].unsqueeze(0)
            neutral = self.kinematics.serial_zero[2 * foot : 2 * foot + 2].to(torch.float32).unsqueeze(0)
            projected, _, feasible = self.kinematics.project(desired, neutral, foot)
            self.assertTrue(bool(torch.all(feasible)))
            self.assertTrue(bool(torch.all(projected >= minimum[list(indexes)])))
            self.assertTrue(bool(torch.all(projected <= maximum[list(indexes)])))

    def test_goal_matches_real_amp_handoff_contract(self) -> None:
        handoff = self.config["handoff_policy"]
        self.assertEqual(handoff["name"], "amp_locomotion_run")
        self.assertEqual(handoff["body_joint_indexes"], list(range(2, 22)))
        self.assertEqual(handoff["action_size"], 20)
        self.assertEqual(handoff["policy_rate_hz"], self.config["policy_rate_hz"])
        self.assertGreater(handoff["maximum_zero_command_xy_drift_m"], 0.0)
        self.assertLessEqual(handoff["maximum_zero_command_xy_drift_m"], 0.1)
        self.assertEqual(
            handoff["maximum_start_linear_speed_mps"],
            self.config["training"]["success_linear_velocity"],
        )
        self.assertEqual(
            handoff["maximum_start_angular_speed_radps"],
            self.config["training"]["success_angular_velocity"],
        )
        self.assertEqual(
            handoff["maximum_start_body_joint_speed_radps"],
            self.config["training"]["success_body_joint_velocity"],
        )
        self.assertLessEqual(handoff["maximum_start_linear_speed_mps"], 0.1)
        self.assertGreater(handoff["commanded_forward_velocity_mps"], 0.0)
        self.assertGreater(handoff["commanded_validation_s"], 0.0)
        self.assertGreaterEqual(
            handoff["minimum_commanded_forward_progress_m"],
            0.2,
        )
        self.assertLessEqual(
            handoff["maximum_commanded_lateral_drift_m"],
            0.1,
        )
        self.assertEqual(
            self.config["training"]["success_body_pose_error"],
            handoff["maximum_start_pose_error"],
        )
        self.assertEqual(handoff["maximum_start_pose_error"], 0.22)
        self.assertEqual(handoff["engage_ramp_s"], 0.2)
        self.assertEqual(
            self.config["goal_position"],
            [
                0.0,
                0.0,
                0.2,
                -1.25,
                0.0,
                -0.5,
                0.2,
                1.25,
                0.0,
                0.5,
                -0.15,
                0.0,
                0.0,
                0.3,
                -0.15,
                0.0,
                -0.15,
                0.0,
                0.0,
                0.3,
                -0.15,
                0.0,
            ],
        )

    def test_real_amp_artifact_hashes(self) -> None:
        handoff = self.config["handoff_policy"]
        model_directory = REPOSITORY_ROOT / handoff["model_directory"]
        for policy_name, filename in handoff["model_files"].items():
            with self.subTest(policy=policy_name):
                digest = hashlib.sha256((model_directory / filename).read_bytes())
                self.assertEqual(digest.hexdigest(), handoff["model_sha256"][policy_name])

    def test_real_amp_config_uses_deployment_joint_order(self) -> None:
        handoff = self.config["handoff_policy"]
        source_path = REPOSITORY_ROOT / handoff["source_config"]
        source = json.loads(source_path.read_text(encoding="utf-8"))
        self.assertEqual(source["joint_names"], self.config["joint_names"])
        self.assertEqual(source["body_joint_indexes"], list(range(2, 22)))
        for name in (
            "default_dof_position",
            "stiffness",
            "damping",
            "maximum_torque",
        ):
            with self.subTest(array=name):
                self.assertEqual(len(source[name]), len(self.config["joint_names"]))

    def test_deployment_observation_layout_is_contiguous(self) -> None:
        contract = self.config["deployment_contract"]
        cursor = 0
        for term in contract["observation_layout"]:
            self.assertEqual(term["start"], cursor)
            cursor += term["size"]
        self.assertEqual(cursor, contract["observation_size"])
        self.assertEqual(contract["action_size"], 22)

    def test_runtime_simulation_order_maps_to_deployment_order(self) -> None:
        simulation_order = [
            "AAHead_yaw",
            "ALeft_Shoulder_Pitch",
            "ARight_Shoulder_Pitch",
            "Left_Hip_Pitch",
            "Right_Hip_Pitch",
            "Head_pitch",
            "Left_Shoulder_Roll",
            "Right_Shoulder_Roll",
            "Left_Hip_Roll",
            "Right_Hip_Roll",
            "Left_Elbow_Pitch",
            "Right_Elbow_Pitch",
            "Left_Hip_Yaw",
            "Right_Hip_Yaw",
            "Left_Elbow_Yaw",
            "Right_Elbow_Yaw",
            "Left_Knee_Pitch",
            "Right_Knee_Pitch",
            "Left_Ankle_Pitch",
            "Right_Ankle_Pitch",
            "Left_Ankle_Roll",
            "Right_Ankle_Roll",
        ]
        expected_ids = [
            0,
            5,
            1,
            6,
            10,
            14,
            2,
            7,
            11,
            15,
            3,
            8,
            12,
            16,
            18,
            20,
            4,
            9,
            13,
            17,
            19,
            21,
        ]
        actual_ids = deployment_joint_ids(simulation_order, self.config["joint_names"])
        self.assertEqual(actual_ids, expected_ids)
        self.assertEqual(
            [simulation_order[index] for index in actual_ids],
            self.config["joint_names"],
        )


if __name__ == "__main__":
    unittest.main()

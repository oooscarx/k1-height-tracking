from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

TASK_DIR = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "booster_train"
    / "booster_train"
    / "tasks"
    / "manager_based"
    / "fall_recovery"
)
SYMMETRY_PATH = TASK_DIR / "wbc_stand_up" / "symmetry.py"
SPEC = importlib.util.spec_from_file_location("k1_wbc_symmetry", SYMMETRY_PATH)
assert SPEC is not None and SPEC.loader is not None
SYMMETRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SYMMETRY)
ORIENTATION_PATH = TASK_DIR / "wbc_stand_up" / "fallen_orientation.py"
ORIENTATION_SPEC = importlib.util.spec_from_file_location(
    "k1_wbc_fallen_orientation",
    ORIENTATION_PATH,
)
assert ORIENTATION_SPEC is not None and ORIENTATION_SPEC.loader is not None
ORIENTATION = importlib.util.module_from_spec(ORIENTATION_SPEC)
ORIENTATION_SPEC.loader.exec_module(ORIENTATION)
NORMALIZER_PATH = (
    Path(__file__).resolve().parents[1]
    / "third_party"
    / "wbc_agile_rsl_rl"
    / "rsl_rl"
    / "modules"
    / "normalizer.py"
)
NORMALIZER_SPEC = importlib.util.spec_from_file_location(
    "wbc_reward_normalizer",
    NORMALIZER_PATH,
)
assert NORMALIZER_SPEC is not None and NORMALIZER_SPEC.loader is not None
NORMALIZER = importlib.util.module_from_spec(NORMALIZER_SPEC)
NORMALIZER_SPEC.loader.exec_module(NORMALIZER)


class K1WbcStandUpTest(unittest.TestCase):
    def test_wbc_action_configuration_matches_reference_semantics(self) -> None:
        env_source = (TASK_DIR / "robots" / "k1" / "stand_up_env_cfg.py").read_text(encoding="utf-8")
        ppo_source = (TASK_DIR / "robots" / "k1" / "stand_up_ppo_cfg.py").read_text(encoding="utf-8")
        self.assertIn('clip={".*": (-1.0, 1.0)}', env_source)
        self.assertIn("normalize_input=False", env_source)
        self.assertIn("clip_actions = None", ppo_source)

    def test_wbc_video_viewer_tracks_the_robot(self) -> None:
        env_source = (TASK_DIR / "robots" / "k1" / "stand_up_env_cfg.py").read_text(encoding="utf-8")
        play_source = (
            Path(__file__).resolve().parents[1] / "scripts" / "rsl_rl" / "play.py"
        ).read_text(encoding="utf-8")
        finalizer_source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "rsl_rl"
            / "finalize_k1_wbc_training.sh"
        ).read_text(encoding="utf-8")
        package_source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "rsl_rl"
            / "package_k1_wbc_checkpoint.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('self.viewer.origin_type = "asset_root"', env_source)
        self.assertIn('self.viewer.asset_name = "robot"', env_source)
        self.assertIn('"--video_output_dir"', play_source)
        self.assertIn("record_successful_k1_wbc_videos.sh", finalizer_source)
        self.assertIn("select_best_k1_wbc_checkpoint.py", finalizer_source)
        self.assertIn('FINAL_CHECKPOINT="$checkpoint"', finalizer_source)
        self.assertIn('MASTERY_REPORT="$mastery_report"', finalizer_source)
        self.assertIn(
            'ROBUSTNESS_MASTERY_REPORT="$robustness_mastery_report"',
            finalizer_source,
        )
        self.assertIn("trainer already exited after writing its final checkpoint", finalizer_source)
        self.assertIn('evaluation_dir/success_videos', package_source)
        self.assertIn('staging/final_training_checkpoint', package_source)
        self.assertIn("k1_wbc_export_manifest.py", package_source)
        self.assertIn('verify "$checkpoint" "$export_dir"', package_source)
        self.assertIn("validate_k1_wbc_exports.py", package_source)

    def test_wbc_finalizer_requires_mastery_confidence_before_packaging(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        finalizer_source = (
            repo_root / "scripts" / "rsl_rl" / "finalize_k1_wbc_training.sh"
        ).read_text(encoding="utf-8")
        package_source = (
            repo_root / "scripts" / "rsl_rl" / "package_k1_wbc_checkpoint.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('run_mastery_gate="${RUN_MASTERY_GATE:-1}"', finalizer_source)
        self.assertIn('minimum_aggregate_wilson="${MINIMUM_AGGREGATE_WILSON:-0.9}"', finalizer_source)
        self.assertIn(
            'minimum_weakest_mode_wilson="${MINIMUM_WEAKEST_MODE_WILSON:-0.9}"',
            finalizer_source,
        )
        self.assertIn("check_k1_wbc_mastery_gate.py", finalizer_source)
        self.assertIn('--terrain-level "$robustness_terrain_level"', finalizer_source)
        self.assertLess(
            finalizer_source.index("check_k1_wbc_mastery_gate.py"),
            finalizer_source.index("record_successful_k1_wbc_videos.sh"),
        )
        self.assertIn('mastery_report="${MASTERY_REPORT:-}"', package_source)
        self.assertIn(
            'robustness_mastery_report="${ROBUSTNESS_MASTERY_REPORT:-}"',
            package_source,
        )

    def test_wbc_legacy_reward_and_curriculum_recipe_is_selected(self) -> None:
        env_source = (TASK_DIR / "robots" / "k1" / "stand_up_env_cfg.py").read_text(encoding="utf-8")
        self.assertIn("func=mdp.standing", env_source)
        self.assertIn('params={"term_keys": "standing"}', env_source)
        self.assertIn("weight=10.0", env_source)
        self.assertIn("func=mdp.remove_harness", env_source)
        self.assertIn('"start": 5_000', env_source)
        self.assertIn('"num_steps": 200_000', env_source)
        self.assertIn("func=mdp.joint_deviation_exp_if_standing", env_source)
        self.assertNotIn("func=mdp.adaptive_force_decay", env_source)
        self.assertNotIn("func=mdp.no_height_progress", env_source)

    def test_wbc_legacy_settling_and_push_recipe_is_selected(self) -> None:
        env_source = (TASK_DIR / "robots" / "k1" / "stand_up_env_cfg.py").read_text(encoding="utf-8")
        runtime_source = (TASK_DIR / "wbc_stand_up" / "env.py").read_text(encoding="utf-8")
        event_source = (TASK_DIR / "wbc_stand_up" / "events.py").read_text(encoding="utf-8")
        terrain_source = (TASK_DIR / "wbc_stand_up" / "terrains.py").read_text(encoding="utf-8")
        self.assertIn("REST_DURATION_S = 2.0", env_source)
        self.assertIn("func=mdp.is_env_inactive", env_source)
        self.assertIn("func=mdp.disable_joints", env_source)
        self.assertGreater(
            runtime_source.index("self.scene.write_data_to_sim()"),
            runtime_source.index("self.action_manager.apply_action()"),
        )
        self.assertGreater(
            runtime_source.index('self.event_manager.apply(mode="pre_sim_step"'),
            runtime_source.index("self.scene.write_data_to_sim()"),
        )
        self.assertGreater(
            runtime_source.index("self.sim.step(render=False)"),
            runtime_source.index('self.event_manager.apply(mode="pre_sim_step"'),
        )
        self.assertIn("_joint_effort_target_sim[rest_env_ids, :] = 0.0", event_source)
        self.assertIn("interval_range_s=(2.0, 4.0)", env_source)
        self.assertIn('"x": (-0.5, 0.5)', env_source)
        self.assertIn('"roll": (-0.25, 0.25)', env_source)
        self.assertIn("num_rows=20", terrain_source)
        self.assertIn("num_cols=16", terrain_source)
        self.assertIn("proportion=1.0", terrain_source)
        self.assertNotIn("boxes_small", terrain_source)
        self.assertNotIn("wave_small", terrain_source)

    def test_wbc_legacy_reward_geometry_matches_reference(self) -> None:
        env_source = (TASK_DIR / "robots" / "k1" / "stand_up_env_cfg.py").read_text(encoding="utf-8")
        reward_source = (TASK_DIR / "wbc_stand_up" / "rewards.py").read_text(encoding="utf-8")
        self.assertIn('"norm": "l2"', env_source)
        self.assertIn("asset.data.root_quat_w.unsqueeze(1)", reward_source)
        self.assertNotIn("max=1e6", reward_source)

    def test_wbc_reward_normalization_keeps_legacy_scale_with_outlier_guard(self) -> None:
        ppo_source = (TASK_DIR / "robots" / "k1" / "stand_up_ppo_cfg.py").read_text(encoding="utf-8")
        self.assertIn("return_scale_decay=None", ppo_source)
        self.assertIn("outlier_threshold=3.0", ppo_source)

        normalizer = NORMALIZER.ReturnVarianceNormalization(
            shape=[1],
            eps=1.0e-2,
            gamma=0.995,
            decay=0.999,
            return_scale_decay=None,
            outlier_threshold=3.0,
        )
        rewards = torch.zeros(4096, 1)
        rewards[0] = -1.0e6
        normalized = normalizer(rewards)
        self.assertTrue(bool(torch.isfinite(normalized).all()))
        self.assertLess(float(normalizer._std), 1.1)
        self.assertGreaterEqual(float(normalized.min()), -3.1)

    def test_wbc_k1_caps_adaptive_learning_rate(self) -> None:
        ppo_source = (TASK_DIR / "robots" / "k1" / "stand_up_ppo_cfg.py").read_text(
            encoding="utf-8"
        )
        algorithm_source = (
            Path(__file__).resolve().parents[1]
            / "third_party"
            / "wbc_agile_rsl_rl"
            / "rsl_rl"
            / "algorithms"
            / "ppo.py"
        ).read_text(encoding="utf-8")
        self.assertIn("max_learning_rate=1.0e-3", ppo_source)
        self.assertIn("self.max_learning_rate", algorithm_source)
        self.assertIn("self.learning_rate * 1.5", algorithm_source)

    def test_wbc_k1_guards_discontinuous_l2c2_and_value_targets(self) -> None:
        env_source = (TASK_DIR / "robots" / "k1" / "stand_up_env_cfg.py").read_text(
            encoding="utf-8"
        )
        ppo_source = (TASK_DIR / "robots" / "k1" / "stand_up_ppo_cfg.py").read_text(
            encoding="utf-8"
        )
        algorithm_source = (
            Path(__file__).resolve().parents[1]
            / "third_party"
            / "wbc_agile_rsl_rl"
            / "rsl_rl"
            / "algorithms"
            / "ppo.py"
        ).read_text(encoding="utf-8")
        self.assertGreaterEqual(env_source.count("clip=(-10.0, 10.0)"), 2)
        self.assertIn("value_loss_huber_delta=10.0", ppo_source)
        self.assertIn("max_actor_observation_delta=10.0", ppo_source)
        self.assertIn("max_critic_observation_delta=50.0", ppo_source)
        self.assertIn("actor_valid &= actor_delta", algorithm_source)
        self.assertIn("critic_valid &= critic_delta", algorithm_source)
        self.assertIn("nn.functional.huber_loss", algorithm_source)

    def test_wbc_k1_resets_exploded_simulation_states(self) -> None:
        env_source = (TASK_DIR / "robots" / "k1" / "stand_up_env_cfg.py").read_text(
            encoding="utf-8"
        )
        terminations_source = (TASK_DIR / "terminations.py").read_text(encoding="utf-8")
        self.assertIn("invalid_state = DoneTerm(", env_source)
        self.assertIn("def invalid_state(", terminations_source)
        self.assertIn("torch.isfinite", terminations_source)

    def test_wbc_resume_restores_curriculum_common_step(self) -> None:
        train_source = (
            Path(__file__).resolve().parents[1] / "scripts" / "rsl_rl" / "train.py"
        ).read_text(encoding="utf-8")
        self.assertIn(
            "curriculum_iteration * runner.num_steps_per_env",
            train_source,
        )
        self.assertIn(
            "base_env.curriculum_manager.compute(env_ids=all_env_ids)",
            train_source,
        )

    def test_wbc_evaluation_records_and_checks_explicit_seed(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        play_source = (repo_root / "scripts" / "rsl_rl" / "play.py").read_text(
            encoding="utf-8"
        )
        evaluate_source = (
            repo_root / "scripts" / "rsl_rl" / "evaluate_k1_wbc_checkpoint.sh"
        ).read_text(encoding="utf-8")
        self.assertIn('metrics["seed"] = int(agent_cfg.seed)', play_source)
        self.assertIn('eval_seed="${EVAL_SEED:-42}"', evaluate_source)
        self.assertIn('--seed "$eval_seed"', evaluate_source)
        self.assertIn('if int(result["seed"]) != int(expected_seed):', evaluate_source)

    def test_wbc_mastery_resume_resets_optimizer_and_curriculum(self) -> None:
        train_source = (
            Path(__file__).resolve().parents[1] / "scripts" / "rsl_rl" / "train.py"
        ).read_text(encoding="utf-8")
        mastery_ppo_source = (
            TASK_DIR / "robots" / "k1" / "stand_up_mastery_ppo_cfg.py"
        ).read_text(encoding="utf-8")
        self.assertIn("load_optimizer_on_resume: bool = False", mastery_ppo_source)
        self.assertIn("reset_learning_iteration_on_resume: bool = True", mastery_ppo_source)
        self.assertIn("restore_wbc_curriculum_on_resume: bool = False", mastery_ppo_source)
        self.assertIn('runner.current_learning_iteration = 0', train_source)
        self.assertIn('load_optimizer=getattr(agent_cfg, "load_optimizer_on_resume", True)', train_source)

    def test_wbc_mastery_uses_strict_balanced_zero_lift_stage(self) -> None:
        mastery_env_source = (
            TASK_DIR / "robots" / "k1" / "stand_up_mastery_env_cfg.py"
        ).read_text(encoding="utf-8")
        self.assertIn('orientation_mode="balanced"', mastery_env_source)
        self.assertIn("standing_ratio=0.0", mastery_env_source)
        self.assertIn("projected_gravity_z_max=-0.85", mastery_env_source)
        self.assertIn("root_linear_speed_max=0.5", mastery_env_source)
        self.assertIn("self.actions.lift.force_limit = 0.0", mastery_env_source)
        self.assertIn("self.curriculum.terrain_levels = None", mastery_env_source)
        self.assertIn(
            "self.rewards.joint_deviation_l1.func = mdp.joint_deviation_strict_success_potential",
            mastery_env_source,
        )
        self.assertIn("successful_termination_term=\"standing\"", mastery_env_source)
        self.assertIn("gamma=0.995", mastery_env_source)
        self.assertIn("root_angular_speed_max=1.0", mastery_env_source)

    def test_wbc_mastery_launcher_keeps_trainer_as_stoppable_child(self) -> None:
        transition_source = (
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "rsl_rl"
            / "transition_k1_wbc_to_mastery.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("trainer_pid=\\$!", transition_source)
        self.assertIn("wait \\$trainer_pid", transition_source)
        self.assertNotIn("echo \\$\\$ > $quoted_pid_file && exec", transition_source)

    def test_wbc_legacy_ppo_enables_official_l2c2(self) -> None:
        ppo_source = (TASK_DIR / "robots" / "k1" / "stand_up_ppo_cfg.py").read_text(encoding="utf-8")
        vendor_root = Path(__file__).resolve().parents[1] / "third_party" / "wbc_agile_rsl_rl"
        algorithm_source = (vendor_root / "rsl_rl" / "algorithms" / "ppo.py").read_text(encoding="utf-8")
        storage_source = (vendor_root / "rsl_rl" / "storage" / "rollout_storage.py").read_text(encoding="utf-8")
        self.assertIn("l2c2_cfg=RslRlL2C2Cfg(", ppo_source)
        self.assertIn("lambda_actor=1.0", ppo_source)
        self.assertIn("lambda_critic=0.1", ppo_source)
        self.assertIn('loss_dict["l2c2_actor"]', algorithm_source)
        self.assertIn('loss_dict["l2c2_critic"]', algorithm_source)
        self.assertIn("previous_observations = self.observations[:-1]", storage_source)
        self.assertIn("l2c2_done_mask = self.dones[:-1]", storage_source)

    def test_joint_mirror_is_an_involution(self) -> None:
        value = torch.arange(44, dtype=torch.float32).reshape(2, 22)
        mirrored = SYMMETRY._mirror_joint_values(value)
        restored = SYMMETRY._mirror_joint_values(mirrored)
        torch.testing.assert_close(restored, value)

    def test_public_joint_mirror_matches_training_transform(self) -> None:
        value = torch.randn(3, 22)
        torch.testing.assert_close(
            SYMMETRY.mirror_joint_values(value),
            SYMMETRY._mirror_joint_values(value),
        )

    def test_joint_mirror_uses_k1_direction_conventions(self) -> None:
        value = torch.zeros(1, 22)
        value[0, 6:10] = torch.tensor([1.0, 2.0, 3.0, 4.0])
        value[0, 16:22] = torch.tensor([5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        mirrored = SYMMETRY._mirror_joint_values(value)
        torch.testing.assert_close(
            mirrored[0, 2:6],
            torch.tensor([1.0, -2.0, 3.0, -4.0]),
        )
        torch.testing.assert_close(
            mirrored[0, 10:16],
            torch.tensor([5.0, -6.0, -7.0, 8.0, 9.0, -10.0]),
        )

    def test_policy_history_mirror_preserves_shape_and_is_reversible(self) -> None:
        width = 5 * (3 + 3 + 22 + 22 + 22)
        obs = torch.randn(8, width)
        env = SimpleNamespace(unwrapped=SimpleNamespace())
        mirrored = SYMMETRY._mirror_flat_observation(obs, env, "policy")
        restored = SYMMETRY._mirror_flat_observation(mirrored, env, "policy")
        self.assertEqual(mirrored.shape, obs.shape)
        torch.testing.assert_close(restored, obs)

    def test_critic_history_mirror_preserves_inactive_flag(self) -> None:
        body_names = (
            "Trunk",
            "Left_Arm",
            "Right_Arm",
            "left_foot_link",
            "right_foot_link",
        )
        sensor = SimpleNamespace(body_names=body_names)
        scene = SimpleNamespace(
            sensors={"contact_forces": sensor},
        )
        env = SimpleNamespace(unwrapped=SimpleNamespace(scene=scene))
        width = 5 * (3 + 3 + 3 + 22 + 22 + 22 + 1 + len(body_names) + 1)
        obs = torch.randn(8, width)
        mirrored = SYMMETRY._mirror_flat_observation(obs, env, "critic")
        restored = SYMMETRY._mirror_flat_observation(mirrored, env, "critic")
        self.assertEqual(mirrored.shape, obs.shape)
        torch.testing.assert_close(restored, obs)

    def test_fallen_orientation_modes_match_k1_body_axes(self) -> None:
        half_sqrt = 2.0**-0.5
        quaternions = {
            "other": torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
            "face_up": torch.tensor([[half_sqrt, 0.0, -half_sqrt, 0.0]]),
            "face_down": torch.tensor([[half_sqrt, 0.0, half_sqrt, 0.0]]),
            "left_side": torch.tensor([[half_sqrt, -half_sqrt, 0.0, 0.0]]),
            "right_side": torch.tensor([[half_sqrt, half_sqrt, 0.0, 0.0]]),
        }
        for expected_mode, quaternion in quaternions.items():
            with self.subTest(expected_mode=expected_mode):
                for mode in quaternions:
                    selected = ORIENTATION.fallen_orientation_mask(
                        quaternion,
                        mode,
                    )
                    self.assertEqual(bool(selected.item()), mode == expected_mode)

    def test_fallen_orientation_is_quaternion_sign_invariant(self) -> None:
        quaternions = torch.randn(128, 4)
        for mode in ORIENTATION.FALLEN_ORIENTATION_MODES:
            selected = ORIENTATION.fallen_orientation_mask(quaternions, mode)
            sign_flipped = ORIENTATION.fallen_orientation_mask(
                -quaternions,
                mode,
            )
            torch.testing.assert_close(selected, sign_flipped)

    def test_fallen_orientation_classes_are_exclusive(self) -> None:
        quaternions = torch.randn(1024, 4)
        masks = torch.stack(
            [
                ORIENTATION.fallen_orientation_mask(quaternions, mode)
                for mode in ORIENTATION.FALLEN_ORIENTATION_MODES
                if mode != "random"
            ],
            dim=0,
        )
        self.assertTrue(bool(torch.all(masks.sum(dim=0) == 1)))

    def test_balanced_fallen_sampling_equalizes_all_orientation_classes(self) -> None:
        half_sqrt = 2.0**-0.5
        class_quaternions = torch.tensor(
            [
                [half_sqrt, 0.0, -half_sqrt, 0.0],
                [half_sqrt, 0.0, half_sqrt, 0.0],
                [half_sqrt, -half_sqrt, 0.0, 0.0],
                [half_sqrt, half_sqrt, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
            ]
        )
        sampled_indices = ORIENTATION.sample_balanced_fallen_orientation_indices(
            class_quaternions,
            103,
        )
        sampled_quaternions = class_quaternions[sampled_indices]
        counts = torch.tensor(
            [
                ORIENTATION.fallen_orientation_mask(sampled_quaternions, mode).sum()
                for mode in ORIENTATION.EXCLUSIVE_FALLEN_ORIENTATION_MODES
            ]
        )
        self.assertEqual(int(counts.sum()), 103)
        self.assertLessEqual(int(counts.max() - counts.min()), 1)


if __name__ == "__main__":
    unittest.main()

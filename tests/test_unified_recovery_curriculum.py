from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest import mock

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "skrl"
    / "run_unified_recovery_curriculum.py"
)
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("run_unified_recovery_curriculum", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
CURRICULUM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CURRICULUM)


class UnifiedRecoveryCurriculumTest(unittest.TestCase):
    def test_clip_stage_modes_require_every_direction_and_half(self) -> None:
        self.assertEqual(
            CURRICULUM.clip_stage_modes([0, 1], 0.6, 0.8),
            [
                "reference_clip_0_phase_0.600_0.700",
                "reference_clip_0_phase_0.700_0.800",
                "reference_clip_1_phase_0.600_0.700",
                "reference_clip_1_phase_0.700_0.800",
            ],
        )

    def test_clip_stage_modes_support_one_direction(self) -> None:
        self.assertEqual(
            CURRICULUM.clip_stage_modes([1], 0.0, 0.2),
            [
                "reference_clip_1_phase_0.000_0.100",
                "reference_clip_1_phase_0.100_0.200",
            ],
        )

    def test_random_fall_stages_keep_five_percent_canonical_resets(self) -> None:
        expected = {
            0.25: (0.75, 0.05, 0.15, 0.0),
            0.50: (0.65, 0.05, 0.25, 0.0),
            0.75: (0.55, 0.05, 0.35, 0.0),
            1.00: (0.45, 0.05, 0.45, 0.0),
        }
        for difficulty, probabilities in expected.items():
            with self.subTest(difficulty=difficulty):
                actual = CURRICULUM.random_reset_probabilities(difficulty)
                for value, target in zip(actual, probabilities):
                    self.assertAlmostEqual(value, target)
                self.assertAlmostEqual(sum(actual), 0.95)

    def test_failure_replay_replaces_reference_resets(self) -> None:
        probabilities = CURRICULUM.random_reset_probabilities(1.0, 0.15)
        self.assertEqual(len(probabilities), 4)
        self.assertAlmostEqual(probabilities[0], 0.30)
        self.assertAlmostEqual(probabilities[3], 0.15)
        self.assertAlmostEqual(sum(probabilities), 0.95)

    def test_randomization_strength_tracks_random_fall_difficulty(self) -> None:
        expected = {
            0.25: (0.25, 0.25, 0.0),
            0.50: (0.50, 0.50, 0.0),
            0.75: (0.75, 0.75, 0.25),
            1.00: (1.00, 1.00, 0.50),
        }
        for difficulty, target in expected.items():
            with self.subTest(difficulty=difficulty):
                actual = CURRICULUM.randomization_scales(
                    difficulty,
                    maximum_domain_scale=1.0,
                    maximum_observation_noise_scale=1.0,
                    maximum_push_scale=0.5,
                    push_start_difficulty=0.5,
                )
                for value, expected_value in zip(actual, target):
                    self.assertAlmostEqual(value, expected_value)

    def test_initial_ground_states_seed_first_random_stage(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            initial = Path(temporary) / "settled_ground_states.npz"
            initial.touch()
            state = {"random_fall_attempts": []}
            self.assertEqual(
                CURRICULUM.latest_failure_state_file(state, initial),
                initial,
            )

    def test_latest_failure_states_replace_initial_ground_states(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            initial = root / "settled_ground_states.npz"
            latest = root / "difficulty025_failures.npz"
            initial.touch()
            latest.touch()
            state = {
                "random_fall_attempts": [
                    {
                        "failure_states": str(latest),
                        "failure_state_count": 128,
                    }
                ]
            }
            self.assertEqual(
                CURRICULUM.latest_failure_state_file(state, initial),
                latest,
            )

    def test_random_stage_video_uses_exact_evaluated_difficulty(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            args = Namespace(
                artifacts_dir=Path(temporary),
                task="task",
                joint_state_tolerance=0.05,
                parallel_state_tolerance=0.04,
                video_length=300,
            )
            with mock.patch.object(CURRICULUM, "run_logged") as run_logged:
                output = CURRICULUM.record_random_fall_video(
                    args,
                    Path("/tmp/agent.pt"),
                    0.75,
                    2,
                )
            command = run_logged.call_args.args[0]
            self.assertEqual(command[command.index("--reset_mode") + 1], "random")
            self.assertEqual(
                command[command.index("--random_fall_difficulty") + 1],
                "0.75",
            )
            self.assertEqual(output.name, "random_difficulty075_attempt2")

    def test_phase_resume_reuses_completed_trainer_instead_of_starting_another(self) -> None:
        args = Namespace(
            poll_seconds=60.0,
            iterations_per_phase_stage=4000,
            checkpoint_interval=400,
        )
        run_dir = Path("/tmp/unified_phase_run")
        checkpoint = run_dir / "checkpoints" / "agent_128000.pt"
        with (
            mock.patch.object(
                CURRICULUM,
                "recover_interrupted_training",
                return_value=(run_dir, checkpoint, True),
            ) as recover_training,
            mock.patch.object(CURRICULUM, "run_logged") as run_logged,
        ):
            actual = CURRICULUM.train_phase_attempt(
                args,
                Path("/tmp/source.pt"),
                0.6,
                0.8,
                1,
                resume=True,
            )
        self.assertEqual(actual, (run_dir, checkpoint))
        recover_training.assert_called_once()
        run_logged.assert_not_called()

    def test_interrupted_phase_without_checkpoint_restarts_from_seed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed = root / "seed.pt"
            seed.touch()
            old_run = root / "old_run"
            new_run = root / "new_run"
            final = new_run / "checkpoints" / "agent_128000.pt"
            final.parent.mkdir(parents=True)
            final.touch()
            args = Namespace(
                poll_seconds=1.0,
                iterations_per_phase_stage=4000,
                checkpoint_interval=400,
                task="task",
                task_reward_scale=5.0,
                style_reward_scale=5.0,
                num_envs=16384,
                joint_state_tolerance=0.05,
                parallel_state_tolerance=0.0,
                artifacts_dir=root,
            )
            with (
                mock.patch.object(
                    CURRICULUM,
                    "recover_interrupted_training",
                    return_value=(old_run, seed, False),
                ),
                mock.patch.object(CURRICULUM, "run_logged") as run_logged,
                mock.patch.object(CURRICULUM, "newest_run", return_value=new_run),
                mock.patch.object(CURRICULUM, "latest_checkpoint", return_value=final),
                mock.patch.object(Path, "iterdir", return_value=iter(())),
            ):
                actual = CURRICULUM.train_phase_attempt(
                    args,
                    seed,
                    0.7,
                    0.9,
                    1,
                    resume=True,
                )

            self.assertEqual(actual, (new_run, final))
            command = run_logged.call_args.args[0]
            self.assertEqual(command[command.index("--checkpoint") + 1], str(seed))
            self.assertEqual(
                run_logged.call_args.args[1].name,
                "unified_phase_070_090_attempt1.log",
            )

    def test_restart_logs_do_not_overwrite_interrupted_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "attempt.log").touch()
            (root / "attempt_restart1.log").touch()
            self.assertEqual(
                CURRICULUM.next_training_log_path(root, "attempt").name,
                "attempt_restart2.log",
            )

    def test_bootstrap_bc_uses_a_separate_small_environment_run(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run_dir = root / "bootstrap_run"
            checkpoint = run_dir / "checkpoints" / "bc_agent.pt"
            checkpoint.parent.mkdir(parents=True)
            checkpoint.touch()
            args = Namespace(
                task="task",
                task_reward_scale=5.0,
                style_reward_scale=5.0,
                bc_iterations=2000,
                bc_num_envs=2048,
                artifacts_dir=root,
                joint_state_tolerance=0.05,
                parallel_state_tolerance=0.0,
            )
            with (
                mock.patch.object(CURRICULUM, "run_logged") as run_logged,
                mock.patch.object(CURRICULUM, "newest_run", return_value=run_dir),
            ):
                actual = CURRICULUM.bootstrap_unified_policy(
                    args,
                    Path("/tmp/source.pt"),
                    0.7,
                    0.9,
                )

            command = run_logged.call_args.args[0]
            self.assertEqual(actual, (run_dir, checkpoint))
            self.assertIn("--bc_only", command)
            self.assertIn("--reset_checkpoint_discriminator", command)
            self.assertEqual(command[command.index("--num_envs") + 1], "2048")
            self.assertEqual(command[command.index("--bc_iterations") + 1], "2000")
            self.assertNotIn("--max_iterations", command)

    def test_handoff_evaluation_uses_real_actor_gate_for_every_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            artifacts = Path(temporary)
            output = (
                artifacts
                / "unified_arbitrary_falls_difficulty100_attempt2_handoff_evaluation.json"
            )
            output.write_text(
                "{}\n",
                encoding="utf-8",
            )
            args = Namespace(
                artifacts_dir=artifacts,
                task="task",
                handoff_evaluation_num_envs=128,
                handoff_evaluation_episodes=512,
                joint_state_tolerance=0.05,
                parallel_state_tolerance=0.04,
                minimum_success_rate=0.9,
                maximum_joint_limit_rate=0.001,
                maximum_parallel_ankle_rate=0.001,
                maximum_parallel_state_violation_fraction=0.001,
                maximum_p90_recovery_time_s=4.0,
                maximum_domain_randomization_scale=1.0,
                maximum_observation_noise_scale=1.0,
                maximum_push_randomization_scale=0.5,
                push_start_difficulty=0.5,
            )
            summary = {"accepted": True, "failures": []}
            with (
                mock.patch.object(CURRICULUM, "run_logged") as run_logged,
                mock.patch.object(
                    CURRICULUM,
                    "handoff_acceptance",
                    return_value=summary,
                ),
            ):
                returned_output, returned_summary = CURRICULUM.evaluate_handoff(
                    args,
                    Path("/tmp/agent.pt"),
                    1.0,
                    2,
                )
            command = run_logged.call_args.args[0]
            self.assertIn("scripts/skrl/evaluate_handoff.py", command)
            self.assertEqual(
                command[command.index("--random-fall-difficulty") + 1],
                "1.0",
            )
            self.assertEqual(
                command[command.index("--domain-randomization-scale") + 1],
                "1.0",
            )
            self.assertEqual(
                command[command.index("--observation-noise-scale") + 1],
                "1.0",
            )
            self.assertEqual(
                command[command.index("--push-randomization-scale") + 1],
                "0.5",
            )
            self.assertNotIn("--start-handoff-at-reset", command)
            self.assertEqual(
                command[
                    command.index("--modes") + 1
                    : command.index("--random-fall-difficulty")
                ],
                list(CURRICULUM.DEFAULT_MODES),
            )
            self.assertEqual(returned_output, output)
            self.assertIs(returned_summary, summary)

    def test_handoff_acceptance_rejects_old_zero_only_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "handoff.json"
            output.write_text(
                json.dumps(
                    {
                        "faceup": {
                            "episodes": 100,
                            "success_rate": 1.0,
                            "handoff_contract": {
                                "actor": "forward",
                            },
                        }
                    }
                ),
                encoding="utf-8",
            )
            args = Namespace()
            base = {"accepted": True, "failures": []}
            with mock.patch.object(
                CURRICULUM,
                "acceptance",
                return_value=base,
            ):
                summary = CURRICULUM.handoff_acceptance(
                    output,
                    ["faceup"],
                    args,
                )
            self.assertFalse(summary["accepted"])
            self.assertTrue(
                any(
                    "commanded forward progress" in failure
                    for failure in summary["failures"]
                )
            )


if __name__ == "__main__":
    unittest.main()

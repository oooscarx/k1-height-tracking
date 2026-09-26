from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "skrl"
    / "run_reverse_phase_curriculum.py"
)
sys.path.insert(0, str(SCRIPT_PATH.parent))
SPEC = importlib.util.spec_from_file_location("run_reverse_phase_curriculum", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
CURRICULUM = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CURRICULUM)


class ReversePhaseCurriculumTest(unittest.TestCase):
    def test_stage_bins_include_context_and_mastery_bins(self) -> None:
        self.assertEqual(
            CURRICULUM.stage_bins(0.6, 0.8),
            [0.5, 0.6, 0.7, 0.8, 0.9],
        )
        self.assertEqual(
            CURRICULUM.stage_bins(0.0, 0.2),
            [0.0, 0.1, 0.2, 0.3],
        )

    def test_required_modes_cover_both_halves_of_stage(self) -> None:
        self.assertEqual(
            CURRICULUM.required_stage_modes(0.6, 0.8),
            [
                "reference_phase_0.600_0.700",
                "reference_phase_0.700_0.800",
            ],
        )

    def test_checkpoint_step_uses_numeric_order(self) -> None:
        self.assertEqual(CURRICULUM.checkpoint_step(Path("agent_12800.pt")), 12800)

    def test_existing_run_selects_latest_restart(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            older = root / "2026-01-01_amp_torch_attempt"
            newer = root / "2026-01-02_amp_torch_attempt"
            older.mkdir()
            newer.mkdir()
            os.utime(older, ns=(1, 1))
            os.utime(newer, ns=(2, 2))
            self.assertEqual(
                CURRICULUM.existing_run(root, "attempt"),
                newer.resolve(),
            )

    def test_recorded_checkpoint_seeds_run_without_periodic_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run_dir = root / "run"
            run_dir.mkdir()
            checkpoint = root / "seed" / "agent_25600.pt"
            checkpoint.parent.mkdir()
            checkpoint.touch()
            selected = CURRICULUM.latest_checkpoint_or_recorded(
                run_dir,
                str(checkpoint),
            )
        self.assertEqual(selected, checkpoint.resolve())

    def test_training_command_matches_only_real_trainer_arguments(self) -> None:
        trainer = [
            "/repo/.venv/bin/python",
            "scripts/skrl/train.py",
            "--experiment_name",
            "reverse_phase_060_080_attempt1",
        ]
        self.assertTrue(
            CURRICULUM.training_command_matches(
                trainer,
                "reverse_phase_060_080_attempt1",
            )
        )
        self.assertFalse(
            CURRICULUM.training_command_matches(
                ["bash", "-c", " ".join(trainer)],
                "reverse_phase_060_080_attempt1",
            )
        )
        self.assertFalse(
            CURRICULUM.training_command_matches(
                trainer,
                "reverse_phase_050_070_attempt1",
            )
        )

    def test_expected_final_checkpoint_step_uses_observed_rollout_length(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            checkpoints = Path(directory) / "checkpoints"
            checkpoints.mkdir()
            (checkpoints / "agent_12800.pt").touch()
            (checkpoints / "agent_25600.pt").touch()
            self.assertEqual(
                CURRICULUM.expected_final_checkpoint_step(
                    Path(directory),
                    iterations=2400,
                    checkpoint_interval=400,
                ),
                76800,
            )

    def test_interrupted_attempt_preserves_partial_checkpoint_for_retry(self) -> None:
        state = CURRICULUM.interrupted_attempt_state(
            minimum=0.6,
            maximum=0.8,
            attempt=1,
            run_dir=Path("run"),
            checkpoint=Path("run/checkpoints/agent_38400.pt"),
            expected_checkpoint_step=128000,
            task_reward_scale=5.0,
            style_reward_scale=5.0,
        )
        self.assertTrue(state["interrupted"])
        self.assertEqual(state["checkpoint"], "run/checkpoints/agent_38400.pt")
        self.assertEqual(state["expected_checkpoint_step"], 128000)
        self.assertNotIn("evaluation", state)

    def test_retry_can_return_to_the_clean_stage_start(self) -> None:
        current = Path("attempt2/checkpoints/agent_12800.pt")
        stage_start = Path("attempt1/checkpoints/agent_51200.pt")
        self.assertEqual(
            CURRICULUM.retry_seed_checkpoint(
                current_checkpoint=current,
                stage_start_checkpoint=stage_start,
                attempt=3,
                retry_from_stage_start=True,
            ),
            stage_start,
        )
        self.assertEqual(
            CURRICULUM.retry_seed_checkpoint(
                current_checkpoint=current,
                stage_start_checkpoint=stage_start,
                attempt=3,
                retry_from_stage_start=False,
            ),
            current,
        )
        self.assertEqual(
            CURRICULUM.retry_seed_checkpoint(
                current_checkpoint=current,
                stage_start_checkpoint=stage_start,
                attempt=1,
                retry_from_stage_start=True,
            ),
            current,
        )

    def test_interrupted_attempt_without_checkpoint_retains_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory)
            (run_dir / "checkpoints").mkdir()
            seed = Path("previous/checkpoints/agent_51200.pt")
            checkpoint, expected, produced = CURRICULUM.interrupted_checkpoint_or_seed(
                run_dir=run_dir,
                seed_checkpoint=seed,
                iterations=4000,
                checkpoint_interval=400,
            )
        self.assertEqual(checkpoint, seed)
        self.assertIsNone(expected)
        self.assertFalse(produced)


if __name__ == "__main__":
    unittest.main()

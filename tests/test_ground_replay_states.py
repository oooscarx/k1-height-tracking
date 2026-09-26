from __future__ import annotations

import hashlib
import importlib.util
import unittest
from pathlib import Path

import numpy as np

SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "filter_ground_replay_states.py"
)
SPEC = importlib.util.spec_from_file_location("filter_ground_replay_states", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
FILTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(FILTER)
ASSET_PATH = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "booster_train"
    / "booster_train"
    / "tasks"
    / "manager_based"
    / "fall_recovery"
    / "motions"
    / "random"
    / "k1_settled_random_ground_states.npz"
)


class GroundReplayStatesTest(unittest.TestCase):
    def test_committed_ground_bank_is_settled_safe_and_four_sided(self) -> None:
        self.assertEqual(
            hashlib.sha256(ASSET_PATH.read_bytes()).hexdigest(),
            "9990bf2d8b7074227147e5e26b2750eb8f112150b25d463381be97ea6975bbdc",
        )
        with np.load(ASSET_PATH, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in FILTER.REQUIRED_FIELDS}
        mask, orientations = FILTER.ground_state_mask(
            arrays,
            minimum_root_height=0.08,
            maximum_root_height=0.45,
            maximum_vertical_gravity=0.75,
            maximum_linear_speed=0.25,
            maximum_angular_speed=0.50,
            maximum_joint_speed=1.0,
        )
        self.assertEqual(mask.shape[0], 210)
        self.assertTrue(np.all(mask))
        np.testing.assert_array_equal(arrays["mode"], orientations)
        for orientation in FILTER.ORIENTATION_NAMES:
            self.assertGreaterEqual(np.count_nonzero(orientations == orientation), 32)

    def test_classifies_all_lying_orientations(self) -> None:
        half = np.sqrt(0.5)
        quaternion = np.asarray(
            [
                [half, 0.0, -half, 0.0],
                [half, 0.0, half, 0.0],
                [half, half, 0.0, 0.0],
                [half, -half, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        gravity = FILTER.projected_gravity(quaternion)
        np.testing.assert_array_equal(
            FILTER.classify_ground_orientation(gravity),
            ["faceup", "facedown", "side_left", "side_right"],
        )

    def test_ground_mask_rejects_airborne_moving_and_upright_states(self) -> None:
        half = np.sqrt(0.5)
        root_pose = np.asarray(
            [
                [0.0, 0.0, 0.25, half, 0.0, -half, 0.0],
                [0.0, 0.0, 0.70, half, 0.0, -half, 0.0],
                [0.0, 0.0, 0.25, half, 0.0, -half, 0.0],
                [0.0, 0.0, 0.25, 1.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        )
        arrays = {
            "joint_position": np.zeros((4, 22), dtype=np.float32),
            "joint_velocity": np.zeros((4, 22), dtype=np.float32),
            "root_pose": root_pose,
            "root_velocity": np.zeros((4, 6), dtype=np.float32),
            "termination_code": np.full(4, 4, dtype=np.int64),
            "mode": np.full(4, "random"),
        }
        arrays["root_velocity"][2, 0] = 0.5
        mask, _ = FILTER.ground_state_mask(
            arrays,
            minimum_root_height=0.08,
            maximum_root_height=0.45,
            maximum_vertical_gravity=0.75,
            maximum_linear_speed=0.25,
            maximum_angular_speed=0.50,
            maximum_joint_speed=1.0,
        )
        np.testing.assert_array_equal(mask, [True, False, False, False])


if __name__ == "__main__":
    unittest.main()

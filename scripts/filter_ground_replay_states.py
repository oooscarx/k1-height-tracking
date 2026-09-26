#!/usr/bin/env python3

"""Filter random-fall evaluation failures into settled ground replay states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REQUIRED_FIELDS = {
    "joint_position",
    "joint_velocity",
    "root_pose",
    "root_velocity",
    "termination_code",
    "mode",
}
ORIENTATION_NAMES = np.asarray(["faceup", "facedown", "side_left", "side_right"])


def projected_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
    """Return world gravity expressed in each root frame."""
    quaternion = quaternion_wxyz / np.linalg.norm(
        quaternion_wxyz,
        axis=-1,
        keepdims=True,
    )
    w, x, y, z = np.moveaxis(quaternion, -1, 0)
    return np.stack(
        (
            2.0 * (w * y - x * z),
            -2.0 * (y * z + w * x),
            -(1.0 - 2.0 * (x * x + y * y)),
        ),
        axis=-1,
    )


def classify_ground_orientation(gravity: np.ndarray) -> np.ndarray:
    """Classify a lying trunk by the dominant horizontal gravity component."""
    use_pitch = np.abs(gravity[:, 0]) >= np.abs(gravity[:, 1])
    classes = np.empty(gravity.shape[0], dtype=np.int64)
    classes[use_pitch] = np.where(gravity[use_pitch, 0] < 0.0, 0, 1)
    classes[~use_pitch] = np.where(gravity[~use_pitch, 1] < 0.0, 2, 3)
    return ORIENTATION_NAMES[classes]


def ground_state_mask(
    arrays: dict[str, np.ndarray],
    *,
    minimum_root_height: float,
    maximum_root_height: float,
    maximum_vertical_gravity: float,
    maximum_linear_speed: float,
    maximum_angular_speed: float,
    maximum_joint_speed: float,
) -> tuple[np.ndarray, np.ndarray]:
    gravity = projected_gravity(arrays["root_pose"][:, 3:])
    root_height = arrays["root_pose"][:, 2]
    root_velocity = arrays["root_velocity"]
    mask = (
        (arrays["termination_code"] == 4)
        & (root_height >= minimum_root_height)
        & (root_height <= maximum_root_height)
        & (np.abs(gravity[:, 2]) <= maximum_vertical_gravity)
        & (np.linalg.norm(root_velocity[:, :3], axis=-1) <= maximum_linear_speed)
        & (np.linalg.norm(root_velocity[:, 3:], axis=-1) <= maximum_angular_speed)
        & (np.max(np.abs(arrays["joint_velocity"]), axis=-1) <= maximum_joint_speed)
    )
    return mask, classify_ground_orientation(gravity)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-root-height", type=float, default=0.08)
    parser.add_argument("--maximum-root-height", type=float, default=0.45)
    parser.add_argument("--maximum-vertical-gravity", type=float, default=0.75)
    parser.add_argument("--maximum-linear-speed", type=float, default=0.25)
    parser.add_argument("--maximum-angular-speed", type=float, default=0.50)
    parser.add_argument("--maximum-joint-speed", type=float, default=1.0)
    parser.add_argument("--minimum-per-orientation", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if args.minimum_root_height < 0.0:
        raise ValueError("--minimum-root-height must be non-negative")
    if args.maximum_root_height <= args.minimum_root_height:
        raise ValueError("--maximum-root-height must exceed --minimum-root-height")
    if not 0.0 <= args.maximum_vertical_gravity < 1.0:
        raise ValueError("--maximum-vertical-gravity must be in [0, 1)")
    for name in ("maximum_linear_speed", "maximum_angular_speed", "maximum_joint_speed"):
        if getattr(args, name) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be non-negative")
    if args.minimum_per_orientation < 1:
        raise ValueError("--minimum-per-orientation must be positive")

    with np.load(input_path, allow_pickle=False) as archive:
        if set(archive.files) != REQUIRED_FIELDS:
            raise ValueError(
                f"unexpected replay fields: expected={sorted(REQUIRED_FIELDS)}, "
                f"actual={sorted(archive.files)}"
            )
        arrays = {name: archive[name] for name in REQUIRED_FIELDS}

    mask, orientations = ground_state_mask(
        arrays,
        minimum_root_height=args.minimum_root_height,
        maximum_root_height=args.maximum_root_height,
        maximum_vertical_gravity=args.maximum_vertical_gravity,
        maximum_linear_speed=args.maximum_linear_speed,
        maximum_angular_speed=args.maximum_angular_speed,
        maximum_joint_speed=args.maximum_joint_speed,
    )
    counts = {
        name: int(np.count_nonzero(mask & (orientations == name)))
        for name in ORIENTATION_NAMES
    }
    missing = [
        name
        for name, count in counts.items()
        if count < args.minimum_per_orientation
    ]
    if missing:
        raise RuntimeError(
            "settled ground replay lacks orientation coverage: "
            + ", ".join(f"{name}={counts[name]}" for name in missing)
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        joint_position=arrays["joint_position"][mask],
        joint_velocity=arrays["joint_velocity"][mask],
        root_pose=arrays["root_pose"][mask],
        root_velocity=arrays["root_velocity"][mask],
        termination_code=arrays["termination_code"][mask],
        mode=orientations[mask],
    )
    print(
        json.dumps(
            {
                "input_states": int(mask.shape[0]),
                "output_states": int(np.count_nonzero(mask)),
                "orientation_counts": counts,
                "output": str(output_path),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

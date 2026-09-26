#!/usr/bin/env python3
"""Retarget one CMU Subject 140 ASF/AMC get-up clip to a hardware-feasible K1 CSV."""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
from retarget_deepmimic_getup import (
    _append_amp_handoff,
    _gmr_joint_names,
    _hardware_project,
    _load_parallel_ankle,
)
from scipy.spatial.transform import Rotation, Slerp

CMU_TO_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ]
)
# CMU documents its stored ASF/AMC lengths as inches divided by 0.45.
CMU_LENGTH_TO_METERS = (1.0 / 0.45) * 2.54 / 100.0
CMU_TO_GMR = {
    "pelvis": "root",
    "left_hip": "lfemur",
    "right_hip": "rfemur",
    "left_knee": "ltibia",
    "right_knee": "rtibia",
    "left_foot": "lfoot",
    "right_foot": "rfoot",
    "left_shoulder": "lhumerus",
    "right_shoulder": "rhumerus",
    "left_elbow": "lradius",
    "right_elbow": "rradius",
    "head": "head",
}


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _convert_to_bvh(
    converter: pathlib.Path,
    asf_path: pathlib.Path,
    amc_path: pathlib.Path,
    bvh_path: pathlib.Path,
    source_fps: int,
) -> None:
    subprocess.run(
        [
            str(converter),
            str(asf_path),
            str(amc_path),
            "--fps",
            str(source_fps),
            "-o",
            str(bvh_path),
        ],
        check=True,
    )
    # GMR's vendored BVH reader splits motion rows on literal spaces only.
    bvh_path.write_text(bvh_path.read_text().replace("\t", " "))


def _load_cmu_frames(
    bvh_path: pathlib.Path,
    source_fps: float,
    output_fps: float,
) -> tuple[list[dict[str, tuple[np.ndarray, np.ndarray]]], int, float]:
    from general_motion_retargeting.utils.lafan1 import read_bvh, utils

    data = read_bvh(str(bvh_path))
    global_quaternion, global_position = utils.quat_fk(data.quats, data.pos, data.parents)
    bone_index = {name: index for index, name in enumerate(data.bones)}
    missing = sorted(set(CMU_TO_GMR.values()) - set(bone_index))
    if missing:
        raise ValueError(f"BVH is missing required CMU bones: {missing}")

    source_times = np.arange(data.pos.shape[0], dtype=np.float64) / source_fps
    duration = source_times[-1]
    output_times = np.arange(0.0, duration + 0.5 / output_fps, 1.0 / output_fps)
    output_times = np.minimum(output_times, duration)
    world_rotation = Rotation.from_matrix(CMU_TO_Z_UP)

    world_positions = {
        human_name: global_position[:, bone_index[cmu_name]]
        @ CMU_TO_Z_UP.T
        * CMU_LENGTH_TO_METERS
        for human_name, cmu_name in CMU_TO_GMR.items()
    }
    world_rotations = {
        human_name: world_rotation
        * Rotation.from_quat(global_quaternion[:, bone_index[cmu_name]], scalar_first=True)
        * world_rotation.inv()
        for human_name, cmu_name in CMU_TO_GMR.items()
    }

    # Every clip ends upright. Use that stable tail to put the subject in K1's
    # x-forward, y-left frame while retaining the captured joint rotations.
    calibration_frames = max(int(round(0.25 * source_fps)), 1)
    calibration_slice = slice(-calibration_frames, None)
    lateral = np.mean(
        world_positions["left_hip"][calibration_slice]
        - world_positions["right_hip"][calibration_slice],
        axis=0,
    )
    lateral[2] = 0.0
    lateral /= np.linalg.norm(lateral)
    up = np.array([0.0, 0.0, 1.0])
    forward = np.cross(lateral, up)
    forward /= np.linalg.norm(forward)
    world_to_canonical = Rotation.from_matrix(np.column_stack((forward, lateral, up)).T)
    origin = np.mean(world_positions["pelvis"][calibration_slice], axis=0)
    origin[2] = 0.0

    resampled: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for human_name in CMU_TO_GMR:
        positions = world_to_canonical.apply(world_positions[human_name] - origin)
        interpolated_position = np.column_stack(
            [np.interp(output_times, source_times, positions[:, axis]) for axis in range(3)]
        )

        aligned_rotation = world_to_canonical * world_rotations[human_name]
        interpolated_rotation = Slerp(source_times, aligned_rotation)(output_times)
        resampled[human_name] = (
            interpolated_position,
            interpolated_rotation.as_quat(scalar_first=True),
        )

    frames = [
        {name: (values[0][frame], values[1][frame]) for name, values in resampled.items()}
        for frame in range(len(output_times))
    ]
    return frames, data.pos.shape[0], duration


def retarget(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(args.gmr_root.resolve()))
    import mink
    import mujoco
    from general_motion_retargeting import GeneralMotionRetargeting

    config = json.loads(args.hardware_config.read_text())
    with tempfile.TemporaryDirectory(prefix="cmu140-") as temporary_dir:
        bvh_path = pathlib.Path(temporary_dir) / f"{args.amc.stem}.bvh"
        _convert_to_bvh(args.amc2bvh, args.asf, args.amc, bvh_path, args.source_fps)
        frames, source_frames, source_duration = _load_cmu_frames(
            bvh_path, args.source_fps, args.output_fps
        )

    solver = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="booster_k1",
        actual_human_height=args.human_height,
        verbose=False,
        use_velocity_limit=False,
    )
    # ASF/AMC FK already gives physical world-frame segment rotations.
    for human_name in solver.rot_offsets1:
        solver.rot_offsets1[human_name] = Rotation.identity()
    for human_name in solver.rot_offsets2:
        solver.rot_offsets2[human_name] = Rotation.identity()
    solver.max_iter = args.ik_iterations

    gmr_joint_names = _gmr_joint_names(solver)
    if gmr_joint_names != config["joint_names"]:
        raise ValueError(f"GMR/K1 joint order mismatch:\nGMR={gmr_joint_names}\nconfig={config['joint_names']}")
    for name, minimum, maximum in zip(
        config["joint_names"], config["position_minimum"], config["position_maximum"]
    ):
        joint_id = mujoco.mj_name2id(solver.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        solver.model.jnt_range[joint_id] = (minimum, maximum)
    solver.ik_limits = [mink.ConfigurationLimit(solver.model)]

    qpos_frames = []
    ik_errors = []
    for frame_index, human_frame in enumerate(frames):
        warmup_steps = args.initial_warmup_steps if frame_index == 0 else 1
        for _ in range(warmup_steps):
            qpos = solver.retarget(human_frame, offset_to_ground=False)
        qpos_frames.append(qpos)
        ik_errors.append(float(solver.error2()))
        if frame_index % 50 == 0 or frame_index == len(frames) - 1:
            print(f"retargeted frame {frame_index + 1}/{len(frames)} (IK error {ik_errors[-1]:.4f})")

    qpos_frames = np.asarray(qpos_frames, dtype=np.float64)
    if qpos_frames.shape != (len(frames), 29):
        raise ValueError(f"unexpected GMR qpos shape: {qpos_frames.shape}")
    root_height_adjustment = np.maximum(args.minimum_root_height - qpos_frames[:, 2], 0.0)
    qpos_frames[:, 2] += root_height_adjustment
    qpos_frames, handoff_transition_frames, handoff_hold_frames = _append_amp_handoff(
        qpos_frames,
        np.asarray(config["goal_position"], dtype=np.float64),
        args.output_fps,
        args.handoff_transition_s,
        args.handoff_hold_s,
        config["training"]["discovery"]["standing_height"],
    )

    parallel_module = (
        args.repo_root
        / "source/booster_train/booster_train/tasks/manager_based/fall_recovery/parallel_ankle.py"
    )
    parallel_ankle_class = _load_parallel_ankle(parallel_module)
    joint_positions, hardware_report = _hardware_project(qpos_frames[:, 7:], config, parallel_ankle_class)

    root_quaternion_wxyz = qpos_frames[:, 3:7]
    root_quaternion_wxyz /= np.linalg.norm(root_quaternion_wxyz, axis=1, keepdims=True)
    csv_motion = np.concatenate(
        (
            qpos_frames[:, :3],
            root_quaternion_wxyz[:, [1, 2, 3, 0]],
            joint_positions,
        ),
        axis=1,
    )
    goal_position = np.asarray(config["goal_position"], dtype=np.float64)
    report = {
        "source_asf": str(args.asf),
        "source_asf_sha256": _sha256(args.asf),
        "source_amc": str(args.amc),
        "source_amc_sha256": _sha256(args.amc),
        "amc2bvh_commit": args.amc2bvh_commit,
        "gmr_commit": args.gmr_commit,
        "source_frames": source_frames,
        "retargeted_frames": len(frames),
        "output_frames": len(qpos_frames),
        "source_fps": args.source_fps,
        "output_fps": args.output_fps,
        "source_duration_s": source_duration,
        "output_duration_s": float((len(qpos_frames) - 1) / args.output_fps),
        "handoff_transition_frames": handoff_transition_frames,
        "handoff_hold_frames": handoff_hold_frames,
        "human_height_m": args.human_height,
        "ik_error_mean": float(np.mean(ik_errors)),
        "ik_error_maximum": float(np.max(ik_errors)),
        "maximum_root_height_adjustment_m": float(np.max(root_height_adjustment)),
        "root_height_minimum_m": float(np.min(csv_motion[:, 2])),
        "root_height_maximum_m": float(np.max(csv_motion[:, 2])),
        "final_goal_maximum_error_rad": float(np.max(np.abs(joint_positions[-1] - goal_position))),
        **hardware_report,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n")

    excessive_adjustment = (
        report["maximum_joint_limit_clamp_rad"] > args.maximum_joint_clamp
        or report["maximum_parallel_ankle_projection_rad"] > args.maximum_ankle_projection
    )
    if excessive_adjustment and not args.allow_large_adjustment:
        raise RuntimeError(
            "retargeted motion requires an excessive hardware projection; inspect "
            f"{args.report} or rerun with --allow-large-adjustment only for visualization"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output, csv_motion, delimiter=",", fmt="%.9f")
    print(f"saved {args.output}")
    print(f"saved {args.report}")
    return report


def main() -> None:
    repo_root = pathlib.Path(__file__).resolve().parents[1]
    default_dataset = repo_root / "datasets/cmu_mocap/subject_140"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--amc", type=pathlib.Path, required=True)
    parser.add_argument("--asf", type=pathlib.Path, default=default_dataset / "140.asf")
    parser.add_argument("--amc2bvh", type=pathlib.Path, required=True)
    parser.add_argument("--amc2bvh-commit", default="f89c7312588a2a4265bf8384dc7309f461be91a7")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    parser.add_argument("--report", type=pathlib.Path)
    parser.add_argument("--gmr-root", type=pathlib.Path, required=True)
    parser.add_argument("--gmr-commit", default="5cd312e")
    parser.add_argument(
        "--hardware-config",
        type=pathlib.Path,
        default=(
            repo_root
            / "source/booster_train/booster_train/tasks/manager_based/fall_recovery/config/k1_fall_recovery.json"
        ),
    )
    parser.add_argument("--source-fps", type=int, default=120)
    parser.add_argument("--output-fps", type=int, default=50)
    parser.add_argument("--human-height", type=float, default=1.8)
    parser.add_argument("--ik-iterations", type=int, default=30)
    parser.add_argument("--initial-warmup-steps", type=int, default=20)
    parser.add_argument("--minimum-root-height", type=float, default=0.06)
    parser.add_argument("--handoff-transition-s", type=float, default=1.5)
    parser.add_argument("--handoff-hold-s", type=float, default=0.5)
    parser.add_argument("--maximum-joint-clamp", type=float, default=0.35)
    parser.add_argument("--maximum-ankle-projection", type=float, default=0.35)
    parser.add_argument("--allow-large-adjustment", action="store_true")
    args = parser.parse_args()
    args.repo_root = repo_root
    args.asf = args.asf.resolve()
    args.amc = args.amc.resolve()
    args.amc2bvh = args.amc2bvh.resolve()
    args.output = args.output.resolve()
    args.report = (args.report or args.output.with_suffix(".report.json")).resolve()
    args.gmr_root = args.gmr_root.resolve()
    args.hardware_config = args.hardware_config.resolve()
    retarget(args)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Retarget a DeepMimic humanoid get-up clip to hardware-feasible Booster K1 CSV."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
from typing import Any

import numpy as np
import torch
from scipy.spatial.transform import Rotation, Slerp

DM_TO_Z_UP = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0],
        [0.0, 1.0, 0.0],
    ]
)
GMR_HUMAN_JOINTS = {
    "pelvis": "root",
    "left_hip": "left_hip",
    "right_hip": "right_hip",
    "left_knee": "left_knee",
    "right_knee": "right_knee",
    "left_shoulder": "left_shoulder",
    "right_shoulder": "right_shoulder",
    "left_elbow": "left_elbow",
    "right_elbow": "right_elbow",
}
GMR_HUMAN_BODIES = {
    "head": "neck",
    "left_foot": "left_ankle",
    "right_foot": "right_ankle",
}


def _sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _transform(position: np.ndarray, orientation: Rotation) -> np.ndarray:
    result = np.eye(4)
    result[:3, :3] = orientation.as_matrix()
    result[:3, 3] = position
    return result


def _rotation_from_wxyz(quaternion: list[float] | np.ndarray) -> Rotation:
    value = np.asarray(quaternion, dtype=np.float64)
    norm = np.linalg.norm(value)
    if not np.isfinite(norm) or norm < 1.0e-8:
        raise ValueError(f"invalid quaternion: {value}")
    return Rotation.from_quat(value / norm, scalar_first=True)


class DeepMimicSkeleton:
    def __init__(self, character_path: pathlib.Path) -> None:
        character = json.loads(character_path.read_text())
        self.joints = sorted(character["Skeleton"]["Joints"], key=lambda joint: joint["ID"])
        self.bodies = {body["Name"]: body for body in character["BodyDefs"]}
        for expected_id, joint in enumerate(self.joints):
            if joint["ID"] != expected_id:
                raise ValueError("DeepMimic joints must have contiguous, ordered IDs")
            if expected_id > 0 and joint["Parent"] >= expected_id:
                raise ValueError(f"joint {joint['Name']} has an invalid parent")

    @staticmethod
    def _attach_transform(item: dict[str, Any]) -> np.ndarray:
        position = np.array([item["AttachX"], item["AttachY"], item["AttachZ"]], dtype=np.float64)
        euler = np.array(
            [item["AttachThetaX"], item["AttachThetaY"], item["AttachThetaZ"]], dtype=np.float64
        )
        return _transform(position, Rotation.from_euler("xyz", euler))

    @staticmethod
    def _frame_rotation(joint_type: str, values: np.ndarray) -> Rotation:
        if joint_type == "spherical":
            return _rotation_from_wxyz(values)
        if joint_type == "revolute":
            return Rotation.from_rotvec(np.array([0.0, 0.0, values[0]]))
        if joint_type in ("fixed", "none"):
            return Rotation.identity()
        raise ValueError(f"unsupported DeepMimic joint type: {joint_type}")

    def forward_kinematics(self, frame: list[float]) -> dict[str, np.ndarray]:
        values = np.asarray(frame, dtype=np.float64)
        cursor = 1
        root_position = values[cursor : cursor + 3]
        cursor += 3
        root_rotation = _rotation_from_wxyz(values[cursor : cursor + 4])
        cursor += 4

        world = {"root": _transform(root_position, root_rotation)}
        for joint in self.joints[1:]:
            joint_type = joint["Type"]
            width = {"spherical": 4, "revolute": 1, "fixed": 0}[joint_type]
            local_rotation = self._frame_rotation(joint_type, values[cursor : cursor + width])
            cursor += width
            parent = self.joints[joint["Parent"]]["Name"]
            world[joint["Name"]] = (
                world[parent] @ self._attach_transform(joint) @ _transform(np.zeros(3), local_rotation)
            )
        if cursor != len(values):
            raise ValueError(f"motion frame has {len(values) - cursor} unconsumed values")
        return world

    def _body_transform(self, joint_world: dict[str, np.ndarray], body_name: str) -> np.ndarray:
        return joint_world[body_name] @ self._attach_transform(self.bodies[body_name])

    @staticmethod
    def _to_z_up(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        position = DM_TO_Z_UP @ transform[:3, 3]
        orientation = DM_TO_Z_UP @ transform[:3, :3] @ DM_TO_Z_UP.T
        quaternion = Rotation.from_matrix(orientation).as_quat(scalar_first=True)
        return position, quaternion

    def gmr_human_frame(self, frame: list[float]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        joint_world = self.forward_kinematics(frame)
        result = {
            human_name: self._to_z_up(joint_world[deepmimic_name])
            for human_name, deepmimic_name in GMR_HUMAN_JOINTS.items()
        }
        result.update(
            {
                human_name: self._to_z_up(self._body_transform(joint_world, deepmimic_name))
                for human_name, deepmimic_name in GMR_HUMAN_BODIES.items()
            }
        )
        return result


def _load_parallel_ankle(module_path: pathlib.Path):
    spec = importlib.util.spec_from_file_location("k1_parallel_ankle", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.K1ParallelAnkleKinematics


def _gmr_joint_names(retarget) -> list[str]:
    import mujoco

    names = []
    for joint_id in range(retarget.model.njnt):
        if retarget.model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
            continue
        names.append(mujoco.mj_id2name(retarget.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id))
    return names


def _hardware_project(
    joint_positions: np.ndarray,
    config: dict[str, Any],
    parallel_ankle_class,
) -> tuple[np.ndarray, dict[str, Any]]:
    minimum = np.asarray(config["position_minimum"], dtype=np.float64)
    maximum = np.asarray(config["position_maximum"], dtype=np.float64)
    limited = np.clip(joint_positions, minimum, maximum)
    clamp = limited - joint_positions

    parallel = parallel_ankle_class(config["parallel_ankle"], device="cpu", dtype=torch.float64)
    ankle_reduction = np.zeros((limited.shape[0], 2), dtype=np.float64)
    for foot, indexes in enumerate(((14, 15), (20, 21))):
        current = parallel.serial_zero[2 * foot : 2 * foot + 2].unsqueeze(0)
        for frame_index in range(limited.shape[0]):
            desired = torch.from_numpy(limited[frame_index, list(indexes)]).unsqueeze(0)
            projected, reduction, feasible = parallel.project(desired, current, foot)
            if not bool(feasible.item()):
                raise RuntimeError(f"parallel ankle projection failed at frame {frame_index}, foot {foot}")
            limited[frame_index, list(indexes)] = projected.squeeze(0).numpy()
            ankle_reduction[frame_index, foot] = reduction.item()
            current = projected

    per_joint_clamp = np.max(np.abs(clamp), axis=0)
    report = {
        "maximum_joint_limit_clamp_rad": float(np.max(per_joint_clamp)),
        "per_joint_limit_clamp_rad": dict(zip(config["joint_names"], per_joint_clamp.tolist())),
        "maximum_parallel_ankle_projection_rad": float(np.max(ankle_reduction)),
        "mean_parallel_ankle_projection_rad": float(np.mean(ankle_reduction)),
    }
    return limited, report


def _append_amp_handoff(
    qpos_frames: np.ndarray,
    goal_position: np.ndarray,
    fps: float,
    transition_s: float,
    hold_s: float,
    standing_height: float,
) -> tuple[np.ndarray, int, int]:
    transition_frames = max(int(round(transition_s * fps)), 0)
    hold_frames = max(int(round(hold_s * fps)), 0)
    if transition_frames == 0 and hold_frames == 0:
        return qpos_frames, 0, 0

    start = qpos_frames[-1].copy()
    start_rotation = _rotation_from_wxyz(start[3:7])
    yaw = start_rotation.as_euler("xyz")[2]
    target_rotation = Rotation.from_euler("z", yaw)
    slerp = Slerp([0.0, 1.0], Rotation.concatenate([start_rotation, target_rotation]))
    target = start.copy()
    target[2] = standing_height
    target[3:7] = target_rotation.as_quat(scalar_first=True)
    target[7:] = goal_position

    transition = []
    for frame in range(1, transition_frames + 1):
        t = frame / transition_frames
        blend = 10.0 * t**3 - 15.0 * t**4 + 6.0 * t**5
        value = (1.0 - blend) * start + blend * target
        value[3:7] = slerp([blend]).as_quat(scalar_first=True)[0]
        transition.append(value)
    hold = [target.copy() for _ in range(hold_frames)]
    appended = np.asarray(transition + hold, dtype=np.float64)
    return np.concatenate((qpos_frames, appended), axis=0), transition_frames, hold_frames


def retarget(args: argparse.Namespace) -> dict[str, Any]:
    sys.path.insert(0, str(args.gmr_root.resolve()))
    import mink
    import mujoco
    from general_motion_retargeting import GeneralMotionRetargeting

    config = json.loads(args.hardware_config.read_text())
    motion = json.loads(args.motion.read_text())
    frames = motion["Frames"]
    skeleton = DeepMimicSkeleton(args.character)

    solver = GeneralMotionRetargeting(
        src_human="smplx",
        tgt_robot="booster_k1",
        actual_human_height=args.human_height,
        verbose=False,
        use_velocity_limit=False,
    )
    # The SMPL-X IK config compensates for SMPL-X's internal joint frames. DeepMimic
    # already provides physical world-frame segment orientations, so those offsets
    # would rotate an upright character onto its side.
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
    for frame_index, frame in enumerate(frames):
        human_frame = skeleton.gmr_human_frame(frame)
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

    nonzero_durations = [frame[0] for frame in frames if frame[0] > 0.0]
    source_fps = 1.0 / float(np.median(nonzero_durations))
    qpos_frames, handoff_transition_frames, handoff_hold_frames = _append_amp_handoff(
        qpos_frames,
        np.asarray(config["goal_position"], dtype=np.float64),
        source_fps,
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

    report = {
        "source_motion": str(args.motion.resolve()),
        "source_motion_sha256": _sha256(args.motion),
        "source_character": str(args.character.resolve()),
        "source_character_sha256": _sha256(args.character),
        "gmr_commit": args.gmr_commit,
        "source_frames": len(frames),
        "output_frames": len(qpos_frames),
        "source_fps": source_fps,
        "source_duration_s": float(sum(frame[0] for frame in frames)),
        "output_duration_s": float((len(qpos_frames) - 1) / source_fps),
        "handoff_transition_frames": handoff_transition_frames,
        "handoff_hold_frames": handoff_hold_frames,
        "human_height_m": args.human_height,
        "ik_error_mean": float(np.mean(ik_errors)),
        "ik_error_maximum": float(np.max(ik_errors)),
        "maximum_root_height_adjustment_m": float(np.max(root_height_adjustment)),
        "root_height_minimum_m": float(np.min(csv_motion[:, 2])),
        "root_height_maximum_m": float(np.max(csv_motion[:, 2])),
        "final_goal_maximum_error_rad": float(
            np.max(np.abs(joint_positions[-1] - np.asarray(config["goal_position"])))
        ),
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motion", type=pathlib.Path, required=True)
    parser.add_argument(
        "--character",
        type=pathlib.Path,
        default=repo_root / "datasets/deepmimic/data/characters/humanoid3d.txt",
    )
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
    args.motion = args.motion.resolve()
    args.character = args.character.resolve()
    args.output = args.output.resolve()
    args.report = (args.report or args.output.with_suffix(".report.json")).resolve()
    args.gmr_root = args.gmr_root.resolve()
    args.hardware_config = args.hardware_config.resolve()
    retarget(args)


if __name__ == "__main__":
    main()

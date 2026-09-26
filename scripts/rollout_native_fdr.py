#!/usr/bin/env python3
"""Run the deployed K1 fall-recovery teacher in Isaac and save its rollout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--direction", choices=("faceup", "facedown"), default="faceup")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--output_dir", type=Path, default=Path("outputs/native_fdr"))
parser.add_argument("--video", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.num_envs < 1:
    parser.error("--num_envs must be positive")
if args_cli.video and args_cli.num_envs != 1:
    parser.error("video recording requires --num_envs 1")
args_cli.enable_cameras = args_cli.video

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import booster_train.tasks  # noqa: F401, E402
import gymnasium as gym  # noqa: E402
import isaaclab_tasks  # noqa: F401, E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils import math as math_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from booster_train.tasks.manager_based.fall_recovery.joint_order import (  # noqa: E402
    deployment_joint_ids,
)
from booster_train.tasks.manager_based.fall_recovery.native_teacher import (  # noqa: E402
    K1NativeRecoveryTeacher,
)
from booster_train.tasks.manager_based.fall_recovery.robots.k1.env_cfg import (  # noqa: E402
    JOINT_NAMES,
)


def _initial_orientation(gravity: torch.Tensor) -> torch.Tensor:
    pitch = torch.asin(torch.clamp(gravity[:, 0], -1.0, 1.0))
    roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
    roll = torch.where(torch.abs(torch.cos(pitch)) < 0.1, torch.zeros_like(roll), roll)
    return math_utils.quat_from_euler_xyz(roll, pitch, torch.zeros_like(roll))


def _as_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def main() -> None:
    output_dir = args_cli.output_dir.expanduser().resolve() / args_cli.direction
    output_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = parse_env_cfg(
        "Booster-K1-Fall-Recovery-Native-Replay-v0",
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.terminations.recovered = None
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.seed = 42
    raw_env = gym.make(
        "Booster-K1-Fall-Recovery-Native-Replay-v0",
        cfg=env_cfg,
        render_mode="rgb_array" if args_cli.video else None,
    )

    teacher = K1NativeRecoveryTeacher(args_cli.direction, args_cli.num_envs, args_cli.device)
    video_length = int(np.ceil(teacher.duration / teacher.period)) + 1
    if args_cli.video:
        raw_env = gym.wrappers.RecordVideo(
            raw_env,
            video_folder=os.fspath(output_dir / "video"),
            step_trigger=lambda step: step == 0,
            video_length=video_length,
            disable_logger=True,
        )

    task_env = raw_env.unwrapped
    robot = task_env.scene["robot"]
    joint_ids = deployment_joint_ids(robot.joint_names, JOINT_NAMES)
    joint_id_tensor = torch.tensor(joint_ids, dtype=torch.long, device=task_env.device)
    action_term = task_env.action_manager.get_term("joint_pos")
    parallel = action_term._parallel

    raw_env.reset()
    initial_joint, initial_gravity, initial_height = teacher.initial_state()
    joint_position = initial_joint.repeat(args_cli.num_envs, 1)
    projection_reduction = torch.zeros(args_cli.num_envs, device=task_env.device)
    for foot, indexes in enumerate(((14, 15), (20, 21))):
        pair = torch.tensor(indexes, dtype=torch.long, device=task_env.device)
        neutral = parallel.serial_zero[2 * foot : 2 * foot + 2].to(torch.float32).expand(
            args_cli.num_envs, -1
        )
        projected, reduction, feasible = parallel.project(joint_position[:, pair], neutral, foot)
        if not torch.all(feasible):
            raise RuntimeError(f"native initial ankle state is infeasible for foot {foot}")
        joint_position[:, pair] = projected
        projection_reduction += reduction

    root_pose = robot.data.default_root_state[:, :7].clone()
    root_pose[:, :2] = task_env.scene.env_origins[:, :2]
    root_pose[:, 2] = task_env.scene.env_origins[:, 2] + initial_height[0]
    gravity = initial_gravity.repeat(args_cli.num_envs, 1)
    root_pose[:, 3:7] = _initial_orientation(gravity)
    zero_joint_velocity = torch.zeros_like(joint_position)
    zero_root_velocity = torch.zeros((args_cli.num_envs, 6), device=task_env.device)
    robot.write_joint_state_to_sim(
        joint_position,
        zero_joint_velocity,
        joint_ids=joint_ids,
    )
    robot.write_root_pose_to_sim(root_pose)
    robot.write_root_velocity_to_sim(zero_root_velocity)
    task_env.scene.write_data_to_sim()
    task_env.sim.render()
    task_env.scene.update(task_env.physics_dt)
    teacher.reset()

    log: dict[str, object] = {
        "fps": np.array([1.0 / teacher.period], dtype=np.float32),
        "joint_names": np.asarray(robot.joint_names),
        "body_names": np.asarray(robot.body_names),
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
        "teacher_action": [],
        "teacher_target": [],
        "teacher_normalized_action": [],
        "teacher_observation": [],
        "trajectory_index": [],
    }

    peak_height = torch.full((args_cli.num_envs,), -torch.inf, device=task_env.device)
    best_gravity_z = torch.full((args_cli.num_envs,), torch.inf, device=task_env.device)
    minimum_parallel_margin = torch.full((args_cli.num_envs,), torch.inf, device=task_env.device)
    termination_reason = "completed"
    step_count = 0
    maximum_steps = video_length
    while step_count < maximum_steps and simulation_app.is_running():
        current_gravity = robot.data.projected_gravity_b
        current_angular_velocity = robot.data.root_ang_vel_b
        current_position = robot.data.joint_pos[:, joint_id_tensor]
        current_velocity = robot.data.joint_vel[:, joint_id_tensor]

        with torch.inference_mode():
            result = teacher.step(
                current_gravity,
                current_angular_velocity,
                current_position,
                current_velocity,
            )

        log["joint_pos"].append(_as_numpy(robot.data.joint_pos[0]))
        log["joint_vel"].append(_as_numpy(robot.data.joint_vel[0]))
        log["body_pos_w"].append(_as_numpy(robot.data.body_pos_w[0]))
        log["body_quat_w"].append(_as_numpy(robot.data.body_quat_w[0]))
        log["body_lin_vel_w"].append(_as_numpy(robot.data.body_lin_vel_w[0]))
        log["body_ang_vel_w"].append(_as_numpy(robot.data.body_ang_vel_w[0]))
        log["teacher_action"].append(_as_numpy(result.action[0]))
        log["teacher_target"].append(_as_numpy(result.target[0]))
        log["teacher_normalized_action"].append(_as_numpy(result.training_normalized_action[0]))
        log["teacher_observation"].append(_as_numpy(result.observation[0]))
        log["trajectory_index"].append(int(result.trajectory_index[0].item()))

        with torch.inference_mode():
            _, _, terminated, truncated, _ = raw_env.step(result.normalized_action)

        height = robot.data.root_pos_w[:, 2] - task_env.scene.env_origins[:, 2]
        peak_height = torch.maximum(peak_height, height)
        best_gravity_z = torch.minimum(best_gravity_z, robot.data.projected_gravity_b[:, 2])
        motor_margins = []
        deployment_position = robot.data.joint_pos[:, joint_id_tensor]
        for foot, indexes in enumerate(((14, 15), (20, 21))):
            motor_margins.append(parallel.motor_margin(deployment_position[:, indexes], foot))
        parallel_margin = torch.amin(torch.cat(motor_margins, dim=-1), dim=-1)
        minimum_parallel_margin = torch.minimum(minimum_parallel_margin, parallel_margin)
        step_count += 1

        done = (terminated | truncated).reshape(-1)
        if torch.any(done):
            for name in task_env.termination_manager.active_terms:
                if torch.any(task_env.termination_manager.get_term(name)):
                    termination_reason = name
                    break
            break
        if torch.all(result.done):
            break

    for key in (
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "teacher_action",
        "teacher_target",
        "teacher_normalized_action",
        "teacher_observation",
        "trajectory_index",
    ):
        log[key] = np.asarray(log[key])
    rollout_path = output_dir / "native_teacher_rollout.npz"
    np.savez(rollout_path, **log)

    final_height = robot.data.root_pos_w[:, 2] - task_env.scene.env_origins[:, 2]
    final_gravity_z = robot.data.projected_gravity_b[:, 2]
    metrics = {
        "direction": args_cli.direction,
        "steps": step_count,
        "duration_s": step_count * teacher.period,
        "termination": termination_reason,
        "initial_ankle_projection_reduction_rad": float(torch.max(projection_reduction).item()),
        "peak_height_m": float(torch.max(peak_height).item()),
        "best_gravity_z": float(torch.min(best_gravity_z).item()),
        "final_height_m": float(torch.mean(final_height).item()),
        "final_gravity_z": float(torch.mean(final_gravity_z).item()),
        "minimum_parallel_motor_margin_rad": float(torch.min(minimum_parallel_margin).item()),
        "rollout": os.fspath(rollout_path),
    }
    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[RESULT] {json.dumps(metrics, sort_keys=True)}")
    raw_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

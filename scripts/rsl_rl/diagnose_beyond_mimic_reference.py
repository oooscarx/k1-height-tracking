"""Diagnose zero-residual tracking of a K1 BeyondMimic reference motion."""

from __future__ import annotations

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True, help="K1 BeyondMimic task ID.")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--seed", type=int, default=1)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import booster_train.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from booster_train.tasks.manager_based.fall_recovery.hardware_config import (  # noqa: E402
    K1_HARDWARE_CONFIG,
)
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
    )
    env_cfg.seed = args_cli.seed
    env_cfg.observations.policy.enable_corruption = False

    motion_cfg = env_cfg.commands.motion
    motion_cfg.pose_range = {}
    motion_cfg.velocity_range = {}
    motion_cfg.joint_position_range = (0.0, 0.0)
    motion_cfg.joint_velocity_scale = 0.0
    motion_cfg.failure_term_names = []

    # Preserve term names because reward terms depend on them, but make every
    # non-timeout failure impossible during this short diagnostic window.
    env_cfg.terminations.recovered = None
    env_cfg.terminations.anchor_pos.params["threshold"] = 1.0e6
    env_cfg.terminations.anchor_ori.params["threshold"] = 1.0e6
    env_cfg.terminations.end_effector_pos.params["threshold"] = 1.0e6
    if env_cfg.terminations.joint_limit is not None:
        env_cfg.terminations.joint_limit.params["margin"] = 1.0e6
    if env_cfg.terminations.parallel_ankle is not None:
        env_cfg.terminations.parallel_ankle.params["motor_tolerance"] = 1.0e6

    env = gym.make(args_cli.task, cfg=env_cfg)
    task_env = env.unwrapped
    action_term = task_env.action_manager.get_term("joint_pos")
    motion = task_env.command_manager.get_term("motion")

    names = K1_HARDWARE_CONFIG["joint_names"]
    minimum = torch.tensor(
        K1_HARDWARE_CONFIG["position_minimum"],
        dtype=torch.float32,
        device=task_env.device,
    )
    maximum = torch.tensor(
        K1_HARDWARE_CONFIG["position_maximum"],
        dtype=torch.float32,
        device=task_env.device,
    )
    zero_action = torch.zeros(task_env.num_envs, len(names), device=task_env.device)

    count = torch.zeros(len(names), dtype=torch.long, device=task_env.device)
    hard_count = torch.zeros_like(count)
    reference_count = torch.zeros_like(count)
    worst_lower = torch.zeros(len(names), device=task_env.device)
    worst_upper = torch.zeros(len(names), device=task_env.device)
    worst_reference_lower = torch.zeros(len(names), device=task_env.device)
    worst_reference_upper = torch.zeros(len(names), device=task_env.device)
    worst_tracking_error = torch.zeros(len(names), device=task_env.device)
    motor_count = torch.zeros(4, dtype=torch.long, device=task_env.device)
    motor_hard_count = torch.zeros_like(motor_count)
    worst_motor_margin = torch.full((4,), torch.inf, device=task_env.device)
    worst_reference_motor_margin = torch.full((4,), torch.inf, device=task_env.device)
    maximum_target_reduction = 0.0

    env.reset(seed=args_cli.seed)
    for step in range(args_cli.steps):
        with torch.inference_mode():
            env.step(zero_action)

        position = action_term.deployment_joint_position
        reference = motion.joint_pos[:, action_term._joint_ids]
        lower_error = torch.clamp(minimum - position, min=0.0)
        upper_error = torch.clamp(position - maximum, min=0.0)
        reference_lower_error = torch.clamp(minimum - reference, min=0.0)
        reference_upper_error = torch.clamp(reference - maximum, min=0.0)

        count += torch.count_nonzero((lower_error > 0.0) | (upper_error > 0.0), dim=0)
        hard_count += torch.count_nonzero((lower_error > 0.05) | (upper_error > 0.05), dim=0)
        reference_count += torch.count_nonzero(
            (reference_lower_error > 0.0) | (reference_upper_error > 0.0), dim=0
        )
        worst_lower = torch.maximum(worst_lower, torch.amax(lower_error, dim=0))
        worst_upper = torch.maximum(worst_upper, torch.amax(upper_error, dim=0))
        worst_reference_lower = torch.maximum(
            worst_reference_lower, torch.amax(reference_lower_error, dim=0)
        )
        worst_reference_upper = torch.maximum(
            worst_reference_upper, torch.amax(reference_upper_error, dim=0)
        )
        worst_tracking_error = torch.maximum(
            worst_tracking_error, torch.amax(torch.abs(position - reference), dim=0)
        )

        for foot, indexes in enumerate(((14, 15), (20, 21))):
            actual_margin = action_term._parallel.motor_margin(position[:, indexes], foot)
            reference_margin = action_term._parallel.motor_margin(reference[:, indexes], foot)
            motor_slice = slice(2 * foot, 2 * foot + 2)
            motor_count[motor_slice] += torch.count_nonzero(actual_margin < 0.0, dim=0)
            motor_hard_count[motor_slice] += torch.count_nonzero(actual_margin < -0.02, dim=0)
            worst_motor_margin[motor_slice] = torch.minimum(
                worst_motor_margin[motor_slice], torch.amin(actual_margin, dim=0)
            )
            worst_reference_motor_margin[motor_slice] = torch.minimum(
                worst_reference_motor_margin[motor_slice], torch.amin(reference_margin, dim=0)
            )

        maximum_target_reduction = max(maximum_target_reduction, action_term.target_reduction.max().item())
        if args_cli.progress_interval > 0 and (step + 1) % args_cli.progress_interval == 0:
            print(f"diagnostic progress: {step + 1}/{args_cli.steps}", flush=True)

    total_samples = task_env.num_envs * args_cli.steps
    print(f"\nZero-residual reference diagnostic: task={args_cli.task}")
    print(f"envs={task_env.num_envs} steps={args_cli.steps} samples={total_samples}")
    print(f"maximum_target_reduction={maximum_target_reduction:.6f} rad")
    print("\nJoint limit and tracking statistics (deployment order):")
    print("id name logical hard ref worst_low worst_high ref_low ref_high max_track")
    for index, name in enumerate(names):
        print(
            f"{index:2d} {name:24s} {count[index].item():7d} {hard_count[index].item():7d} "
            f"{reference_count[index].item():7d} {worst_lower[index].item():9.5f} "
            f"{worst_upper[index].item():10.5f} {worst_reference_lower[index].item():8.5f} "
            f"{worst_reference_upper[index].item():9.5f} {worst_tracking_error[index].item():9.5f}"
        )

    print("\nParallel ankle motor statistics:")
    print("motor infeasible hard(<-0.02) worst_actual_margin worst_reference_margin")
    for motor in range(4):
        print(
            f"{motor:5d} {motor_count[motor].item():10d} {motor_hard_count[motor].item():13d} "
            f"{worst_motor_margin[motor].item():19.6f} "
            f"{worst_reference_motor_margin[motor].item():22.6f}"
        )

    sys.stdout.flush()
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

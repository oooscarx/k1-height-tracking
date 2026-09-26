"""Evaluate a deterministic K1 BeyondMimic recovery checkpoint."""

from __future__ import annotations

import argparse
import faulthandler
import json
import signal
import sys
import traceback
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=1024)
parser.add_argument("--episodes", type=int, default=4096)
parser.add_argument("--mode", choices=("canonical", "reference"), default="canonical")
parser.add_argument("--seed", type=int, default=1)
parser.add_argument("--max_vector_steps", type=int, default=600)
parser.add_argument("--progress_interval", type=int, default=25)
parser.add_argument("--trace_env", type=int, default=-1, help="Environment index to trace; negative disables tracing.")
parser.add_argument("--trace_steps", type=int, default=40, help="Maximum vector steps printed for --trace_env.")
parser.add_argument("--zero_actions", action="store_true", help="Evaluate the reference-centered zero residual baseline.")
parser.add_argument(
    "--safety_stride",
    type=int,
    default=0,
    help="Sample detailed logical/parallel safety metrics every N steps; zero disables them.",
)
parser.add_argument("--output", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
faulthandler.register(signal.SIGUSR1)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import booster_train.tasks  # noqa: E402, F401
import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from booster_train.tasks.manager_based.fall_recovery.hardware_config import (  # noqa: E402
    K1_HARDWARE_CONFIG,
)
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402


def tensor_stats(value: torch.Tensor) -> dict[str, float]:
    value = value.to(torch.float32)
    return {
        "mean": torch.mean(value).item(),
        "p50": torch.quantile(value, 0.5).item(),
        "p90": torch.quantile(value, 0.9).item(),
        "worst": torch.amax(value).item(),
    }


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    agent_cfg = load_cfg_from_registry(args_cli.task, "rsl_rl_cfg_entry_point")
    env_cfg.seed = args_cli.seed
    agent_cfg.seed = args_cli.seed
    agent_cfg.device = args_cli.device or agent_cfg.device
    env_cfg.observations.policy.enable_corruption = False

    motion_cfg = env_cfg.commands.motion
    motion_cfg.play = args_cli.mode == "canonical"
    motion_cfg.pose_range = {}
    motion_cfg.velocity_range = {}
    motion_cfg.joint_position_range = (0.0, 0.0)
    motion_cfg.joint_velocity_scale = 0.0

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    checkpoint = args_cli.checkpoint.expanduser().resolve()
    print(f"[INFO] Loading checkpoint: {checkpoint}")
    runner.load(str(checkpoint))
    policy = runner.get_inference_policy(device=raw_env.unwrapped.device)

    task_env = raw_env.unwrapped
    robot = task_env.scene["robot"]
    action_term = task_env.action_manager.get_term("joint_pos")
    motion = task_env.command_manager.get_term("motion")
    success_cfg = task_env.cfg.terminations.recovered.params
    goal = torch.tensor(
        K1_HARDWARE_CONFIG["goal_position"],
        dtype=torch.float32,
        device=task_env.device,
    )
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

    observations = env.get_observations()
    if isinstance(observations, tuple):
        observations = observations[0]

    peak_height = torch.full((task_env.num_envs,), -torch.inf, device=task_env.device)
    best_gravity_z = torch.full_like(peak_height, torch.inf)
    best_pose_error = torch.full_like(peak_height, torch.inf)
    best_joint_speed = torch.full_like(peak_height, torch.inf)
    maximum_limit_violation = torch.zeros_like(peak_height)
    minimum_motor_margin = torch.full_like(peak_height, torch.inf)
    reached_instant_success = torch.zeros(task_env.num_envs, dtype=torch.bool, device=task_env.device)

    completed: dict[str, list[torch.Tensor]] = {
        "peak_height": [],
        "best_gravity_z": [],
        "best_pose_error": [],
        "best_joint_speed": [],
        "maximum_limit_violation": [],
        "minimum_motor_margin": [],
        "instant_success": [],
    }
    term_names = ("recovered", "anchor_pos", "anchor_ori", "end_effector_pos", "time_out")
    termination_counts_gpu = {name: torch.zeros((), dtype=torch.long, device=task_env.device) for name in term_names}
    episode_count_gpu = torch.zeros((), dtype=torch.long, device=task_env.device)

    for vector_step in range(args_cli.max_vector_steps):
        with torch.inference_mode():
            position = action_term.deployment_joint_position
            velocity = robot.data.joint_vel[:, action_term._joint_ids]
            height = robot.data.root_pos_w[:, 2] - task_env.scene.env_origins[:, 2]
            gravity_z = robot.data.projected_gravity_b[:, 2]
            pose_error = torch.amax(torch.abs(position[:, 2:] - goal[2:]), dim=-1)
            joint_speed = torch.amax(torch.abs(velocity[:, 2:]), dim=-1)
            peak_height = torch.maximum(peak_height, height)
            best_gravity_z = torch.minimum(best_gravity_z, gravity_z)
            best_pose_error = torch.minimum(best_pose_error, pose_error)
            best_joint_speed = torch.minimum(best_joint_speed, joint_speed)
            if args_cli.safety_stride > 0 and vector_step % args_cli.safety_stride == 0:
                limit_violation = torch.amax(
                    torch.maximum(
                        torch.clamp(minimum - position, min=0.0),
                        torch.clamp(position - maximum, min=0.0),
                    ),
                    dim=-1,
                )
                motor_margin = torch.full_like(height, torch.inf)
                for foot, indexes in enumerate(((14, 15), (20, 21))):
                    margin = action_term._parallel.motor_margin(position[:, indexes], foot)
                    motor_margin = torch.minimum(motor_margin, torch.amin(margin, dim=-1))
                maximum_limit_violation = torch.maximum(maximum_limit_violation, limit_violation)
                minimum_motor_margin = torch.minimum(minimum_motor_margin, motor_margin)
            reached_instant_success |= (
                (height >= success_cfg["minimum_height"])
                & (gravity_z <= success_cfg["maximum_gravity_z"])
                & (
                    torch.linalg.vector_norm(robot.data.root_ang_vel_b, dim=-1)
                    <= success_cfg["maximum_angular_velocity"]
                )
                & (pose_error <= success_cfg["maximum_body_pose_error"])
                & (joint_speed <= success_cfg["maximum_body_joint_velocity"])
            )

            actions = torch.zeros(task_env.num_envs, env.num_actions, device=task_env.device)
            if not args_cli.zero_actions:
                actions = policy(observations)
            if 0 <= args_cli.trace_env < task_env.num_envs and vector_step < args_cli.trace_steps:
                trace_env = args_cli.trace_env
                end_effector_indexes = [
                    motion.cfg.body_names.index(name)
                    for name in ("left_hand_link", "right_hand_link", "left_foot_link", "right_foot_link")
                ]
                end_effector_z_error = torch.abs(
                    motion.body_pos_relative_w[trace_env, end_effector_indexes, 2]
                    - motion.robot_body_pos_w[trace_env, end_effector_indexes, 2]
                )
                reference_position = motion.joint_pos[trace_env, action_term._joint_ids]
                print(
                    "[TRACE pre] "
                    f"step={vector_step} frame={motion.time_steps[trace_env].item()} "
                    f"height={height[trace_env].item():.4f} gravity_z={gravity_z[trace_env].item():.4f} "
                    f"eef_z={[round(value, 4) for value in end_effector_z_error.tolist()]} "
                    f"q_error={torch.amax(torch.abs(position[trace_env] - reference_position)).item():.4f} "
                    f"qd_max={torch.amax(torch.abs(velocity[trace_env])).item():.4f} "
                    f"action_max={torch.amax(torch.abs(actions[trace_env])).item():.4f}",
                    flush=True,
                )
            observations, _, done, _ = env.step(actions)
            done = done.reshape(-1).bool()
            if 0 <= args_cli.trace_env < task_env.num_envs and vector_step < args_cli.trace_steps:
                trace_env = args_cli.trace_env
                active_terms = [
                    name for name in term_names if task_env.termination_manager.get_term(name)[trace_env].item()
                ]
                print(
                    "[TRACE post] "
                    f"step={vector_step} target_reduction={action_term.target_reduction[trace_env].item():.4f} "
                    f"done={done[trace_env].item()} terms={active_terms}",
                    flush=True,
                )

            nan = torch.full_like(peak_height, torch.nan)
            for name, value in (
                ("peak_height", peak_height),
                ("best_gravity_z", best_gravity_z),
                ("best_pose_error", best_pose_error),
                ("best_joint_speed", best_joint_speed),
                ("maximum_limit_violation", maximum_limit_violation),
                ("minimum_motor_margin", minimum_motor_margin),
                ("instant_success", reached_instant_success.to(torch.float32)),
            ):
                completed[name].append(torch.where(done, value, nan))
            for name in term_names:
                termination_counts_gpu[name] += torch.count_nonzero(task_env.termination_manager.get_term(name))
            episode_count_gpu += torch.count_nonzero(done)
            peak_height = torch.where(done, -torch.inf, peak_height)
            best_gravity_z = torch.where(done, torch.inf, best_gravity_z)
            best_pose_error = torch.where(done, torch.inf, best_pose_error)
            best_joint_speed = torch.where(done, torch.inf, best_joint_speed)
            maximum_limit_violation = torch.where(done, 0.0, maximum_limit_violation)
            minimum_motor_margin = torch.where(done, torch.inf, minimum_motor_margin)
            reached_instant_success &= ~done

        vector_steps = vector_step + 1
        if args_cli.progress_interval > 0 and vector_steps % args_cli.progress_interval == 0:
            print(f"evaluation progress: vector_steps={vector_steps}/{args_cli.max_vector_steps}", flush=True)

    actual_episodes = int(episode_count_gpu.item())
    completed_tensors = {}
    for name, frames in completed.items():
        value = torch.stack(frames).flatten()
        completed_tensors[name] = value[torch.isfinite(value)].detach().cpu()
    completed = completed_tensors
    if actual_episodes == 0:
        raise RuntimeError("evaluation completed no episodes; increase --max_vector_steps")
    termination_counts = {name: int(value.item()) for name, value in termination_counts_gpu.items()}
    instant_successes = int(torch.count_nonzero(completed["instant_success"]).item())
    result = {
        "task": args_cli.task,
        "mode": args_cli.mode,
        "checkpoint": str(checkpoint),
        "requested_episodes": args_cli.episodes,
        "episodes": actual_episodes,
        "vector_steps": vector_steps,
        "stable_success_rate": termination_counts["recovered"] / max(actual_episodes, 1),
        "instant_success_rate": instant_successes / max(actual_episodes, 1),
        "termination_counts": termination_counts,
        "peak_height_m": tensor_stats(completed["peak_height"]),
        "best_gravity_z": tensor_stats(completed["best_gravity_z"]),
        "minimum_pose_error_rad": tensor_stats(completed["best_pose_error"]),
        "minimum_joint_speed_rad_s": tensor_stats(completed["best_joint_speed"]),
    }
    if args_cli.safety_stride > 0:
        result["maximum_logical_limit_violation_rad"] = tensor_stats(completed["maximum_limit_violation"])
        result["maximum_parallel_motor_shortfall_rad"] = tensor_stats(-completed["minimum_motor_margin"])
    print(f"[RESULT] {json.dumps(result, indent=2, sort_keys=True)}")
    if args_cli.output is not None:
        args_cli.output.parent.mkdir(parents=True, exist_ok=True)
        args_cli.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    sys.stdout.flush()
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        raise
    finally:
        simulation_app.close()

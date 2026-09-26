"""Export an unassisted Stage-I recovery rollout as a Stage-II reference."""

from __future__ import annotations

import argparse
import os
import sys

from isaaclab.app import AppLauncher

import cli_args  # isort: skip

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True, help="FaceUp or FaceDown discovery task.")
parser.add_argument("--output", required=True, help="Output .npz reference path.")
parser.add_argument("--max_steps", type=int, default=500, help="Maximum rollout length at 50 Hz.")
parser.add_argument("--agent", default="rsl_rl_cfg_entry_point", help="Agent configuration entry point.")
parser.add_argument("--seed", type=int, default=1)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import booster_train.tasks  # noqa: F401
import gymnasium as gym
import numpy as np
import torch
from booster_train.tasks.manager_based.fall_recovery.hardware_config import K1_HARDWARE_CONFIG
from booster_train.tasks.manager_based.fall_recovery.joint_order import deployment_joint_ids
from isaaclab.utils.assets import retrieve_file_path
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg, agent_cfg: RslRlOnPolicyRunnerCfg) -> None:
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = 1
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    discovery = env_cfg.commands.recovery
    discovery.maximum_assistance_force = 0.0
    discovery.joint_noise = 0.0
    discovery.root_position_noise = 0.0
    discovery.orientation_noise = 0.0
    discovery.standing_reset_fraction_start = 0.0
    discovery.standing_reset_fraction_end = 0.0
    env_cfg.observations.policy.enable_corruption = False
    env_cfg.terminations.recovered = None

    log_root = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        checkpoint = retrieve_file_path(args_cli.checkpoint)
    else:
        checkpoint = get_checkpoint_path(log_root, agent_cfg.load_run, agent_cfg.load_checkpoint)

    env = gym.make(args_cli.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(checkpoint)
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    robot = env.unwrapped.scene["robot"]
    action_term = env.unwrapped.action_manager.get_term("joint_pos")
    joint_ids = deployment_joint_ids(robot.joint_names, K1_HARDWARE_CONFIG["joint_names"])
    training = K1_HARDWARE_CONFIG["training"]
    goal = torch.tensor(K1_HARDWARE_CONFIG["goal_position"], device=env.unwrapped.device)
    origin_z = env.unwrapped.scene.env_origins[0, 2]
    hold_steps = round(training["success_hold_s"] * K1_HARDWARE_CONFIG["policy_rate_hz"])

    joint_frames: list[np.ndarray] = []
    gravity_frames: list[np.ndarray] = []
    height_frames: list[np.ndarray] = []
    stable_steps = 0
    best_stable_steps = 0
    best_progress = float("-inf")
    best_snapshot: dict[str, float | int | bool] = {}
    last_snapshot: dict[str, float | int | bool] = {}
    first_action: list[float] = []
    maximum_action = 0.0

    def recovery_snapshot(step: int) -> tuple[dict[str, float | int | bool], bool]:
        joint_pos = robot.data.joint_pos[0, joint_ids]
        joint_vel = robot.data.joint_vel[0, joint_ids]
        gravity = robot.data.projected_gravity_b[0]
        height = robot.data.root_pos_w[0, 2] - origin_z
        angular_speed = torch.linalg.vector_norm(robot.data.root_ang_vel_b[0])
        pose_error = torch.amax(torch.abs(joint_pos[2:] - goal[2:]))
        joint_speed = torch.amax(torch.abs(joint_vel[2:]))
        target_error = torch.amax(torch.abs(robot.data.joint_pos_target[0, joint_ids] - joint_pos))
        processed_error = torch.amax(torch.abs(action_term.processed_actions[0] - joint_pos))
        computed_torque = torch.amax(torch.abs(robot.data.computed_torque[0, joint_ids]))
        applied_torque = torch.amax(torch.abs(robot.data.applied_torque[0, joint_ids]))
        conditions = {
            "height_ok": bool(height >= training["success_height"]),
            "upright_ok": bool(gravity[2] <= training["success_gravity_z"]),
            "angular_speed_ok": bool(angular_speed <= training["success_angular_velocity"]),
            "pose_ok": bool(pose_error <= training["success_body_pose_error"]),
            "joint_speed_ok": bool(joint_speed <= training["success_body_joint_velocity"]),
        }
        snapshot: dict[str, float | int | bool] = {
            "step": step,
            "height": height.item(),
            "gravity_z": gravity[2].item(),
            "angular_speed": angular_speed.item(),
            "pose_error": pose_error.item(),
            "joint_speed": joint_speed.item(),
            "target_error": target_error.item(),
            "processed_error": processed_error.item(),
            "computed_torque": computed_torque.item(),
            "applied_torque": applied_torque.item(),
            **conditions,
        }
        return snapshot, all(conditions.values())

    def failure_summary(reason: str, final_snapshot: dict[str, float | int | bool]) -> str:
        return (
            f"{reason}; best_stable_steps={best_stable_steps}/{hold_steps}; "
            f"maximum_action={maximum_action:.6f}; first_action={first_action}; "
            f"best={best_snapshot}; final={final_snapshot}"
        )

    obs = wrapped.get_observations()
    if isinstance(obs, tuple):
        obs = obs[0]

    for step in range(args_cli.max_steps):
        joint_pos = robot.data.joint_pos[0, joint_ids]
        gravity = robot.data.projected_gravity_b[0]
        height = robot.data.root_pos_w[0, 2] - origin_z

        joint_frames.append(joint_pos.detach().cpu().numpy().copy())
        gravity_frames.append(gravity.detach().cpu().numpy().copy())
        height_frames.append(np.asarray([height.item()], dtype=np.float32))

        snapshot, stable = recovery_snapshot(step)
        last_snapshot = snapshot
        stable_steps = stable_steps + 1 if stable else 0
        best_stable_steps = max(best_stable_steps, stable_steps)
        progress = float(snapshot["height"]) / training["success_height"] - float(snapshot["gravity_z"])
        if progress > best_progress:
            best_progress = progress
            best_snapshot = snapshot
        if stable_steps >= hold_steps:
            break

        with torch.inference_mode():
            actions = policy(obs)
            if not first_action:
                first_action.extend(actions[0].detach().cpu().tolist())
            maximum_action = max(maximum_action, torch.amax(torch.abs(actions)).item())
            obs, _, done, _ = wrapped.step(actions)
        if bool(done[0]):
            manager = env.unwrapped.termination_manager
            reasons = [
                name for name in ("time_out", "joint_limit", "parallel_ankle") if bool(manager.get_term(name)[0])
            ]
            raise RuntimeError(failure_summary(f"rollout terminated by {reasons}", last_snapshot))
    else:
        final_snapshot, _ = recovery_snapshot(args_cli.max_steps)
        raise RuntimeError(
            failure_summary(
                f"checkpoint did not recover without assistance in {args_cli.max_steps} steps",
                final_snapshot,
            )
        )

    output = os.path.abspath(args_cli.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    np.savez_compressed(
        output,
        joint=np.asarray(joint_frames, dtype=np.float32),
        gravity=np.asarray(gravity_frames, dtype=np.float32),
        height=np.asarray(height_frames, dtype=np.float32),
        fps=np.asarray(K1_HARDWARE_CONFIG["policy_rate_hz"], dtype=np.float32),
        source_task=np.asarray(args_cli.task),
        source_checkpoint=np.asarray(os.path.abspath(checkpoint)),
        assistance_force=np.asarray(0.0, dtype=np.float32),
    )
    duration = (len(joint_frames) - 1) / K1_HARDWARE_CONFIG["policy_rate_hz"]
    print(f"Exported {len(joint_frames)} frames ({duration:.2f} s) to {output}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

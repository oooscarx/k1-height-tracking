# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Record deterministic K1 AMP recovery from a selected fall mode."""

import argparse
import os
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play and record a K1 AMP recovery checkpoint.")
parser.add_argument("--task", default="Booster-K1-Fall-Recovery-AMP-v0")
parser.add_argument("--checkpoint", required=True)
parser.add_argument(
    "--reset_mode",
    default="faceup",
    choices=["faceup", "facedown", "side_left", "side_right", "random", "failure", "reference"],
)
parser.add_argument("--failure_state_blend", type=float, default=None)
parser.add_argument("--failure_state_index", type=int, default=None)
parser.add_argument("--random_fall_difficulty", type=float, default=None)
parser.add_argument("--reference_phase_min", type=float, default=None)
parser.add_argument("--reference_phase_max", type=float, default=None)
parser.add_argument("--reference_clip_index", type=int, default=None)
parser.add_argument("--joint_state_tolerance", type=float, default=None)
parser.add_argument("--parallel_state_tolerance", type=float, default=None)
parser.add_argument("--episode_length_s", type=float, default=None)
parser.add_argument(
    "--observe_after_unsafe",
    action="store_true",
    help="Record through simulated safety violations; never use this mode for acceptance.",
)
parser.add_argument("--video_length", type=int, default=500)
parser.add_argument("--output_dir", type=Path, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import booster_train.tasks  # noqa: F401
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from rollout_numeric_guard import configure_normalized_policy_action_bounds
from skrl.utils.runner.torch import Runner


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.amp_reset_mode = args_cli.reset_mode
    env_cfg.observations.policy.enable_corruption = False
    if args_cli.episode_length_s is not None:
        if args_cli.episode_length_s <= 0.0:
            raise ValueError("--episode_length_s must be positive")
        env_cfg.episode_length_s = args_cli.episode_length_s
    if args_cli.failure_state_blend is not None:
        if not 0.0 <= args_cli.failure_state_blend <= 1.0:
            raise ValueError("--failure_state_blend must be in [0, 1]")
        env_cfg.amp_failure_state_blend_start = args_cli.failure_state_blend
        env_cfg.amp_failure_state_blend_end = args_cli.failure_state_blend
    if args_cli.random_fall_difficulty is not None:
        if not 0.0 <= args_cli.random_fall_difficulty <= 1.0:
            raise ValueError("--random_fall_difficulty must be in [0, 1]")
        env_cfg.amp_random_fall_difficulty_start = args_cli.random_fall_difficulty
        env_cfg.amp_random_fall_difficulty_end = args_cli.random_fall_difficulty
    for argument, attribute in (
        ("reference_phase_min", "amp_reference_phase_minimum"),
        ("reference_phase_max", "amp_reference_phase_maximum"),
    ):
        value = getattr(args_cli, argument)
        if value is not None:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"--{argument} must be in [0, 1]")
            setattr(env_cfg, attribute, value)
    if env_cfg.amp_reference_phase_minimum > env_cfg.amp_reference_phase_maximum:
        raise ValueError("reference phase minimum must not exceed maximum")
    if args_cli.reference_clip_index is not None:
        if args_cli.reference_clip_index < 0:
            raise ValueError("--reference_clip_index must be non-negative")
        env_cfg.amp_reference_clip_indices = [args_cli.reference_clip_index]
    if args_cli.joint_state_tolerance is not None:
        if args_cli.joint_state_tolerance < 0.0:
            raise ValueError("--joint_state_tolerance must be non-negative")
        params = env_cfg.terminations.joint_limit.params
        if "margin_start" in params:
            params["margin_start"] = args_cli.joint_state_tolerance
            params["margin_end"] = args_cli.joint_state_tolerance
        else:
            params["margin"] = args_cli.joint_state_tolerance
    if args_cli.parallel_state_tolerance is not None:
        if args_cli.parallel_state_tolerance < 0.0:
            raise ValueError("--parallel_state_tolerance must be non-negative")
        params = env_cfg.terminations.parallel_ankle.params
        if "motor_tolerance_start" in params:
            params["motor_tolerance_start"] = args_cli.parallel_state_tolerance
            params["motor_tolerance_end"] = args_cli.parallel_state_tolerance
        else:
            params["motor_tolerance"] = args_cli.parallel_state_tolerance
    if args_cli.observe_after_unsafe:
        env_cfg.terminations.recovered = None
        env_cfg.terminations.joint_limit = None
        env_cfg.terminations.parallel_ankle = None
        env_cfg.terminations.excessive_impact = None
        if hasattr(env_cfg.rewards, "unsafe_termination"):
            env_cfg.rewards.unsafe_termination = None
    agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_amp_cfg_entry_point")
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    agent_cfg["agent"]["experiment"]["write_interval"] = 0
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = 0

    checkpoint = Path(args_cli.checkpoint).expanduser().resolve()
    output_dir = args_cli.output_dir or checkpoint.parent.parent / "videos" / args_cli.reset_mode
    raw_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    raw_env = gym.wrappers.RecordVideo(
        raw_env,
        video_folder=os.fspath(output_dir),
        step_trigger=lambda step: step == 0,
        video_length=args_cli.video_length,
        disable_logger=True,
    )
    env = SkrlVecEnvWrapper(raw_env, ml_framework="torch")
    runner = Runner(env, agent_cfg)
    configure_normalized_policy_action_bounds(runner.agent)
    print(f"[INFO] Loading checkpoint: {checkpoint}")
    runner.agent.load(os.fspath(checkpoint))
    runner.agent.enable_training_mode(False, apply_to_models=True)

    task_env = raw_env.unwrapped
    if args_cli.failure_state_index is not None:
        if args_cli.reset_mode != "failure":
            raise ValueError("--failure_state_index requires --reset_mode failure")
        state_count = task_env._failure_states["joint_position"].shape[0]
        if not 0 <= args_cli.failure_state_index < state_count:
            raise ValueError(
                f"--failure_state_index must be in [0, {state_count - 1}]"
            )
        index = args_cli.failure_state_index
        task_env._failure_states = {
            name: value[index : index + 1] for name, value in task_env._failure_states.items()
        }

    observations, _ = env.reset()
    episode_finished = False
    for _ in range(args_cli.video_length):
        if not simulation_app.is_running():
            break
        with torch.inference_mode():
            outputs = runner.agent.act(observations, None, timestep=0, timesteps=0)
            actions = outputs[-1].get("mean_actions", outputs[0])
            observations, _, terminated, truncated, _ = env.step(actions)
            if torch.any(terminated | truncated):
                episode_finished = True
                break

    if raw_env.recording:
        if episode_finished and len(raw_env.recorded_frames) > 1:
            # Isaac auto-resets before returning the terminal step, so its render
            # belongs to the next episode rather than the completed recovery.
            raw_env.recorded_frames.pop()
        while len(raw_env.recorded_frames) > 1 and not np.any(raw_env.recorded_frames[0]):
            raw_env.recorded_frames.pop(0)

    env.close()
    print(f"[INFO] Wrote {args_cli.reset_mode} video under: {output_dir}")


if __name__ == "__main__":
    main()
    simulation_app.close()

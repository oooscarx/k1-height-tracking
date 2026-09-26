# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Distill the native recovery trajectory into a phase-free proprioceptive policy."""

import argparse
import copy
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Phase-free behavior cloning for K1 recovery.")
parser.add_argument(
    "--task",
    default="Booster-K1-Fall-Recovery-AMP-FaceUp-Native-TaskDriven-v0",
)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--num_envs", type=int, default=4096)
parser.add_argument("--iterations", type=int, default=12000)
parser.add_argument("--learning_rate", type=float, default=1.0e-4)
parser.add_argument("--student_rollout_fraction", type=float, default=0.0)
parser.add_argument("--save_interval", type=int, default=3000)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import booster_train.tasks  # noqa: F401
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import numpy as np
import torch
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg
from skrl.utils.model_instantiators.torch import gaussian_model


def _checkpoint_path(output: Path, iteration: int | None = None) -> Path:
    if iteration is None:
        return output
    return output.with_name(f"{output.stem}_{iteration}{output.suffix}")


def _save_checkpoint(
    source_checkpoint: dict,
    policy,
    output: Path,
    iteration: int | None = None,
) -> None:
    checkpoint = copy.deepcopy(source_checkpoint)
    checkpoint["policy"] = policy.state_dict()
    path = _checkpoint_path(output, iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    print(f"[BC] saved checkpoint: {path}", flush=True)


def main() -> None:
    if args_cli.num_envs < 1:
        raise ValueError("--num_envs must be positive")
    if args_cli.iterations < 1:
        raise ValueError("--iterations must be positive")
    if args_cli.learning_rate <= 0.0:
        raise ValueError("--learning_rate must be positive")
    if not 0.0 <= args_cli.student_rollout_fraction <= 1.0:
        raise ValueError("--student_rollout_fraction must be in [0, 1]")
    if args_cli.save_interval < 1:
        raise ValueError("--save_interval must be positive")

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env_cfg.amp_reset_mode = "reference"
    env_cfg.amp_reference_terminal_reset_probability = 0.0
    env_cfg.observations.policy.enable_corruption = False
    agent_cfg = load_cfg_from_registry(args_cli.task, "skrl_amp_cfg_entry_point")
    env_cfg.seed = int(agent_cfg["seed"])

    raw_env = gym.make(args_cli.task, cfg=env_cfg)
    task_env = raw_env.unwrapped
    observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(raw_env.observation_space["policy"].shape[-1],),
        dtype=np.float32,
    )
    state_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(raw_env.observation_space["critic"].shape[-1],),
        dtype=np.float32,
    )
    action_space = gym.spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(task_env.action_manager.total_action_dim,),
        dtype=np.float32,
    )
    model_cfg = copy.deepcopy(agent_cfg["models"]["policy"])
    model_cfg.pop("class")
    policy = gaussian_model(
        observation_space=observation_space,
        state_space=state_space,
        action_space=action_space,
        device=task_env.device,
        **model_cfg,
    )
    policy.init_state_dict(role="policy")

    checkpoint_path = Path(args_cli.checkpoint).expanduser().resolve()
    source_checkpoint = torch.load(checkpoint_path, map_location=task_env.device, weights_only=False)
    policy.load_state_dict(source_checkpoint["policy"])
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=args_cli.learning_rate)

    observations, _ = raw_env.reset()
    print(
        f"[BC] phase-free distillation: envs={args_cli.num_envs}, "
        f"iterations={args_cli.iterations}, lr={args_cli.learning_rate:g}, "
        f"student_rollout_fraction={args_cli.student_rollout_fraction:g}",
        flush=True,
    )
    for iteration in range(1, args_cli.iterations + 1):
        targets = task_env._motion_loader.teacher_action[task_env._reference_frame_index]
        _, outputs = policy.act({"observations": observations["policy"]}, role="policy")
        predictions = outputs["mean_actions"]
        if predictions.shape != targets.shape and predictions.transpose(0, 1).shape == targets.shape:
            predictions = predictions.transpose(0, 1)
        if predictions.shape != targets.shape:
            raise RuntimeError(
                f"policy output shape {tuple(predictions.shape)} does not match "
                f"teacher shape {tuple(targets.shape)}"
            )
        loss = torch.mean(torch.square(predictions - targets))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        optimizer.step()

        with torch.no_grad():
            rollout_fraction = args_cli.student_rollout_fraction * iteration / args_cli.iterations
            student_actions = torch.clamp(predictions.detach(), -1.0, 1.0)
            actions = (1.0 - rollout_fraction) * targets + rollout_fraction * student_actions
            observations, _, _, _, _ = raw_env.step(actions)

        if iteration == 1 or iteration % 100 == 0 or iteration == args_cli.iterations:
            maximum_error = torch.amax(torch.abs(predictions - targets))
            print(
                f"[BC] iteration={iteration} loss={loss.item():.6f} "
                f"max_abs_error={maximum_error.item():.6f}",
                flush=True,
            )
        if iteration % args_cli.save_interval == 0:
            _save_checkpoint(source_checkpoint, policy, args_cli.output, iteration)

    _save_checkpoint(source_checkpoint, policy, args_cli.output)
    raw_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

"""Frozen-policy transition audit, independent of all training processes."""

import argparse
import sys

from isaaclab.app import AppLauncher
import cli_args

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Booster-K1-Height-Tracking-v0")
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=5000)
parser.add_argument("--output", required=True)
parser.add_argument("--fallen_cache", required=True)
parser.add_argument("--lift", required=True, type=float)
parser.add_argument("--seed", type=int, default=731)
parser.add_argument("--training_sampling", action="store_true", help="Use current training sampler for a code smoke test, not original-distribution evaluation.")
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args()
args.device = args.device or "cpu"
if args.checkpoint is None or args.steps < 1 or args.num_envs < 1:
    parser.error("--checkpoint and positive --steps/--num_envs are required")
if args.device.startswith("cuda") and args.num_envs > 64:
    parser.error("GPU diagnostic buffers are sized for at most 64 environments")
sys.argv = [sys.argv[0]] + hydra_args
app = AppLauncher(args).app

import hashlib
import importlib
import json
import time
from pathlib import Path

import booster_train.tasks  # noqa: F401
import gymnasium as gym
import torch
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner
from booster_train.tasks.manager_based.fall_recovery.wbc_stand_up import WbcStandUpVecEnvWrapper
from height_transition_stats import TransitionCollector


@hydra_task_config(args.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = agent_cfg.seed = args.seed
    env_cfg.sim.device = agent_cfg.device = args.device
    if args.device.startswith("cuda"):
        # Capacity only, for <=64 diagnostic environments sharing the training GPU.
        # Any PhysX overflow invalidates the run; physical solver settings stay unchanged.
        env_cfg.sim.physx.gpu_max_rigid_contact_count = 2**18
        env_cfg.sim.physx.gpu_max_rigid_patch_count = 2**16
        env_cfg.sim.physx.gpu_found_lost_pairs_capacity = 2**17
        env_cfg.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 2**19
        env_cfg.sim.physx.gpu_total_aggregate_pairs_capacity = 2**17
        env_cfg.sim.physx.gpu_collision_stack_size = 2**25
    agent_cfg.fallen_state_dataset_cfg.cache_path_override = args.fallen_cache
    if not args.training_sampling:
        for name in ("progress_sampling", "delta_curriculum", "staircase", "ramp_curriculum", "governed_commands"):
            if hasattr(env_cfg.commands.height, name):
                setattr(env_cfg.commands.height, name, False)
    # Freeze only this diagnostic environment; retain randomization and commands.
    for name, value in vars(env_cfg.curriculum).items():
        if hasattr(value, "func"):
            setattr(env_cfg.curriculum, name, None)
    env_cfg.scene.terrain.max_init_terrain_level = 0
    env = gym.make(args.task, cfg=env_cfg)
    spec = gym.spec(args.task)
    module, function = spec.kwargs["pre_learn_entry_point"].split(":")
    getattr(importlib.import_module(module), function)(env.unwrapped, args.task, agent_cfg)
    env = WbcStandUpVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    base = env.unwrapped
    base.action_manager.get_term("lift").scale_forces(args.lift)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=args.device)
    runner.load(args.checkpoint, load_optimizer=False)
    runner.get_inference_policy(device=args.device)
    collector = TransitionCollector(base, args.output)
    obs, _ = env.reset()
    metadata = dict(
        checkpoint=str(Path(args.checkpoint).resolve()),
        checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
        iteration=runner.current_learning_iteration, lift=args.lift, seed=args.seed,
        num_envs=args.num_envs, steps=args.steps, device=args.device, dt=base.step_dt,
        stochastic_actions=True, curriculum_frozen=True, terrain_level=0,
        settle_time=collector.command.cfg.settle_time_s,
        threshold=collector.command.cfg.success_error_threshold,
        command_range=list(collector.command.cfg.ranges.height),
        hold_range=list(collector.command.cfg.resampling_time_range),
        standing_ratio=collector.command.cfg.standing_ratio,
        flat_ratio=collector.command.cfg.flat_ratio,
        progress_sampling=getattr(collector.command.cfg, "progress_sampling", False),
        delta_curriculum_enabled=getattr(collector.command.cfg, "delta_curriculum", False),
        contact_threshold_n=5.0, stable_contact_window_s=0.2,
    )
    metadata_path = Path(args.output + ".meta.json")
    metadata_path.write_text(json.dumps(metadata, indent=2))
    start = time.monotonic()
    try:
        for step in range(args.steps):
            with torch.inference_mode():
                actions = runner.alg.policy.act(runner.obs_normalizer(obs))
                obs, _, _, _ = env.step(actions)
            if (step + 1) % 250 == 0:
                collector.stream.flush()
                print(f"TRANSITION_DIAG step={step+1} segments={collector.count} "
                      f"elapsed={time.monotonic()-start:.1f}s", flush=True)
        metadata["completed"] = True
    finally:
        collector.close()
        metadata["elapsed_s"] = time.monotonic() - start
        metadata["segments"] = collector.count
        metadata_path.write_text(json.dumps(metadata, indent=2))
        env.close()


if __name__ == "__main__":
    main()
    app.close()

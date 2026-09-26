# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import math
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--resume_path",
    type=str,
    default=None,
    help="Explicit checkpoint path. Requires --resume and may cross experiment directories.",
)
parser.add_argument(
    "--wbc_fallen_cache",
    type=str,
    default=None,
    help="Explicit fallen-state cache for a WBC stand-up task.",
)
parser.add_argument(
    "--wbc_max_terrain_level",
    type=int,
    default=None,
    help="Maximum initially sampled terrain level for staged WBC training.",
)
parser.add_argument(
    "--wbc_initial_lift_scale",
    type=float,
    default=None,
    help="Initial adaptive lift scale for resuming a WBC height-tracking run.",
)
parser.add_argument(
    "--wbc_initial_lift_ema",
    type=float,
    default=None,
    help="Initial height-error EMA for resuming a WBC height-tracking run.",
)
parser.add_argument(
    "--wbc_lift_resume_warmup_steps",
    type=int,
    default=1000,
    help="Simulation steps to freeze adaptive lift after restoring its state.",
)
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for minimum supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# for distributed training, check minimum supported rsl-rl version
RSL_RL_VERSION = "2.3.1"
installed_version = metadata.version("rsl-rl-lib")
if args_cli.distributed and version.parse(installed_version) < version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import os
from datetime import datetime

import booster_train.tasks  # noqa: F401
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import omni
import torch
from booster_train.tasks.manager_based.fall_recovery.wbc_stand_up import (
    WbcStandUpVecEnvWrapper,
)
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def configure_action_std(runner: OnPolicyRunner, agent_cfg: RslRlOnPolicyRunnerCfg) -> None:
    freeze_action_std = getattr(agent_cfg, "freeze_action_std", False)
    freeze_resumed_action_std = getattr(agent_cfg, "freeze_resumed_action_std", False)
    reset_optimizer_state = getattr(agent_cfg, "reset_action_std_optimizer_state_on_resume", False)
    minimum_action_std = getattr(agent_cfg, "minimum_action_std", None)
    if freeze_action_std and freeze_resumed_action_std:
        raise ValueError("configured and resumed action std cannot both be frozen")
    if (freeze_action_std or freeze_resumed_action_std) and reset_optimizer_state:
        raise ValueError("action std cannot be frozen while resetting its optimizer state")
    if (freeze_action_std or freeze_resumed_action_std) and minimum_action_std is not None:
        raise ValueError("action std cannot be frozen while enforcing a minimum")
    if freeze_resumed_action_std and not getattr(agent_cfg, "resume", False):
        raise ValueError("resumed action std can only be frozen when resuming a checkpoint")
    if minimum_action_std is not None and (
        not math.isfinite(minimum_action_std) or minimum_action_std <= 0.0
    ):
        raise ValueError("minimum action std must be a finite positive value")
    if (
        not freeze_action_std
        and not freeze_resumed_action_std
        and not reset_optimizer_state
        and minimum_action_std is None
    ):
        return
    policy = runner.alg.policy
    parameter = policy.log_std if policy.noise_std_type == "log" else policy.std
    if freeze_resumed_action_std:
        parameter.requires_grad_(False)
        values = torch.exp(parameter) if policy.noise_std_type == "log" else parameter
        print(f"[INFO] Frozen resumed action std: {values.detach().tolist()}")
        return
    if freeze_action_std:
        values = torch.tensor(agent_cfg.action_std, dtype=torch.float32, device=runner.device)
        if values.shape != (runner.env.num_actions,):
            raise ValueError(
                f"fixed action std has shape {tuple(values.shape)}, expected {(runner.env.num_actions,)}"
            )
        with torch.no_grad():
            parameter.copy_(torch.log(values) if policy.noise_std_type == "log" else values)
        parameter.requires_grad_(False)
        print(f"[INFO] Fixed action std: {values.tolist()}")
        return

    parameter.requires_grad_(True)
    if reset_optimizer_state and getattr(agent_cfg, "resume", False):
        removed_state = runner.alg.optimizer.state.pop(parameter, None)
        print(
            "[INFO] Action std is learnable; reset resumed optimizer state: "
            f"{removed_state is not None}"
        )
    if minimum_action_std is not None:
        minimum_parameter = (
            math.log(minimum_action_std)
            if policy.noise_std_type == "log"
            else minimum_action_std
        )

        def clamp_action_std(*_args, **_kwargs) -> None:
            with torch.no_grad():
                parameter.clamp_(min=minimum_parameter)

        clamp_action_std()
        runner.alg.optimizer.register_step_post_hook(clamp_action_std)
        print(f"[INFO] Minimum learnable action std: {minimum_action_std}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )
    if args_cli.wbc_fallen_cache is not None:
        dataset_cfg = getattr(agent_cfg, "fallen_state_dataset_cfg", None)
        if dataset_cfg is None:
            raise ValueError("--wbc_fallen_cache requires a WBC fallen-state dataset config")
        dataset_cfg.cache_path_override = os.path.abspath(
            os.path.expanduser(args_cli.wbc_fallen_cache)
        )
    if args_cli.wbc_max_terrain_level is not None:
        terrain_cfg = getattr(env_cfg.scene, "terrain", None)
        terrain_generator = getattr(terrain_cfg, "terrain_generator", None)
        if terrain_generator is None:
            raise ValueError("--wbc_max_terrain_level requires generated terrain")
        maximum_level = int(args_cli.wbc_max_terrain_level)
        if not 0 <= maximum_level < terrain_generator.num_rows:
            raise ValueError(
                "--wbc_max_terrain_level must be within "
                f"0..{terrain_generator.num_rows - 1}"
            )
        terrain_cfg.max_init_terrain_level = maximum_level
    if args_cli.wbc_initial_lift_scale is not None or args_cli.wbc_initial_lift_ema is not None:
        curriculum_cfg = getattr(env_cfg, "curriculum", None)
        adaptive_lift = getattr(curriculum_cfg, "adaptive_lift", None)
        if adaptive_lift is None:
            raise ValueError("initial lift state requires an adaptive_lift curriculum")
        if args_cli.wbc_initial_lift_scale is not None:
            lift_scale = float(args_cli.wbc_initial_lift_scale)
            if not 0.0 <= lift_scale <= 1.0:
                raise ValueError("--wbc_initial_lift_scale must be within [0, 1]")
            adaptive_lift.params["initial_force_scale"] = lift_scale
        if args_cli.wbc_initial_lift_ema is not None:
            lift_ema = float(args_cli.wbc_initial_lift_ema)
            if not torch.isfinite(torch.tensor(lift_ema)) or lift_ema < 0.0:
                raise ValueError("--wbc_initial_lift_ema must be finite and non-negative")
            adaptive_lift.params["initial_ema"] = lift_ema

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors output directory if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
        env_cfg.io_descriptors_output_dir = log_dir
    else:
        omni.log.warn(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # Resolve the source checkpoint before the new run directory is populated.
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        if args_cli.resume_path is not None:
            resume_path = os.path.abspath(os.path.expanduser(args_cli.resume_path))
            if not os.path.isfile(resume_path):
                raise FileNotFoundError(f"explicit checkpoint does not exist: {resume_path}")
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    elif args_cli.resume_path is not None:
        raise ValueError("--resume_path requires --resume")

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # Stand-up tasks collect/load fallen states after scene creation and before
    # the RSL wrapper performs its first reset.
    task_spec = gym.spec(args_cli.task)
    pre_learn_entry_point = task_spec.kwargs.get("pre_learn_entry_point")
    if pre_learn_entry_point is not None:
        import importlib

        module_name, function_name = pre_learn_entry_point.split(":")
        pre_learn = getattr(importlib.import_module(module_name), function_name)
        pre_learn(env.unwrapped, args_cli.task, agent_cfg)

    # wrap around environment for rsl-rl
    is_wbc_task = task_spec.kwargs.get("wbc_stand_up", False) or task_spec.kwargs.get(
        "wbc_height_tracking", False
    )
    wrapper_type = WbcStandUpVecEnvWrapper if is_wbc_task else RslRlVecEnvWrapper
    env = wrapper_type(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=agent_cfg.device)
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    # load the checkpoint
    if agent_cfg.resume or agent_cfg.algorithm.class_name == "Distillation":
        print(f"[INFO]: Loading model checkpoint from: {resume_path}")
        # load previously trained model
        runner.load(
            resume_path,
            load_optimizer=getattr(agent_cfg, "load_optimizer_on_resume", True),
        )
        loaded_iteration = runner.current_learning_iteration
        if getattr(agent_cfg, "reset_learning_iteration_on_resume", False):
            runner.current_learning_iteration = 0
            print(
                "[INFO]: Reset learning iteration after loading checkpoint "
                f"from iteration {loaded_iteration}"
            )
        if agent_cfg.resume and (
            task_spec.kwargs.get("wbc_stand_up", False)
            or task_spec.kwargs.get("wbc_height_tracking", False)
        ):
            base_env = env.unwrapped
            restore_curriculum = getattr(
                agent_cfg,
                "restore_wbc_curriculum_on_resume",
                True,
            )
            curriculum_iteration = (
                runner.current_learning_iteration if restore_curriculum else 0
            )
            base_env.common_step_counter = (
                curriculum_iteration * runner.num_steps_per_env
            )
            if args_cli.wbc_initial_lift_scale is not None:
                warmup_steps = int(args_cli.wbc_lift_resume_warmup_steps)
                if warmup_steps < 0:
                    raise ValueError("--wbc_lift_resume_warmup_steps must be non-negative")
                for term_name, term_cfg in zip(
                    base_env.curriculum_manager._term_names,
                    base_env.curriculum_manager._term_cfgs,
                ):
                    if term_name == "adaptive_lift":
                        term_cfg.func.arm_resume_warmup(
                            base_env.common_step_counter,
                            warmup_steps,
                        )
                        break
                else:
                    raise RuntimeError("adaptive_lift curriculum term is unavailable")
            all_env_ids = torch.arange(
                base_env.num_envs,
                dtype=torch.long,
                device=base_env.device,
            )
            base_env.curriculum_manager.compute(env_ids=all_env_ids)
            print(
                "[INFO]: Initialized WBC curriculum at common step "
                f"{base_env.common_step_counter}"
            )

    configure_action_std(runner, agent_cfg)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

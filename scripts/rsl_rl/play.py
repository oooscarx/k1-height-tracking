# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
from importlib.metadata import version

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--video_output_dir",
    type=str,
    default=None,
    help="Optional root directory for recorded videos instead of the checkpoint run directory.",
)
parser.add_argument(
    "--state_output",
    type=str,
    default=None,
    help="Optional NPZ path for a renderer-free playback state trajectory.",
)
parser.add_argument(
    "--state_length",
    type=int,
    default=750,
    help="Number of policy steps to export with --state_output.",
)
parser.add_argument(
    "--height_diagnostic_output",
    type=str,
    default=None,
    help="Write fixed-command height tracking errors to this JSON file.",
)
parser.add_argument(
    "--height_diagnostic_steps",
    type=int,
    default=750,
    help="Number of policy steps to sample for fixed-command height diagnostics.",
)
parser.add_argument(
    "--height_diagnostic_warmup_steps",
    type=int,
    default=150,
    help="Warmup policy steps before collecting fixed-command height diagnostics.",
)
parser.add_argument(
    "--height_diagnostic_levels",
    type=int,
    default=8,
    help="Number of evenly spaced non-negative height targets to diagnose.",
)
parser.add_argument(
    "--height_diagnostic_standing_resets",
    action="store_true",
    help="Force all height-diagnostic resets to the configured default standing pose.",
)
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--wbc_reset_mode",
    choices=("random", "face_up", "face_down", "left_side", "right_side", "other"),
    default=None,
    help="Restrict WBC stand-up resets to one fallen orientation class.",
)
parser.add_argument(
    "--wbc_fallen_cache",
    type=str,
    default=None,
    help="Explicit WBC fallen-state cache to use regardless of evaluation environment count.",
)
parser.add_argument(
    "--wbc_disable_external_disturbances",
    action="store_true",
    help="Disable interval pushes and external wrenches for baseline WBC videos.",
)
parser.add_argument(
    "--wbc_disable_domain_randomization",
    action="store_true",
    help="Disable startup physical randomization for a WBC baseline evaluation.",
)
parser.add_argument(
    "--wbc_eval_output",
    type=str,
    default=None,
    help="Write one-episode WBC stand-up metrics to this JSON file.",
)
parser.add_argument(
    "--wbc_stochastic_actions",
    action="store_true",
    help="Sample WBC actions from the training distribution during diagnostics.",
)
parser.add_argument(
    "--wbc_stability_seconds",
    type=float,
    default=1.0,
    help="Continuous stable-standing time required for WBC evaluation success.",
)
parser.add_argument(
    "--wbc_lift_scale",
    type=float,
    default=None,
    help="Override WBC lift curriculum scale; use zero for unaided evaluation.",
)
parser.add_argument(
    "--wbc_terrain_level",
    type=int,
    default=None,
    help="WBC evaluation terrain level, or -1 to distribute envs across all levels.",
)
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument(
    "--web_control",
    action="store_true",
    help="Serve a local browser UI with live video, height commands, reset, and pushes.",
)
parser.add_argument("--web_host", type=str, default="127.0.0.1", help="Web-control bind address.")
parser.add_argument("--web_port", type=int, default=8765, help="Web-control TCP port.")
parser.add_argument("--web_frame_rate", type=float, default=12.0, help="Maximum web video frame rate.")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video_output_dir is not None and not args_cli.video:
    parser.error("--video_output_dir requires --video")
if args_cli.state_output is not None and args_cli.video:
    parser.error("--state_output and --video cannot be used together")
if args_cli.state_length <= 0:
    parser.error("--state_length must be positive")
if args_cli.height_diagnostic_steps <= 0:
    parser.error("--height_diagnostic_steps must be positive")
if args_cli.height_diagnostic_warmup_steps < 0:
    parser.error("--height_diagnostic_warmup_steps must be non-negative")
if args_cli.height_diagnostic_levels <= 1:
    parser.error("--height_diagnostic_levels must be greater than one")
if args_cli.web_control and (args_cli.video or args_cli.state_output or args_cli.height_diagnostic_output):
    parser.error("--web_control cannot be combined with video, state export, or height diagnostics")
if args_cli.web_control and args_cli.num_envs not in (None, 1):
    parser.error("--web_control requires --num_envs 1")
if not 0 < args_cli.web_port <= 65535:
    parser.error("--web_port must be between 1 and 65535")
if not 0 < args_cli.web_frame_rate <= 30:
    parser.error("--web_frame_rate must be in (0, 30]")
if args_cli.web_control:
    args_cli.num_envs = 1
# always enable cameras to record video
if args_cli.video or args_cli.web_control:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import json
import os
import time
from pathlib import Path

import numpy as np
import booster_train.tasks  # noqa: F401
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import torch
from booster_train.tasks.manager_based.fall_recovery.wbc_stand_up import (
    WbcStandUpVecEnvWrapper,
    fallen_orientation_mask,
    mirror_flat_observation,
    mirror_joint_values,
    reset_from_fallen_dataset,
)
from k1_wbc_export_manifest import write_export_manifest
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper, export_policy_as_jit, export_policy_as_onnx
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl.runners import OnPolicyRunner


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", train_task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)
    task_spec = gym.spec(args_cli.task)
    is_wbc_stand_up = task_spec.kwargs.get("wbc_stand_up", False)
    is_wbc_task = is_wbc_stand_up or task_spec.kwargs.get("wbc_height_tracking", False)
    if args_cli.wbc_fallen_cache is not None:
        if not is_wbc_task:
            raise ValueError("--wbc_fallen_cache requires a WBC task")
        agent_cfg.fallen_state_dataset_cfg.cache_path_override = args_cli.wbc_fallen_cache
    if args_cli.wbc_disable_external_disturbances:
        if not is_wbc_task:
            raise ValueError(
                "--wbc_disable_external_disturbances requires a WBC stand-up task"
            )
        for event_name in (
            "apply_external_force_torque",
            "apply_external_force_torque_extremities",
            "push_robot",
        ):
            setattr(env_cfg.events, event_name, None)
    if args_cli.wbc_disable_domain_randomization:
        if not is_wbc_task:
            raise ValueError(
                "--wbc_disable_domain_randomization requires a WBC stand-up task"
            )
        for event_name in (
            "randomize_physics_material",
            "randomize_actuator_gains",
            "randomize_joint_friction",
            "randomize_joint_armature",
            "randomize_bodies_mass",
            "randomize_base_mass",
            "randomize_bodies_com",
            "randomize_base_com",
        ):
            setattr(env_cfg.events, event_name, None)
    if args_cli.wbc_lift_scale is not None:
        if not is_wbc_task:
            raise ValueError("--wbc_lift_scale requires a WBC stand-up task")
        # The curriculum term owns its own scale and would overwrite a playback
        # override on the first environment reset.
        for curriculum_name in ("remove_lift", "adaptive_lift"):
            if hasattr(env_cfg.curriculum, curriculum_name):
                setattr(env_cfg.curriculum, curriculum_name, None)
    if args_cli.height_diagnostic_standing_resets:
        if args_cli.height_diagnostic_output is None:
            raise ValueError("--height_diagnostic_standing_resets requires --height_diagnostic_output")
        env_cfg.events.reset_base.params["standing_ratio"] = 1.0
        env_cfg.events.reset_base.params["random_fallen_ratio"] = 0.0

    # create isaac environment
    render_mode = "rgb_array" if args_cli.video or args_cli.web_control else None
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=render_mode)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    evaluation_mode = args_cli.wbc_reset_mode
    if args_cli.wbc_eval_output is not None and evaluation_mode is None:
        evaluation_mode = "random"
    if args_cli.video:
        video_subdir = evaluation_mode or "play"
        video_root = args_cli.video_output_dir or os.path.join(log_dir, "videos")
        video_kwargs = {
            "video_folder": os.path.join(video_root, video_subdir),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    pre_learn_entry_point = task_spec.kwargs.get("pre_learn_entry_point")
    if pre_learn_entry_point is not None:
        import importlib

        module_name, function_name = pre_learn_entry_point.split(":")
        pre_learn = getattr(importlib.import_module(module_name), function_name)
        pre_learn(env.unwrapped, args_cli.task, agent_cfg)
    if evaluation_mode is not None:
        if not is_wbc_stand_up:
            raise ValueError("--wbc_reset_mode requires a WBC stand-up task")
        _configure_wbc_reset(env.unwrapped, evaluation_mode)
    if args_cli.wbc_terrain_level is not None:
        if not is_wbc_stand_up:
            raise ValueError("--wbc_terrain_level requires a WBC stand-up task")
        _configure_wbc_terrain_level(
            env.unwrapped,
            args_cli.wbc_terrain_level,
        )

    # wrap around environment for rsl-rl
    wrapper_type = (
        WbcStandUpVecEnvWrapper
        if is_wbc_task
        else RslRlVecEnvWrapper
    )
    env = wrapper_type(env, clip_actions=agent_cfg.clip_actions)
    if args_cli.wbc_lift_scale is not None:
        lift_action = env.unwrapped.action_manager.get_term("lift")
        lift_action.scale_forces(args_cli.wbc_lift_scale)
        # Force one reset after the fallen-state dataset, orientation filter,
        # terrain level, and lift override have all been configured.
        env.reset()
        if lift_action.force_scale != args_cli.wbc_lift_scale:
            raise RuntimeError(
                "WBC lift override was changed during evaluation reset: "
                f"requested={args_cli.wbc_lift_scale}, "
                f"actual={lift_action.force_scale}"
            )
        print(f"[INFO] WBC lift force scale: {lift_action.force_scale:.6f}")

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    ppo_runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    ppo_runner.load(resume_path)

    # obtain the trained policy for inference
    policy = ppo_runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = ppo_runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = ppo_runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    elif hasattr(ppo_runner, "obs_normalizer"):     # compatibility for older versions
        normalizer = ppo_runner.obs_normalizer
    else:
        normalizer = None

    if args_cli.height_diagnostic_output is not None:
        if not task_spec.kwargs.get("wbc_height_tracking", False):
            raise ValueError("--height_diagnostic_output requires a WBC height-tracking task")
        base_env = env.unwrapped
        if base_env.num_envs < args_cli.height_diagnostic_levels:
            raise ValueError(
                "height diagnostics require at least one environment per level: "
                f"num_envs={base_env.num_envs}, levels={args_cli.height_diagnostic_levels}"
            )
        height_command = base_env.command_manager.get_term("height")
        maximum_height = float(height_command.cfg.ranges.height[1])
        levels = torch.linspace(
            0.0,
            maximum_height,
            args_cli.height_diagnostic_levels,
            device=base_env.device,
        )
        target_indices = torch.arange(base_env.num_envs, device=base_env.device) % len(levels)
        targets = levels[target_indices]

        def force_height_targets() -> None:
            height_command.set_manual_height(targets)

        def policy_observations():
            current = env.get_observations()
            if version("rsl-rl-lib").startswith("2.3."):
                return current[0]
            return current

        force_height_targets()
        initial_measured = height_command.measured_height.detach().cpu()
        obs = policy_observations()
        error_samples = [[] for _ in levels]
        measured_samples = [[] for _ in levels]
        termination_counts = torch.zeros(len(levels), dtype=torch.long)
        total_steps = args_cli.height_diagnostic_warmup_steps + args_cli.height_diagnostic_steps
        for step in range(total_steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
            force_height_targets()
            if torch.any(dones):
                done_levels = target_indices[dones]
                termination_counts += torch.bincount(done_levels.cpu(), minlength=len(levels))
                # Reset environments briefly contain a freshly sampled command in
                # the observation returned by step(). Recompute after restoring
                # the fixed targets; this only happens on termination steps.
                obs = policy_observations()
            if step >= args_cli.height_diagnostic_warmup_steps:
                measured = height_command.measured_height.detach()
                errors = torch.abs(measured - targets)
                valid = height_command.settled
                for level_index in range(len(levels)):
                    mask = valid & (target_indices == level_index)
                    if torch.any(mask):
                        error_samples[level_index].append(errors[mask].cpu())
                        measured_samples[level_index].append(measured[mask].cpu())

        bins = []
        for level_index, target in enumerate(levels.cpu()):
            mask = target_indices.cpu() == level_index
            if not error_samples[level_index]:
                raise RuntimeError(f"no settled diagnostic samples for height level {float(target):.6f}")
            level_errors = torch.cat(error_samples[level_index])
            level_measured = torch.cat(measured_samples[level_index])
            bins.append(
                {
                    "target_height_m": float(target.item()),
                    "environment_count": int(mask.sum().item()),
                    "sample_count": int(level_errors.numel()),
                    "termination_count": int(termination_counts[level_index].item()),
                    "initial_measured_height_m": float(initial_measured[mask].mean().item()),
                    "mean_measured_height_m": float(level_measured.mean().item()),
                    "mean_abs_error_m": float(level_errors.mean().item()),
                    "median_abs_error_m": float(torch.quantile(level_errors, 0.5).item()),
                    "p95_abs_error_m": float(torch.quantile(level_errors, 0.95).item()),
                    "max_abs_error_m": float(level_errors.max().item()),
                }
            )
        report = {
            "checkpoint": resume_path,
            "task": args_cli.task,
            "lift_force_scale": float(base_env.action_manager.get_term("lift").force_scale),
            "num_envs": int(base_env.num_envs),
            "warmup_steps": args_cli.height_diagnostic_warmup_steps,
            "sample_steps": args_cli.height_diagnostic_steps,
            "bins": bins,
        }
        output_path = os.path.abspath(args_cli.height_diagnostic_output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(report, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        env.close()
        return

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    run_name = os.path.basename(log_dir)
    jit_filename = f"{agent_cfg.experiment_name}_{run_name}.pt"
    onnx_filename = f"{agent_cfg.experiment_name}_{run_name}.onnx"
    export_policy_as_jit(
        policy_nn,
        normalizer=normalizer,
        path=export_model_dir,
        filename=jit_filename,
    )
    export_policy_as_onnx(
        policy_nn,
        normalizer=normalizer,
        path=export_model_dir,
        filename=onnx_filename,
    )
    export_manifest = write_export_manifest(
        checkpoint=Path(resume_path),
        export_dir=Path(export_model_dir),
        jit_filename=jit_filename,
        onnx_filename=onnx_filename,
    )
    print(f"[INFO] Export provenance manifest: {export_manifest}")

    if (
        args_cli.headless
        and not args_cli.video
        and not args_cli.web_control
        and args_cli.wbc_eval_output is None
        and args_cli.state_output is None
    ):
        print("[INFO] Headless mode and no video recording. Exiting after model export.")
        env.close()
        return

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    observation_extras = {}
    if version("rsl-rl-lib").startswith("2.3."):
        obs, observation_extras = obs

    def policy_observations():
        current = env.get_observations()
        if version("rsl-rl-lib").startswith("2.3."):
            return current[0]
        return current
    if args_cli.wbc_eval_output is not None:
        if not is_wbc_stand_up:
            raise ValueError("--wbc_eval_output requires a WBC stand-up task")
        metrics = _evaluate_wbc_policy(
            env,
            policy_nn.act if args_cli.wbc_stochastic_actions else policy,
            policy,
            obs,
            evaluation_mode,
            args_cli.wbc_stability_seconds,
            _wbc_effective_raw_action_clip(
                env.unwrapped,
                agent_cfg.clip_actions,
            ),
            policy_nn,
            observation_extras,
            ppo_runner.alg.reward_normalizer,
            args_cli.wbc_stochastic_actions,
        )
        metrics["checkpoint"] = resume_path
        metrics["task"] = args_cli.task
        metrics["seed"] = int(agent_cfg.seed)
        metrics["lift_force_scale"] = float(
            env.unwrapped.action_manager.get_term("lift").force_scale
        )
        metrics["stochastic_actions"] = bool(args_cli.wbc_stochastic_actions)
        metrics["external_disturbances_disabled"] = bool(
            args_cli.wbc_disable_external_disturbances
        )
        metrics["domain_randomization_disabled"] = bool(
            args_cli.wbc_disable_domain_randomization
        )
        output_path = os.path.abspath(args_cli.wbc_eval_output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(metrics, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
        print(f"[INFO] WBC stand-up evaluation written to: {output_path}")
        print(json.dumps(metrics, indent=2, sort_keys=True))
        env.close()
        return

    state_frames = None
    if args_cli.state_output is not None:
        state_frames = {
            "body_pos_w": [],
            "body_quat_w": [],
            "joint_pos": [],
            "height_command": [],
            "measured_height": [],
        }
        state_robot = env.unwrapped.scene["robot"]
        state_height_command = env.unwrapped.command_manager.get_term("height")

        def capture_state_frame() -> None:
            state_frames["body_pos_w"].append(state_robot.data.body_pos_w[0].detach().cpu().numpy().copy())
            state_frames["body_quat_w"].append(state_robot.data.body_quat_w[0].detach().cpu().numpy().copy())
            state_frames["joint_pos"].append(state_robot.data.joint_pos[0].detach().cpu().numpy().copy())
            state_frames["height_command"].append(float(state_height_command.target_height[0].item()))
            state_frames["measured_height"].append(float(state_height_command.measured_height[0].item()))

    web_control = None
    web_height_command = None
    web_robot = None
    next_web_frame_time = 0.0
    web_frame_count = 0
    web_frame_clock = time.monotonic()
    if args_cli.web_control:
        if not task_spec.kwargs.get("wbc_height_tracking", False):
            raise ValueError("--web_control requires a WBC height-tracking task")
        from height_web_control import HeightWebControlServer

        base_env = env.unwrapped
        web_height_command = base_env.command_manager.get_term("height")
        web_robot = base_env.scene["robot"]
        initial_target = float(web_height_command.target_height[0].item())
        web_height_command.set_manual_height(initial_target, [0])
        obs = policy_observations()
        terrain_levels = getattr(base_env.scene.terrain, "terrain_levels", None)
        terrain_level = int(terrain_levels[0].item()) if terrain_levels is not None else 0
        lift_scale = float(base_env.action_manager.get_term("lift").force_scale)
        web_control = HeightWebControlServer(args_cli.web_host, args_cli.web_port)
        web_control.update_state(
            target_height=initial_target,
            measured_height=float(web_height_command.measured_height[0].item()),
            height_error=abs(float(web_height_command.measured_height[0].item()) - initial_target),
            minimum_height=float(web_height_command.cfg.ranges.height[0]),
            maximum_height=float(web_height_command.cfg.ranges.height[1]),
            terrain_level=terrain_level,
            lift=lift_scale,
            domain_randomization=not args_cli.wbc_disable_domain_randomization,
            external_disturbances=not args_cli.wbc_disable_external_disturbances,
            policy_hz=1.0 / dt,
        )
        initial_frame = env.unwrapped.render()
        if initial_frame is not None:
            web_control.update_frame(initial_frame)
        web_control.start()
        print(f"[INFO] K1 web control: http://{args_cli.web_host}:{args_cli.web_port}")

    timestep = 0
    # simulate environment
    try:
        while simulation_app.is_running():
            start_time = time.time()
            if web_control is not None:
                observations_changed = False
                with torch.inference_mode():
                    for command in web_control.drain_commands():
                        if command["action"] == "set_height":
                            web_height_command.set_manual_height(command["value"], [0])
                            observations_changed = True
                        elif command["action"] == "reset":
                            env.reset()
                            web_height_command.set_manual_height(web_control.snapshot()["target_height"], [0])
                            observations_changed = True
                        elif command["action"] == "push":
                            root_velocity = web_robot.data.root_vel_w.clone()
                            root_velocity[0, 0] += command["x"]
                            root_velocity[0, 1] += command["y"]
                            root_velocity[0, 5] += command["yaw"]
                            web_robot.write_root_velocity_to_sim(root_velocity)
                            observations_changed = True
                if observations_changed:
                    obs = policy_observations()

            # run everything in inference mode
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, _, _ = env.step(actions)
            if state_frames is not None:
                capture_state_frame()
            if web_control is not None:
                now = time.monotonic()
                measured = float(web_height_command.measured_height[0].item())
                target = float(web_height_command.target_height[0].item())
                terrain_levels = getattr(env.unwrapped.scene.terrain, "terrain_levels", None)
                terrain_level = int(terrain_levels[0].item()) if terrain_levels is not None else 0
                if now >= next_web_frame_time:
                    frame = env.unwrapped.render()
                    if frame is not None:
                        web_control.update_frame(frame)
                        web_frame_count += 1
                    next_web_frame_time = now + 1.0 / args_cli.web_frame_rate
                elapsed = now - web_frame_clock
                frame_hz = web_frame_count / elapsed if elapsed > 0 else 0.0
                web_control.update_state(
                    target_height=target,
                    measured_height=measured,
                    height_error=abs(measured - target),
                    terrain_level=terrain_level,
                    frame_hz=frame_hz,
                )
            if args_cli.video:
                timestep += 1
                # Exit the play loop after recording one video
                if timestep == args_cli.video_length:
                    break
            elif state_frames is not None:
                timestep += 1
                if timestep == args_cli.state_length:
                    break

            # Interactive playback is paced to policy time even without --real-time.
            sleep_time = dt - (time.time() - start_time)
            if (args_cli.real_time or web_control is not None) and sleep_time > 0:
                time.sleep(sleep_time)
    finally:
        if web_control is not None:
            web_control.close()

    if state_frames is not None:
        output_path = os.path.abspath(args_cli.state_output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        np.savez_compressed(
            output_path,
            body_pos_w=np.asarray(state_frames["body_pos_w"], dtype=np.float32),
            body_quat_w=np.asarray(state_frames["body_quat_w"], dtype=np.float32),
            body_names=np.asarray(state_robot.body_names),
            joint_pos=np.asarray(state_frames["joint_pos"], dtype=np.float32),
            joint_names=np.asarray(state_robot.joint_names),
            height_command=np.asarray(state_frames["height_command"], dtype=np.float32),
            measured_height=np.asarray(state_frames["measured_height"], dtype=np.float32),
            lift_force_scale=np.float32(env.unwrapped.action_manager.get_term("lift").force_scale),
            step_dt=np.float32(dt),
        )
        print(f"[INFO] Playback state trajectory written to: {output_path}")

    # close the simulator
    env.close()


def _evaluate_wbc_policy(
    env,
    policy,
    mirror_policy,
    obs: torch.Tensor,
    orientation_mode: str,
    stability_seconds: float,
    action_clip: float | None,
    policy_nn,
    observation_extras: dict,
    reward_normalizer,
    stochastic_actions: bool,
) -> dict:
    """Evaluate one initial fallen episode for every parallel environment."""
    if stability_seconds <= 0.0:
        raise ValueError("--wbc_stability_seconds must be positive")

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    height_sensor = base_env.scene.sensors["height_measurement_sensor"]
    action_term = base_env.action_manager.get_term("joint_pos")
    num_envs = base_env.num_envs
    dt = base_env.step_dt
    stability_steps = max(1, round(stability_seconds / dt))
    max_steps = base_env.max_episode_length
    target_height = float(
        base_env.cfg.rewards.base_height_fine.params["target_height"]
    )
    standing_height = target_height * 0.8

    initial_mode_match = fallen_orientation_mask(
        robot.data.root_quat_w,
        orientation_mode,
    )
    active = torch.ones(num_envs, dtype=torch.bool, device=base_env.device)
    successful = torch.zeros_like(active)
    timed_out = torch.zeros_like(active)
    no_progress = torch.zeros_like(active)
    stable_count = torch.zeros(
        num_envs,
        dtype=torch.long,
        device=base_env.device,
    )
    recovery_time = torch.full(
        (num_envs,),
        torch.nan,
        dtype=torch.float32,
        device=base_env.device,
    )
    max_height = torch.full_like(recovery_time, -torch.inf)
    nonfinite_action = torch.zeros_like(active)
    joint_limit_braking = torch.zeros_like(active)
    parallel_fallback = torch.zeros_like(active)
    maximum_motor_utilization = torch.zeros_like(recovery_time)
    maximum_motor_velocity_ratio = torch.zeros_like(recovery_time)
    maximum_motor_velocity_ratio_by_motor = torch.zeros(
        num_envs,
        4,
        dtype=torch.float32,
        device=base_env.device,
    )
    minimum_motor_margin = torch.full_like(recovery_time, torch.inf)
    maximum_raw_action_abs = torch.zeros_like(recovery_time)
    raw_action_clip_exceeded = torch.zeros_like(active)
    raw_action_elements = 0
    clipped_raw_action_elements = 0
    maximum_policy_observation_abs = torch.zeros((), device=base_env.device)
    maximum_critic_observation_abs = torch.zeros((), device=base_env.device)
    maximum_critic_value_abs = torch.zeros((), device=base_env.device)
    maximum_critic_value_abs_by_env = torch.zeros_like(recovery_time)
    nonfinite_critic_value = torch.zeros_like(active)
    maximum_root_linear_velocity_abs = torch.zeros((), device=base_env.device)
    maximum_root_angular_velocity_abs = torch.zeros((), device=base_env.device)
    maximum_joint_velocity_abs = torch.zeros((), device=base_env.device)
    maximum_contact_force_norm = torch.zeros((), device=base_env.device)
    maximum_policy_observation_delta_abs = torch.zeros((), device=base_env.device)
    maximum_critic_observation_delta_abs = torch.zeros((), device=base_env.device)
    maximum_raw_reward_abs = torch.zeros((), device=base_env.device)
    minimum_raw_reward = torch.full((), torch.inf, device=base_env.device)
    maximum_normalized_reward_abs = torch.zeros((), device=base_env.device)
    minimum_normalized_reward = torch.full((), torch.inf, device=base_env.device)
    num_actions = len(action_term._joint_names)
    policy_mirror_error_sum = torch.zeros(num_actions, device=base_env.device)
    policy_mirror_error_maximum = torch.zeros_like(policy_mirror_error_sum)
    policy_mirror_sample_count = 0

    def update_finite_max(
        current: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        selected = values[mask]
        if selected.numel() == 0:
            return current
        finite_abs = torch.where(
            torch.isfinite(selected),
            torch.abs(selected),
            torch.zeros_like(selected),
        )
        return torch.maximum(current, torch.amax(finite_abs))

    steps_executed = 0
    with torch.inference_mode():
        for step in range(max_steps):
            previous_policy_obs = obs
            maximum_policy_observation_abs = update_finite_max(
                maximum_policy_observation_abs,
                obs,
                active,
            )
            critic_obs = observation_extras.get("observations", {}).get("critic")
            if critic_obs is not None:
                maximum_critic_observation_abs = update_finite_max(
                    maximum_critic_observation_abs,
                    critic_obs,
                    active,
                )
                critic_values = policy_nn.evaluate(critic_obs)
                maximum_critic_value_abs = update_finite_max(
                    maximum_critic_value_abs,
                    critic_values,
                    active,
                )
                finite_critic_value_abs = torch.where(
                    torch.isfinite(critic_values),
                    torch.abs(critic_values),
                    torch.zeros_like(critic_values),
                )
                critic_value_abs_by_env = finite_critic_value_abs.reshape(
                    num_envs,
                    -1,
                ).amax(dim=-1)
                maximum_critic_value_abs_by_env = torch.where(
                    active,
                    torch.maximum(
                        maximum_critic_value_abs_by_env,
                        critic_value_abs_by_env,
                    ),
                    maximum_critic_value_abs_by_env,
                )
                nonfinite_critic_value |= active & torch.any(
                    ~torch.isfinite(critic_values).reshape(num_envs, -1),
                    dim=-1,
                )
            maximum_root_linear_velocity_abs = update_finite_max(
                maximum_root_linear_velocity_abs,
                robot.data.root_lin_vel_b,
                active,
            )
            maximum_root_angular_velocity_abs = update_finite_max(
                maximum_root_angular_velocity_abs,
                robot.data.root_ang_vel_b,
                active,
            )
            maximum_joint_velocity_abs = update_finite_max(
                maximum_joint_velocity_abs,
                robot.data.joint_vel,
                active,
            )
            contact_force_norm = torch.linalg.vector_norm(
                base_env.scene.sensors["contact_forces"].data.net_forces_w,
                dim=-1,
            )
            maximum_contact_force_norm = update_finite_max(
                maximum_contact_force_norm,
                contact_force_norm,
                active,
            )
            actions = policy(obs)
            source_actions = mirror_policy(obs) if stochastic_actions else actions
            mirrored_actions = mirror_policy(
                mirror_flat_observation(obs, env, "policy")
            )
            policy_mirror_error = torch.abs(
                mirrored_actions - mirror_joint_values(source_actions)
            )
            active_mirror_error = policy_mirror_error[active]
            if active_mirror_error.numel() > 0:
                policy_mirror_error_sum += torch.sum(active_mirror_error, dim=0)
                policy_mirror_error_maximum = torch.maximum(
                    policy_mirror_error_maximum,
                    torch.amax(active_mirror_error, dim=0),
                )
                policy_mirror_sample_count += active_mirror_error.shape[0]
            finite_action_abs = torch.where(
                torch.isfinite(actions),
                torch.abs(actions),
                torch.zeros_like(actions),
            )
            maximum_raw_action_abs = torch.where(
                active,
                torch.maximum(
                    maximum_raw_action_abs,
                    torch.amax(finite_action_abs, dim=-1),
                ),
                maximum_raw_action_abs,
            )
            if action_clip is not None:
                clipped = finite_action_abs > action_clip
                raw_action_clip_exceeded |= active & torch.any(clipped, dim=-1)
                clipped_raw_action_elements += int(clipped[active].sum())
                raw_action_elements += int(active.sum()) * actions.shape[-1]
            obs, rewards, dones, infos = env.step(actions)
            observation_extras = infos
            steps_executed = step + 1
            done_mask = dones.bool()
            valid = active & ~done_mask
            maximum_raw_reward_abs = update_finite_max(
                maximum_raw_reward_abs,
                rewards,
                active,
            )
            finite_rewards = rewards[active & torch.isfinite(rewards)]
            if finite_rewards.numel() > 0:
                minimum_raw_reward = torch.minimum(
                    minimum_raw_reward,
                    torch.amin(finite_rewards),
                )
            if reward_normalizer is not None:
                scale = (
                    reward_normalizer._std
                    * reward_normalizer.gamma_factor
                    * reward_normalizer._return_correction
                    + reward_normalizer.eps
                )
                normalized_rewards = rewards / scale
                if reward_normalizer.outlier_threshold is not None:
                    normalized_mean = reward_normalizer._mean / scale
                    normalized_rewards = torch.clamp(
                        normalized_rewards,
                        normalized_mean - reward_normalizer.outlier_threshold,
                        normalized_mean + reward_normalizer.outlier_threshold,
                    )
                maximum_normalized_reward_abs = update_finite_max(
                    maximum_normalized_reward_abs,
                    normalized_rewards,
                    active,
                )
                finite_normalized_rewards = normalized_rewards[
                    active & torch.isfinite(normalized_rewards)
                ]
                if finite_normalized_rewards.numel() > 0:
                    minimum_normalized_reward = torch.minimum(
                        minimum_normalized_reward,
                        torch.amin(finite_normalized_rewards),
                    )
            maximum_policy_observation_delta_abs = update_finite_max(
                maximum_policy_observation_delta_abs,
                obs - previous_policy_obs,
                valid,
            )
            next_critic_obs = infos.get("observations", {}).get("critic")
            if critic_obs is not None and next_critic_obs is not None:
                maximum_critic_observation_delta_abs = update_finite_max(
                    maximum_critic_observation_delta_abs,
                    next_critic_obs - critic_obs,
                    valid,
                )

            ground_height = torch.mean(
                height_sensor.data.ray_hits_w[..., 2],
                dim=-1,
            )
            height = robot.data.root_pos_w[:, 2] - ground_height
            max_height = torch.where(
                active,
                torch.maximum(max_height, height),
                max_height,
            )

            gravity = robot.data.projected_gravity_b
            stable = (
                (height >= standing_height)
                & (gravity[:, 2] <= -0.85)
                & (torch.linalg.vector_norm(gravity[:, :2], dim=-1) <= 0.4)
                & (torch.linalg.vector_norm(robot.data.root_lin_vel_b, dim=-1) <= 0.5)
                & (torch.linalg.vector_norm(robot.data.root_ang_vel_b, dim=-1) <= 1.0)
            )
            stable_count = torch.where(
                valid & stable,
                stable_count + 1,
                torch.zeros_like(stable_count),
            )
            new_success = valid & (stable_count >= stability_steps)
            recovery_time[new_success] = (step + 1) * dt
            successful |= new_success

            nonfinite_action |= valid & action_term.nonfinite_action
            joint_limit_braking |= valid & torch.any(
                action_term.joint_limit_braking,
                dim=-1,
            )
            parallel_fallback |= valid & action_term.parallel_target_fallback
            maximum_motor_utilization = torch.where(
                valid,
                torch.maximum(
                    maximum_motor_utilization,
                    torch.amax(action_term.parallel_motor_utilization, dim=-1),
                ),
                maximum_motor_utilization,
            )
            maximum_motor_velocity_ratio = torch.where(
                valid,
                torch.maximum(
                    maximum_motor_velocity_ratio,
                    torch.amax(action_term.parallel_motor_velocity_ratio, dim=-1),
                ),
                maximum_motor_velocity_ratio,
            )
            maximum_motor_velocity_ratio_by_motor = torch.where(
                valid.unsqueeze(-1),
                torch.maximum(
                    maximum_motor_velocity_ratio_by_motor,
                    action_term.parallel_motor_velocity_ratio,
                ),
                maximum_motor_velocity_ratio_by_motor,
            )
            minimum_motor_margin = torch.where(
                valid,
                torch.minimum(
                    minimum_motor_margin,
                    torch.amin(action_term.parallel_motor_margin, dim=-1),
                ),
                minimum_motor_margin,
            )

            failed = active & done_mask & ~new_success
            timeout_mask = infos.get(
                "time_outs",
                torch.zeros_like(dones),
            ).bool()
            timed_out |= failed & timeout_mask
            no_progress |= failed & ~timeout_mask
            active &= ~(new_success | done_mask)
            if not torch.any(active):
                break

    successful_times = recovery_time[successful]
    finite_motor_margin = minimum_motor_margin[torch.isfinite(minimum_motor_margin)]
    terrain_levels, terrain_counts = torch.unique(
        base_env.scene.terrain.terrain_levels,
        return_counts=True,
    )
    result = {
        "orientation_mode": orientation_mode,
        "num_envs": num_envs,
        "steps_executed": steps_executed,
        "episode_seconds": max_steps * dt,
        "stability_seconds": stability_seconds,
        "standing_height_threshold": standing_height,
        "initial_mode_match_rate": float(initial_mode_match.float().mean()),
        "terrain_level_counts": {
            str(int(level)): int(count)
            for level, count in zip(terrain_levels, terrain_counts)
        },
        "success_count": int(successful.sum()),
        "success_rate": float(successful.float().mean()),
        "timeout_failure_count": int(timed_out.sum()),
        "no_progress_failure_count": int(no_progress.sum()),
        "incomplete_count": int(active.sum()),
        "nonfinite_action_count": int(nonfinite_action.sum()),
        "joint_limit_braking_count": int(joint_limit_braking.sum()),
        "parallel_fallback_count": int(parallel_fallback.sum()),
        "mean_max_height": float(max_height.mean()),
        "p10_max_height": float(torch.quantile(max_height, 0.1)),
        "maximum_raw_action_abs": float(maximum_raw_action_abs.max()),
        "maximum_policy_observation_abs": float(maximum_policy_observation_abs),
        "maximum_critic_observation_abs": float(maximum_critic_observation_abs),
        "maximum_critic_value_abs": float(maximum_critic_value_abs),
        "p95_maximum_critic_value_abs": float(
            torch.quantile(maximum_critic_value_abs_by_env, 0.95)
        ),
        "p99_maximum_critic_value_abs": float(
            torch.quantile(maximum_critic_value_abs_by_env, 0.99)
        ),
        "critic_value_abs_above_50_count": int(
            (maximum_critic_value_abs_by_env > 50.0).sum()
        ),
        "nonfinite_critic_value_count": int(nonfinite_critic_value.sum()),
        "maximum_root_linear_velocity_abs": float(maximum_root_linear_velocity_abs),
        "maximum_root_angular_velocity_abs": float(maximum_root_angular_velocity_abs),
        "maximum_joint_velocity_abs": float(maximum_joint_velocity_abs),
        "maximum_contact_force_norm": float(maximum_contact_force_norm),
        "maximum_policy_observation_delta_abs": float(
            maximum_policy_observation_delta_abs
        ),
        "maximum_critic_observation_delta_abs": float(
            maximum_critic_observation_delta_abs
        ),
        "maximum_raw_reward_abs": float(maximum_raw_reward_abs),
        "minimum_raw_reward": (
            float(minimum_raw_reward) if torch.isfinite(minimum_raw_reward) else None
        ),
        "maximum_normalized_reward_abs": float(maximum_normalized_reward_abs),
        "minimum_normalized_reward": (
            float(minimum_normalized_reward)
            if torch.isfinite(minimum_normalized_reward)
            else None
        ),
        "p95_maximum_raw_action_abs": float(
            torch.quantile(maximum_raw_action_abs, 0.95)
        ),
        "raw_action_clip": action_clip,
        "raw_action_clip_exceeded_count": int(raw_action_clip_exceeded.sum()),
        "raw_action_clipped_element_rate": (
            clipped_raw_action_elements / raw_action_elements
            if raw_action_elements > 0
            else None
        ),
        "maximum_motor_utilization": float(maximum_motor_utilization.max()),
        "p95_motor_utilization": float(
            torch.quantile(maximum_motor_utilization, 0.95)
        ),
        "maximum_motor_velocity_ratio": float(maximum_motor_velocity_ratio.max()),
        "p95_motor_velocity_ratio": float(
            torch.quantile(maximum_motor_velocity_ratio, 0.95)
        ),
        "maximum_motor_velocity_ratio_by_motor": [
            float(value)
            for value in torch.amax(
                maximum_motor_velocity_ratio_by_motor,
                dim=0,
            )
        ],
        "p95_motor_velocity_ratio_by_motor": [
            float(value)
            for value in torch.quantile(
                maximum_motor_velocity_ratio_by_motor,
                0.95,
                dim=0,
            )
        ],
        "motor_velocity_limit_exceeded_count_by_motor": [
            int(value)
            for value in torch.sum(
                maximum_motor_velocity_ratio_by_motor > 1.0,
                dim=0,
            )
        ],
        "motor_velocity_limit_exceeded_count": int(
            (maximum_motor_velocity_ratio > 1.0).sum()
        ),
        "fallback_env_maximum_motor_velocity_ratio": (
            float(maximum_motor_velocity_ratio[parallel_fallback].max())
            if torch.any(parallel_fallback)
            else None
        ),
        "minimum_motor_margin": (
            float(finite_motor_margin.min())
            if finite_motor_margin.numel() > 0
            else None
        ),
        "mean_recovery_time": (
            float(successful_times.mean())
            if successful_times.numel() > 0
            else None
        ),
        "p90_recovery_time": (
            float(torch.quantile(successful_times, 0.9))
            if successful_times.numel() > 0
            else None
        ),
        "success_criteria": {
            "height": standing_height,
            "projected_gravity_z_max": -0.85,
            "projected_gravity_xy_norm_max": 0.4,
            "root_linear_speed_max": 0.5,
            "root_angular_speed_max": 1.0,
        },
        "policy_mirror_joint_names": list(action_term._joint_names),
        "policy_mirror_sample_count": policy_mirror_sample_count,
        "mean_policy_mirror_error_abs": (
            float(
                torch.sum(policy_mirror_error_sum)
                / (policy_mirror_sample_count * num_actions)
            )
            if policy_mirror_sample_count > 0
            else None
        ),
        "maximum_policy_mirror_error_abs": float(
            torch.amax(policy_mirror_error_maximum)
        ),
        "mean_policy_mirror_error_abs_by_joint": (
            [
                float(value)
                for value in policy_mirror_error_sum / policy_mirror_sample_count
            ]
            if policy_mirror_sample_count > 0
            else None
        ),
        "maximum_policy_mirror_error_abs_by_joint": [
            float(value) for value in policy_mirror_error_maximum
        ],
    }
    return result


def _wbc_effective_raw_action_clip(
    env,
    wrapper_clip: float | None,
) -> float | None:
    """Resolve the symmetric raw-action limit used by the WBC action chain."""
    if wrapper_clip is not None:
        return float(wrapper_clip)

    action_term = env.action_manager.get_term("joint_pos")
    if getattr(action_term.cfg, "normalize_input", True):
        return 1.0
    if action_term.cfg.clip is None:
        return None

    scale = action_term._scale[0]
    delta_minimum = action_term._clip[0, :, 0]
    delta_maximum = action_term._clip[0, :, 1]
    raw_limit = torch.minimum(
        torch.abs(delta_minimum),
        torch.abs(delta_maximum),
    ) / scale
    if not torch.allclose(raw_limit, raw_limit[0].expand_as(raw_limit)):
        raise ValueError(
            "WBC evaluator requires one symmetric raw-action limit"
        )
    return float(raw_limit[0])


def _configure_wbc_reset(env, orientation_mode: str) -> None:
    """Use only fallen states from one orientation class during playback."""
    if "reset" not in env.event_manager.active_terms:
        raise RuntimeError("WBC stand-up task has no reset event group")
    for term_name in env.event_manager.active_terms["reset"]:
        term_cfg = env.event_manager.get_term_cfg(term_name)
        if isinstance(term_cfg.func, reset_from_fallen_dataset):
            term_cfg.func.set_orientation_mode(orientation_mode)
            term_cfg.params["orientation_mode"] = orientation_mode
            term_cfg.params["standing_ratio"] = 0.0
            print(
                "[INFO] WBC fallen-state playback mode: "
                f"{orientation_mode} (standing_ratio=0)"
            )
            return
    raise RuntimeError("WBC stand-up task has no fallen-state reset term")


def _configure_wbc_terrain_level(env, requested_level: int) -> None:
    """Set one terrain level or balance evaluation envs across every level."""
    terrain = env.scene.terrain
    terrain_generator = terrain.cfg.terrain_generator
    if terrain_generator is None:
        if requested_level not in (0, -1):
            raise ValueError("flat WBC terrain only supports level zero")
        return
    num_levels = terrain_generator.num_rows
    if requested_level == -1:
        terrain.terrain_levels[:] = (
            torch.arange(env.num_envs, device=env.device) % num_levels
        )
        print(f"[INFO] WBC terrain levels distributed across 0..{num_levels - 1}")
        return
    if not 0 <= requested_level < num_levels:
        raise ValueError(
            f"WBC terrain level must be -1 or within 0..{num_levels - 1}"
        )
    terrain.terrain_levels.fill_(requested_level)
    print(f"[INFO] WBC terrain level: {requested_level}")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

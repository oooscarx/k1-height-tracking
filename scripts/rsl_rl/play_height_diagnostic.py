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
    "--height_diagnostic_min",
    type=float,
    default=None,
    help="Override the minimum fixed-command diagnostic target.",
)
parser.add_argument(
    "--height_diagnostic_max",
    type=float,
    default=None,
    help="Override the maximum fixed-command diagnostic target.",
)
parser.add_argument(
    "--height_diagnostic_standing_resets",
    action="store_true",
    help="Force all height-diagnostic resets to the configured default standing pose.",
)
parser.add_argument(
    "--height_transition_output",
    type=str,
    default=None,
    help="Write a no-lift rate-limited height-sequence stability report to this JSON file.",
)
parser.add_argument(
    "--height_transition_hold_s",
    type=float,
    default=7.0,
    help="Duration of each requested height in the transition diagnostic.",
)
parser.add_argument(
    "--height_transition_rate_mps",
    type=float,
    default=0.08,
    help="Height-command slew rate used by the transition diagnostic.",
)
parser.add_argument(
    "--height_policy_command_offset_m",
    type=float,
    default=0.0,
    help="Add an offset to the five height-command history samples seen by the policy only.",
)
parser.add_argument(
    "--height_policy_command_minimum_m",
    type=float,
    default=None,
    help="Clamp the five height-command history samples seen by the policy to this minimum.",
)
parser.add_argument(
    "--height_transition_sequence",
    type=str,
    default="0.54,0.60,0.66,0.72,0.60,0.54,0.72",
    help="Comma-separated requested heights for the transition diagnostic.",
)
parser.add_argument(
    "--height_transition_command_range",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=None,
    help="Override the command range without changing the action contract.",
)
parser.add_argument(
    "--height_transition_zero_sagittal_actions",
    action="store_true",
    help="Zero hip, knee, and ankle pitch policy residuals during transition diagnostics.",
)
parser.add_argument(
    "--height_transition_zero_ankle_roll_actions",
    action="store_true",
    help="Zero both ankle-roll policy residuals during transition diagnostics.",
)
parser.add_argument(
    "--height_transition_ankle_roll_action_scale",
    type=float,
    default=None,
    help="Scale both ankle-roll policy residuals during transition diagnostics.",
)
parser.add_argument(
    "--height_transition_ankle_roll_position_scale",
    type=float,
    default=None,
    help="Scale both ankle-roll physical residual ranges during transition diagnostics.",
)
parser.add_argument(
    "--height_transition_ankle_roll_contract_scale",
    type=float,
    default=None,
    help="Scale ankle-roll physical residuals and matching last-action observations.",
)
parser.add_argument(
    "--height_transition_ankle_roll_history_scale",
    type=float,
    default=None,
    help="Override ankle-roll scaling in the policy and critic last-action history.",
)
parser.add_argument(
    "--height_transition_zero_policy_actions",
    action="store_true",
    help="Zero every policy residual during transition diagnostics.",
)
parser.add_argument(
    "--height_transition_initial_yaw_rad",
    type=float,
    default=None,
    help="Override the initial standing-reset yaw for every diagnostic environment.",
)
parser.add_argument(
    "--height_transition_residual_base_maximum_scale",
    type=float,
    default=None,
    help="Override the low-height residual scale for non-amplified leg joints.",
)
parser.add_argument(
    "--height_transition_residual_handoff_scale",
    type=float,
    default=None,
    help="Override the low-height handoff scale for amplified sagittal leg joints.",
)
parser.add_argument(
    "--height_transition_leg_damping_multiplier",
    type=float,
    default=None,
    help="Multiply leg and foot actuator damping for transition diagnostics.",
)
parser.add_argument(
    "--height_transition_leg_target_velocity_limit",
    type=float,
    default=None,
    help="Override the leg position-target velocity limit in rad/s.",
)
parser.add_argument(
    "--height_transition_bad_orientation_limit_rad",
    type=float,
    default=None,
    help="Override the bad-orientation termination angle for transition diagnostics.",
)
parser.add_argument(
    "--height_transition_minimum_root_height_m",
    type=float,
    default=None,
    help="Override the root-height termination threshold for transition diagnostics.",
)
parser.add_argument(
    "--height_transition_env_spacing",
    type=float,
    default=None,
    help="Override cloned-environment spacing for transition diagnostics.",
)
parser.add_argument(
    "--height_transition_posture_depth_scale",
    type=float,
    default=None,
    help="Scale the low posture away from the high posture without moving the high endpoint.",
)
parser.add_argument(
    "--height_transition_posture_exponent",
    type=float,
    default=None,
    help="Override the height-conditioned posture interpolation exponent.",
)
parser.add_argument(
    "--height_transition_posture_low_exponent",
    type=float,
    default=None,
    help="Override the dedicated low-segment posture interpolation exponent.",
)
parser.add_argument(
    "--height_transition_posture_low_handoff_height",
    type=float,
    default=None,
    help="Enable a dedicated low posture segment below this height.",
)
parser.add_argument(
    "--height_transition_posture_phase_knots",
    type=str,
    default=None,
    help=(
        "Override posture phase knots as comma-separated height:phase pairs."
    ),
)
parser.add_argument(
    "--height_transition_posture_minimum",
    type=float,
    default=None,
    help="Override the command height mapped to the low posture endpoint.",
)
parser.add_argument(
    "--height_transition_deep_handoff_scale",
    type=float,
    default=None,
    help="Override the deep residual handoff minimum scale.",
)
parser.add_argument(
    "--height_transition_deep_handoff_range",
    type=float,
    nargs=2,
    metavar=("MIN", "MAX"),
    default=None,
    help="Override the deep residual handoff height range.",
)
parser.add_argument(
    "--height_transition_deep_handoff_sagittal_only",
    action="store_true",
    help="Apply the deep residual handoff only to sagittal leg joints.",
)
parser.add_argument(
    "--height_transition_measured_conditioning_blend",
    type=float,
    default=None,
    help="Blend measured height into both posture and residual-gain conditioning.",
)
parser.add_argument(
    "--height_transition_posture_measured_blend",
    type=float,
    default=None,
    help="Blend measured height into posture conditioning only.",
)
parser.add_argument(
    "--height_transition_residual_measured_blend",
    type=float,
    default=None,
    help="Blend measured height into residual-gain conditioning only.",
)
parser.add_argument(
    "--height_transition_residual_conditioning_maximum",
    action="store_true",
    help="Condition residual gain on max(commanded height, measured height).",
)
parser.add_argument(
    "--startup_trace_output",
    type=str,
    default=None,
    help="Write the first policy steps after a standing reset to this JSON file.",
)
parser.add_argument(
    "--startup_trace_steps",
    type=int,
    default=10,
    help="Number of policy steps to record with --startup_trace_output.",
)
parser.add_argument(
    "--sanitize_actor_output",
    type=str,
    default=None,
    help="Write a checkpoint whose actor is distilled to the configured executed action range.",
)
parser.add_argument(
    "--sanitize_actor_steps",
    type=int,
    default=300,
    help="Policy steps collected per actor-head distillation round.",
)
parser.add_argument(
    "--sanitize_actor_rounds",
    type=int,
    default=3,
    help="Teacher-forced bounded-actor distillation rounds.",
)
parser.add_argument(
    "--sanitize_actor_learning_rate",
    type=float,
    default=3.0e-4,
    help="Learning rate used to distill the bounded actor.",
)
parser.add_argument(
    "--distill_height_response_output",
    type=str,
    default=None,
    help="Write a checkpoint with a supervised low/high height action response.",
)
parser.add_argument("--distill_height_response_warmup_steps", type=int, default=150)
parser.add_argument("--distill_height_response_collect_steps", type=int, default=200)
parser.add_argument("--distill_height_response_optimization_steps", type=int, default=1000)
parser.add_argument("--distill_height_response_batch_size", type=int, default=2048)
parser.add_argument("--distill_height_response_learning_rate", type=float, default=1.0e-3)
parser.add_argument(
    "--distill_height_response_train_all_layers",
    action="store_true",
    help="Fit the complete actor instead of only its output layer.",
)
parser.add_argument(
    "--distill_height_response_target_weight",
    type=float,
    default=1.0,
    help="Extra loss weight for the selected height-response action dimensions.",
)
parser.add_argument("--distill_height_response_low", type=float, default=0.68)
parser.add_argument("--distill_height_response_high", type=float, default=0.72)
parser.add_argument(
    "--distill_height_response_anchor",
    choices=("low", "high"),
    default="low",
    help="Preserve the teacher at this endpoint and shape the opposite endpoint.",
)
parser.add_argument(
    "--distill_height_response_delta",
    type=str,
    default="10:0.10,13:-0.35,14:0.10,16:0.10,19:-0.35,20:0.10",
    help="Comma-separated ACTION_INDEX:HIGH_MINUS_LOW_ACTION pairs.",
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
if args_cli.height_transition_hold_s <= 2.0:
    parser.error("--height_transition_hold_s must be greater than two seconds")
if args_cli.height_transition_rate_mps <= 0.0:
    parser.error("--height_transition_rate_mps must be positive")
if (
    args_cli.height_transition_ankle_roll_action_scale is not None
    and args_cli.height_transition_ankle_roll_action_scale < 0.0
):
    parser.error("--height_transition_ankle_roll_action_scale must be non-negative")
try:
    args_cli.height_transition_sequence = tuple(
        float(value.strip())
        for value in args_cli.height_transition_sequence.split(",")
        if value.strip()
    )
except ValueError as error:
    parser.error(f"--height_transition_sequence contains a non-numeric height: {error}")
if not args_cli.height_transition_sequence:
    parser.error("--height_transition_sequence must contain at least one height")
if args_cli.startup_trace_steps <= 0:
    parser.error("--startup_trace_steps must be positive")
if args_cli.sanitize_actor_steps <= 0:
    parser.error("--sanitize_actor_steps must be positive")
if args_cli.sanitize_actor_rounds <= 0:
    parser.error("--sanitize_actor_rounds must be positive")
if args_cli.sanitize_actor_learning_rate <= 0.0:
    parser.error("--sanitize_actor_learning_rate must be positive")
for name in (
    "distill_height_response_collect_steps",
    "distill_height_response_optimization_steps",
    "distill_height_response_batch_size",
):
    if getattr(args_cli, name) <= 0:
        parser.error(f"--{name} must be positive")
if args_cli.distill_height_response_warmup_steps < 0:
    parser.error("--distill_height_response_warmup_steps must be non-negative")
if args_cli.distill_height_response_learning_rate <= 0.0:
    parser.error("--distill_height_response_learning_rate must be positive")
if args_cli.distill_height_response_target_weight < 1.0:
    parser.error("--distill_height_response_target_weight must be at least one")
if args_cli.distill_height_response_low >= args_cli.distill_height_response_high:
    parser.error("height-response low target must be below the high target")
try:
    args_cli.distill_height_response_delta = tuple(
        (int(index), float(delta))
        for item in args_cli.distill_height_response_delta.split(",")
        for index, delta in (item.strip().split(":"),)
    )
except (TypeError, ValueError) as error:
    parser.error(f"invalid --distill_height_response_delta: {error}")
if not args_cli.distill_height_response_delta:
    parser.error("--distill_height_response_delta must not be empty")
# always enable cameras to record video
if args_cli.video:
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
from booster_train.tasks.manager_based.height_tracking.observations import (
    scaled_last_action,
)
from k1_wbc_export_manifest import write_export_manifest
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils import math as math_utils
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
    if args_cli.height_transition_env_spacing is not None:
        if args_cli.height_transition_env_spacing < 0.0:
            raise ValueError("--height_transition_env_spacing must be non-negative")
        env_cfg.scene.env_spacing = args_cli.height_transition_env_spacing

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
    if args_cli.height_transition_output is not None:
        # The default K1 episode is 32 seconds, shorter than the complete
        # seven-command acceptance sequence. Keep timeout resets out of the
        # transition metrics so only genuine invalid states count as failures.
        env_cfg.episode_length_s = max(
            env_cfg.episode_length_s,
            len(args_cli.height_transition_sequence) * args_cli.height_transition_hold_s + 5.0,
        )
        action_cfg = env_cfg.actions.joint_pos
        if args_cli.height_transition_command_range is not None:
            command_minimum, command_maximum = args_cli.height_transition_command_range
            if command_minimum >= command_maximum:
                raise ValueError(
                    "--height_transition_command_range MIN must be below MAX"
                )
            env_cfg.commands.height.ranges.height = (
                command_minimum,
                command_maximum,
            )
        if args_cli.height_transition_posture_exponent is not None:
            if args_cli.height_transition_posture_exponent <= 0.0:
                raise ValueError(
                    "--height_transition_posture_exponent must be positive"
                )
            action_cfg.height_posture_exponent = (
                args_cli.height_transition_posture_exponent
            )
        if args_cli.height_transition_ankle_roll_position_scale is not None:
            multiplier = args_cli.height_transition_ankle_roll_position_scale
            if multiplier < 0.0:
                raise ValueError(
                    "--height_transition_ankle_roll_position_scale must be non-negative"
                )
            position_scale = list(action_cfg.position_scale)
            for joint_index in (15, 21):
                position_scale[joint_index] *= multiplier
            action_cfg.position_scale = position_scale
        if args_cli.height_transition_ankle_roll_contract_scale is not None:
            multiplier = args_cli.height_transition_ankle_roll_contract_scale
            if multiplier < 0.0:
                raise ValueError(
                    "--height_transition_ankle_roll_contract_scale must be non-negative"
                )
            position_scale = list(action_cfg.position_scale)
            for joint_index in (15, 21):
                position_scale[joint_index] *= multiplier
            action_cfg.position_scale = position_scale
            for observation_group in (
                env_cfg.observations.policy,
                env_cfg.observations.critic,
            ):
                observation_group.actions.func = scaled_last_action
                observation_group.actions.params = {
                    "joint_indices": (15, 21),
                    "scale": multiplier,
                }
        if args_cli.height_transition_ankle_roll_history_scale is not None:
            multiplier = args_cli.height_transition_ankle_roll_history_scale
            if multiplier < 0.0:
                raise ValueError(
                    "--height_transition_ankle_roll_history_scale must be non-negative"
                )
            for observation_group in (
                env_cfg.observations.policy,
                env_cfg.observations.critic,
            ):
                observation_group.actions.func = scaled_last_action
                observation_group.actions.params = {
                    "joint_indices": (15, 21),
                    "scale": multiplier,
                }
        if args_cli.height_transition_posture_low_exponent is not None:
            if args_cli.height_transition_posture_low_exponent <= 0.0:
                raise ValueError(
                    "--height_transition_posture_low_exponent must be positive"
                )
            action_cfg.height_posture_low_exponent = (
                args_cli.height_transition_posture_low_exponent
            )
        if args_cli.height_transition_posture_low_handoff_height is not None:
            posture_minimum, posture_maximum = action_cfg.height_posture_range
            handoff_height = args_cli.height_transition_posture_low_handoff_height
            if not posture_minimum < handoff_height < posture_maximum:
                raise ValueError(
                    "--height_transition_posture_low_handoff_height must be inside "
                    "the posture range"
                )
            action_cfg.height_posture_low_handoff_height = handoff_height
        if args_cli.height_transition_posture_phase_knots is not None:
            try:
                phase_knots = [
                    tuple(float(value) for value in pair.split(":"))
                    for pair in args_cli.height_transition_posture_phase_knots.split(",")
                ]
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "--height_transition_posture_phase_knots must use height:phase pairs"
                ) from exc
            if any(len(pair) != 2 for pair in phase_knots):
                raise ValueError(
                    "--height_transition_posture_phase_knots must use height:phase pairs"
                )
            action_cfg.height_posture_phase_knots = phase_knots
        if args_cli.height_transition_posture_minimum is not None:
            _, posture_maximum = action_cfg.height_posture_range
            if args_cli.height_transition_posture_minimum >= posture_maximum:
                raise ValueError(
                    "--height_transition_posture_minimum must be below the maximum"
                )
            action_cfg.height_posture_range = (
                args_cli.height_transition_posture_minimum,
                posture_maximum,
            )
        if args_cli.height_transition_posture_depth_scale is not None:
            depth_scale = args_cli.height_transition_posture_depth_scale
            if depth_scale <= 0.0:
                raise ValueError(
                    "--height_transition_posture_depth_scale must be positive"
                )
            low_center = action_cfg.height_posture_center
            high_center = action_cfg.height_posture_high_center
            if low_center is None or high_center is None:
                raise ValueError(
                    "posture depth scaling requires low and high posture centers"
                )
            scaled_low_center = list(low_center)
            for joint_index in (10, 13, 16, 19):
                candidate = (
                    high_center[joint_index]
                    + depth_scale
                    * (low_center[joint_index] - high_center[joint_index])
                )
                minimum = (
                    action_cfg.position_minimum[joint_index]
                    + action_cfg.position_target_margin
                )
                maximum = (
                    action_cfg.position_maximum[joint_index]
                    - action_cfg.position_target_margin
                )
                scaled_low_center[joint_index] = min(
                    max(candidate, minimum), maximum
                )
            action_cfg.height_posture_center = scaled_low_center
        if args_cli.height_transition_residual_base_maximum_scale is not None:
            action_cfg.height_residual_base_maximum_scale = (
                args_cli.height_transition_residual_base_maximum_scale
            )
        if args_cli.height_transition_residual_handoff_scale is not None:
            action_cfg.height_residual_handoff_minimum_scale = (
                args_cli.height_transition_residual_handoff_scale
            )
        if args_cli.height_transition_deep_handoff_scale is not None:
            action_cfg.height_residual_deep_handoff_minimum_scale = (
                args_cli.height_transition_deep_handoff_scale
            )
        if args_cli.height_transition_deep_handoff_range is not None:
            handoff_minimum, handoff_maximum = (
                args_cli.height_transition_deep_handoff_range
            )
            if handoff_minimum >= handoff_maximum:
                raise ValueError(
                    "--height_transition_deep_handoff_range MIN must be below MAX"
                )
            action_cfg.height_residual_deep_handoff_range = (
                handoff_minimum,
                handoff_maximum,
            )
        if args_cli.height_transition_deep_handoff_sagittal_only:
            action_cfg.height_residual_deep_handoff_joint_names = list(
                action_cfg.height_residual_amplify_joint_names
            )
        if args_cli.height_transition_measured_conditioning_blend is not None:
            blend = args_cli.height_transition_measured_conditioning_blend
            if not 0.0 <= blend <= 1.0:
                raise ValueError(
                    "--height_transition_measured_conditioning_blend must be within [0, 1]"
                )
            action_cfg.height_posture_measured_blend = blend
            action_cfg.height_residual_measured_blend = blend
        for argument_name, attribute_name in (
            (
                "height_transition_posture_measured_blend",
                "height_posture_measured_blend",
            ),
            (
                "height_transition_residual_measured_blend",
                "height_residual_measured_blend",
            ),
        ):
            blend = getattr(args_cli, argument_name)
            if blend is None:
                continue
            if not 0.0 <= blend <= 1.0:
                raise ValueError(f"--{argument_name} must be within [0, 1]")
            setattr(action_cfg, attribute_name, blend)
        if args_cli.height_transition_residual_conditioning_maximum:
            action_cfg.height_residual_conditioning_maximum = True
        if args_cli.height_transition_leg_damping_multiplier is not None:
            multiplier = args_cli.height_transition_leg_damping_multiplier
            if multiplier <= 0.0:
                raise ValueError(
                    "--height_transition_leg_damping_multiplier must be positive"
                )
            for actuator_name in ("legs", "feet"):
                actuator_cfg = env_cfg.scene.robot.actuators[actuator_name]
                if isinstance(actuator_cfg.damping, dict):
                    actuator_cfg.damping = {
                        name: value * multiplier
                        for name, value in actuator_cfg.damping.items()
                    }
                else:
                    actuator_cfg.damping *= multiplier
        if args_cli.height_transition_leg_target_velocity_limit is not None:
            limit = args_cli.height_transition_leg_target_velocity_limit
            if limit <= 0.0:
                raise ValueError(
                    "--height_transition_leg_target_velocity_limit must be positive"
                )
            velocity_limit = list(action_cfg.position_target_velocity_limit)
            velocity_limit[10:] = [limit] * (len(velocity_limit) - 10)
            action_cfg.position_target_velocity_limit = velocity_limit
        if args_cli.height_transition_bad_orientation_limit_rad is not None:
            env_cfg.terminations.bad_orientation.params["limit_angle"] = (
                args_cli.height_transition_bad_orientation_limit_rad
            )
        if args_cli.height_transition_minimum_root_height_m is not None:
            env_cfg.terminations.base_too_low.params["minimum_height"] = (
                args_cli.height_transition_minimum_root_height_m
            )
    if any(
        output is not None
        for output in (
            args_cli.startup_trace_output,
            args_cli.height_transition_output,
            args_cli.height_diagnostic_output,
        )
    ):
        # Deployment uses deterministic sensors. Keep acceptance diagnostics on
        # that same observation contract instead of masking a brittle policy
        # with fresh training noise at every step.
        env_cfg.observations.policy.enable_corruption = False

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

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

    if args_cli.startup_trace_output is not None:
        if not task_spec.kwargs.get("wbc_height_tracking", False):
            raise ValueError("--startup_trace_output requires a WBC height-tracking task")
        base_env = env.unwrapped
        height_command = base_env.command_manager.get_term("height")
        action_term = base_env.action_manager.get_term("joint_pos")
        robot = base_env.scene["robot"]
        target = torch.full((base_env.num_envs,), 0.72, device=base_env.device)
        height_command.set_manual_height(target)
        env.reset()
        height_command.set_manual_height(target)

        def trace_observations():
            current = env.get_observations()
            if version("rsl-rl-lib").startswith("2.3."):
                return current[0]
            return current

        observations = trace_observations()
        initial_observation = observations[0].detach().cpu().tolist()
        samples = []
        for step in range(args_cli.startup_trace_steps):
            with torch.inference_mode():
                actions = policy(observations)
                observations, _, dones, _ = env.step(actions)
            height_command.set_manual_height(target)
            samples.append(
                {
                    "step": step + 1,
                    "time_s": (step + 1) * float(base_env.step_dt),
                    "root_position": robot.data.root_pos_w[0].detach().cpu().tolist(),
                    "root_quaternion": robot.data.root_quat_w[0].detach().cpu().tolist(),
                    "root_linear_velocity_body": robot.data.root_lin_vel_b[0].detach().cpu().tolist(),
                    "root_angular_velocity_body": robot.data.root_ang_vel_b[0].detach().cpu().tolist(),
                    "joint_position": robot.data.joint_pos[0, action_term._joint_ids].detach().cpu().tolist(),
                    "joint_velocity": robot.data.joint_vel[0, action_term._joint_ids].detach().cpu().tolist(),
                    "raw_action": actions[0].detach().cpu().tolist(),
                    "executed_action": base_env.action_manager.action[0].detach().cpu().tolist(),
                    "processed_target": action_term.processed_actions[0].detach().cpu().tolist(),
                    "measured_height": float(height_command.measured_height[0]),
                    "done": bool(dones[0]),
                }
            )
        actuator_delay_steps = {}
        for actuator_name, actuator in robot.actuators.items():
            delay_buffer = getattr(actuator, "positions_delay_buffer", None)
            if delay_buffer is not None:
                actuator_delay_steps[actuator_name] = (
                    delay_buffer.time_lags.detach().cpu().tolist()
                )

        report = {
            "checkpoint": resume_path,
            "task": args_cli.task,
            "actuator_delay_steps": actuator_delay_steps,
            "step_dt": float(base_env.step_dt),
            "joint_names": list(action_term._joint_names),
            "initial_observation": initial_observation,
            "samples": samples,
        }
        output_path = os.path.abspath(args_cli.startup_trace_output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(report, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        env.close()
        return

    if args_cli.sanitize_actor_output is not None:
        if not task_spec.kwargs.get("wbc_height_tracking", False):
            raise ValueError("--sanitize_actor_output requires a WBC height-tracking task")
        import copy
        import math

        base_env = env.unwrapped
        action_clip = float(agent_cfg.clip_actions)
        if not math.isfinite(action_clip) or action_clip <= 0.0:
            raise ValueError(f"actor sanitization requires a positive finite action clip, got {action_clip}")
        actor = policy_nn.actor
        if not isinstance(actor, torch.nn.Sequential) or not isinstance(actor[-1], torch.nn.Linear):
            raise TypeError("actor sanitization requires a Sequential actor with a final Linear layer")
        teacher_actor = copy.deepcopy(actor).eval()
        output_layer: torch.nn.Linear = actor[-1]
        observation_manager = base_env.observation_manager
        term_names = observation_manager.active_terms["policy"]
        term_dims = observation_manager.group_obs_term_dim["policy"]
        action_term_index = term_names.index("actions")
        action_history_start = sum(math.prod(dim) for dim in term_dims[:action_term_index])
        action_history_width = math.prod(term_dims[action_term_index])
        action_dim = output_layer.out_features
        if action_history_width % action_dim:
            raise ValueError(
                "action observation history is not divisible by the policy action dimension: "
                f"{action_history_width} vs {action_dim}"
            )
        history_length = action_history_width // action_dim
        action_history_stop = action_history_start + action_history_width
        optimizer = torch.optim.Adam(actor.parameters(), lr=args_cli.sanitize_actor_learning_rate)
        sample_count = 0
        round_reports = []

        def policy_observations():
            current = env.get_observations()
            if version("rsl-rl-lib").startswith("2.3."):
                return current[0]
            return current

        for round_index in range(args_cli.sanitize_actor_rounds):
            env.reset()
            observations = policy_observations()
            teacher_history = torch.zeros(
                (base_env.num_envs, history_length, action_dim),
                device=base_env.device,
            )
            raw_loss_sum = 0.0
            executed_error_sum = 0.0
            saturation_sum = 0.0
            for _ in range(args_cli.sanitize_actor_steps):
                with torch.no_grad():
                    teacher_observations = observations.clone()
                    teacher_observations[:, action_history_start:action_history_stop] = (
                        teacher_history.reshape(base_env.num_envs, -1)
                    )
                    teacher_raw_actions = teacher_actor(teacher_observations)
                    teacher_targets = torch.clamp(teacher_raw_actions, -action_clip, action_clip)

                student_actions = actor(observations.detach())
                loss = torch.nn.functional.smooth_l1_loss(
                    student_actions,
                    teacher_targets,
                    beta=0.25,
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), 5.0)
                optimizer.step()
                with torch.no_grad():
                    updated_student_actions = actor(observations)
                    executed_student_actions = torch.clamp(updated_student_actions, -action_clip, action_clip)
                    raw_loss_sum += float(loss)
                    executed_error_sum += float(
                        torch.mean(torch.square(executed_student_actions - teacher_targets))
                    )
                    saturation_sum += float((torch.abs(updated_student_actions) >= action_clip).float().mean())
                    sample_count += base_env.num_envs
                    # Keep data on the known-good teacher state distribution.
                    # A partially distilled student otherwise falls immediately
                    # and poisons the remaining supervised batches.
                    executed_actions = teacher_targets
                    observations, _, dones, _ = env.step(executed_actions)
                    teacher_history = torch.roll(teacher_history, shifts=-1, dims=1)
                    teacher_history[:, -1] = teacher_targets
                    if torch.any(dones):
                        teacher_history[dones.bool()] = 0.0
            round_report = {
                "round": round_index + 1,
                "samples": sample_count,
                "action_clip": action_clip,
                "raw_actor_smooth_l1": raw_loss_sum / args_cli.sanitize_actor_steps,
                "executed_action_mse": executed_error_sum / args_cli.sanitize_actor_steps,
                "raw_actor_saturation_fraction": saturation_sum / args_cli.sanitize_actor_steps,
            }
            round_reports.append(round_report)
            print(f"[INFO] Actor sanitization round: {round_report}")

        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        checkpoint["model_state_dict"] = {
            key: value.detach().cpu() for key, value in policy_nn.state_dict().items()
        }
        checkpoint["infos"] = {
            "actor_output_sanitization": {
                "source_checkpoint": resume_path,
                "action_clip": [-action_clip, action_clip],
                "rounds": round_reports,
            }
        }
        output_path = os.path.abspath(args_cli.sanitize_actor_output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(checkpoint, output_path)
        print(f"[INFO] Wrote sanitized actor checkpoint: {output_path}")
        env.close()
        return

    if args_cli.distill_height_response_output is not None:
        if not task_spec.kwargs.get("wbc_height_tracking", False):
            raise ValueError(
                "--distill_height_response_output requires a WBC height-tracking task"
            )
        import copy
        import math

        base_env = env.unwrapped
        actor = policy_nn.actor
        if not isinstance(actor, torch.nn.Sequential) or not isinstance(
            actor[-1], torch.nn.Linear
        ):
            raise TypeError("height-response distillation requires a Sequential actor")
        output_layer: torch.nn.Linear = actor[-1]
        action_count = output_layer.out_features
        response_indices = tuple(index for index, _ in args_cli.distill_height_response_delta)
        if len(set(response_indices)) != len(response_indices) or any(
            index < 0 or index >= action_count for index in response_indices
        ):
            raise ValueError("height-response action indices are invalid")
        response_target = torch.tensor(
            [delta for _, delta in args_cli.distill_height_response_delta],
            dtype=torch.float32,
            device=base_env.device,
        )
        teacher_actor = copy.deepcopy(actor).eval()
        for parameter in actor.parameters():
            parameter.requires_grad_(args_cli.distill_height_response_train_all_layers)
        if not args_cli.distill_height_response_train_all_layers:
            for parameter in output_layer.parameters():
                parameter.requires_grad_(True)

        observation_manager = base_env.observation_manager
        term_names = observation_manager.active_terms["policy"]
        term_dims = observation_manager.group_obs_term_dim["policy"]
        command_term_index = term_names.index("height_command")
        command_history_start = sum(
            math.prod(dim) for dim in term_dims[:command_term_index]
        )
        command_history_width = math.prod(term_dims[command_term_index])
        command_history_stop = command_history_start + command_history_width
        height_command = base_env.command_manager.get_term("height")
        low_targets = torch.full(
            (base_env.num_envs,),
            args_cli.distill_height_response_low,
            device=base_env.device,
        )

        def policy_observations():
            current = env.get_observations()
            if version("rsl-rl-lib").startswith("2.3."):
                return current[0]
            return current

        height_command.set_manual_height(low_targets)
        env.reset()
        height_command.set_manual_height(low_targets)
        observations = policy_observations()
        collected = []
        collection_steps = (
            args_cli.distill_height_response_warmup_steps
            + args_cli.distill_height_response_collect_steps
        )
        for step in range(collection_steps):
            with torch.inference_mode():
                teacher_actions = teacher_actor(observations)
                observations, _, dones, _ = env.step(teacher_actions)
            height_command.set_manual_height(low_targets)
            if torch.any(dones):
                observations = policy_observations()
            if step >= args_cli.distill_height_response_warmup_steps:
                collected.append(observations.detach().clone())
        observation_dataset = torch.cat(collected, dim=0)

        trainable_parameters = [parameter for parameter in actor.parameters() if parameter.requires_grad]
        optimizer = torch.optim.Adam(
            trainable_parameters,
            lr=args_cli.distill_height_response_learning_rate,
        )
        loss_history = []
        for _ in range(args_cli.distill_height_response_optimization_steps):
            indexes = torch.randint(
                observation_dataset.shape[0],
                (min(args_cli.distill_height_response_batch_size, observation_dataset.shape[0]),),
                device=base_env.device,
            )
            sampled = observation_dataset[indexes]
            low_observations = sampled.clone()
            high_observations = sampled.clone()
            low_observations[:, command_history_start:command_history_stop] = (
                args_cli.distill_height_response_low
            )
            high_observations[:, command_history_start:command_history_stop] = (
                args_cli.distill_height_response_high
            )
            with torch.no_grad():
                teacher_low = teacher_actor(low_observations)
                teacher_high = teacher_actor(high_observations)
                low_target = teacher_low
                high_target = teacher_high
                if args_cli.distill_height_response_anchor == "low":
                    high_target = teacher_high.clone()
                    high_target[:, response_indices] = (
                        teacher_low[:, response_indices] + response_target
                    )
                else:
                    low_target = teacher_low.clone()
                    low_target[:, response_indices] = (
                        teacher_high[:, response_indices] - response_target
                    )
            student_low = actor(low_observations)
            student_high = actor(high_observations)
            low_loss = torch.nn.functional.mse_loss(student_low, low_target)
            selected_low_loss = torch.nn.functional.mse_loss(
                student_low[:, response_indices],
                low_target[:, response_indices],
            )
            high_loss = torch.nn.functional.mse_loss(student_high, high_target)
            loss = (
                low_loss
                + high_loss
                + (args_cli.distill_height_response_target_weight - 1.0) * selected_low_loss
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_parameters, 5.0)
            optimizer.step()
            loss_history.append((float(low_loss), float(high_loss)))

        with torch.inference_mode():
            validation = observation_dataset[: min(4096, observation_dataset.shape[0])]
            low_validation = validation.clone()
            high_validation = validation.clone()
            low_validation[:, command_history_start:command_history_stop] = (
                args_cli.distill_height_response_low
            )
            high_validation[:, command_history_start:command_history_stop] = (
                args_cli.distill_height_response_high
            )
            learned_delta = (
                actor(high_validation)[:, response_indices]
                - actor(low_validation)[:, response_indices]
            ).mean(dim=0)

        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        checkpoint["model_state_dict"] = {
            key: value.detach().cpu() for key, value in policy_nn.state_dict().items()
        }
        infos = dict(checkpoint.get("infos") or {})
        infos["height_response_distillation"] = {
            "source_checkpoint": resume_path,
            "observation_count": int(observation_dataset.shape[0]),
            "low_height_m": args_cli.distill_height_response_low,
            "high_height_m": args_cli.distill_height_response_high,
            "anchor": args_cli.distill_height_response_anchor,
            "train_all_layers": args_cli.distill_height_response_train_all_layers,
            "target_weight": args_cli.distill_height_response_target_weight,
            "action_indices": list(response_indices),
            "target_action_delta": response_target.detach().cpu().tolist(),
            "learned_action_delta": learned_delta.detach().cpu().tolist(),
            "final_low_loss": loss_history[-1][0],
            "final_high_loss": loss_history[-1][1],
        }
        checkpoint["infos"] = infos
        output_path = os.path.abspath(args_cli.distill_height_response_output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        torch.save(checkpoint, output_path)
        print(f"[INFO] Wrote height-response checkpoint: {output_path}")
        print(json.dumps(infos["height_response_distillation"], indent=2))
        env.close()
        return

    if args_cli.height_transition_output is not None:
        if not task_spec.kwargs.get("wbc_height_tracking", False):
            raise ValueError("--height_transition_output requires a WBC height-tracking task")
        base_env = env.unwrapped
        height_command = base_env.command_manager.get_term("height")
        robot = base_env.scene["robot"]
        action_term = base_env.action_manager.get_term("joint_pos")
        reward_term_names = tuple(base_env.reward_manager._term_names)
        termination_term_names = tuple(base_env.termination_manager._term_names)
        sequence = list(args_cli.height_transition_sequence)
        dt = float(base_env.step_dt)
        steps_per_segment = round(args_cli.height_transition_hold_s / dt)
        steady_steps = round(2.0 / dt)
        settle_window_steps = round(1.0 / dt)
        command_values = torch.full(
            (base_env.num_envs,), sequence[0], device=base_env.device
        )
        height_command.set_manual_height(command_values)
        env.reset()
        height_command.set_manual_height(command_values)
        if args_cli.height_transition_initial_yaw_rad is not None:
            root_pose = robot.data.root_state_w[:, :7].clone()
            zero = torch.zeros(base_env.num_envs, device=base_env.device)
            fixed_yaw = torch.full_like(
                zero, args_cli.height_transition_initial_yaw_rad
            )
            root_pose[:, 3:7] = math_utils.quat_from_euler_xyz(
                zero, zero, fixed_yaw
            )
            robot.write_root_pose_to_sim(root_pose)
        _, _, initial_root_yaw = math_utils.euler_xyz_from_quat(
            robot.data.root_quat_w
        )

        def policy_observations():
            current = env.get_observations()
            if version("rsl-rl-lib").startswith("2.3."):
                current = current[0]
            if args_cli.height_policy_command_offset_m:
                current = current.clone()
                current[:, -5:] += args_cli.height_policy_command_offset_m
            if args_cli.height_policy_command_minimum_m is not None:
                current = current.clone()
                current[:, -5:] = torch.clamp_min(
                    current[:, -5:], args_cli.height_policy_command_minimum_m
                )
            return current

        observations = policy_observations()
        previous_targets = action_term.processed_actions.detach().clone()
        previous_policy_actions = base_env.action_manager.action.detach().clone()
        segment_reports = []
        total_terminations = 0
        total_invalid_states = 0
        total_timeouts = 0
        for segment_index, requested_height in enumerate(sequence):
            measured_rows = []
            command_rows = []
            planar_speed_rows = []
            angular_speed_rows = []
            joint_speed_rows = []
            joint_velocity_rows = []
            joint_position_rows = []
            joint_target_rows = []
            joint_torque_rows = []
            target_delta_rows = []
            joint_target_delta_rows = []
            policy_action_delta_rows = []
            joint_policy_action_delta_rows = []
            policy_action_clip_rows = []
            joint_policy_action_clip_rows = []
            raw_action_rows = []
            executed_action_rows = []
            command_center_rows = []
            reward_term_rows = []
            tilt_rows = []
            done_rows = []
            invalid_rows = []
            timeout_rows = []
            termination_term_rows = {
                name: [] for name in termination_term_names
            }
            for _ in range(steps_per_segment):
                maximum_step = args_cli.height_transition_rate_mps * dt
                difference = requested_height - command_values
                command_values += torch.clamp(difference, -maximum_step, maximum_step)
                height_command.set_manual_height(command_values)
                observations = policy_observations()
                with torch.inference_mode():
                    actions = policy(observations)
                    if args_cli.height_transition_ankle_roll_action_scale is not None:
                        actions = actions.clone()
                        actions[:, [15, 21]] *= (
                            args_cli.height_transition_ankle_roll_action_scale
                        )
                    if args_cli.height_transition_zero_policy_actions:
                        actions = torch.zeros_like(actions)
                    elif args_cli.height_transition_zero_sagittal_actions:
                        actions = actions.clone()
                        actions[:, [10, 13, 14, 16, 19, 20]] = 0.0
                    elif args_cli.height_transition_zero_ankle_roll_actions:
                        actions = actions.clone()
                        actions[:, [15, 21]] = 0.0
                    observations, _, dones, _ = env.step(actions)
                height_command.set_manual_height(command_values)
                reward_term_rows.append(
                    base_env.reward_manager._step_reward.detach().cpu().clone()
                )
                processed_targets = action_term.processed_actions.detach()
                policy_actions = base_env.action_manager.action.detach()
                raw_action_rows.append(actions.detach().cpu())
                executed_action_rows.append(policy_actions.detach().cpu())
                command_center_rows.append(action_term.command_center.detach().cpu())
                measured_rows.append(height_command.measured_height.detach().cpu())
                command_rows.append(height_command.target_height.detach().cpu())
                planar_speed_rows.append(
                    torch.linalg.vector_norm(robot.data.root_lin_vel_b[:, :2], dim=1).detach().cpu()
                )
                angular_speed_rows.append(
                    torch.linalg.vector_norm(robot.data.root_ang_vel_b[:, :2], dim=1).detach().cpu()
                )
                joint_velocity = robot.data.joint_vel[:, action_term._joint_ids]
                joint_velocity_rows.append(joint_velocity.detach().cpu())
                joint_position_rows.append(
                    robot.data.joint_pos[:, action_term._joint_ids].detach().cpu()
                )
                joint_target_rows.append(processed_targets.detach().cpu())
                joint_torque_rows.append(
                    robot.data.applied_torque[:, action_term._joint_ids].detach().cpu()
                )
                joint_speed_rows.append(
                    torch.sqrt(torch.mean(torch.square(joint_velocity), dim=1)).detach().cpu()
                )
                joint_target_delta = processed_targets - previous_targets
                joint_target_delta_rows.append(joint_target_delta.detach().cpu())
                target_delta_rows.append(
                    torch.sqrt(torch.mean(torch.square(joint_target_delta), dim=1))
                    .detach()
                    .cpu()
                )
                joint_policy_action_delta = policy_actions - previous_policy_actions
                joint_policy_action_delta_rows.append(joint_policy_action_delta.detach().cpu())
                policy_action_delta_rows.append(
                    torch.sqrt(torch.mean(torch.square(joint_policy_action_delta), dim=1))
                    .detach()
                    .cpu()
                )
                joint_policy_action_clip = (torch.abs(policy_actions) >= 3.999).float()
                joint_policy_action_clip_rows.append(joint_policy_action_clip.detach().cpu())
                policy_action_clip_rows.append(
                    torch.mean(joint_policy_action_clip, dim=1).detach().cpu()
                )
                tilt_rows.append(
                    torch.acos(torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0))
                    .detach()
                    .cpu()
                )
                done_rows.append(dones.detach().cpu().bool())
                invalid_rows.append(base_env.termination_manager.terminated.detach().cpu().bool())
                timeout_rows.append(base_env.termination_manager.time_outs.detach().cpu().bool())
                for name in termination_term_names:
                    termination_term_rows[name].append(
                        base_env.termination_manager.get_term(name)
                        .detach()
                        .cpu()
                        .bool()
                    )
                previous_targets = processed_targets.clone()
                previous_policy_actions = policy_actions.clone()

            measured = torch.stack(measured_rows)
            commanded = torch.stack(command_rows)
            errors = torch.abs(measured - commanded)
            done_matrix = torch.stack(done_rows)
            invalid_matrix = torch.stack(invalid_rows)
            timeout_matrix = torch.stack(timeout_rows)
            termination_term_matrices = {
                name: torch.stack(rows)
                for name, rows in termination_term_rows.items()
            }
            total_terminations += int(done_matrix.sum())
            total_invalid_states += int(invalid_matrix.sum())
            total_timeouts += int(timeout_matrix.sum())
            at_target = torch.abs(commanded[:, 0] - requested_height) < 1.0e-4
            target_step = int(torch.where(at_target)[0][0]) if torch.any(at_target) else steps_per_segment
            settled_times = []
            for env_index in range(base_env.num_envs):
                settled_time = None
                for step in range(target_step, steps_per_segment - settle_window_steps + 1):
                    window = errors[step : step + settle_window_steps, env_index]
                    if torch.all(window <= 0.03):
                        settled_time = (step - target_step) * dt
                        break
                settled_times.append(settled_time)
            all_joint_velocity = torch.stack(joint_velocity_rows)
            all_joint_target_delta = torch.stack(joint_target_delta_rows)
            all_joint_policy_action_delta = torch.stack(joint_policy_action_delta_rows)
            all_raw_action = torch.stack(raw_action_rows)
            steady_slice = slice(max(target_step, steps_per_segment - steady_steps), steps_per_segment)
            steady_measured = measured[steady_slice]
            steady_commanded = commanded[steady_slice]
            planar_speed = torch.stack(planar_speed_rows)[steady_slice]
            angular_speed = torch.stack(angular_speed_rows)[steady_slice]
            joint_speed = torch.stack(joint_speed_rows)[steady_slice]
            joint_velocity = all_joint_velocity[steady_slice]
            joint_position = torch.stack(joint_position_rows)[steady_slice]
            joint_target = torch.stack(joint_target_rows)[steady_slice]
            joint_torque = torch.stack(joint_torque_rows)[steady_slice]
            target_delta = torch.stack(target_delta_rows)[steady_slice]
            joint_target_delta = all_joint_target_delta[steady_slice]
            policy_action_delta = torch.stack(policy_action_delta_rows)[steady_slice]
            joint_policy_action_delta = all_joint_policy_action_delta[steady_slice]
            policy_action_clip = torch.stack(policy_action_clip_rows)[steady_slice]
            joint_policy_action_clip = torch.stack(joint_policy_action_clip_rows)[steady_slice]
            raw_action = all_raw_action[steady_slice]
            executed_action = torch.stack(executed_action_rows)[steady_slice]
            command_center = torch.stack(command_center_rows)[steady_slice]
            reward_terms = torch.stack(reward_term_rows)[steady_slice]
            tilt = torch.stack(tilt_rows)[steady_slice]
            valid_settling = [value for value in settled_times if value is not None]
            signed_error = measured - commanded
            per_env_error = torch.mean(torch.abs(steady_measured - steady_commanded), dim=0)
            per_env_p95 = torch.quantile(torch.abs(steady_measured - steady_commanded), 0.95, dim=0)
            failed_env_indices = torch.where(per_env_p95 > 0.03)[0].tolist()
            terminated_envs = torch.where(torch.any(done_matrix, dim=0))[0]
            stable_envs = torch.where(~torch.any(done_matrix, dim=0))[0]
            precursor_window_steps = max(1, round(0.5 / dt))
            onset_window_steps = max(1, round(2.0 / dt))
            failed_precursor = {"velocity": [], "target_delta": [], "action_delta": [], "raw": []}
            stable_precursor = {"velocity": [], "target_delta": [], "action_delta": [], "raw": []}
            action_onset_leads = [[] for _ in action_term._joint_names]
            speed_onset_leads = [[] for _ in action_term._joint_names]
            for failed_rank, env_index_tensor in enumerate(terminated_envs):
                env_index = int(env_index_tensor)
                first_step = int(torch.where(done_matrix[:, env_index])[0][0])
                start = max(0, first_step - precursor_window_steps)
                end = max(start + 1, first_step)
                failed_precursor["velocity"].append(all_joint_velocity[start:end, env_index])
                failed_precursor["target_delta"].append(all_joint_target_delta[start:end, env_index])
                failed_precursor["action_delta"].append(
                    all_joint_policy_action_delta[start:end, env_index]
                )
                failed_precursor["raw"].append(all_raw_action[start:end, env_index])
                onset_start = max(0, first_step - onset_window_steps)
                onset_action_delta = torch.abs(
                    all_joint_policy_action_delta[onset_start:first_step, env_index]
                )
                onset_speed = torch.abs(
                    all_joint_velocity[onset_start:first_step, env_index]
                )
                for joint_index in range(len(action_term._joint_names)):
                    action_crossings = torch.where(
                        onset_action_delta[:, joint_index] > 0.02
                    )[0]
                    if action_crossings.numel() > 0:
                        action_onset_leads[joint_index].append(
                            (first_step - onset_start - int(action_crossings[0])) * dt
                        )
                    speed_crossings = torch.where(onset_speed[:, joint_index] > 0.2)[0]
                    if speed_crossings.numel() > 0:
                        speed_onset_leads[joint_index].append(
                            (first_step - onset_start - int(speed_crossings[0])) * dt
                        )
                if stable_envs.numel() > 0:
                    stable_index = int(stable_envs[failed_rank % stable_envs.numel()])
                    stable_precursor["velocity"].append(
                        all_joint_velocity[start:end, stable_index]
                    )
                    stable_precursor["target_delta"].append(
                        all_joint_target_delta[start:end, stable_index]
                    )
                    stable_precursor["action_delta"].append(
                        all_joint_policy_action_delta[start:end, stable_index]
                    )
                    stable_precursor["raw"].append(all_raw_action[start:end, stable_index])

            def precursor_joint_report(samples, *, include_onset=False):
                if not samples["velocity"]:
                    return []
                velocity_samples = torch.cat(samples["velocity"], dim=0)
                target_delta_samples = torch.cat(samples["target_delta"], dim=0)
                action_delta_samples = torch.cat(samples["action_delta"], dim=0)
                raw_samples = torch.cat(samples["raw"], dim=0)
                return [
                    {
                        "joint": joint_name,
                        "speed_rms_radps": float(
                            torch.sqrt(torch.mean(torch.square(velocity_samples[:, joint_index])))
                        ),
                        "target_delta_rms_rad": float(
                            torch.sqrt(torch.mean(torch.square(target_delta_samples[:, joint_index])))
                        ),
                        "policy_action_delta_rms": float(
                            torch.sqrt(torch.mean(torch.square(action_delta_samples[:, joint_index])))
                        ),
                        "raw_action_std": float(torch.std(raw_samples[:, joint_index])),
                        **(
                            {
                                "action_delta_onset_lead_s_median": (
                                    float(np.median(action_onset_leads[joint_index]))
                                    if action_onset_leads[joint_index]
                                    else None
                                ),
                                "action_delta_onset_fraction": (
                                    len(action_onset_leads[joint_index])
                                    / max(len(terminated_envs), 1)
                                ),
                                "speed_onset_lead_s_median": (
                                    float(np.median(speed_onset_leads[joint_index]))
                                    if speed_onset_leads[joint_index]
                                    else None
                                ),
                                "speed_onset_fraction": (
                                    len(speed_onset_leads[joint_index])
                                    / max(len(terminated_envs), 1)
                                ),
                            }
                            if include_onset
                            else {}
                        ),
                    }
                    for joint_index, joint_name in enumerate(action_term._joint_names)
                ]

            segment_reports.append(
                {
                    "segment": segment_index,
                    "requested_height_m": requested_height,
                    "effective_height_m": float(torch.mean(commanded[-1])),
                    "ramp_duration_s": target_step * dt,
                    "settled_fraction": len(valid_settling) / base_env.num_envs,
                    "mean_settling_time_after_ramp_s": (
                        float(np.mean(valid_settling)) if valid_settling else None
                    ),
                    "steady_mean_abs_error_m": float(torch.mean(torch.abs(steady_measured - steady_commanded))),
                    "steady_p95_abs_error_m": float(torch.quantile(torch.abs(steady_measured - steady_commanded), 0.95)),
                    "steady_height_std_m": float(torch.mean(torch.std(steady_measured, dim=0))),
                    "maximum_overshoot_m": float(torch.clamp(signed_error, min=0.0).max()),
                    "maximum_undershoot_m": float(torch.clamp(-signed_error, min=0.0).max()),
                    "steady_planar_speed_rms_mps": float(torch.sqrt(torch.mean(torch.square(planar_speed)))),
                    "steady_roll_pitch_rate_rms_radps": float(torch.sqrt(torch.mean(torch.square(angular_speed)))),
                    "steady_joint_speed_rms_radps": float(torch.sqrt(torch.mean(torch.square(joint_speed)))),
                    "steady_joint_target_delta_rms_rad": float(torch.sqrt(torch.mean(torch.square(target_delta)))),
                    "steady_policy_action_delta_rms": float(
                        torch.sqrt(torch.mean(torch.square(policy_action_delta)))
                    ),
                    "steady_policy_action_clip_fraction": float(torch.mean(policy_action_clip)),
                    "steady_torso_tilt_rms_rad": float(torch.sqrt(torch.mean(torch.square(tilt)))),
                    "maximum_torso_tilt_rad": float(torch.max(tilt)),
                    "steady_reward_total_per_s": float(torch.mean(torch.sum(reward_terms, dim=-1))),
                    "steady_reward_terms_per_s": {
                        name: float(torch.mean(reward_terms[:, :, term_index]))
                        for term_index, name in enumerate(reward_term_names)
                    },
                    "per_joint": [
                        {
                            "joint": joint_name,
                            "speed_rms_radps": float(
                                torch.sqrt(torch.mean(torch.square(joint_velocity[:, :, joint_index])))
                            ),
                            "position_mean_rad": float(
                                torch.mean(joint_position[:, :, joint_index])
                            ),
                            "position_std_rad": float(
                                torch.std(joint_position[:, :, joint_index])
                            ),
                            "target_mean_rad": float(
                                torch.mean(joint_target[:, :, joint_index])
                            ),
                            "tracking_error_rms_rad": float(
                                torch.sqrt(
                                    torch.mean(
                                        torch.square(
                                            joint_target[:, :, joint_index]
                                            - joint_position[:, :, joint_index]
                                        )
                                    )
                                )
                            ),
                            "torque_rms_nm": float(
                                torch.sqrt(torch.mean(torch.square(joint_torque[:, :, joint_index])))
                            ),
                            "target_delta_rms_rad": float(
                                torch.sqrt(torch.mean(torch.square(joint_target_delta[:, :, joint_index])))
                            ),
                            "policy_action_delta_rms": float(
                                torch.sqrt(
                                    torch.mean(torch.square(joint_policy_action_delta[:, :, joint_index]))
                                )
                            ),
                            "policy_action_clip_fraction": float(
                                torch.mean(joint_policy_action_clip[:, :, joint_index])
                            ),
                            "raw_action_mean": float(
                                torch.mean(raw_action[:, :, joint_index])
                            ),
                            "raw_action_std": float(
                                torch.std(raw_action[:, :, joint_index])
                            ),
                            "executed_action_mean": float(
                                torch.mean(executed_action[:, :, joint_index])
                            ),
                            "command_center_mean_rad": float(
                                torch.mean(command_center[:, :, joint_index])
                            ),
                            "action_delta_clip_min_rad": float(
                                action_term._clip[0, joint_index, 0].detach().cpu()
                            ),
                            "action_delta_clip_max_rad": float(
                                action_term._clip[0, joint_index, 1].detach().cpu()
                            ),
                        }
                        for joint_index, joint_name in enumerate(action_term._joint_names)
                    ],
                    "termination_count": int(done_matrix.sum()),
                    "invalid_state_count": int(invalid_matrix.sum()),
                    "timeout_count": int(timeout_matrix.sum()),
                    "termination_terms": {
                        name: {
                            "count": int(matrix.sum()),
                            "env_indices": torch.where(torch.any(matrix, dim=0))[0].tolist(),
                        }
                        for name, matrix in termination_term_matrices.items()
                    },
                    "failed_env_indices": failed_env_indices,
                    "terminated_env_indices": terminated_envs.tolist(),
                    "first_termination_precursor_per_joint": precursor_joint_report(
                        failed_precursor,
                        include_onset=True,
                    ),
                    "matched_stable_precursor_per_joint": precursor_joint_report(
                        stable_precursor
                    ),
                    "per_env": [
                        {
                            "env": env_index,
                            "mean_abs_error_m": float(per_env_error[env_index]),
                            "p95_abs_error_m": float(per_env_p95[env_index]),
                            "mean_height_m": float(torch.mean(steady_measured[:, env_index])),
                            "joint_speed_rms_radps": float(
                                torch.sqrt(torch.mean(torch.square(joint_speed[:, env_index])))
                            ),
                            "joint_target_delta_rms_rad": float(
                                torch.sqrt(torch.mean(torch.square(target_delta[:, env_index])))
                            ),
                            "roll_pitch_rate_rms_radps": float(
                                torch.sqrt(torch.mean(torch.square(angular_speed[:, env_index])))
                            ),
                            "torso_tilt_rms_rad": float(
                                torch.sqrt(torch.mean(torch.square(tilt[:, env_index])))
                            ),
                            "maximum_torso_tilt_rad": float(
                                torch.max(tilt[:, env_index])
                            ),
                            "policy_action_delta_rms": float(
                                torch.sqrt(torch.mean(torch.square(policy_action_delta[:, env_index])))
                            ),
                            "policy_action_clip_fraction": float(
                                torch.mean(policy_action_clip[:, env_index])
                            ),
                        }
                        for env_index in range(base_env.num_envs)
                    ],
                }
            )

        actuator_delay_steps = {}
        for actuator_name, actuator in robot.actuators.items():
            delay_buffer = getattr(actuator, "positions_delay_buffer", None)
            if delay_buffer is not None:
                actuator_delay_steps[actuator_name] = (
                    delay_buffer.time_lags.detach().cpu().tolist()
                )

        report = {
            "checkpoint": resume_path,
            "task": args_cli.task,
            "actuator_delay_steps": actuator_delay_steps,
            "initial_root_yaw_rad": initial_root_yaw.detach().cpu().tolist(),
            "initial_root_yaw_override_rad": args_cli.height_transition_initial_yaw_rad,
            "residual_base_maximum_scale": (
                action_term.cfg.height_residual_base_maximum_scale
            ),
            "residual_handoff_scale": (
                action_term.cfg.height_residual_handoff_minimum_scale
            ),
            "leg_damping_multiplier": args_cli.height_transition_leg_damping_multiplier,
            "leg_target_velocity_limit": (
                args_cli.height_transition_leg_target_velocity_limit
            ),
            "bad_orientation_limit_rad": (
                env_cfg.terminations.bad_orientation.params["limit_angle"]
            ),
            "minimum_root_height_m": (
                env_cfg.terminations.base_too_low.params["minimum_height"]
            ),
            "env_spacing_m": env_cfg.scene.env_spacing,
            "posture_depth_scale": args_cli.height_transition_posture_depth_scale,
            "posture_exponent": action_term.cfg.height_posture_exponent,
            "posture_low_exponent": action_term.cfg.height_posture_low_exponent,
            "posture_phase_knots": action_term.cfg.height_posture_phase_knots,
            "deep_handoff_scale": (
                action_term.cfg.height_residual_deep_handoff_minimum_scale
            ),
            "deep_handoff_joint_names": (
                action_term.cfg.height_residual_deep_handoff_joint_names
            ),
            "posture_measured_blend": action_term.cfg.height_posture_measured_blend,
            "residual_measured_blend": action_term.cfg.height_residual_measured_blend,
            "residual_conditioning_maximum": (
                action_term.cfg.height_residual_conditioning_maximum
            ),
            "posture_range": list(action_term.cfg.height_posture_range),
            "height_policy_command_offset_m": args_cli.height_policy_command_offset_m,
            "height_policy_command_minimum_m": args_cli.height_policy_command_minimum_m,
            "zero_sagittal_policy_actions": args_cli.height_transition_zero_sagittal_actions,
            "zero_ankle_roll_policy_actions": (
                args_cli.height_transition_zero_ankle_roll_actions
            ),
            "ankle_roll_policy_action_scale": (
                args_cli.height_transition_ankle_roll_action_scale
            ),
            "ankle_roll_position_scale": (
                args_cli.height_transition_ankle_roll_position_scale
            ),
            "ankle_roll_contract_scale": (
                args_cli.height_transition_ankle_roll_contract_scale
            ),
            "ankle_roll_history_scale": (
                args_cli.height_transition_ankle_roll_history_scale
            ),
            "zero_policy_actions": args_cli.height_transition_zero_policy_actions,
            "lift_force_scale": float(base_env.action_manager.get_term("lift").force_scale),
            "num_envs": int(base_env.num_envs),
            "rate_mps": args_cli.height_transition_rate_mps,
            "hold_s": args_cli.height_transition_hold_s,
            "sequence": sequence,
            "total_terminations": total_terminations,
            "total_invalid_states": total_invalid_states,
            "total_timeouts": total_timeouts,
            "segments": segment_reports,
        }
        output_path = os.path.abspath(args_cli.height_transition_output)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as output_file:
            json.dump(report, output_file, indent=2, sort_keys=True)
            output_file.write("\n")
        print(json.dumps(report, indent=2, sort_keys=True))
        env.close()
        return

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
        maximum_height = (
            args_cli.height_diagnostic_max
            if args_cli.height_diagnostic_max is not None
            else float(height_command.cfg.ranges.height[1])
        )
        minimum_height = (
            args_cli.height_diagnostic_min
            if args_cli.height_diagnostic_min is not None
            else 0.0
        )
        if minimum_height >= maximum_height:
            raise ValueError(
                "height diagnostic minimum must be below maximum: "
                f"{minimum_height} >= {maximum_height}"
            )
        levels = torch.linspace(
            minimum_height,
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
        robot = base_env.scene["robot"]
        action_term = base_env.action_manager.get_term("joint_pos")
        joint_names = list(action_term._joint_names)
        obs = policy_observations()
        error_samples = [[] for _ in levels]
        measured_samples = [[] for _ in levels]
        raw_action_samples = [[] for _ in levels]
        processed_target_samples = [[] for _ in levels]
        joint_position_samples = [[] for _ in levels]
        environment_error_sum = torch.zeros(base_env.num_envs, device=base_env.device)
        environment_success_sum = torch.zeros(base_env.num_envs, device=base_env.device)
        environment_sample_count = torch.zeros(base_env.num_envs, device=base_env.device)
        environment_termination_count = torch.zeros(
            base_env.num_envs, dtype=torch.long, device=base_env.device
        )
        peak_measured = torch.full((base_env.num_envs,), -torch.inf, device=base_env.device)
        termination_counts = torch.zeros(len(levels), dtype=torch.long)
        total_steps = args_cli.height_diagnostic_warmup_steps + args_cli.height_diagnostic_steps
        for step in range(total_steps):
            with torch.inference_mode():
                actions = policy(obs)
                obs, _, dones, _ = env.step(actions)
            force_height_targets()
            if torch.any(dones):
                done_mask = dones.bool()
                done_levels = target_indices[done_mask]
                termination_counts += torch.bincount(done_levels.cpu(), minlength=len(levels))
                environment_termination_count += done_mask.long()
                # Reset environments briefly contain a freshly sampled command in
                # the observation returned by step(). Recompute after restoring
                # the fixed targets; this only happens on termination steps.
                obs = policy_observations()
            if step >= args_cli.height_diagnostic_warmup_steps:
                measured = height_command.measured_height.detach()
                peak_measured = torch.maximum(peak_measured, measured)
                errors = torch.abs(measured - targets)
                valid = height_command.settled
                environment_error_sum += errors * valid
                environment_success_sum += (errors <= 0.08).float() * valid
                environment_sample_count += valid.float()
                for level_index in range(len(levels)):
                    mask = valid & (target_indices == level_index)
                    if torch.any(mask):
                        error_samples[level_index].append(errors[mask].cpu())
                        measured_samples[level_index].append(measured[mask].cpu())
                        raw_action_samples[level_index].append(actions[mask].detach().cpu())
                        processed_target_samples[level_index].append(
                            action_term.processed_actions[mask].detach().cpu()
                        )
                        joint_position_samples[level_index].append(
                            robot.data.joint_pos[mask][:, action_term._joint_ids].detach().cpu()
                        )

        terrain = base_env.scene.terrain
        terrain_levels = getattr(
            terrain,
            "terrain_levels",
            torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device),
        )
        terrain_types = getattr(
            terrain,
            "terrain_types",
            torch.zeros(base_env.num_envs, dtype=torch.long, device=base_env.device),
        )
        bins = []
        for level_index, target in enumerate(levels.cpu()):
            mask = target_indices.cpu() == level_index
            if not error_samples[level_index]:
                raise RuntimeError(f"no settled diagnostic samples for height level {float(target):.6f}")
            level_errors = torch.cat(error_samples[level_index])
            level_measured = torch.cat(measured_samples[level_index])
            level_raw_actions = torch.cat(raw_action_samples[level_index])
            level_processed_targets = torch.cat(processed_target_samples[level_index])
            level_joint_positions = torch.cat(joint_position_samples[level_index])
            level_peak = peak_measured[mask.to(device=base_env.device)]
            level_signed_errors = level_measured - target
            level_environment_count = environment_sample_count[mask.to(device=base_env.device)].clamp(min=1.0)
            level_environment_errors = (
                environment_error_sum[mask.to(device=base_env.device)] / level_environment_count
            )
            level_environment_success = (
                environment_success_sum[mask.to(device=base_env.device)] / level_environment_count
            )
            level_environment_ids = torch.arange(base_env.num_envs)[mask]
            bins.append(
                {
                    "target_height_m": float(target.item()),
                    "environment_count": int(mask.sum().item()),
                    "sample_count": int(level_errors.numel()),
                    "termination_count": int(termination_counts[level_index].item()),
                    "initial_measured_height_m": float(initial_measured[mask].mean().item()),
                    "mean_measured_height_m": float(level_measured.mean().item()),
                    "maximum_measured_height_m": float(level_peak.max().item()),
                    "mean_environment_peak_height_m": float(level_peak.mean().item()),
                    "sample_fraction_within_0_01_m": float(
                        (torch.abs(level_measured - target) <= 0.01).float().mean().item()
                    ),
                    "sample_fraction_within_0_05_m": float(
                        (level_errors <= 0.05).float().mean().item()
                    ),
                    "sample_fraction_within_0_08_m": float(
                        (level_errors <= 0.08).float().mean().item()
                    ),
                    "sample_fraction_undershooting_by_over_0_08_m": float(
                        (level_signed_errors < -0.08).float().mean().item()
                    ),
                    "sample_fraction_overshooting_by_over_0_08_m": float(
                        (level_signed_errors > 0.08).float().mean().item()
                    ),
                    "mean_abs_error_m": float(level_errors.mean().item()),
                    "median_abs_error_m": float(torch.quantile(level_errors, 0.5).item()),
                    "p95_abs_error_m": float(torch.quantile(level_errors, 0.95).item()),
                    "max_abs_error_m": float(level_errors.max().item()),
                    "mean_raw_action_by_joint": dict(
                        zip(joint_names, level_raw_actions.mean(dim=0).tolist(), strict=True)
                    ),
                    "raw_action_clip_fraction_by_joint": dict(
                        zip(
                            joint_names,
                            (torch.abs(level_raw_actions) >= 1.0).float().mean(dim=0).tolist(),
                            strict=True,
                        )
                    ),
                    "mean_processed_target_by_joint_rad": dict(
                        zip(joint_names, level_processed_targets.mean(dim=0).tolist(), strict=True)
                    ),
                    "mean_joint_position_by_joint_rad": dict(
                        zip(joint_names, level_joint_positions.mean(dim=0).tolist(), strict=True)
                    ),
                    "environment_mean_abs_error_m": [
                        float(value) for value in level_environment_errors.cpu()
                    ],
                    "environment_fraction_within_0_08_m": [
                        float(value) for value in level_environment_success.cpu()
                    ],
                    "environment_termination_count": [
                        int(value)
                        for value in environment_termination_count[
                            mask.to(device=base_env.device)
                        ].cpu()
                    ],
                    "environment_ids": [int(value) for value in level_environment_ids],
                    "environment_terrain_level": [
                        int(value) for value in terrain_levels[mask.to(device=base_env.device)].cpu()
                    ],
                    "environment_terrain_type": [
                        int(value) for value in terrain_types[mask.to(device=base_env.device)].cpu()
                    ],
                }
            )
        terrain_generator_cfg = base_env.cfg.scene.terrain.terrain_generator
        if terrain_generator_cfg is None:
            terrain_column_names = [base_env.cfg.scene.terrain.terrain_type]
        else:
            terrain_names = list(terrain_generator_cfg.sub_terrains)
            terrain_proportions = np.array(
                [cfg.proportion for cfg in terrain_generator_cfg.sub_terrains.values()]
            )
            terrain_proportions /= terrain_proportions.sum()
            terrain_column_names = []
            for column in range(terrain_generator_cfg.num_cols):
                terrain_index = np.min(
                    np.where(
                        column / terrain_generator_cfg.num_cols + 0.001
                        < np.cumsum(terrain_proportions)
                    )[0]
                )
                terrain_column_names.append(terrain_names[int(terrain_index)])
        report = {
            "checkpoint": resume_path,
            "task": args_cli.task,
            "lift_force_scale": float(base_env.action_manager.get_term("lift").force_scale),
            "num_envs": int(base_env.num_envs),
            "warmup_steps": args_cli.height_diagnostic_warmup_steps,
            "sample_steps": args_cli.height_diagnostic_steps,
            "terrain_column_names": terrain_column_names,
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

    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, _, _, _ = env.step(actions)
        if state_frames is not None:
            capture_state_frame()
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        elif state_frames is not None:
            timestep += 1
            if timestep == args_cli.state_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

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

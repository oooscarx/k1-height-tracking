# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Train the K1 recovery policy with skrl AMP."""

import argparse
import copy
import hashlib
import sys
import types
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Train a K1 AMP recovery policy.")
parser.add_argument("--video", action="store_true", default=False)
parser.add_argument("--video_length", type=int, default=500)
parser.add_argument("--video_interval", type=int, default=5000)
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", type=str, default="Booster-K1-Fall-Recovery-AMP-v0")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument(
    "--reset_checkpoint_learning_rate",
    action="store_true",
    help="Use the current agent config learning rate after loading a checkpoint",
)
parser.add_argument(
    "--reset_checkpoint_optimizer",
    action="store_true",
    help="Discard loaded optimizer moments while retaining the checkpoint model weights",
)
parser.add_argument(
    "--reset_checkpoint_value",
    action="store_true",
    help=(
        "Reinitialize the value model and return scaler while preserving the critic "
        "input normalization statistics"
    ),
)
parser.add_argument(
    "--reset_checkpoint_discriminator",
    action="store_true",
    help="Reinitialize the AMP discriminator and its observation scaler",
)
parser.add_argument("--agent_learning_rate", type=float, default=None)
parser.add_argument(
    "--disable_mixed_precision",
    action="store_true",
    help="Run AMP updates in float32 for numerical stability",
)
parser.add_argument("--episode_length_s", type=float, default=None)
parser.add_argument("--task_reward_scale", type=float, default=None)
parser.add_argument("--style_reward_scale", type=float, default=None)
parser.add_argument(
    "--maximum_amp_style_fraction",
    type=float,
    default=None,
    help="Bound the weighted AMP style fraction per rollout update",
)
parser.add_argument(
    "--minimum_amp_style_reward_scale",
    type=float,
    default=0.0,
    help="Minimum style scale retained when the AMP fraction cap is active",
)
parser.add_argument("--max_iterations", type=int, default=None, help="Number of AMP rollout/update iterations")
parser.add_argument("--experiment_name", type=str, default=None, help="Suffix for the timestamped run directory")
parser.add_argument(
    "--bc_iterations",
    type=int,
    default=0,
    help="Teacher-forced behavior-cloning updates before AMP training",
)
parser.add_argument("--bc_learning_rate", type=float, default=3.0e-4)
parser.add_argument(
    "--bc_policy_anchor_weight",
    type=float,
    default=0.0,
    help="Weight an installed pre-BC policy anchor in the behavior-cloning loss",
)
parser.add_argument(
    "--bc_physical_teacher_target_blend",
    type=float,
    default=0.0,
    help="Blend recorded physical teacher targets over next-frame position targets",
)
parser.add_argument("--bc_teacher", choices=("motion", "native"), default="motion")
parser.add_argument(
    "--bc_reset_mode",
    choices=("reference", "faceup", "failure"),
    default=None,
    help="Temporarily force this reset distribution while collecting BC samples",
)
parser.add_argument(
    "--bc_only",
    action="store_true",
    help="Save the BC checkpoint and exit before AMP/PPO training",
)
parser.add_argument(
    "--bc_student_rollout_fraction",
    type=float,
    default=0.0,
    help="Final fraction of student action used to collect native-teacher BC states",
)
parser.add_argument("--bc_canonical_joint_noise", type=float, default=None)
parser.add_argument("--bc_canonical_root_xy_noise", type=float, default=None)
parser.add_argument("--bc_canonical_height_noise", type=float, default=None)
parser.add_argument("--bc_canonical_orientation_noise", type=float, default=None)
parser.add_argument(
    "--policy_anchor_samples",
    type=int,
    default=0,
    help="Number of clean reference observations used to preserve the loaded/BC policy during AMP",
)
parser.add_argument("--policy_anchor_batch_size", type=int, default=4096)
parser.add_argument("--policy_anchor_steps", type=int, default=1)
parser.add_argument("--policy_anchor_learning_rate", type=float, default=1.0e-5)
parser.add_argument("--policy_anchor_canonical_fraction", type=float, default=0.0)
parser.add_argument("--policy_anchor_canonical_noise_scale", type=float, default=0.6)
parser.add_argument("--policy_anchor_reference_phase_min", type=float, default=None)
parser.add_argument("--policy_anchor_reference_phase_max", type=float, default=None)
parser.add_argument(
    "--policy_anchor_before_bc",
    action="store_true",
    help="Capture the loaded policy before behavior cloning and reuse it during AMP",
)
parser.add_argument("--failure_reset_probability_start", type=float, default=None)
parser.add_argument("--failure_reset_probability_end", type=float, default=None)
parser.add_argument("--failure_state_blend_start", type=float, default=None)
parser.add_argument("--failure_state_blend_end", type=float, default=None)
parser.add_argument("--canonical_noise_scale_start", type=float, default=None)
parser.add_argument("--canonical_noise_scale_end", type=float, default=None)
parser.add_argument("--robust_reset_curriculum_steps", type=int, default=None)
parser.add_argument("--joint_state_tolerance", type=float, default=None)
parser.add_argument("--parallel_state_tolerance", type=float, default=None)
parser.add_argument("--joint_state_tolerance_start", type=float, default=None)
parser.add_argument("--joint_state_tolerance_end", type=float, default=None)
parser.add_argument("--parallel_state_tolerance_start", type=float, default=None)
parser.add_argument("--parallel_state_tolerance_end", type=float, default=None)
parser.add_argument("--state_tolerance_curriculum_steps", type=int, default=None)
parser.add_argument("--early_success_weight", type=float, default=None)
parser.add_argument("--stability_hold_progress_weight", type=float, default=None)
parser.add_argument("--standing_pose_max_error_hold_weight", type=float, default=None)
parser.add_argument("--handoff_curriculum_initial_progress", type=float, default=None)
parser.add_argument(
    "--fixed_handoff_curriculum_progress",
    type=float,
    default=None,
    help="Disable adaptive handoff gates and train at one fixed curriculum progress",
)
parser.add_argument("--policy_min_log_std", type=float, default=None)
parser.add_argument("--policy_max_log_std", type=float, default=None)
parser.add_argument("--adaptive_failure_curriculum", action="store_true")
parser.add_argument("--adaptive_failure_blend_start", type=float, default=None)
parser.add_argument("--adaptive_failure_blend_end", type=float, default=None)
parser.add_argument("--adaptive_failure_promotion", type=float, default=None)
parser.add_argument("--adaptive_failure_demotion", type=float, default=None)
parser.add_argument("--adaptive_failure_success_threshold", type=float, default=None)
parser.add_argument("--adaptive_failure_demotion_threshold", type=float, default=None)
parser.add_argument("--adaptive_failure_min_trials", type=int, default=None)
parser.add_argument("--adaptive_failure_promotion_windows", type=int, default=None)
parser.add_argument("--reference_reset_probability", type=float, default=None)
parser.add_argument("--reference_reset_probability_start", type=float, default=None)
parser.add_argument("--reference_reset_probability_end", type=float, default=None)
parser.add_argument("--standing_reset_probability_start", type=float, default=None)
parser.add_argument("--standing_reset_probability_end", type=float, default=None)
parser.add_argument("--random_fall_probability_start", type=float, default=None)
parser.add_argument("--random_fall_probability_end", type=float, default=None)
parser.add_argument("--reset_distribution_curriculum_steps", type=int, default=None)
parser.add_argument("--random_fall_difficulty_start", type=float, default=None)
parser.add_argument("--random_fall_difficulty_end", type=float, default=None)
parser.add_argument(
    "--domain_randomization_scale",
    type=float,
    default=0.0,
    help="Scale rigid-body material, mass, trunk COM, and actuator-gain randomization",
)
parser.add_argument(
    "--push_randomization_scale",
    type=float,
    default=0.0,
    help="Scale intermittent root velocity pushes; zero disables them",
)
parser.add_argument(
    "--observation_noise_scale",
    type=float,
    default=0.0,
    help="Scale policy IMU and joint-state observation corruption",
)
parser.add_argument("--push_interval_min_s", type=float, default=0.8)
parser.add_argument("--push_interval_max_s", type=float, default=2.0)
parser.add_argument("--failure_state_file", type=str, default=None)
parser.add_argument("--reference_clip_indices", type=int, nargs="+", default=None)
parser.add_argument("--reference_clip_weights", type=float, nargs="+", default=None)
parser.add_argument("--reference_phase_min", type=float, default=None)
parser.add_argument("--reference_phase_max", type=float, default=None)
parser.add_argument("--reference_phase_bin_edges", type=float, nargs="+", default=None)
parser.add_argument("--reference_phase_bin_weights", type=float, nargs="+", default=None)
parser.add_argument("--adaptive_failure_state_resume", type=str, default=None)
parser.add_argument("--adaptive_failure_reset_statistics", action="store_true")
parser.add_argument("--adaptive_failure_state_interval", type=int, default=None)
parser.add_argument(
    "--checkpoint_interval",
    type=int,
    default=None,
    help="Checkpoint interval in AMP rollout/update iterations",
)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
if args_cli.video:
    args_cli.enable_cameras = True
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os
import random
from datetime import datetime

import booster_train.tasks  # noqa: F401
import gymnasium as gym
import isaaclab_tasks  # noqa: F401
import skrl
import torch
from amp_reward_diagnostics import install_amp_reward_diagnostics
from booster_train.tasks.manager_based.fall_recovery.robots.k1.env_cfg import (
    configure_training_randomization,
)
from checkpoint_loading import (
    clamp_gaussian_log_std_parameter,
    restore_amp_checkpoint_components,
)
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml
from isaaclab_rl.skrl import SkrlVecEnvWrapper
from isaaclab_tasks.utils.hydra import hydra_task_config
from packaging import version
from rollout_numeric_guard import (
    configure_normalized_policy_action_bounds,
    finite_batch_rows,
    sanitize_batch_rows,
    tensors_are_finite,
)
from skrl.utils.runner.torch import Runner

if version.parse(skrl.__version__) < version.parse("2.1.0"):
    raise RuntimeError(f"skrl >= 2.1.0 is required, found {skrl.__version__}")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _model_inputs(observations):
    if isinstance(observations, dict):
        return {
            "observations": observations["policy"],
            "states": observations.get("critic"),
        }
    return {"observations": observations, "states": None}


def pretrain_from_teacher(
    raw_env,
    runner: Runner,
    iterations: int,
    learning_rate: float,
    teacher_mode: str,
    student_rollout_fraction: float,
    reset_mode: str | None,
    policy_anchor_weight: float,
    physical_teacher_target_blend: float,
) -> None:
    if iterations <= 0:
        return
    if learning_rate <= 0.0:
        raise ValueError("--bc_learning_rate must be positive")
    if not 0.0 <= student_rollout_fraction <= 1.0:
        raise ValueError("--bc_student_rollout_fraction must be in [0, 1]")
    if policy_anchor_weight < 0.0:
        raise ValueError("--bc_policy_anchor_weight must be non-negative")
    if not 0.0 <= physical_teacher_target_blend <= 1.0:
        raise ValueError("--bc_physical_teacher_target_blend must be in [0, 1]")
    if teacher_mode != "native" and student_rollout_fraction > 0.0:
        raise ValueError("--bc_student_rollout_fraction requires --bc_teacher native")

    task_env = raw_env.unwrapped
    motion_teacher_action = task_env._motion_loader.teacher_action
    generated_motion_teacher = motion_teacher_action is None
    native_teacher = None
    action_term = None
    if teacher_mode == "motion":
        if generated_motion_teacher:
            action_term = task_env.action_manager.get_term("joint_pos")
            print(
                "[INFO] AMP motion pool has no consistent teacher_normalized_action; "
                "deriving all next-frame targets from the configured action center and scale"
            )
    elif teacher_mode == "native":
        from booster_train.tasks.manager_based.fall_recovery.native_teacher import K1NativeRecoveryTeacher

        native_teacher = K1NativeRecoveryTeacher("faceup", task_env.num_envs, task_env.device)
        action_term = task_env.action_manager.get_term("joint_pos")
    else:
        raise ValueError(f"unsupported BC teacher mode: {teacher_mode}")

    policy = runner.agent.models["policy"]
    anchor_observations = getattr(runner.agent, "_policy_anchor_observations", None)
    anchor_actions = getattr(runner.agent, "_policy_anchor_actions", None)
    if policy_anchor_weight > 0.0 and (
        anchor_observations is None or anchor_actions is None
    ):
        raise ValueError(
            "--bc_policy_anchor_weight requires --policy_anchor_before_bc "
            "and positive --policy_anchor_samples"
        )
    anchor_batch_size = min(
        int(getattr(runner.agent, "_policy_anchor_batch_size", 0)),
        0 if anchor_observations is None else anchor_observations.shape[0],
    )
    policy.train()
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    original_reset_mode = task_env.cfg.amp_reset_mode
    try:
        if reset_mode is not None:
            task_env.cfg.amp_reset_mode = reset_mode
        observations, _ = raw_env.reset()
        model_inputs = _model_inputs(observations)
        print(
            f"[INFO] Behavior cloning for {iterations} iterations "
            f"({task_env.num_envs} samples/iteration, lr={learning_rate:g}, teacher={teacher_mode}, "
            f"reset_mode={task_env.cfg.amp_reset_mode}, "
            f"final_student_rollout_fraction={student_rollout_fraction:g}, "
            f"physical_teacher_target_blend={physical_teacher_target_blend:g})",
            flush=True,
        )
        for iteration in range(1, iterations + 1):
            if native_teacher is None:
                if generated_motion_teacher:
                    target_indexes = task_env._motion_loader.advance_indexes(
                        task_env._reference_frame_index
                    )
                    target_joint_position = task_env._motion_loader.joint_position[
                        target_indexes
                    ].clone()
                    teacher_target_mask = task_env._motion_loader.teacher_target_mask[
                        task_env._reference_frame_index
                    ]
                    if physical_teacher_target_blend > 0.0 and torch.any(
                        teacher_target_mask
                    ):
                        physical_target = (
                            task_env._motion_loader.teacher_target[
                                task_env._reference_frame_index[teacher_target_mask]
                            ]
                        )
                        target_joint_position[teacher_target_mask] = torch.lerp(
                            target_joint_position[teacher_target_mask],
                            physical_target,
                            physical_teacher_target_blend,
                        )
                    targets = torch.clamp(
                        (target_joint_position - action_term._center) / action_term._scale,
                        -1.0,
                        1.0,
                    )
                else:
                    targets = motion_teacher_action[task_env._reference_frame_index]
            else:
                robot = task_env.robot
                with torch.no_grad():
                    teacher_step = native_teacher.step(
                        robot.data.projected_gravity_b,
                        robot.data.root_ang_vel_b,
                        robot.data.joint_pos[:, task_env._joint_ids],
                        robot.data.joint_vel[:, task_env._joint_ids],
                    )
                    targets = torch.clamp(
                        (teacher_step.target - action_term._center) / action_term._scale,
                        -1.0,
                        1.0,
                    )
            policy_inputs = dict(model_inputs)
            policy_inputs["observations"] = runner.agent._observation_preprocessor(
                model_inputs["observations"]
            )
            _, outputs = policy.act(policy_inputs, role="policy")
            predictions = outputs["mean_actions"]
            teacher_loss = torch.mean(torch.square(predictions - targets))
            anchor_loss = torch.zeros((), device=teacher_loss.device)
            if policy_anchor_weight > 0.0:
                indexes = torch.randint(
                    anchor_observations.shape[0],
                    (anchor_batch_size,),
                    device=anchor_observations.device,
                )
                processed_anchor = runner.agent._observation_preprocessor(
                    anchor_observations[indexes]
                )
                _, anchor_outputs = policy.act(
                    {"observations": processed_anchor}, role="policy"
                )
                anchor_loss = torch.mean(
                    torch.square(anchor_outputs["mean_actions"] - anchor_actions[indexes])
                )
            loss = teacher_loss + policy_anchor_weight * anchor_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            optimizer.step()

            with torch.no_grad():
                rollout_fraction = student_rollout_fraction * iteration / iterations
                student_actions = torch.clamp(predictions.detach(), -1.0, 1.0)
                rollout_actions = (1.0 - rollout_fraction) * targets + rollout_fraction * student_actions
                observations, _, terminated, truncated, _ = raw_env.step(rollout_actions)
                model_inputs = _model_inputs(observations)
                if native_teacher is not None:
                    done = (terminated | truncated).reshape(-1)
                    if torch.any(done):
                        native_teacher.reset(torch.nonzero(done, as_tuple=False).squeeze(-1))
            if iteration == 1 or iteration % 100 == 0 or iteration == iterations:
                maximum_error = torch.amax(torch.abs(predictions - targets))
                print(
                    f"[BC] iteration={iteration} loss={loss.item():.6f} "
                    f"teacher_loss={teacher_loss.item():.6f} "
                    f"anchor_loss={anchor_loss.item():.6f} "
                    f"max_abs_error={maximum_error.item():.6f}",
                    flush=True,
                )
    finally:
        task_env.cfg.amp_reset_mode = original_reset_mode
        raw_env.reset()
        policy.eval()


def install_policy_anchor(
    raw_env,
    runner: Runner,
    sample_count: int,
    batch_size: int,
    steps: int,
    learning_rate: float,
    canonical_fraction: float,
    canonical_noise_scale: float,
    reference_phase_minimum: float | None,
    reference_phase_maximum: float | None,
) -> None:
    if sample_count <= 0:
        return
    if batch_size <= 0:
        raise ValueError("--policy_anchor_batch_size must be positive")
    if steps <= 0:
        raise ValueError("--policy_anchor_steps must be positive")
    if learning_rate <= 0.0:
        raise ValueError("--policy_anchor_learning_rate must be positive")
    if not 0.0 <= canonical_fraction <= 1.0:
        raise ValueError("--policy_anchor_canonical_fraction must be in [0, 1]")
    if not 0.0 <= canonical_noise_scale <= 1.0:
        raise ValueError("--policy_anchor_canonical_noise_scale must be in [0, 1]")

    task_env = raw_env.unwrapped
    if not hasattr(task_env.cfg, "amp_reset_mode"):
        raise RuntimeError("policy anchoring requires an AMP environment")
    policy = runner.agent.models["policy"]
    original_reset_mode = task_env.cfg.amp_reset_mode
    original_ankle_neutral_fraction = task_env.cfg.amp_reset_ankle_neutral_fraction
    original_reference_phase_minimum = task_env.cfg.amp_reference_phase_minimum
    original_reference_phase_maximum = task_env.cfg.amp_reference_phase_maximum
    original_reference_phase_bin_edges = task_env.cfg.amp_reference_phase_bin_edges
    original_reference_phase_bin_weights = task_env.cfg.amp_reference_phase_bin_weights
    override_reference_phase = (
        reference_phase_minimum is not None or reference_phase_maximum is not None
    )
    anchor_reference_phase_minimum = (
        original_reference_phase_minimum
        if reference_phase_minimum is None
        else reference_phase_minimum
    )
    anchor_reference_phase_maximum = (
        original_reference_phase_maximum
        if reference_phase_maximum is None
        else reference_phase_maximum
    )
    if not 0.0 <= anchor_reference_phase_minimum <= anchor_reference_phase_maximum <= 1.0:
        raise ValueError("policy anchor reference phases must satisfy 0 <= min <= max <= 1")
    canonical_attributes = (
        "amp_canonical_joint_noise",
        "amp_canonical_ankle_joint_noise",
        "amp_canonical_root_xy_noise",
        "amp_canonical_height_noise",
        "amp_canonical_orientation_noise",
    )
    original_canonical_noise = {
        attribute: getattr(task_env.cfg, attribute) for attribute in canonical_attributes
    }
    anchor_observations = []
    anchor_actions = []

    def collect(mode: str, count: int) -> None:
        if count <= 0:
            return
        task_env.cfg.amp_reset_mode = mode
        collected = 0
        while collected < count:
            observations, _ = raw_env.reset()
            with torch.no_grad():
                policy_observations = _model_inputs(observations)["observations"]
                processed = runner.agent._observation_preprocessor(policy_observations)
                _, outputs = policy.act({"observations": processed}, role="policy")
                take = min(count - collected, policy_observations.shape[0])
                anchor_observations.append(policy_observations[:take].detach().clone())
                anchor_actions.append(outputs["mean_actions"][:take].detach().clone())
                collected += take

    try:
        task_env.cfg.amp_reset_ankle_neutral_fraction = 0.0
        task_env.cfg.amp_reference_phase_minimum = anchor_reference_phase_minimum
        task_env.cfg.amp_reference_phase_maximum = anchor_reference_phase_maximum
        if override_reference_phase:
            task_env.cfg.amp_reference_phase_bin_edges = []
            task_env.cfg.amp_reference_phase_bin_weights = []
        policy.eval()
        canonical_count = round(sample_count * canonical_fraction)
        reference_count = sample_count - canonical_count
        collect("reference", reference_count)
        for attribute, value in original_canonical_noise.items():
            setattr(task_env.cfg, attribute, value * canonical_noise_scale)
        collect("faceup", canonical_count)
    finally:
        task_env.cfg.amp_reset_mode = original_reset_mode
        task_env.cfg.amp_reset_ankle_neutral_fraction = original_ankle_neutral_fraction
        task_env.cfg.amp_reference_phase_minimum = original_reference_phase_minimum
        task_env.cfg.amp_reference_phase_maximum = original_reference_phase_maximum
        task_env.cfg.amp_reference_phase_bin_edges = original_reference_phase_bin_edges
        task_env.cfg.amp_reference_phase_bin_weights = original_reference_phase_bin_weights
        for attribute, value in original_canonical_noise.items():
            setattr(task_env.cfg, attribute, value)
        raw_env.reset()

    observations = torch.cat(anchor_observations, dim=0)[:sample_count]
    actions = torch.cat(anchor_actions, dim=0)[:sample_count]
    optimizer = torch.optim.Adam(policy.parameters(), lr=learning_rate)
    original_update = runner.agent.update

    def anchored_update(self, *, timestep: int, timesteps: int) -> None:
        original_update(timestep=timestep, timesteps=timesteps)
        cumulative_loss = 0.0
        for _ in range(steps):
            indexes = torch.randint(observations.shape[0], (batch_size,), device=observations.device)
            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                processed = self._observation_preprocessor(observations[indexes])
                _, outputs = self.policy.act({"observations": processed}, role="policy")
                loss = torch.mean(torch.square(outputs["mean_actions"] - actions[indexes]))
            optimizer.zero_grad(set_to_none=True)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.grad_norm_clip)
            self.scaler.step(optimizer)
            self.scaler.update()
            cumulative_loss += loss.item()
        self.track_data("Loss / Policy anchor loss", cumulative_loss / steps)

    runner.agent.update = types.MethodType(anchored_update, runner.agent)
    runner.agent._policy_anchor_optimizer = optimizer
    runner.agent._policy_anchor_observations = observations
    runner.agent._policy_anchor_actions = actions
    runner.agent._policy_anchor_batch_size = batch_size
    print(
        f"[INFO] Installed policy anchor: samples={sample_count}, batch_size={batch_size}, "
        f"steps={steps}, lr={learning_rate:g}, canonical_fraction={canonical_fraction:g}, "
        f"canonical_noise_scale={canonical_noise_scale:g}, "
        f"reference_phase=[{anchor_reference_phase_minimum:g}, "
        f"{anchor_reference_phase_maximum:g}]"
    )


def ensure_amp_reply_buffer_capacity(agent_cfg: dict, num_envs: int) -> None:
    reply_buffer_cfg = agent_cfg.get("reply_buffer")
    if reply_buffer_cfg is None:
        return

    rollouts = int(agent_cfg["agent"]["rollouts"])
    rollout_samples = int(num_envs) * rollouts
    minimum_capacity = 2 * rollout_samples
    configured_capacity = int(reply_buffer_cfg["memory_size"])
    if configured_capacity < minimum_capacity:
        reply_buffer_cfg["memory_size"] = minimum_capacity
        print(
            "[INFO] Increased AMP reply buffer capacity "
            f"from {configured_capacity} to {minimum_capacity} "
            f"for {num_envs} envs x {rollouts} rollout steps"
        )


def install_nonfinite_rollout_isolation(agent, raw_env) -> None:
    """Keep a bad vectorized environment out of policy scalers and rollout memory."""
    if getattr(agent, "_nonfinite_rollout_isolation_installed", False):
        return
    task_env = raw_env.unwrapped
    action_term = task_env.action_manager.get_term("joint_pos")
    original_act = agent.act
    original_record_transition = agent.record_transition

    def act(self, observations, states, *, timestep: int, timesteps: int):
        finite_input = finite_batch_rows(
            observations,
            task_env.num_envs,
            task_env.device,
            maximum_absolute_value=1.0e6,
        )
        finite_input &= finite_batch_rows(
            states,
            task_env.num_envs,
            task_env.device,
            maximum_absolute_value=1.0e6,
        )
        actions, outputs = original_act(
            sanitize_batch_rows(observations, ~finite_input),
            sanitize_batch_rows(states, ~finite_input),
            timestep=timestep,
            timesteps=timesteps,
        )
        finite_output = finite_batch_rows(actions, task_env.num_envs, task_env.device)
        if self._current_log_prob is not None:
            finite_output &= finite_batch_rows(
                self._current_log_prob,
                task_env.num_envs,
                task_env.device,
            )
        if self._current_values is not None:
            finite_output &= finite_batch_rows(
                self._current_values,
                task_env.num_envs,
                task_env.device,
            )
        failed = ~(finite_input & finite_output)
        action_term.mark_nonfinite_policy_input(failed)
        self._current_nonfinite_policy_rows = failed
        actions = torch.clamp(sanitize_batch_rows(actions, failed), -1.0, 1.0)
        if self._current_log_prob is not None:
            self._current_log_prob = sanitize_batch_rows(
                self._current_log_prob,
                failed,
            )
        if self._current_values is not None:
            self._current_values = sanitize_batch_rows(self._current_values, failed)
        return actions, outputs

    def record_transition(self, **kwargs) -> None:
        failed = self._current_nonfinite_policy_rows.clone()
        for name in (
            "observations",
            "states",
            "actions",
            "rewards",
            "next_observations",
            "next_states",
        ):
            failed |= ~finite_batch_rows(
                kwargs[name],
                task_env.num_envs,
                task_env.device,
                maximum_absolute_value=1.0e6,
            )
        infos = dict(kwargs["infos"])
        if "amp_obs" in infos:
            failed |= ~finite_batch_rows(
                infos["amp_obs"],
                task_env.num_envs,
                task_env.device,
                maximum_absolute_value=1.0e6,
            )
        action_term.mark_nonfinite_policy_input(failed)
        for name in (
            "observations",
            "states",
            "actions",
            "rewards",
            "next_observations",
            "next_states",
        ):
            kwargs[name] = sanitize_batch_rows(kwargs[name], failed)
        if "amp_obs" in infos:
            infos["amp_obs"] = sanitize_batch_rows(infos["amp_obs"], failed)
        kwargs["infos"] = infos
        termination_mask = failed.reshape(
            (failed.numel(),) + (1,) * (kwargs["terminated"].ndim - 1)
        )
        kwargs["terminated"] = kwargs["terminated"] | termination_mask
        kwargs["truncated"] = kwargs["truncated"] & ~termination_mask
        if self._current_log_prob is not None:
            self._current_log_prob = sanitize_batch_rows(self._current_log_prob, failed)
        if self._current_values is not None:
            self._current_values = sanitize_batch_rows(self._current_values, failed)
        original_record_transition(**kwargs)

    agent.act = types.MethodType(act, agent)
    agent.record_transition = types.MethodType(record_transition, agent)
    agent._current_nonfinite_policy_rows = torch.zeros(
        task_env.num_envs,
        dtype=torch.bool,
        device=task_env.device,
    )
    agent._nonfinite_rollout_isolation_installed = True


def install_finite_update_guard(agent) -> None:
    """Rollback an AMP update if it contaminates a model or running scaler."""
    if getattr(agent, "_finite_update_guard_installed", False):
        return
    original_update = agent.update
    guarded_modules = {
        name: module
        for name, module in agent.checkpoint_modules.items()
        if name != "optimizer" and hasattr(module, "state_dict")
    }

    def update(self, *, timestep: int, timesteps: int) -> None:
        memory_names = (
            "observations",
            "states",
            "actions",
            "rewards",
            "values",
            "amp_observations",
        )
        invalid_memory = [
            name
            for name in memory_names
            if not tensors_are_finite(self.memory.get_tensor_by_name(name))
        ]
        if invalid_memory:
            raise RuntimeError(
                "non-finite AMP rollout tensors reached the update: "
                + ", ".join(invalid_memory)
            )
        invalid_before = [
            name
            for name, module in guarded_modules.items()
            if not tensors_are_finite(module.state_dict())
        ]
        if invalid_before:
            raise RuntimeError(
                "non-finite AMP state before update: " + ", ".join(invalid_before)
            )
        snapshots = {
            name: copy.deepcopy(module.state_dict())
            for name, module in guarded_modules.items()
        }
        original_update(timestep=timestep, timesteps=timesteps)
        invalid_after = [
            name
            for name, module in guarded_modules.items()
            if not tensors_are_finite(module.state_dict())
        ]
        if not invalid_after:
            self.track_data("Numeric / Rolled back update", 0.0)
            return
        for name, module in guarded_modules.items():
            module.load_state_dict(snapshots[name])
        self.optimizer.state.clear()
        if version.parse(torch.__version__) >= version.parse("2.4"):
            self.scaler = torch.amp.GradScaler(
                device=self._device_type,
                enabled=self.cfg.mixed_precision,
            )
        else:
            self.scaler = torch.cuda.amp.GradScaler(
                enabled=self.cfg.mixed_precision
            )
        self.track_data("Numeric / Rolled back update", 1.0)
        print(
            "[WARNING] Rolled back non-finite AMP update affecting: "
            + ", ".join(invalid_after)
        )

    agent.update = types.MethodType(update, agent)
    agent._finite_update_guard_installed = True


def load_amp_checkpoint(
    runner,
    checkpoint: str,
    *,
    reset_optimizer: bool,
    reset_value: bool,
    reset_discriminator: bool,
) -> None:
    """Load compatible AMP components while allowing architecture-specific resets."""
    if (reset_value or reset_discriminator) and not reset_optimizer:
        raise ValueError(
            "Resetting the value model or discriminator also requires "
            "--reset_checkpoint_optimizer"
        )
    if not reset_value and not reset_discriminator:
        runner.agent.load(checkpoint)
        if reset_optimizer:
            runner.agent.optimizer.state.clear()
            print("[INFO] Reset loaded optimizer state")
    else:
        data = torch.load(
            checkpoint,
            map_location=runner.agent.device,
            weights_only=False,
        )
        loaded = restore_amp_checkpoint_components(
            runner.agent,
            data,
            reset_value=reset_value,
            reset_discriminator=reset_discriminator,
        )
        if reset_value:
            print("[INFO] Kept initialized value model and return preprocessor")
        if reset_discriminator:
            print("[INFO] Kept initialized discriminator and AMP observation preprocessor")
        print(f"[INFO] Selectively loaded checkpoint components: {', '.join(loaded)}")
        print("[INFO] Kept initialized optimizer state")

    log_std = clamp_gaussian_log_std_parameter(runner.agent.models["policy"])
    if log_std is not None and (
        log_std["before_minimum"] != log_std["after_minimum"]
        or log_std["before_maximum"] != log_std["after_maximum"]
    ):
        print(
            "[INFO] Projected loaded policy log-std into configured bounds: "
            f"[{log_std['before_minimum']:g}, {log_std['before_maximum']:g}] -> "
            f"[{log_std['after_minimum']:g}, {log_std['after_maximum']:g}]"
        )


@hydra_task_config(args_cli.task, "skrl_amp_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: dict) -> None:
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    configure_training_randomization(
        env_cfg,
        domain_scale=args_cli.domain_randomization_scale,
        push_scale=args_cli.push_randomization_scale,
        observation_noise_scale=args_cli.observation_noise_scale,
        push_interval_range_s=(
            args_cli.push_interval_min_s,
            args_cli.push_interval_max_s,
        ),
    )
    if args_cli.max_iterations is not None:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    if args_cli.bc_only:
        # BC uses the policy directly; PPO and discriminator buffers are wasted work.
        agent_cfg["agent"]["rollouts"] = 1
        agent_cfg["trainer"]["timesteps"] = 0
        agent_cfg.pop("motion_dataset", None)
        agent_cfg.pop("reply_buffer", None)
    else:
        ensure_amp_reply_buffer_capacity(agent_cfg, env_cfg.scene.num_envs)
    if args_cli.agent_learning_rate is not None:
        if args_cli.agent_learning_rate <= 0.0:
            raise ValueError("--agent_learning_rate must be positive")
        agent_cfg["agent"]["learning_rate"] = args_cli.agent_learning_rate
    if args_cli.disable_mixed_precision:
        agent_cfg["agent"]["mixed_precision"] = False
    if args_cli.episode_length_s is not None:
        if args_cli.episode_length_s <= 0.0:
            raise ValueError("--episode_length_s must be positive")
        env_cfg.episode_length_s = args_cli.episode_length_s
    for argument in ("task_reward_scale", "style_reward_scale"):
        value = getattr(args_cli, argument)
        if value is not None:
            if value < 0.0:
                raise ValueError(f"--{argument} must be non-negative")
            agent_cfg["agent"][argument] = value
    policy_cfg = agent_cfg["models"]["policy"]
    if args_cli.policy_min_log_std is not None:
        if args_cli.policy_min_log_std > 0.0:
            raise ValueError("--policy_min_log_std must be non-positive")
        policy_cfg["min_log_std"] = args_cli.policy_min_log_std
    if args_cli.policy_max_log_std is not None:
        if args_cli.policy_max_log_std > 0.0:
            raise ValueError("--policy_max_log_std must be non-positive")
        if args_cli.policy_max_log_std < float(policy_cfg["min_log_std"]):
            raise ValueError("--policy_max_log_std must not be below the configured minimum")
        policy_cfg["max_log_std"] = args_cli.policy_max_log_std
    if float(policy_cfg["min_log_std"]) > float(policy_cfg["max_log_std"]):
        raise ValueError("configured policy min_log_std must not exceed max_log_std")
    if args_cli.checkpoint_interval is not None:
        if args_cli.checkpoint_interval < 1:
            raise ValueError("--checkpoint_interval must be positive")
        agent_cfg["agent"]["experiment"]["checkpoint_interval"] = (
            args_cli.checkpoint_interval * agent_cfg["agent"]["rollouts"]
        )
    for argument, attribute in (
        ("failure_reset_probability_start", "amp_failure_reset_probability_start"),
        ("failure_reset_probability_end", "amp_failure_reset_probability_end"),
        ("failure_state_blend_start", "amp_failure_state_blend_start"),
        ("failure_state_blend_end", "amp_failure_state_blend_end"),
        ("canonical_noise_scale_start", "amp_canonical_noise_scale_start"),
        ("canonical_noise_scale_end", "amp_canonical_noise_scale_end"),
    ):
        value = getattr(args_cli, argument)
        if value is not None:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"--{argument} must be in [0, 1]")
            setattr(env_cfg, attribute, value)
    if args_cli.reference_reset_probability is not None:
        if not 0.0 <= args_cli.reference_reset_probability <= 1.0:
            raise ValueError("--reference_reset_probability must be in [0, 1]")
        env_cfg.amp_reference_reset_probability_start = args_cli.reference_reset_probability
        env_cfg.amp_reference_reset_probability_end = args_cli.reference_reset_probability
    for argument, attribute in (
        ("reference_reset_probability_start", "amp_reference_reset_probability_start"),
        ("reference_reset_probability_end", "amp_reference_reset_probability_end"),
        ("standing_reset_probability_start", "amp_standing_reset_probability_start"),
        ("standing_reset_probability_end", "amp_standing_reset_probability_end"),
        ("random_fall_probability_start", "amp_random_fall_probability_start"),
        ("random_fall_probability_end", "amp_random_fall_probability_end"),
    ):
        value = getattr(args_cli, argument)
        if value is not None:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"--{argument} must be in [0, 1]")
            setattr(env_cfg, attribute, value)
    if args_cli.reset_distribution_curriculum_steps is not None:
        if args_cli.reset_distribution_curriculum_steps < 1:
            raise ValueError("--reset_distribution_curriculum_steps must be positive")
        env_cfg.amp_reference_reset_curriculum_steps = args_cli.reset_distribution_curriculum_steps
    for argument, attribute in (
        ("random_fall_difficulty_start", "amp_random_fall_difficulty_start"),
        ("random_fall_difficulty_end", "amp_random_fall_difficulty_end"),
    ):
        value = getattr(args_cli, argument)
        if value is not None:
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"--{argument} must be in [0, 1]")
            setattr(env_cfg, attribute, value)
    if (
        env_cfg.amp_random_fall_difficulty_start
        > env_cfg.amp_random_fall_difficulty_end
    ):
        raise ValueError("random fall difficulty start must not exceed end")
    if args_cli.failure_state_file is not None:
        failure_state_path = Path(args_cli.failure_state_file).expanduser().resolve()
        if not failure_state_path.is_file():
            raise FileNotFoundError(f"failure-state dataset does not exist: {failure_state_path}")
        env_cfg.amp_failure_state_file = str(failure_state_path)
        env_cfg.amp_failure_state_sha256 = hashlib.sha256(
            failure_state_path.read_bytes()
        ).hexdigest()
    if args_cli.reference_clip_indices is not None:
        env_cfg.amp_reference_clip_indices = args_cli.reference_clip_indices
    if args_cli.reference_clip_weights is not None:
        env_cfg.amp_reference_clip_weights = args_cli.reference_clip_weights
    for suffix in ("start", "end"):
        total_probability = sum(
            float(getattr(env_cfg, f"{attribute}_{suffix}"))
            for attribute in (
                "amp_reference_reset_probability",
                "amp_standing_reset_probability",
                "amp_random_fall_probability",
                "amp_failure_reset_probability",
            )
        )
        if total_probability > 1.0 + 1.0e-6:
            raise ValueError(
                f"AMP reset probabilities at {suffix} sum to {total_probability:g}, exceeding one"
            )
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
    if bool(args_cli.reference_phase_bin_edges) != bool(
        args_cli.reference_phase_bin_weights
    ):
        raise ValueError(
            "--reference_phase_bin_edges and --reference_phase_bin_weights "
            "must be configured together"
        )
    if args_cli.reference_phase_bin_edges is not None:
        env_cfg.amp_reference_phase_bin_edges = args_cli.reference_phase_bin_edges
        env_cfg.amp_reference_phase_bin_weights = args_cli.reference_phase_bin_weights
    if args_cli.robust_reset_curriculum_steps is not None:
        if args_cli.robust_reset_curriculum_steps < 1:
            raise ValueError("--robust_reset_curriculum_steps must be positive")
        env_cfg.amp_robust_reset_curriculum_steps = args_cli.robust_reset_curriculum_steps
    if args_cli.joint_state_tolerance is not None:
        if args_cli.joint_state_tolerance < 0.0:
            raise ValueError("--joint_state_tolerance must be non-negative")
        params = env_cfg.terminations.joint_limit.params
        params["margin_start"] = args_cli.joint_state_tolerance
        params["margin_end"] = args_cli.joint_state_tolerance
    if args_cli.parallel_state_tolerance is not None:
        if args_cli.parallel_state_tolerance < 0.0:
            raise ValueError("--parallel_state_tolerance must be non-negative")
        params = env_cfg.terminations.parallel_ankle.params
        params["motor_tolerance_start"] = args_cli.parallel_state_tolerance
        params["motor_tolerance_end"] = args_cli.parallel_state_tolerance
    for argument, term_name, parameter in (
        ("joint_state_tolerance_start", "joint_limit", "margin_start"),
        ("joint_state_tolerance_end", "joint_limit", "margin_end"),
        ("parallel_state_tolerance_start", "parallel_ankle", "motor_tolerance_start"),
        ("parallel_state_tolerance_end", "parallel_ankle", "motor_tolerance_end"),
    ):
        value = getattr(args_cli, argument)
        if value is not None:
            if value < 0.0:
                raise ValueError(f"--{argument} must be non-negative")
            getattr(env_cfg.terminations, term_name).params[parameter] = value
    if args_cli.state_tolerance_curriculum_steps is not None:
        if args_cli.state_tolerance_curriculum_steps < 1:
            raise ValueError("--state_tolerance_curriculum_steps must be positive")
        env_cfg.terminations.joint_limit.params["curriculum_steps"] = (
            args_cli.state_tolerance_curriculum_steps
        )
        env_cfg.terminations.parallel_ankle.params["curriculum_steps"] = (
            args_cli.state_tolerance_curriculum_steps
        )
    if args_cli.early_success_weight is not None:
        if args_cli.early_success_weight <= 0.0:
            raise ValueError("--early_success_weight must be positive")
        env_cfg.rewards.early_success.weight = args_cli.early_success_weight
    if args_cli.stability_hold_progress_weight is not None:
        if args_cli.stability_hold_progress_weight <= 0.0:
            raise ValueError("--stability_hold_progress_weight must be positive")
        if not hasattr(env_cfg.rewards, "stability_hold"):
            raise ValueError("selected task does not define a stability-hold reward")
        env_cfg.rewards.stability_hold.weight = args_cli.stability_hold_progress_weight
    if args_cli.standing_pose_max_error_hold_weight is not None:
        if args_cli.standing_pose_max_error_hold_weight <= 0.0:
            raise ValueError("--standing_pose_max_error_hold_weight must be positive")
        if not hasattr(env_cfg.rewards, "standing_pose_max_error_hold"):
            raise ValueError("selected task does not define a standing max-error hold reward")
        env_cfg.rewards.standing_pose_max_error_hold.weight = (
            args_cli.standing_pose_max_error_hold_weight
        )
    if (
        args_cli.handoff_curriculum_initial_progress is not None
        and args_cli.fixed_handoff_curriculum_progress is not None
    ):
        raise ValueError(
            "--handoff_curriculum_initial_progress and "
            "--fixed_handoff_curriculum_progress cannot be used together"
        )
    if args_cli.handoff_curriculum_initial_progress is not None:
        if not 0.0 <= args_cli.handoff_curriculum_initial_progress <= 1.0:
            raise ValueError("--handoff_curriculum_initial_progress must be in [0, 1]")
        env_cfg.amp_handoff_curriculum_initial_progress = (
            args_cli.handoff_curriculum_initial_progress
        )
    if args_cli.fixed_handoff_curriculum_progress is not None:
        if not 0.0 <= args_cli.fixed_handoff_curriculum_progress <= 1.0:
            raise ValueError("--fixed_handoff_curriculum_progress must be in [0, 1]")
        env_cfg.amp_handoff_curriculum_initial_progress = (
            args_cli.fixed_handoff_curriculum_progress
        )
        env_cfg.amp_handoff_curriculum_adaptive = False
    if args_cli.adaptive_failure_curriculum:
        env_cfg.amp_failure_adaptive_curriculum = True
        for argument, attribute in (
            ("adaptive_failure_blend_start", "amp_failure_adaptive_blend_start"),
            ("adaptive_failure_blend_end", "amp_failure_adaptive_blend_end"),
            ("adaptive_failure_promotion", "amp_failure_adaptive_promotion"),
            ("adaptive_failure_demotion", "amp_failure_adaptive_demotion"),
            ("adaptive_failure_success_threshold", "amp_failure_adaptive_success_threshold"),
            ("adaptive_failure_demotion_threshold", "amp_failure_adaptive_demotion_threshold"),
            ("adaptive_failure_min_trials", "amp_failure_adaptive_min_trials"),
            ("adaptive_failure_promotion_windows", "amp_failure_adaptive_promotion_windows"),
            ("adaptive_failure_state_interval", "amp_failure_adaptive_state_interval"),
        ):
            value = getattr(args_cli, argument)
            if value is not None:
                setattr(env_cfg, attribute, value)
        for attribute in (
            "amp_failure_adaptive_blend_start",
            "amp_failure_adaptive_blend_end",
            "amp_failure_adaptive_success_threshold",
            "amp_failure_adaptive_demotion_threshold",
        ):
            value = getattr(env_cfg, attribute)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{attribute} must be in [0, 1]")
        if env_cfg.amp_failure_adaptive_blend_start > env_cfg.amp_failure_adaptive_blend_end:
            raise ValueError("adaptive failure blend start must not exceed the end")
        for attribute in ("amp_failure_adaptive_promotion", "amp_failure_adaptive_demotion"):
            if getattr(env_cfg, attribute) <= 0.0:
                raise ValueError(f"{attribute} must be positive")
        if (
            env_cfg.amp_failure_adaptive_demotion_threshold
            >= env_cfg.amp_failure_adaptive_success_threshold
        ):
            raise ValueError("adaptive failure demotion threshold must be below success threshold")
        if env_cfg.amp_failure_adaptive_min_trials < 1:
            raise ValueError("--adaptive_failure_min_trials must be positive")
        if env_cfg.amp_failure_adaptive_promotion_windows < 1:
            raise ValueError("--adaptive_failure_promotion_windows must be positive")
        if env_cfg.amp_failure_adaptive_state_interval < 1:
            raise ValueError("--adaptive_failure_state_interval must be positive")
        if args_cli.adaptive_failure_state_resume is not None:
            env_cfg.amp_failure_adaptive_state_resume = os.path.abspath(
                os.path.expanduser(args_cli.adaptive_failure_state_resume)
            )
        env_cfg.amp_failure_adaptive_reset_statistics = (
            args_cli.adaptive_failure_reset_statistics
        )
    if args_cli.bc_teacher == "native":
        env_cfg.amp_reset_mode = "faceup"
        env_cfg.amp_canonical_from_reference = True
        for argument, attribute in (
            ("bc_canonical_joint_noise", "amp_canonical_joint_noise"),
            ("bc_canonical_root_xy_noise", "amp_canonical_root_xy_noise"),
            ("bc_canonical_height_noise", "amp_canonical_height_noise"),
            ("bc_canonical_orientation_noise", "amp_canonical_orientation_noise"),
        ):
            value = getattr(args_cli, argument)
            if value is not None:
                if value < 0.0:
                    raise ValueError(f"--{argument} must be non-negative")
                setattr(env_cfg, attribute, value)
    agent_cfg["trainer"]["close_environment_at_exit"] = False

    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    log_root = os.path.abspath(os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]))
    log_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_amp_torch")
    suffix = args_cli.experiment_name or agent_cfg["agent"]["experiment"].get("experiment_name")
    if suffix:
        log_name += f"_{suffix}"
    log_dir = os.path.join(log_root, log_name)
    agent_cfg["agent"]["experiment"]["directory"] = log_root
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_name
    if env_cfg.amp_failure_adaptive_curriculum:
        env_cfg.amp_failure_adaptive_state_output = os.path.join(
            log_dir,
            "adaptive_failure_curriculum.npz",
        )
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    print(f"Exact experiment name requested from command line: {log_name}")
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    raw_env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        raw_env = gym.wrappers.RecordVideo(raw_env, **video_kwargs)

    env = SkrlVecEnvWrapper(raw_env, ml_framework="torch")
    runner = Runner(env, agent_cfg)
    configure_normalized_policy_action_bounds(runner.agent)
    install_nonfinite_rollout_isolation(runner.agent, raw_env)
    configured_style_scale = float(runner.agent.cfg.style_reward_scale)
    if (
        args_cli.maximum_amp_style_fraction is not None
        and not 0.0 < args_cli.maximum_amp_style_fraction < 1.0
    ):
        raise ValueError("--maximum_amp_style_fraction must be inside (0, 1)")
    if not 0.0 <= args_cli.minimum_amp_style_reward_scale <= configured_style_scale:
        raise ValueError(
            "--minimum_amp_style_reward_scale must be between zero and the configured style scale"
        )
    install_amp_reward_diagnostics(
        runner.agent,
        maximum_style_fraction=args_cli.maximum_amp_style_fraction,
        minimum_style_reward_scale=args_cli.minimum_amp_style_reward_scale,
    )
    if args_cli.checkpoint:
        checkpoint = retrieve_file_path(args_cli.checkpoint)
        print(f"[INFO] Loading model checkpoint from: {checkpoint}")
        load_amp_checkpoint(
            runner,
            checkpoint,
            reset_optimizer=args_cli.reset_checkpoint_optimizer,
            reset_value=args_cli.reset_checkpoint_value,
            reset_discriminator=args_cli.reset_checkpoint_discriminator,
        )
        if args_cli.reset_checkpoint_learning_rate or args_cli.agent_learning_rate is not None:
            learning_rate = float(agent_cfg["agent"]["learning_rate"])
            for group in runner.agent.optimizer.param_groups:
                group["lr"] = learning_rate
            print(f"[INFO] Reset loaded optimizer learning rate to {learning_rate:g}")
    if args_cli.policy_anchor_before_bc:
        install_policy_anchor(
            raw_env,
            runner,
            args_cli.policy_anchor_samples,
            args_cli.policy_anchor_batch_size,
            args_cli.policy_anchor_steps,
            args_cli.policy_anchor_learning_rate,
            args_cli.policy_anchor_canonical_fraction,
            args_cli.policy_anchor_canonical_noise_scale,
            args_cli.policy_anchor_reference_phase_min,
            args_cli.policy_anchor_reference_phase_max,
        )
    pretrain_from_teacher(
        raw_env,
        runner,
        args_cli.bc_iterations,
        args_cli.bc_learning_rate,
        args_cli.bc_teacher,
        args_cli.bc_student_rollout_fraction,
        args_cli.bc_reset_mode,
        args_cli.bc_policy_anchor_weight,
        args_cli.bc_physical_teacher_target_blend,
    )
    if args_cli.bc_iterations > 0:
        bc_checkpoint = os.path.join(log_dir, "checkpoints", "bc_agent.pt")
        os.makedirs(os.path.dirname(bc_checkpoint), exist_ok=True)
        runner.agent.save(bc_checkpoint)
        print(f"[INFO] Saved behavior-cloned checkpoint: {bc_checkpoint}")
    if args_cli.bc_only:
        if args_cli.bc_iterations <= 0:
            raise ValueError("--bc_only requires --bc_iterations")
        env.close()
        return
    if not args_cli.policy_anchor_before_bc:
        install_policy_anchor(
            raw_env,
            runner,
            args_cli.policy_anchor_samples,
            args_cli.policy_anchor_batch_size,
            args_cli.policy_anchor_steps,
            args_cli.policy_anchor_learning_rate,
            args_cli.policy_anchor_canonical_fraction,
            args_cli.policy_anchor_canonical_noise_scale,
            args_cli.policy_anchor_reference_phase_min,
            args_cli.policy_anchor_reference_phase_max,
        )
    install_finite_update_guard(runner.agent)
    runner.run()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

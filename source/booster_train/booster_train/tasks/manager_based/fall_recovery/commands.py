from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import MISSING

import isaaclab.utils.math as math_utils
import numpy as np
import torch
from isaaclab.assets import Articulation
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

from .joint_order import deployment_joint_ids
from .parallel_ankle import K1ParallelAnkleKinematics


class RecoveryReferenceCommand(CommandTerm):
    """Reference-state curriculum that gradually opens into arbitrary falls."""

    cfg: "RecoveryReferenceCommandCfg"

    def __init__(self, cfg: "RecoveryReferenceCommandCfg", env) -> None:
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._simulation_joint_ids = deployment_joint_ids(self.robot.joint_names, cfg.joint_names)

        self._joint_minimum = torch.tensor(cfg.position_minimum, device=self.device).unsqueeze(0)
        self._joint_maximum = torch.tensor(cfg.position_maximum, device=self.device).unsqueeze(0)
        self._parallel = K1ParallelAnkleKinematics(cfg.parallel_ankle, device=self.device, dtype=torch.float32)
        self._ankle_pairs = ((14, 15), (20, 21))

        self._trajectories: list[dict[str, torch.Tensor]] = []
        for path in cfg.reference_files:
            with np.load(path) as data:
                joint = torch.tensor(data["joint"], dtype=torch.float32, device=self.device)
                gravity = torch.tensor(data["gravity"], dtype=torch.float32, device=self.device)
                height = torch.tensor(data["height"], dtype=torch.float32, device=self.device)
            if joint.ndim != 2 or joint.shape[1] != 22:
                raise ValueError(f"{path}: joint reference must have shape (frames, 22)")
            if gravity.shape != (joint.shape[0], 3) or height.shape != (
                joint.shape[0],
                1,
            ):
                raise ValueError(f"{path}: gravity/height reference length mismatch")
            if (
                not torch.all(torch.isfinite(joint))
                or not torch.all(torch.isfinite(gravity))
                or not torch.all(torch.isfinite(height))
            ):
                raise ValueError(f"{path}: reference contains a non-finite value")
            joint = torch.clamp(joint, self._joint_minimum, self._joint_maximum)
            for foot, indexes in enumerate(self._ankle_pairs):
                pair = torch.tensor(indexes, dtype=torch.long, device=self.device)
                desired = joint[:, pair]
                neutral = self._parallel.serial_zero[2 * foot : 2 * foot + 2].expand_as(desired)
                projected, _, feasible = self._parallel.project(desired, neutral, foot)
                if not torch.all(feasible):
                    raise RuntimeError(f"{path}: reference ankle projection failed for foot {foot}")
                joint[:, pair] = projected
            joint_velocity = torch.zeros_like(joint)
            joint_velocity[:-1] = (joint[1:] - joint[:-1]) * cfg.reference_fps
            joint_velocity[-1] = joint_velocity[-2]
            self._trajectories.append(
                {
                    "joint": joint,
                    "joint_velocity": joint_velocity,
                    "gravity": gravity,
                    "height": height,
                }
            )

        self._trajectory_lengths = torch.tensor(
            [item["joint"].shape[0] for item in self._trajectories],
            dtype=torch.long,
            device=self.device,
        )
        self.trajectory_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self.phase_speed = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        self.reference_weight = torch.ones(self.num_envs, dtype=torch.float32, device=self.device)
        self.reference_joint_pos = torch.zeros(self.num_envs, 22, device=self.device)
        self.reference_joint_vel = torch.zeros_like(self.reference_joint_pos)
        self.reference_gravity = torch.zeros(self.num_envs, 3, device=self.device)
        self.reference_height = torch.zeros(self.num_envs, 1, device=self.device)
        self.reference_phase = torch.zeros(self.num_envs, 1, device=self.device)

        self.metrics["curriculum_progress"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reference_reset_fraction"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["reference_phase"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return torch.cat(
            (
                self.reference_joint_pos,
                self.reference_gravity,
                self.reference_height,
                self.reference_phase,
            ),
            dim=-1,
        )

    def _as_env_ids(self, env_ids: Sequence[int] | slice) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.tensor(env_ids, dtype=torch.long, device=self.device)

    def _curriculum_progress(self) -> float:
        transitions = float(self._env.common_step_counter * self.num_envs)
        return min(transitions / float(self.cfg.curriculum_steps), 1.0)

    def _update_reference(self, env_ids: torch.Tensor) -> None:
        for trajectory_id, trajectory in enumerate(self._trajectories):
            mask = self.trajectory_ids[env_ids] == trajectory_id
            selected_envs = env_ids[mask]
            if selected_envs.numel() == 0:
                continue
            frames = torch.clamp(
                self.time_steps[selected_envs].long(),
                min=0,
                max=trajectory["joint"].shape[0] - 1,
            )
            self.reference_joint_pos[selected_envs] = trajectory["joint"][frames]
            self.reference_joint_vel[selected_envs] = trajectory["joint_velocity"][frames]
            self.reference_gravity[selected_envs] = trajectory["gravity"][frames]
            self.reference_height[selected_envs] = trajectory["height"][frames]
            self.reference_phase[selected_envs, 0] = frames / max(trajectory["joint"].shape[0] - 1, 1)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        env_ids = self._as_env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        count = env_ids.numel()
        if hasattr(self._env, "_recovery_stable_steps"):
            self._env._recovery_stable_steps[env_ids] = 0
        progress = self._curriculum_progress()

        self.trajectory_ids[env_ids] = torch.randint(0, len(self._trajectories), (count,), device=self.device)
        selected_lengths = self._trajectory_lengths[self.trajectory_ids[env_ids]]
        sampled_phase = torch.rand(count, device=self.device)
        self.time_steps[env_ids] = sampled_phase * (selected_lengths - 1).to(torch.float32)

        reference_probability = self.cfg.reference_reset_probability_start + progress * (
            self.cfg.reference_reset_probability_end - self.cfg.reference_reset_probability_start
        )
        reference_reset = torch.rand(count, device=self.device) < reference_probability
        if self.cfg.play:
            self.time_steps[env_ids] = 0.0
            reference_reset[:] = not self.cfg.play_all_falls

        phase_low, phase_high = self.cfg.reference_phase_speed
        self.phase_speed[env_ids] = torch.empty(count, device=self.device).uniform_(phase_low, phase_high)
        self._update_reference(env_ids)

        joint_pos = self.reference_joint_pos[env_ids].clone()
        reference_noise = 0.03 + 0.07 * progress
        broad_noise = 0.20 + 0.15 * progress
        noise_scale = torch.where(reference_reset, reference_noise, broad_noise).unsqueeze(-1)
        joint_pos += (2.0 * torch.rand_like(joint_pos) - 1.0) * noise_scale
        joint_pos = torch.clamp(joint_pos, self._joint_minimum, self._joint_maximum)
        for foot, indexes in enumerate(self._ankle_pairs):
            pair = torch.tensor(indexes, dtype=torch.long, device=self.device)
            desired = joint_pos[:, pair]
            neutral = self._parallel.serial_zero[2 * foot : 2 * foot + 2].to(torch.float32).expand_as(desired)
            projected, _, feasible = self._parallel.project(desired, neutral, foot)
            if not torch.all(feasible):
                raise RuntimeError(f"reset ankle projection failed for foot {foot}")
            joint_pos[:, pair] = projected

        joint_velocity = self.reference_joint_vel[env_ids].clone()
        joint_velocity += (2.0 * torch.rand_like(joint_velocity) - 1.0) * (0.3 + 1.2 * progress)
        joint_velocity[~reference_reset] *= 0.5

        gravity = self.reference_gravity[env_ids]
        pitch = torch.asin(torch.clamp(gravity[:, 0], -1.0, 1.0))
        roll = torch.atan2(-gravity[:, 1], -gravity[:, 2])
        roll = torch.where(torch.abs(torch.cos(pitch)) < 0.1, torch.zeros_like(roll), roll)

        if self.cfg.play and self.cfg.play_all_falls:
            fall_mode = torch.remainder(env_ids, 4)
        else:
            fall_mode = torch.randint(0, 4, (count,), device=self.device)
        random_roll = torch.zeros(count, device=self.device)
        random_pitch = torch.zeros(count, device=self.device)
        random_pitch = torch.where(fall_mode == 0, -0.5 * math.pi, random_pitch)
        random_pitch = torch.where(fall_mode == 1, 0.5 * math.pi, random_pitch)
        random_roll = torch.where(fall_mode == 2, 0.5 * math.pi, random_roll)
        random_roll = torch.where(fall_mode == 3, -0.5 * math.pi, random_roll)
        roll = torch.where(reference_reset, roll, random_roll)
        pitch = torch.where(reference_reset, pitch, random_pitch)
        orientation_noise = torch.where(reference_reset, 0.10 + 0.10 * progress, 0.25)
        roll += (2.0 * torch.rand(count, device=self.device) - 1.0) * orientation_noise
        pitch += (2.0 * torch.rand(count, device=self.device) - 1.0) * orientation_noise
        yaw = (2.0 * torch.rand(count, device=self.device) - 1.0) * math.pi
        orientation = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

        root_pose = self.robot.data.default_root_state[env_ids, :7].clone()
        root_pose[:, :2] = self._env.scene.env_origins[env_ids, :2]
        root_pose[:, :2] += (2.0 * torch.rand(count, 2, device=self.device) - 1.0) * 0.15
        random_height = torch.empty(count, device=self.device).uniform_(0.06, 0.14)
        root_pose[:, 2] = self._env.scene.env_origins[env_ids, 2] + torch.where(
            reference_reset,
            torch.clamp(
                self.reference_height[env_ids, 0] + 0.01 * torch.randn(count, device=self.device),
                0.055,
                0.54,
            ),
            random_height,
        )
        root_pose[:, 3:7] = orientation

        root_velocity = torch.zeros(count, 6, device=self.device)
        linear_scale = 0.15 + 0.55 * progress
        angular_scale = 0.4 + 1.8 * progress
        root_velocity[:, :3].uniform_(-linear_scale, linear_scale)
        root_velocity[:, 3:].uniform_(-angular_scale, angular_scale)
        root_velocity[reference_reset] *= 0.5

        self.robot.write_joint_state_to_sim(
            joint_pos,
            joint_velocity,
            joint_ids=self._simulation_joint_ids,
            env_ids=env_ids,
        )
        self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)

        reference_strength = 0.15 + 0.85 * (1.0 - progress)
        self.reference_weight[env_ids] = torch.where(
            reference_reset,
            torch.full((count,), reference_strength, device=self.device),
            torch.full((count,), 0.05 * (1.0 - progress), device=self.device),
        )
        self.metrics["curriculum_progress"][env_ids] = progress
        self.metrics["reference_reset_fraction"][env_ids] = reference_reset.to(torch.float32)

    def _update_command(self) -> None:
        self.time_steps += self.phase_speed
        maximum = self._trajectory_lengths[self.trajectory_ids].to(torch.float32) - 1.0
        self.time_steps = torch.minimum(self.time_steps, maximum)
        env_ids = torch.arange(self.num_envs, device=self.device)
        self._update_reference(env_ids)
        self.metrics["reference_phase"][:] = self.reference_phase[:, 0]

    def _update_metrics(self) -> None:
        pass


@configclass
class RecoveryReferenceCommandCfg(CommandTermCfg):
    class_type: type = RecoveryReferenceCommand

    asset_name: str = "robot"
    reference_files: list[str] = MISSING
    reference_fps: float = 50.0
    joint_names: list[str] = MISSING
    position_minimum: list[float] = MISSING
    position_maximum: list[float] = MISSING
    parallel_ankle: dict = MISSING
    curriculum_steps: int = 150_000_000
    reference_reset_probability_start: float = 1.0
    reference_reset_probability_end: float = 0.35
    reference_phase_speed: tuple[float, float] = (1.0, 1.8)
    play: bool = False
    play_all_falls: bool = False


class RecoveryDiscoveryCommand(CommandTerm):
    """Generate fixed fallen starts and HumanUP-style lift assistance."""

    cfg: "RecoveryDiscoveryCommandCfg"

    def __init__(self, cfg: "RecoveryDiscoveryCommandCfg", env) -> None:
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._simulation_joint_ids = deployment_joint_ids(self.robot.joint_names, cfg.joint_names)
        self._joint_minimum = torch.tensor(cfg.position_minimum, device=self.device).unsqueeze(0)
        self._joint_maximum = torch.tensor(cfg.position_maximum, device=self.device).unsqueeze(0)
        self._initial_joint_position = torch.tensor(
            cfg.initial_joint_position,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        self._standing_joint_position = torch.tensor(
            cfg.standing_joint_position,
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        self._parallel = K1ParallelAnkleKinematics(cfg.parallel_ankle, device=self.device, dtype=torch.float32)
        self._ankle_pairs = ((14, 15), (20, 21))
        body_ids, body_names = self.robot.find_bodies(cfg.assistance_body_name, preserve_order=True)
        if len(body_ids) != 1:
            raise ValueError(f"assistance body must resolve once, got {body_names} ({body_ids})")
        self._assistance_body_ids = torch.tensor(body_ids, dtype=torch.long, device=self.device)
        self._command = torch.zeros(self.num_envs, 1, device=self.device)
        self.assistance_force = torch.zeros(self.num_envs, device=self.device)
        self.metrics["assistance_force"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["assistance_height_scale"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["standing_reset_fraction"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _as_env_ids(self, env_ids: Sequence[int] | slice) -> torch.Tensor:
        if isinstance(env_ids, slice):
            return torch.arange(self.num_envs, device=self.device)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.tensor(env_ids, dtype=torch.long, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        env_ids = self._as_env_ids(env_ids)
        if env_ids.numel() == 0:
            return
        count = env_ids.numel()
        if hasattr(self._env, "_recovery_stable_steps"):
            self._env._recovery_stable_steps[env_ids] = 0

        transitions = float(self._env.common_step_counter * self.num_envs)
        curriculum_progress = min(transitions / float(self.cfg.standing_reset_curriculum_steps), 1.0)
        standing_fraction = self.cfg.standing_reset_fraction_start + curriculum_progress * (
            self.cfg.standing_reset_fraction_end - self.cfg.standing_reset_fraction_start
        )
        standing_reset = torch.rand(count, device=self.device) < standing_fraction

        joint_pos = self._initial_joint_position.expand(count, -1).clone()
        joint_pos[standing_reset] = self._standing_joint_position
        if self.cfg.joint_noise > 0.0:
            joint_pos += (2.0 * torch.rand_like(joint_pos) - 1.0) * self.cfg.joint_noise
        joint_pos = torch.clamp(joint_pos, self._joint_minimum, self._joint_maximum)
        for foot, indexes in enumerate(self._ankle_pairs):
            pair = torch.tensor(indexes, dtype=torch.long, device=self.device)
            neutral = self._parallel.serial_zero[2 * foot : 2 * foot + 2].to(torch.float32).expand(count, -1)
            projected, _, feasible = self._parallel.project(joint_pos[:, pair], neutral, foot)
            if not torch.all(feasible):
                raise RuntimeError(f"discovery reset ankle projection failed for foot {foot}")
            joint_pos[:, pair] = projected

        joint_vel = torch.zeros_like(joint_pos)
        root_pose = self.robot.data.default_root_state[env_ids, :7].clone()
        root_pose[:, :2] = self._env.scene.env_origins[env_ids, :2]
        if self.cfg.root_position_noise > 0.0:
            root_pose[:, :2] += (2.0 * torch.rand(count, 2, device=self.device) - 1.0) * self.cfg.root_position_noise
        root_height = torch.full((count,), self.cfg.initial_height, device=self.device)
        root_height[standing_reset] = self.cfg.standing_height
        root_pose[:, 2] = self._env.scene.env_origins[env_ids, 2] + root_height
        angle_noise = self.cfg.orientation_noise
        roll = (2.0 * torch.rand(count, device=self.device) - 1.0) * angle_noise
        pitch = torch.full((count,), self.cfg.initial_pitch, device=self.device)
        pitch[standing_reset] = 0.0
        pitch += (2.0 * torch.rand(count, device=self.device) - 1.0) * angle_noise
        yaw = (2.0 * torch.rand(count, device=self.device) - 1.0) * angle_noise
        root_pose[:, 3:7] = math_utils.quat_from_euler_xyz(roll, pitch, yaw)

        self.robot.write_joint_state_to_sim(
            joint_pos,
            joint_vel,
            joint_ids=self._simulation_joint_ids,
            env_ids=env_ids,
        )
        self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(torch.zeros(count, 6, device=self.device), env_ids=env_ids)
        self.metrics["standing_reset_fraction"][env_ids] = standing_reset.to(torch.float32)

    def _update_command(self) -> None:
        height = self.robot.data.root_pos_w[:, 2] - self._env.scene.env_origins[:, 2]
        height_progress = torch.clamp(height / self.cfg.assistance_target_height, 0.0, 1.0)
        height_scale = 1.0 - torch.sin(0.5 * math.pi * height_progress)
        pulse = self._env.common_step_counter % self.cfg.assistance_interval_steps == 0
        self.assistance_force = self.cfg.maximum_assistance_force * height_scale if pulse else torch.zeros_like(height)

        force = torch.zeros(self.num_envs, 1, 3, device=self.device)
        force[:, 0, 2] = self.assistance_force
        self.robot.set_external_force_and_torque(
            forces=force,
            torques=torch.zeros_like(force),
            body_ids=self._assistance_body_ids,
            is_global=True,
        )
        self.metrics["assistance_force"][:] = self.assistance_force
        self.metrics["assistance_height_scale"][:] = height_scale

    def _update_metrics(self) -> None:
        pass


@configclass
class RecoveryDiscoveryCommandCfg(CommandTermCfg):
    class_type: type = RecoveryDiscoveryCommand

    asset_name: str = "robot"
    joint_names: list[str] = MISSING
    position_minimum: list[float] = MISSING
    position_maximum: list[float] = MISSING
    initial_joint_position: list[float] = MISSING
    standing_joint_position: list[float] = MISSING
    parallel_ankle: dict = MISSING
    initial_pitch: float = -0.5 * math.pi
    initial_height: float = 0.08
    standing_height: float = 0.52
    standing_reset_fraction_start: float = 0.2
    standing_reset_fraction_end: float = 0.1
    standing_reset_curriculum_steps: int = 6_553_600_000
    joint_noise: float = 0.025
    root_position_noise: float = 0.03
    orientation_noise: float = 0.04
    assistance_body_name: str = "Trunk"
    maximum_assistance_force: float = 200.0
    assistance_target_height: float = 0.47
    assistance_interval_steps: int = 50

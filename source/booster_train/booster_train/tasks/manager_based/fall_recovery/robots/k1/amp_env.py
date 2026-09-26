from __future__ import annotations

import hashlib
import math
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import math as math_utils

from booster_train.tasks.manager_based.fall_recovery.amp import (
    K1AmpMotionLoader,
    compute_amp_frame,
    quat_multiply,
    quat_rotate,
    quat_slerp_batch,
    sample_reference_clip_ids,
    sample_reference_phase_offsets,
)
from booster_train.tasks.manager_based.fall_recovery.joint_order import deployment_joint_ids
from booster_train.tasks.manager_based.fall_recovery.recovery_math import (
    limit_reset_joint_velocity,
    mastery_curriculum_update,
    mirror_bilateral_joint_position,
    optional_termination_mask,
    reference_phase_bin_indices,
)

from .env_cfg import GOAL_POSITION, JOINT_NAMES, K1FallRecoveryAmpEnvCfg


class K1FallRecoveryAmpEnv(ManagerBasedRLEnv):
    """Manager-based K1 recovery task with the interfaces expected by skrl AMP."""

    cfg: K1FallRecoveryAmpEnvCfg
    _reference_phase_bin_count = 10

    def __init__(self, cfg: K1FallRecoveryAmpEnvCfg, render_mode: str | None = None, **kwargs) -> None:
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)
        self.robot = self.scene["robot"]
        self._joint_ids = deployment_joint_ids(self.robot.joint_names, JOINT_NAMES)
        self._amp_joint_ids = [self._joint_ids[JOINT_NAMES.index(name)] for name in cfg.amp_joint_names]
        self._root_body_id = self._body_id(cfg.amp_root_body_name)
        self._key_body_ids = [self._body_id(name) for name in cfg.amp_key_body_names]
        self._motion_loader = K1AmpMotionLoader(
            motion_files=cfg.amp_motion_files,
            joint_names=JOINT_NAMES,
            amp_joint_names=cfg.amp_joint_names,
            root_body_name=cfg.amp_root_body_name,
            key_body_names=tuple(cfg.amp_key_body_names),
            history_length=cfg.amp_history_length,
            expected_fps=1.0 / self.step_dt,
            device=self.device,
        )
        if cfg.amp_style_motion_files:
            self._style_motion_loader = K1AmpMotionLoader(
                motion_files=cfg.amp_style_motion_files,
                joint_names=JOINT_NAMES,
                amp_joint_names=cfg.amp_joint_names,
                root_body_name=cfg.amp_root_body_name,
                key_body_names=tuple(cfg.amp_key_body_names),
                history_length=cfg.amp_history_length,
                expected_fps=1.0 / self.step_dt,
                device=self.device,
            )
        else:
            self._style_motion_loader = self._motion_loader

        expected_frame_size = 2 * len(cfg.amp_joint_names) + 1 + 6 + 3 + 3 + 3 * len(cfg.amp_key_body_names)
        if self._motion_loader.amp_frame_size != expected_frame_size:
            raise ValueError(
                f"AMP frame has {self._motion_loader.amp_frame_size} values, expected {expected_frame_size}"
            )
        self._reference_frame_index = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._action_reference_frame_index = torch.zeros_like(self._reference_frame_index)
        self.amp_observation_size = cfg.amp_history_length * expected_frame_size
        self.amp_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.amp_observation_size,),
            dtype=np.float32,
        )
        self._amp_history = torch.zeros(
            (self.num_envs, cfg.amp_history_length, expected_frame_size),
            dtype=torch.float32,
            device=self.device,
        )
        self._joint_minimum = torch.tensor(cfg.actions.joint_pos.position_minimum, device=self.device).unsqueeze(0)
        self._joint_maximum = torch.tensor(cfg.actions.joint_pos.position_maximum, device=self.device).unsqueeze(0)
        self._safe_joint_minimum = self._joint_minimum + cfg.amp_reset_joint_margin
        self._safe_joint_maximum = self._joint_maximum - cfg.amp_reset_joint_margin
        self._random_safe_joint_minimum = (
            self._joint_minimum + cfg.amp_random_reset_joint_margin
        )
        self._random_safe_joint_maximum = (
            self._joint_maximum - cfg.amp_random_reset_joint_margin
        )
        self._parallel = self.action_manager.get_term("joint_pos")._parallel
        self._failure_states = self._load_failure_states()
        failure_state_count = 0 if self._failure_states is None else self._failure_states["joint_position"].shape[0]
        self._failure_state_index = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._reset_mode_code = torch.full(
            (self.num_envs,), -1, dtype=torch.long, device=self.device
        )
        self._completed_mode_counts = torch.zeros(5, dtype=torch.long, device=self.device)
        self._recovered_mode_counts = torch.zeros_like(self._completed_mode_counts)
        self._reference_reset_phase = torch.full(
            (self.num_envs,),
            torch.nan,
            dtype=torch.float32,
            device=self.device,
        )
        self._reference_reset_clip_id = torch.full(
            (self.num_envs,),
            -1,
            dtype=torch.long,
            device=self.device,
        )
        self._terminal_state_valid = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._terminal_joint_position = torch.zeros(
            (self.num_envs, len(self._joint_ids)),
            dtype=torch.float32,
            device=self.device,
        )
        self._terminal_joint_velocity = torch.zeros_like(
            self._terminal_joint_position
        )
        self._terminal_root_pose = torch.zeros(
            (self.num_envs, 7),
            dtype=torch.float32,
            device=self.device,
        )
        self._terminal_root_velocity = torch.zeros(
            (self.num_envs, 6),
            dtype=torch.float32,
            device=self.device,
        )
        self._completed_reference_phase_counts = torch.zeros(
            self._reference_phase_bin_count,
            dtype=torch.long,
            device=self.device,
        )
        self._recovered_reference_phase_counts = torch.zeros_like(
            self._completed_reference_phase_counts
        )
        clip_count = self._motion_loader.clip_start_indexes.numel()
        self._completed_reference_clip_phase_counts = torch.zeros(
            (clip_count, self._reference_phase_bin_count),
            dtype=torch.long,
            device=self.device,
        )
        self._recovered_reference_clip_phase_counts = torch.zeros_like(
            self._completed_reference_clip_phase_counts
        )
        self._adaptive_failure_blend = torch.full(
            (failure_state_count,),
            cfg.amp_failure_adaptive_blend_start,
            dtype=torch.float32,
            device=self.device,
        )
        self._adaptive_failure_trials = torch.zeros(
            failure_state_count, dtype=torch.long, device=self.device
        )
        self._adaptive_failure_successes = torch.zeros_like(self._adaptive_failure_trials)
        self._adaptive_failure_promotion_streak = torch.zeros_like(self._adaptive_failure_trials)
        self._adaptive_failure_last_success_rate = torch.full(
            (failure_state_count,),
            torch.nan,
            dtype=torch.float32,
            device=self.device,
        )
        self._adaptive_failure_state_last_saved_step = -1
        self._load_adaptive_failure_curriculum()
        if not 0.0 <= cfg.amp_handoff_curriculum_initial_progress <= 1.0:
            raise ValueError("initial handoff curriculum progress must be in [0, 1]")
        self._handoff_curriculum_progress = float(
            cfg.amp_handoff_curriculum_initial_progress
        )
        self._handoff_curriculum_promotion_streak = 0
        self._handoff_curriculum_demotion_streak = 0
        self._handoff_curriculum_trials = torch.zeros(
            (), dtype=torch.long, device=self.device
        )
        self._handoff_curriculum_successes = torch.zeros_like(
            self._handoff_curriculum_trials
        )
        self._handoff_curriculum_safety_samples = torch.zeros_like(
            self._handoff_curriculum_trials
        )
        self._handoff_curriculum_parallel_state_violations = torch.zeros_like(
            self._handoff_curriculum_trials
        )
        self._handoff_curriculum_hard_joint_limit_violations = torch.zeros_like(
            self._handoff_curriculum_trials
        )
        self._handoff_curriculum_joint_limit_terminations = torch.zeros_like(
            self._handoff_curriculum_trials
        )
        self._handoff_curriculum_parallel_ankle_terminations = torch.zeros_like(
            self._handoff_curriculum_trials
        )
        self._handoff_curriculum_nonfinite_action_terminations = torch.zeros_like(
            self._handoff_curriculum_trials
        )
        self._handoff_curriculum_last_update_step = self.common_step_counter
        self._handoff_curriculum_last_success_rate = float("nan")
        self._handoff_curriculum_last_parallel_state_violation_fraction = float("nan")
        self._handoff_curriculum_last_hard_joint_limit_fraction = float("nan")
        self._handoff_curriculum_last_joint_limit_termination_rate = float("nan")
        self._handoff_curriculum_last_parallel_ankle_termination_rate = float("nan")
        self._handoff_curriculum_last_nonfinite_action_termination_rate = float("nan")
        self._handoff_curriculum_last_safety_passed = False
        self._tracking_frame_cache_step = -1
        self._tracking_frame_cache: torch.Tensor | None = None

    def _load_failure_states(self) -> dict[str, torch.Tensor] | None:
        if not self.cfg.amp_failure_state_file:
            if max(
                self.cfg.amp_failure_reset_probability_start,
                self.cfg.amp_failure_reset_probability_end,
            ) > 0.0:
                raise ValueError("AMP failure resets require amp_failure_state_file")
            return None

        path = Path(self.cfg.amp_failure_state_file)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != self.cfg.amp_failure_state_sha256:
            raise RuntimeError(
                f"{path}: SHA256 {digest} does not match {self.cfg.amp_failure_state_sha256}"
            )
        required = {
            "joint_position": (22,),
            "joint_velocity": (22,),
            "root_pose": (7,),
            "root_velocity": (6,),
            "termination_code": (),
            "mode": (),
        }
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != set(required):
                raise ValueError(
                    f"{path}: unexpected failure-state fields; "
                    f"expected={sorted(required)}, actual={sorted(archive.files)}"
                )
            sample_count = int(archive["joint_position"].shape[0])
            if sample_count < 1:
                raise ValueError(f"{path}: failure-state dataset is empty")
            for name, trailing_shape in required.items():
                if archive[name].shape != (sample_count, *trailing_shape):
                    raise ValueError(
                        f"{path}: {name} has shape {archive[name].shape}, "
                        f"expected {(sample_count, *trailing_shape)}"
                    )
            for name in ("joint_position", "joint_velocity", "root_pose", "root_velocity"):
                if not np.all(np.isfinite(archive[name])):
                    raise ValueError(f"{path}: {name} contains non-finite values")
            quaternion_norm = np.linalg.norm(archive["root_pose"][:, 3:], axis=-1)
            if not np.allclose(quaternion_norm, 1.0, atol=1.0e-4, rtol=0.0):
                raise ValueError(f"{path}: root quaternions are not normalized")
            allowed_modes = {
                "faceup",
                "facedown",
                "side_left",
                "side_right",
                "random",
                "reference",
            }
            if not np.all(np.isin(archive["mode"], tuple(allowed_modes))):
                raise ValueError(f"{path}: unsupported failure-state reset mode")
            if not np.all(np.isin(archive["termination_code"], tuple(range(1, 8)))):
                raise ValueError(f"{path}: unsupported failure termination code")
            return {
                name: torch.as_tensor(archive[name], device=self.device)
                for name in ("joint_position", "joint_velocity", "root_pose", "root_velocity")
            }

    def _body_id(self, name: str) -> int:
        try:
            return self.robot.data.body_names.index(name)
        except ValueError as error:
            raise ValueError(f"K1 AMP body {name!r} is missing: {self.robot.data.body_names}") from error

    def _compute_amp_frame(self) -> torch.Tensor:
        root_position = self.robot.data.body_pos_w[:, self._root_body_id]
        frame = compute_amp_frame(
            self.robot.data.joint_pos[:, self._amp_joint_ids],
            self.robot.data.joint_vel[:, self._amp_joint_ids],
            root_position,
            self.robot.data.body_quat_w[:, self._root_body_id],
            self.robot.data.body_lin_vel_w[:, self._root_body_id],
            self.robot.data.body_ang_vel_w[:, self._root_body_id],
            self.robot.data.body_pos_w[:, self._key_body_ids],
        )
        frame[:, 2 * len(self.cfg.amp_joint_names)] -= self.scene.env_origins[:, 2]
        return frame

    def _compute_amp_tracking_frame(self) -> torch.Tensor:
        if self._tracking_frame_cache_step != self.common_step_counter:
            self._tracking_frame_cache = self._compute_amp_frame()
            self._tracking_frame_cache_step = self.common_step_counter
        if self._tracking_frame_cache is None:
            raise RuntimeError("AMP tracking frame cache was not initialized")
        return self._tracking_frame_cache

    def _set_amp_history(self, current: torch.Tensor, reset_mask: torch.Tensor | None = None) -> None:
        if reset_mask is None:
            self._amp_history[:] = current.unsqueeze(1)
            return
        if self.cfg.amp_history_length > 1:
            self._amp_history[:, 1:] = self._amp_history[:, :-1].clone()
        self._amp_history[:, 0] = current
        if torch.any(reset_mask):
            self._amp_history[reset_mask] = current[reset_mask].unsqueeze(1)

    def reset(self, seed: int | None = None, env_ids=None, options=None):
        observations, extras = super().reset(seed=seed, env_ids=env_ids, options=options)
        self._set_amp_history(self._compute_amp_frame())
        extras["amp_obs"] = self._amp_history.flatten(start_dim=1)
        return observations, extras

    def step(self, action: torch.Tensor):
        self._completed_mode_counts.zero_()
        self._recovered_mode_counts.zero_()
        self._completed_reference_phase_counts.zero_()
        self._recovered_reference_phase_counts.zero_()
        self._completed_reference_clip_phase_counts.zero_()
        self._recovered_reference_clip_phase_counts.zero_()
        self._action_reference_frame_index.copy_(self._reference_frame_index)
        self._reference_frame_index = self._motion_loader.advance_indexes(self._reference_frame_index)
        observations, rewards, terminated, truncated, extras = super().step(action)
        self._set_amp_history(self._compute_amp_frame(), terminated | truncated)
        extras["amp_obs"] = self._amp_history.flatten(start_dim=1)
        termination_counts: dict[str, torch.Tensor] = {}
        for name in (
            "recovered",
            "joint_limit",
            "parallel_ankle",
            "nonfinite_action",
            "time_out",
        ):
            if name in self.termination_manager.active_terms:
                count = torch.count_nonzero(self.termination_manager.get_term(name))
                termination_counts[name] = count
                extras["log"][f"Termination/{name}"] = count
        for code, name in enumerate(("reference", "standing", "random_fall", "failure", "canonical")):
            completed = self._completed_mode_counts[code]
            recovered = self._recovered_mode_counts[code]
            extras["log"][f"Recovery/{name}_completed"] = completed
            extras["log"][f"Recovery/{name}_recovered"] = recovered
            extras["log"][f"Recovery/{name}_success_rate"] = recovered.to(torch.float32) / torch.clamp(
                completed,
                min=1,
            )
        for index in range(self._reference_phase_bin_count):
            minimum = index / self._reference_phase_bin_count
            maximum = (index + 1) / self._reference_phase_bin_count
            name = f"reference_phase_{minimum:.1f}_{maximum:.1f}"
            completed = self._completed_reference_phase_counts[index]
            recovered = self._recovered_reference_phase_counts[index]
            extras["log"][f"Recovery/{name}_completed"] = completed
            extras["log"][f"Recovery/{name}_recovered"] = recovered
            extras["log"][f"Recovery/{name}_success_rate"] = recovered.to(torch.float32) / torch.clamp(
                completed,
                min=1,
            )
        for clip_index in range(self._completed_reference_clip_phase_counts.shape[0]):
            for phase_index in range(self._reference_phase_bin_count):
                minimum = phase_index / self._reference_phase_bin_count
                maximum = (phase_index + 1) / self._reference_phase_bin_count
                name = (
                    f"reference_clip_{clip_index}_phase_"
                    f"{minimum:.1f}_{maximum:.1f}"
                )
                completed = self._completed_reference_clip_phase_counts[
                    clip_index,
                    phase_index,
                ]
                recovered = self._recovered_reference_clip_phase_counts[
                    clip_index,
                    phase_index,
                ]
                extras["log"][f"Recovery/{name}_completed"] = completed
                extras["log"][f"Recovery/{name}_recovered"] = recovered
                extras["log"][f"Recovery/{name}_success_rate"] = (
                    recovered.to(torch.float32) / torch.clamp(completed, min=1)
                )
        motor_margins = []
        joint_action = self.action_manager.get_term("joint_pos")
        joint_position = joint_action.deployment_joint_position
        for foot, indexes in enumerate(((14, 15), (20, 21))):
            motor_margins.append(self._parallel.motor_margin(joint_position[:, indexes], foot))
        minimum_margin = torch.amin(torch.cat(motor_margins, dim=-1), dim=-1)
        parallel_state_violation = minimum_margin < 0.0
        extras["log"]["Metrics/parallel_min_motor_margin"] = torch.mean(minimum_margin)
        extras["log"]["Metrics/parallel_state_violation_fraction"] = torch.mean(
            parallel_state_violation.to(torch.float32)
        )
        extras["log"]["Metrics/parallel_target_fallback_fraction"] = torch.mean(
            joint_action.parallel_target_fallback.to(torch.float32)
        )
        nonfinite_action_count = termination_counts.get("nonfinite_action")
        extras["log"]["Metrics/nonfinite_action_fraction"] = (
            torch.zeros((), device=self.device)
            if nonfinite_action_count is None
            else nonfinite_action_count.to(torch.float32) / self.num_envs
        )
        lower_violation = torch.clamp(self._joint_minimum - joint_position, min=0.0)
        upper_violation = torch.clamp(joint_position - self._joint_maximum, min=0.0)
        maximum_joint_violation = torch.amax(lower_violation + upper_violation, dim=-1)
        hard_joint_limit_violation = maximum_joint_violation > 0.05
        extras["log"]["Metrics/joint_limit_max_violation"] = torch.mean(maximum_joint_violation)
        joint_limit_count = termination_counts.get("joint_limit")
        extras["log"]["Metrics/hard_joint_limit_fraction"] = (
            torch.zeros((), device=self.device)
            if joint_limit_count is None
            else joint_limit_count.to(torch.float32) / self.num_envs
        )
        extras["log"]["Metrics/joint_limit_braking_fraction"] = torch.mean(
            torch.any(joint_action.joint_limit_braking, dim=-1).to(torch.float32)
        )
        extras["log"]["Metrics/joint_limit_target_reduction"] = torch.mean(
            joint_action.joint_limit_target_reduction
        )
        joint_limit_mask = getattr(self, "_hard_joint_limit_violation_mask", None)
        if (
            isinstance(joint_limit_mask, torch.Tensor)
            and joint_limit_mask.shape == (self.num_envs, len(JOINT_NAMES))
        ):
            for index, name in enumerate(JOINT_NAMES):
                extras["log"][f"Termination/joint_limit_by_joint/{name}"] = (
                    torch.count_nonzero(joint_limit_mask[:, index])
                )
        self._update_handoff_success_curriculum(
            parallel_state_violation=parallel_state_violation,
            hard_joint_limit_violation=hard_joint_limit_violation,
            joint_limit_terminations=termination_counts.get("joint_limit"),
            parallel_ankle_terminations=termination_counts.get("parallel_ankle"),
            nonfinite_action_terminations=termination_counts.get("nonfinite_action"),
        )
        extras["log"]["Curriculum/handoff_progress"] = torch.tensor(
            self._handoff_curriculum_progress,
            device=self.device,
        )
        extras["log"]["Curriculum/handoff_promotion_streak"] = torch.tensor(
            self._handoff_curriculum_promotion_streak,
            device=self.device,
        )
        extras["log"]["Curriculum/handoff_demotion_streak"] = torch.tensor(
            self._handoff_curriculum_demotion_streak,
            device=self.device,
        )
        if math.isfinite(self._handoff_curriculum_last_success_rate):
            extras["log"]["Curriculum/handoff_window_success_rate"] = torch.tensor(
                self._handoff_curriculum_last_success_rate,
                device=self.device,
            )
            extras["log"][
                "Curriculum/handoff_window_parallel_state_violation_fraction"
            ] = torch.tensor(
                self._handoff_curriculum_last_parallel_state_violation_fraction,
                device=self.device,
            )
            extras["log"][
                "Curriculum/handoff_window_hard_joint_limit_fraction"
            ] = torch.tensor(
                self._handoff_curriculum_last_hard_joint_limit_fraction,
                device=self.device,
            )
            extras["log"][
                "Curriculum/handoff_window_joint_limit_termination_rate"
            ] = torch.tensor(
                self._handoff_curriculum_last_joint_limit_termination_rate,
                device=self.device,
            )
            extras["log"][
                "Curriculum/handoff_window_parallel_ankle_termination_rate"
            ] = torch.tensor(
                self._handoff_curriculum_last_parallel_ankle_termination_rate,
                device=self.device,
            )
            extras["log"][
                "Curriculum/handoff_window_nonfinite_action_termination_rate"
            ] = torch.tensor(
                self._handoff_curriculum_last_nonfinite_action_termination_rate,
                device=self.device,
            )
            extras["log"]["Curriculum/handoff_window_safety_passed"] = torch.tensor(
                float(self._handoff_curriculum_last_safety_passed),
                device=self.device,
            )
        failure_probability, failure_blend, canonical_noise_scale = self._robust_reset_curriculum()
        extras["log"]["Curriculum/failure_reset_probability"] = torch.tensor(
            failure_probability, device=self.device
        )
        extras["log"]["Curriculum/failure_state_blend"] = torch.tensor(
            failure_blend, device=self.device
        )
        extras["log"]["Curriculum/canonical_noise_scale"] = torch.tensor(
            canonical_noise_scale, device=self.device
        )
        (
            random_linear_velocity,
            random_angular_velocity,
            random_joint_noise,
            random_ankle_joint_noise,
            random_uniform_blend,
            random_joint_velocity,
            random_ankle_joint_velocity,
            random_fall_difficulty,
        ) = self._random_fall_curriculum()
        for name, value in (
            ("random_fall_difficulty", random_fall_difficulty),
            ("random_fall_linear_velocity", random_linear_velocity),
            ("random_fall_angular_velocity", random_angular_velocity),
            ("random_joint_noise", random_joint_noise),
            ("random_ankle_joint_noise", random_ankle_joint_noise),
            ("random_joint_uniform_blend", random_uniform_blend),
            ("random_joint_velocity", random_joint_velocity),
            ("random_ankle_joint_velocity", random_ankle_joint_velocity),
        ):
            extras["log"][f"Curriculum/{name}"] = torch.tensor(
                value,
                device=self.device,
            )
        if self._adaptive_failure_blend.numel() > 0:
            extras["log"]["Curriculum/adaptive_failure_blend_mean"] = torch.mean(
                self._adaptive_failure_blend
            )
            extras["log"]["Curriculum/adaptive_failure_blend_min"] = torch.amin(
                self._adaptive_failure_blend
            )
            extras["log"]["Curriculum/adaptive_failure_blend_max"] = torch.amax(
                self._adaptive_failure_blend
            )
            extras["log"]["Curriculum/adaptive_failure_at_max_fraction"] = torch.mean(
                (
                    self._adaptive_failure_blend
                    >= self.cfg.amp_failure_adaptive_blend_end - 1.0e-6
                ).to(torch.float32)
            )
            extras["log"]["Curriculum/adaptive_failure_trial_mean"] = torch.mean(
                self._adaptive_failure_trials.to(torch.float32)
            )
            extras["log"]["Curriculum/adaptive_failure_promotion_streak_mean"] = torch.mean(
                self._adaptive_failure_promotion_streak.to(torch.float32)
            )
            evaluated = torch.isfinite(self._adaptive_failure_last_success_rate)
            extras["log"]["Curriculum/adaptive_failure_evaluated_fraction"] = torch.mean(
                evaluated.to(torch.float32)
            )
            if torch.any(evaluated):
                success_rates = self._adaptive_failure_last_success_rate[evaluated]
                extras["log"]["Curriculum/adaptive_failure_last_success_rate_mean"] = torch.mean(
                    success_rates
                )
                extras["log"]["Curriculum/adaptive_failure_last_success_rate_min"] = torch.amin(
                    success_rates
                )
                extras["log"]["Curriculum/adaptive_failure_last_success_rate_max"] = torch.amax(
                    success_rates
                )
        self._save_adaptive_failure_curriculum()
        return observations, rewards, terminated, truncated, extras

    def _update_handoff_success_curriculum(
        self,
        *,
        parallel_state_violation: torch.Tensor,
        hard_joint_limit_violation: torch.Tensor,
        joint_limit_terminations: torch.Tensor | None,
        parallel_ankle_terminations: torch.Tensor | None,
        nonfinite_action_terminations: torch.Tensor | None,
    ) -> None:
        if (
            not self.cfg.amp_handoff_curriculum_adaptive
            or self.cfg.amp_reset_mode != "train"
        ):
            return
        if parallel_state_violation.dtype != torch.bool:
            raise ValueError("parallel-state curriculum mask must be boolean")
        if hard_joint_limit_violation.dtype != torch.bool:
            raise ValueError("hard-joint-limit curriculum mask must be boolean")
        if parallel_state_violation.shape != hard_joint_limit_violation.shape:
            raise ValueError("handoff curriculum safety masks must have the same shape")
        self._handoff_curriculum_trials += torch.sum(self._completed_mode_counts)
        self._handoff_curriculum_successes += torch.sum(self._recovered_mode_counts)
        self._handoff_curriculum_safety_samples += parallel_state_violation.numel()
        self._handoff_curriculum_parallel_state_violations += torch.count_nonzero(
            parallel_state_violation
        )
        self._handoff_curriculum_hard_joint_limit_violations += torch.count_nonzero(
            hard_joint_limit_violation
        )
        if joint_limit_terminations is not None:
            self._handoff_curriculum_joint_limit_terminations += joint_limit_terminations
        if parallel_ankle_terminations is not None:
            self._handoff_curriculum_parallel_ankle_terminations += parallel_ankle_terminations
        if nonfinite_action_terminations is not None:
            self._handoff_curriculum_nonfinite_action_terminations += nonfinite_action_terminations
        elapsed = self.common_step_counter - self._handoff_curriculum_last_update_step
        if elapsed < self.cfg.amp_handoff_curriculum_window_steps:
            return

        trials = int(self._handoff_curriculum_trials.item())
        if trials >= self.cfg.amp_handoff_curriculum_minimum_trials:
            successes = int(self._handoff_curriculum_successes.item())
            success_rate = successes / trials
            safety_samples = max(
                int(self._handoff_curriculum_safety_samples.item()),
                1,
            )
            parallel_state_violation_fraction = (
                int(self._handoff_curriculum_parallel_state_violations.item())
                / safety_samples
            )
            hard_joint_limit_fraction = (
                int(self._handoff_curriculum_hard_joint_limit_violations.item())
                / safety_samples
            )
            joint_limit_termination_rate = (
                int(self._handoff_curriculum_joint_limit_terminations.item())
                / trials
            )
            parallel_ankle_termination_rate = (
                int(self._handoff_curriculum_parallel_ankle_terminations.item())
                / trials
            )
            nonfinite_action_termination_rate = (
                int(self._handoff_curriculum_nonfinite_action_terminations.item())
                / trials
            )
            safety_passed = (
                parallel_state_violation_fraction
                <= self.cfg.amp_handoff_curriculum_maximum_parallel_state_violation_fraction
                and hard_joint_limit_fraction
                <= self.cfg.amp_handoff_curriculum_maximum_hard_joint_limit_fraction
                and joint_limit_termination_rate
                <= self.cfg.amp_handoff_curriculum_maximum_joint_limit_termination_rate
                and parallel_ankle_termination_rate
                <= self.cfg.amp_handoff_curriculum_maximum_parallel_ankle_termination_rate
                and nonfinite_action_termination_rate == 0.0
            )
            (
                self._handoff_curriculum_progress,
                self._handoff_curriculum_promotion_streak,
                self._handoff_curriculum_demotion_streak,
            ) = mastery_curriculum_update(
                progress=self._handoff_curriculum_progress,
                promotion_streak=self._handoff_curriculum_promotion_streak,
                demotion_streak=self._handoff_curriculum_demotion_streak,
                success_rate=success_rate,
                promotion_threshold=(
                    self.cfg.amp_handoff_curriculum_success_threshold
                ),
                demotion_threshold=(
                    self.cfg.amp_handoff_curriculum_demotion_threshold
                ),
                promotion_windows=(
                    self.cfg.amp_handoff_curriculum_promotion_windows
                ),
                demotion_windows=(
                    self.cfg.amp_handoff_curriculum_demotion_windows
                ),
                promotion_step=self.cfg.amp_handoff_curriculum_promotion_step,
                demotion_step=self.cfg.amp_handoff_curriculum_demotion_step,
                promotion_allowed=safety_passed,
            )
            self._handoff_curriculum_last_success_rate = success_rate
            self._handoff_curriculum_last_parallel_state_violation_fraction = (
                parallel_state_violation_fraction
            )
            self._handoff_curriculum_last_hard_joint_limit_fraction = (
                hard_joint_limit_fraction
            )
            self._handoff_curriculum_last_joint_limit_termination_rate = (
                joint_limit_termination_rate
            )
            self._handoff_curriculum_last_parallel_ankle_termination_rate = (
                parallel_ankle_termination_rate
            )
            self._handoff_curriculum_last_nonfinite_action_termination_rate = (
                nonfinite_action_termination_rate
            )
            self._handoff_curriculum_last_safety_passed = safety_passed
        self._handoff_curriculum_trials.zero_()
        self._handoff_curriculum_successes.zero_()
        self._handoff_curriculum_safety_samples.zero_()
        self._handoff_curriculum_parallel_state_violations.zero_()
        self._handoff_curriculum_hard_joint_limit_violations.zero_()
        self._handoff_curriculum_joint_limit_terminations.zero_()
        self._handoff_curriculum_parallel_ankle_terminations.zero_()
        self._handoff_curriculum_nonfinite_action_terminations.zero_()
        self._handoff_curriculum_last_update_step = self.common_step_counter

    def collect_reference_motions(self, num_samples: int) -> torch.Tensor:
        return self._style_motion_loader.sample_reference_observations(num_samples)

    @property
    def reference_amp_frame(self) -> torch.Tensor:
        return self._motion_loader.amp_frame[self._reference_frame_index]

    @property
    def reference_amp_phase(self) -> torch.Tensor:
        return self._motion_loader.phase_at(self._reference_frame_index).unsqueeze(-1)

    @property
    def future_reference_amp_frame(self) -> torch.Tensor:
        indexes = self._motion_loader.advance_indexes(self._reference_frame_index)
        return self._motion_loader.amp_frame[indexes]

    @property
    def future_reference_amp_phase(self) -> torch.Tensor:
        indexes = self._motion_loader.advance_indexes(self._reference_frame_index)
        return self._motion_loader.phase_at(indexes).unsqueeze(-1)

    @property
    def reference_teacher_action(self) -> torch.Tensor:
        if self._motion_loader.teacher_action is None:
            raise RuntimeError("the configured AMP motion does not contain teacher actions")
        return self._motion_loader.teacher_action[self._action_reference_frame_index]

    def _robust_reset_curriculum(self) -> tuple[float, float, float]:
        transition_count = float(self.common_step_counter * self.num_envs)
        progress = min(transition_count / float(self.cfg.amp_robust_reset_curriculum_steps), 1.0)

        def interpolate(start: float, end: float) -> float:
            return start + progress * (end - start)

        return (
            interpolate(
                self.cfg.amp_failure_reset_probability_start,
                self.cfg.amp_failure_reset_probability_end,
            ),
            interpolate(
                self.cfg.amp_failure_state_blend_start,
                self.cfg.amp_failure_state_blend_end,
            ),
            interpolate(
                self.cfg.amp_canonical_noise_scale_start,
                self.cfg.amp_canonical_noise_scale_end,
            ),
        )

    def _random_fall_curriculum(
        self,
    ) -> tuple[float, float, float, float, float, float, float, float]:
        if self.cfg.amp_reset_mode == "random":
            schedule_progress = 1.0
        else:
            transition_count = float(self.common_step_counter * self.num_envs)
            schedule_progress = min(
                transition_count
                / float(self.cfg.amp_reference_reset_curriculum_steps),
                1.0,
            )

        difficulty = (
            self.cfg.amp_random_fall_difficulty_start
            + schedule_progress
            * (
                self.cfg.amp_random_fall_difficulty_end
                - self.cfg.amp_random_fall_difficulty_start
            )
        )

        def interpolate(start: float, end: float) -> float:
            return start + difficulty * (end - start)

        return (
            interpolate(
                self.cfg.amp_random_fall_linear_velocity_start,
                self.cfg.amp_random_fall_linear_velocity_end,
            ),
            interpolate(
                self.cfg.amp_random_fall_angular_velocity_start,
                self.cfg.amp_random_fall_angular_velocity_end,
            ),
            interpolate(
                self.cfg.amp_random_joint_noise_start,
                self.cfg.amp_random_joint_noise_end,
            ),
            interpolate(
                self.cfg.amp_random_ankle_joint_noise_start,
                self.cfg.amp_random_ankle_joint_noise_end,
            ),
            interpolate(
                self.cfg.amp_random_joint_uniform_blend_start,
                self.cfg.amp_random_joint_uniform_blend_end,
            ),
            interpolate(
                self.cfg.amp_random_joint_velocity_start,
                self.cfg.amp_random_joint_velocity_end,
            ),
            interpolate(
                self.cfg.amp_random_ankle_joint_velocity_start,
                self.cfg.amp_random_ankle_joint_velocity_end,
            ),
            difficulty,
        )

    def _reset_idx(self, env_ids) -> None:
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if hasattr(self, "_terminal_state_valid"):
            done = torch.zeros(
                env_ids.numel(),
                dtype=torch.bool,
                device=self.device,
            )
            for name in (
                "recovered",
                "joint_limit",
                "parallel_ankle",
                "nonfinite_action",
                "time_out",
            ):
                done |= optional_termination_mask(
                    self.termination_manager,
                    name,
                    env_ids,
                )
            self._terminal_state_valid[env_ids] = done
            terminal_ids = env_ids[done]
            if terminal_ids.numel() > 0:
                self._terminal_joint_position[terminal_ids] = self.robot.data.joint_pos[
                    terminal_ids
                ][:, self._joint_ids]
                self._terminal_joint_velocity[terminal_ids] = self.robot.data.joint_vel[
                    terminal_ids
                ][:, self._joint_ids]
                self._terminal_root_pose[terminal_ids, :3] = (
                    self.robot.data.root_pos_w[terminal_ids]
                    - self.scene.env_origins[terminal_ids]
                )
                self._terminal_root_pose[terminal_ids, 3:] = (
                    self.robot.data.root_quat_w[terminal_ids]
                )
                self._terminal_root_velocity[terminal_ids] = torch.cat(
                    (
                        self.robot.data.root_lin_vel_w[terminal_ids],
                        self.robot.data.root_ang_vel_w[terminal_ids],
                    ),
                    dim=-1,
                )
        if hasattr(self, "_reset_mode_code"):
            previous_modes = self._reset_mode_code[env_ids]
            valid = previous_modes >= 0
            if torch.any(valid):
                recovered = optional_termination_mask(
                    self.termination_manager,
                    "recovered",
                    env_ids,
                )
                for code in range(self._completed_mode_counts.numel()):
                    selected = valid & (previous_modes == code)
                    self._completed_mode_counts[code] += torch.count_nonzero(selected)
                    self._recovered_mode_counts[code] += torch.count_nonzero(selected & recovered)
                previous_reference_phase = self._reference_reset_phase[env_ids]
                reference = (
                    valid
                    & (previous_modes == 0)
                    & torch.isfinite(previous_reference_phase)
                )
                if torch.any(reference):
                    phase_bins = reference_phase_bin_indices(
                        previous_reference_phase[reference],
                        self._reference_phase_bin_count,
                    )
                    self._completed_reference_phase_counts.index_add_(
                        0,
                        phase_bins,
                        torch.ones_like(phase_bins),
                    )
                    self._recovered_reference_phase_counts.index_add_(
                        0,
                        phase_bins,
                        recovered[reference].to(torch.long),
                    )
                    clip_ids = self._reference_reset_clip_id[env_ids][reference]
                    valid_clip = (
                        (clip_ids >= 0)
                        & (clip_ids < self._completed_reference_clip_phase_counts.shape[0])
                    )
                    if torch.any(valid_clip):
                        flat_indexes = (
                            clip_ids[valid_clip] * self._reference_phase_bin_count
                            + phase_bins[valid_clip]
                        )
                        self._completed_reference_clip_phase_counts.view(-1).index_add_(
                            0,
                            flat_indexes,
                            torch.ones_like(flat_indexes),
                        )
                        self._recovered_reference_clip_phase_counts.view(-1).index_add_(
                            0,
                            flat_indexes,
                            recovered[reference][valid_clip].to(torch.long),
                        )
        if hasattr(self, "_failure_state_index"):
            self._update_adaptive_failure_curriculum(env_ids)
        super()._reset_idx(env_ids)
        if not hasattr(self, "_motion_loader"):
            return

        self._failure_state_index[env_ids] = -1
        self._reference_reset_phase[env_ids] = torch.nan
        self._reference_reset_clip_id[env_ids] = -1
        reset_mode = self.cfg.amp_reset_mode
        canonical_mode = self.cfg.amp_canonical_reset_mode
        if reset_mode == "train":
            transition_count = float(self.common_step_counter * self.num_envs)
            progress = min(transition_count / float(self.cfg.amp_reference_reset_curriculum_steps), 1.0)
            reference_probability = self.cfg.amp_reference_reset_probability_start + progress * (
                self.cfg.amp_reference_reset_probability_end - self.cfg.amp_reference_reset_probability_start
            )
            standing_probability = self.cfg.amp_standing_reset_probability_start + progress * (
                self.cfg.amp_standing_reset_probability_end - self.cfg.amp_standing_reset_probability_start
            )
            random_fall_probability = self.cfg.amp_random_fall_probability_start + progress * (
                self.cfg.amp_random_fall_probability_end - self.cfg.amp_random_fall_probability_start
            )
            failure_probability, failure_blend, canonical_noise_scale = self._robust_reset_curriculum()
            if (
                reference_probability
                + standing_probability
                + random_fall_probability
                + failure_probability
                > 1.0 + 1.0e-6
            ):
                raise ValueError("AMP reset probabilities exceed one")
            reset_sample = torch.rand(env_ids.numel(), device=self.device)
            use_reference = reset_sample < reference_probability
            use_standing = (reset_sample >= reference_probability) & (
                reset_sample < reference_probability + standing_probability
            )
            use_random_fall = (reset_sample >= reference_probability + standing_probability) & (
                reset_sample < reference_probability + standing_probability + random_fall_probability
            )
            failure_start = reference_probability + standing_probability + random_fall_probability
            use_failure = (reset_sample >= failure_start) & (
                reset_sample < failure_start + failure_probability
            )
            use_canonical = ~(use_reference | use_standing | use_random_fall | use_failure)
        elif reset_mode == "reference":
            use_reference = torch.ones(env_ids.numel(), dtype=torch.bool, device=self.device)
            use_standing = torch.zeros_like(use_reference)
            use_random_fall = torch.zeros_like(use_reference)
            use_failure = torch.zeros_like(use_reference)
            use_canonical = torch.zeros_like(use_reference)
            failure_blend = self.cfg.amp_failure_state_blend_end
            canonical_noise_scale = 1.0
        elif reset_mode == "random":
            use_reference = torch.zeros(env_ids.numel(), dtype=torch.bool, device=self.device)
            use_standing = torch.zeros_like(use_reference)
            use_random_fall = torch.ones_like(use_reference)
            use_failure = torch.zeros_like(use_reference)
            use_canonical = torch.zeros_like(use_reference)
            failure_blend = self.cfg.amp_failure_state_blend_end
            canonical_noise_scale = 1.0
        elif reset_mode == "failure":
            use_reference = torch.zeros(env_ids.numel(), dtype=torch.bool, device=self.device)
            use_standing = torch.zeros_like(use_reference)
            use_random_fall = torch.zeros_like(use_reference)
            use_failure = torch.ones_like(use_reference)
            use_canonical = torch.zeros_like(use_reference)
            failure_blend = self.cfg.amp_failure_state_blend_end
            canonical_noise_scale = 1.0
        elif reset_mode == "standing":
            use_reference = torch.zeros(env_ids.numel(), dtype=torch.bool, device=self.device)
            use_standing = torch.ones_like(use_reference)
            use_random_fall = torch.zeros_like(use_reference)
            use_failure = torch.zeros_like(use_reference)
            use_canonical = torch.zeros_like(use_reference)
            failure_blend = self.cfg.amp_failure_state_blend_end
            canonical_noise_scale = 1.0
        elif reset_mode in {"faceup", "facedown", "side_left", "side_right", "canonical"}:
            use_reference = torch.zeros(env_ids.numel(), dtype=torch.bool, device=self.device)
            use_standing = torch.zeros_like(use_reference)
            use_random_fall = torch.zeros_like(use_reference)
            use_failure = torch.zeros_like(use_reference)
            use_canonical = torch.ones_like(use_reference)
            canonical_mode = reset_mode
            failure_blend = self.cfg.amp_failure_state_blend_end
            canonical_noise_scale = 1.0
        else:
            raise ValueError(f"Unsupported AMP reset mode: {reset_mode}")

        reference_env_ids = env_ids[use_reference]
        if reference_env_ids.numel() > 0:
            self._write_reference_resets(reference_env_ids)
        standing_env_ids = env_ids[use_standing]
        if standing_env_ids.numel() > 0:
            self._write_standing_resets(standing_env_ids)
        random_fall_env_ids = env_ids[use_random_fall]
        if random_fall_env_ids.numel() > 0:
            self._write_random_fall_resets(random_fall_env_ids)
        failure_env_ids = env_ids[use_failure]
        if failure_env_ids.numel() > 0:
            self._write_failure_resets(failure_env_ids, failure_blend)
        canonical_env_ids = env_ids[use_canonical]
        if canonical_env_ids.numel() > 0:
            self._write_canonical_resets(canonical_env_ids, canonical_mode, canonical_noise_scale)
        self._reset_mode_code[reference_env_ids] = 0
        self._reset_mode_code[standing_env_ids] = 1
        self._reset_mode_code[random_fall_env_ids] = 2
        self._reset_mode_code[failure_env_ids] = 3
        self._reset_mode_code[canonical_env_ids] = 4
        self._action_reference_frame_index[env_ids] = self._reference_frame_index[env_ids]

    def _random_yaw(self, count: int) -> torch.Tensor:
        yaw = (2.0 * torch.rand(count, device=self.device) - 1.0) * np.pi
        rotation = torch.zeros((count, 4), device=self.device)
        rotation[:, 0] = torch.cos(0.5 * yaw)
        rotation[:, 3] = torch.sin(0.5 * yaw)
        return rotation

    def _write_reference_resets(self, env_ids: torch.Tensor) -> None:
        minimum_phase = self.cfg.amp_reference_phase_minimum
        maximum_phase = self.cfg.amp_reference_phase_maximum
        if not 0.0 <= minimum_phase <= maximum_phase <= 1.0:
            raise ValueError("AMP reference phase range must satisfy 0 <= minimum <= maximum <= 1")
        count = env_ids.numel()
        clip_ids = self._sample_reference_clip_ids(count)
        clip_start = self._motion_loader.clip_start_indexes[clip_ids]
        clip_end = self._motion_loader.clip_end_indexes[clip_ids]
        clip_span = clip_end - clip_start
        offsets = sample_reference_phase_offsets(
            clip_span,
            minimum_phase,
            maximum_phase,
            self.cfg.amp_reference_phase_bin_edges,
            self.cfg.amp_reference_phase_bin_weights,
        )
        indexes = clip_start + offsets
        terminal_probability = self.cfg.amp_reference_terminal_reset_probability
        terminal_reset = torch.rand(env_ids.numel(), device=self.device) < terminal_probability
        if torch.any(terminal_reset):
            terminal_count = int(torch.count_nonzero(terminal_reset).item())
            terminal_clip_ids = self._sample_reference_clip_ids(terminal_count)
            clip_ids[terminal_reset] = terminal_clip_ids
            clip_end = self._motion_loader.clip_end_indexes[terminal_clip_ids]
            window_frames = max(round(self.cfg.amp_reference_terminal_reset_window_s / self.step_dt), 1)
            clip_start = torch.maximum(
                self._motion_loader.clip_start_indexes[terminal_clip_ids],
                clip_end - window_frames + 1,
            )
            window_size = clip_end - clip_start + 1
            offsets = torch.floor(torch.rand(terminal_count, device=self.device) * window_size).to(torch.long)
            indexes[terminal_reset] = clip_start + offsets
        self._reference_frame_index[env_ids] = indexes
        self._reference_reset_phase[env_ids] = self._motion_loader.phase_at(indexes)
        self._reference_reset_clip_id[env_ids] = clip_ids
        state = self._motion_loader.states_at(indexes)
        yaw_rotation = self._random_yaw(env_ids.numel())
        root_pose = torch.empty((env_ids.numel(), 7), dtype=torch.float32, device=self.device)
        root_pose[:, :2] = self.scene.env_origins[env_ids, :2]
        root_pose[:, 2] = (
            self.scene.env_origins[env_ids, 2]
            + state.root_position[:, 2]
            + self.cfg.amp_reference_reset_height_offset
        )
        root_pose[:, 3:] = quat_multiply(yaw_rotation, state.root_rotation)
        root_velocity = torch.cat(
            (
                quat_rotate(yaw_rotation, state.root_linear_velocity),
                quat_rotate(yaw_rotation, state.root_angular_velocity),
            ),
            dim=-1,
        )
        joint_position = self._prepare_reset_joint_position(state.joint_position)
        joint_velocity = state.joint_velocity * self.cfg.amp_reference_reset_joint_velocity_scale
        self._write_state(env_ids, joint_position, joint_velocity, root_pose, root_velocity)

    def _sample_reference_clip_ids(self, count: int) -> torch.Tensor:
        clip_count = self._motion_loader.clip_start_indexes.numel()
        return sample_reference_clip_ids(
            clip_count=clip_count,
            sample_count=count,
            device=self.device,
            allowed_clip_ids=self.cfg.amp_reference_clip_indices,
            clip_weights=self.cfg.amp_reference_clip_weights,
        )

    def _write_random_fall_resets(self, env_ids: torch.Tensor) -> None:
        count = env_ids.numel()
        indexes = self._motion_loader.sample_state_indexes(count)
        self._reference_frame_index[env_ids] = indexes
        state = self._motion_loader.states_at(indexes)
        (
            linear_limit,
            angular_limit,
            joint_noise_limit,
            ankle_noise_limit,
            uniform_blend_limit,
            joint_velocity_limit,
            ankle_joint_velocity_limit,
            _,
        ) = self._random_fall_curriculum()
        joint_position = state.joint_position.clone()
        joint_noise_scale = torch.full_like(joint_position, joint_noise_limit)
        joint_noise_scale[:, [14, 15, 20, 21]] = ankle_noise_limit
        joint_position += (
            2.0 * torch.rand_like(joint_position) - 1.0
        ) * joint_noise_scale
        uniform_position = self._random_safe_joint_minimum + torch.rand_like(
            joint_position
        ) * (
            self._random_safe_joint_maximum
            - self._random_safe_joint_minimum
        )
        uniform_blend = uniform_blend_limit * torch.rand(
            (count, 1),
            dtype=joint_position.dtype,
            device=self.device,
        )
        joint_position = self._prepare_reset_joint_position(
            torch.lerp(joint_position, uniform_position, uniform_blend),
            position_minimum=self._random_safe_joint_minimum,
            position_maximum=self._random_safe_joint_maximum,
            parallel_motor_margin=self.cfg.amp_random_reset_parallel_motor_margin,
        )
        joint_velocity_scale = torch.full_like(
            joint_position,
            joint_velocity_limit,
        )
        joint_velocity_scale[:, [14, 15, 20, 21]] = ankle_joint_velocity_limit
        joint_velocity = (
            2.0 * torch.rand_like(joint_position) - 1.0
        ) * joint_velocity_scale
        joint_velocity = limit_reset_joint_velocity(
            joint_position,
            joint_velocity,
            self._random_safe_joint_minimum,
            self._random_safe_joint_maximum,
            self.cfg.amp_random_joint_velocity_safety_horizon_s,
        )

        root_pose = torch.empty((count, 7), dtype=torch.float32, device=self.device)
        root_pose[:, :2] = self.scene.env_origins[env_ids, :2]
        minimum_height, maximum_height = self.cfg.amp_random_fall_height_range
        root_pose[:, 2] = self.scene.env_origins[env_ids, 2] + minimum_height + (
            maximum_height - minimum_height
        ) * torch.rand(count, device=self.device)
        root_pose[:, 3:] = torch.nn.functional.normalize(torch.randn((count, 4), device=self.device), dim=-1)
        root_velocity = torch.cat(
            (
                (2.0 * torch.rand((count, 3), device=self.device) - 1.0) * linear_limit,
                (2.0 * torch.rand((count, 3), device=self.device) - 1.0) * angular_limit,
            ),
            dim=-1,
        )
        self._write_state(
            env_ids,
            joint_position,
            joint_velocity,
            root_pose,
            root_velocity,
        )

    def _write_failure_resets(self, env_ids: torch.Tensor, blend: float) -> None:
        if self._failure_states is None:
            raise RuntimeError("AMP failure reset requested without a loaded failure-state dataset")
        count = env_ids.numel()
        sample_count = self._failure_states["joint_position"].shape[0]
        indexes = torch.randint(sample_count, (count,), device=self.device)
        self._failure_state_index[env_ids] = indexes
        if self.cfg.amp_failure_adaptive_curriculum:
            blend_value = self._adaptive_failure_blend[indexes].unsqueeze(-1)
        else:
            blend_value = torch.full((count, 1), blend, dtype=torch.float32, device=self.device)
        reference_indexes = self._motion_loader.clip_start_indexes[0].repeat(count)
        self._reference_frame_index[env_ids] = reference_indexes
        reference = self._motion_loader.states_at(reference_indexes)
        joint_position = self._prepare_reset_joint_position(
            torch.lerp(
                reference.joint_position,
                self._failure_states["joint_position"][indexes],
                blend_value,
            ),
            ankle_neutral_fraction=0.0,
        )
        joint_velocity = self._failure_states["joint_velocity"][indexes] * blend_value
        root_pose = self._failure_states["root_pose"][indexes].clone()
        reference_root_position = torch.zeros((count, 3), dtype=torch.float32, device=self.device)
        reference_root_position[:, 2] = reference.root_position[:, 2]
        root_pose[:, :3] = torch.lerp(reference_root_position, root_pose[:, :3], blend_value)
        root_pose[:, 3:] = quat_slerp_batch(
            reference.root_rotation,
            root_pose[:, 3:],
            blend_value,
        )
        root_pose[:, :3] += self.scene.env_origins[env_ids]
        root_velocity = self._failure_states["root_velocity"][indexes] * blend_value
        self._write_state(env_ids, joint_position, joint_velocity, root_pose, root_velocity)

    def _update_adaptive_failure_curriculum(self, env_ids) -> None:
        if not self.cfg.amp_failure_adaptive_curriculum or self._adaptive_failure_blend.numel() == 0:
            return
        env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        previous_index = self._failure_state_index[env_ids]
        valid = previous_index >= 0
        if not torch.any(valid):
            return

        done = torch.zeros(env_ids.numel(), dtype=torch.bool, device=self.device)
        for name in (
            "recovered",
            "joint_limit",
            "parallel_ankle",
            "nonfinite_action",
            "time_out",
        ):
            done |= optional_termination_mask(
                self.termination_manager,
                name,
                env_ids,
            )
        valid &= done
        if not torch.any(valid):
            return

        indexes = previous_index[valid]
        recovered = optional_termination_mask(
            self.termination_manager,
            "recovered",
            env_ids,
        )[valid]
        self._adaptive_failure_trials.index_add_(
            0,
            indexes,
            torch.ones_like(indexes),
        )
        self._adaptive_failure_successes.index_add_(
            0,
            indexes,
            recovered.to(torch.long),
        )
        ready = self._adaptive_failure_trials >= self.cfg.amp_failure_adaptive_min_trials
        if not torch.any(ready):
            return

        ready_indexes = torch.nonzero(ready, as_tuple=False).squeeze(-1)
        success_rate = (
            self._adaptive_failure_successes[ready_indexes].to(torch.float32)
            / self._adaptive_failure_trials[ready_indexes].to(torch.float32)
        )
        self._adaptive_failure_last_success_rate[ready_indexes] = success_rate
        successful_window = success_rate >= self.cfg.amp_failure_adaptive_success_threshold
        failed_window = success_rate <= self.cfg.amp_failure_adaptive_demotion_threshold
        streak = self._adaptive_failure_promotion_streak[ready_indexes]
        streak = torch.where(successful_window, streak + 1, torch.zeros_like(streak))
        self._adaptive_failure_promotion_streak[ready_indexes] = streak

        promote = successful_window & (
            streak >= self.cfg.amp_failure_adaptive_promotion_windows
        )
        demote = failed_window
        if torch.any(promote):
            promote_indexes = ready_indexes[promote]
            self._adaptive_failure_blend[promote_indexes] = torch.clamp(
                self._adaptive_failure_blend[promote_indexes]
                + self.cfg.amp_failure_adaptive_promotion,
                max=self.cfg.amp_failure_adaptive_blend_end,
            )
            self._adaptive_failure_promotion_streak[promote_indexes] = 0
        if torch.any(demote):
            demote_indexes = ready_indexes[demote]
            self._adaptive_failure_blend[demote_indexes] = torch.clamp(
                self._adaptive_failure_blend[demote_indexes]
                - self.cfg.amp_failure_adaptive_demotion,
                min=self.cfg.amp_failure_adaptive_blend_start,
            )
            self._adaptive_failure_promotion_streak[demote_indexes] = 0

        self._adaptive_failure_trials[ready_indexes] = 0
        self._adaptive_failure_successes[ready_indexes] = 0

    def _load_adaptive_failure_curriculum(self) -> None:
        path_value = self.cfg.amp_failure_adaptive_state_resume
        if not path_value:
            return
        path = Path(path_value)
        required = {
            "blend",
            "trials",
            "successes",
            "promotion_streak",
        }
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != required:
                raise ValueError(
                    f"{path}: unexpected adaptive curriculum fields "
                    f"{sorted(archive.files)}, expected {sorted(required)}"
                )
            expected_shape = self._adaptive_failure_blend.shape
            for name in required:
                if archive[name].shape != expected_shape:
                    raise ValueError(
                        f"{path}: {name} has shape {archive[name].shape}, expected {expected_shape}"
                    )
            self._adaptive_failure_blend.copy_(
                torch.as_tensor(archive["blend"], dtype=torch.float32, device=self.device)
            )
            if not self.cfg.amp_failure_adaptive_reset_statistics:
                for name, target in (
                    ("trials", self._adaptive_failure_trials),
                    ("successes", self._adaptive_failure_successes),
                    ("promotion_streak", self._adaptive_failure_promotion_streak),
                ):
                    target.copy_(torch.as_tensor(archive[name], dtype=torch.long, device=self.device))
        suffix = " (blend only; reset window statistics)" if self.cfg.amp_failure_adaptive_reset_statistics else ""
        print(f"[INFO] Resumed adaptive failure curriculum: {path}{suffix}")

    def _save_adaptive_failure_curriculum(self) -> None:
        if not self.cfg.amp_failure_adaptive_curriculum:
            return
        path_value = self.cfg.amp_failure_adaptive_state_output
        interval = self.cfg.amp_failure_adaptive_state_interval
        if (
            not path_value
            or interval < 1
            or self.common_step_counter == self._adaptive_failure_state_last_saved_step
            or self.common_step_counter % interval != 0
        ):
            return
        path = Path(path_value)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                blend=self._adaptive_failure_blend.detach().cpu().numpy(),
                trials=self._adaptive_failure_trials.detach().cpu().numpy(),
                successes=self._adaptive_failure_successes.detach().cpu().numpy(),
                promotion_streak=self._adaptive_failure_promotion_streak.detach().cpu().numpy(),
            )
        temporary.replace(path)
        self._adaptive_failure_state_last_saved_step = self.common_step_counter

    def _write_standing_resets(self, env_ids: torch.Tensor) -> None:
        count = env_ids.numel()
        self._reference_frame_index[env_ids] = self._motion_loader.clip_end_indexes[0]
        joint_position = torch.tensor(GOAL_POSITION, dtype=torch.float32, device=self.device).repeat(count, 1)
        joint_position += 0.02 * (2.0 * torch.rand_like(joint_position) - 1.0)
        joint_position = self._prepare_reset_joint_position(joint_position)

        root_pose = torch.empty((count, 7), dtype=torch.float32, device=self.device)
        root_pose[:, :2] = self.scene.env_origins[env_ids, :2]
        root_pose[:, :2] += 0.03 * (2.0 * torch.rand((count, 2), device=self.device) - 1.0)
        root_pose[:, 2] = self.scene.env_origins[env_ids, 2] + self.cfg.commands.recovery.standing_height
        root_pose[:, 3:] = self._random_yaw(count)
        root_velocity = torch.zeros((count, 6), dtype=torch.float32, device=self.device)
        self._write_state(env_ids, joint_position, torch.zeros_like(joint_position), root_pose, root_velocity)

    def _write_canonical_resets(
        self,
        env_ids: torch.Tensor,
        mode: str,
        noise_scale: float = 1.0,
    ) -> None:
        if self.cfg.amp_canonical_from_reference:
            self._write_reference_aligned_canonical_resets(env_ids, noise_scale)
            return

        count = env_ids.numel()
        self._reference_frame_index[env_ids] = self._motion_loader.clip_start_indexes[0]
        joint_position = torch.tensor(
            self.cfg.commands.recovery.initial_joint_position,
            dtype=torch.float32,
            device=self.device,
        ).repeat(count, 1)
        if mode == "side_left":
            joint_position = mirror_bilateral_joint_position(
                joint_position,
                self.cfg.amp_bilateral_left_indices,
                self.cfg.amp_bilateral_right_indices,
                self.cfg.amp_bilateral_mirror_signs,
            )
        joint_position = self._prepare_reset_joint_position(joint_position)
        root_pose = torch.empty((count, 7), dtype=torch.float32, device=self.device)
        root_pose[:, :2] = self.scene.env_origins[env_ids, :2]
        if mode in {"side_left", "side_right"}:
            root_height = self.cfg.amp_canonical_side_height
        elif mode == "canonical":
            root_height = self.cfg.amp_canonical_random_height
        else:
            root_height = self.cfg.commands.recovery.initial_height
        root_pose[:, 2] = self.scene.env_origins[env_ids, 2] + root_height
        orientation_noise = self.cfg.commands.recovery.orientation_noise * noise_scale
        roll_noise = (2.0 * torch.rand(count, device=self.device) - 1.0) * orientation_noise
        pitch_noise = (2.0 * torch.rand(count, device=self.device) - 1.0) * orientation_noise
        if mode == "faceup":
            roll = roll_noise
            pitch = torch.full((count,), -0.5 * np.pi, device=self.device) + pitch_noise
        elif mode == "facedown":
            roll = roll_noise
            pitch = torch.full((count,), 0.5 * np.pi, device=self.device) + pitch_noise
        elif mode == "side_left":
            roll = torch.full((count,), 0.5 * np.pi, device=self.device) + roll_noise
            pitch = pitch_noise
        elif mode == "side_right":
            roll = torch.full((count,), -0.5 * np.pi, device=self.device) + roll_noise
            pitch = pitch_noise
        else:
            roll = roll_noise
            pitch_sign = torch.where(
                torch.rand(count, device=self.device) < 0.5,
                torch.full((count,), -1.0, device=self.device),
                torch.ones(count, device=self.device),
            )
            pitch = 0.5 * np.pi * pitch_sign + pitch_noise
        yaw = (2.0 * torch.rand(count, device=self.device) - 1.0) * np.pi
        root_pose[:, 3:] = math_utils.quat_from_euler_xyz(roll, pitch, yaw)
        self._write_state(
            env_ids,
            joint_position,
            torch.zeros_like(joint_position),
            root_pose,
            torch.zeros((count, 6), device=self.device),
        )

    def _write_reference_aligned_canonical_resets(
        self,
        env_ids: torch.Tensor,
        noise_scale: float = 1.0,
    ) -> None:
        count = env_ids.numel()
        indexes = self._motion_loader.clip_start_indexes[0].repeat(count)
        self._reference_frame_index[env_ids] = indexes
        state = self._motion_loader.states_at(indexes)

        joint_noise_scale = torch.full_like(
            state.joint_position,
            self.cfg.amp_canonical_joint_noise * noise_scale,
        )
        joint_noise_scale[:, [14, 15, 20, 21]] = (
            self.cfg.amp_canonical_ankle_joint_noise * noise_scale
        )
        joint_noise = joint_noise_scale * (2.0 * torch.rand_like(state.joint_position) - 1.0)
        joint_position = self._prepare_reset_joint_position(state.joint_position + joint_noise)

        root_pose = torch.empty((count, 7), dtype=torch.float32, device=self.device)
        root_pose[:, :2] = self.scene.env_origins[env_ids, :2]
        root_pose[:, :2] += self.cfg.amp_canonical_root_xy_noise * noise_scale * (
            2.0 * torch.rand((count, 2), device=self.device) - 1.0
        )
        root_pose[:, 2] = self.scene.env_origins[env_ids, 2] + state.root_position[:, 2]
        root_pose[:, 2] += self.cfg.amp_canonical_height_noise * noise_scale * (
            2.0 * torch.rand(count, device=self.device) - 1.0
        )

        orientation_noise = self.cfg.amp_canonical_orientation_noise * noise_scale
        roll = orientation_noise * (2.0 * torch.rand(count, device=self.device) - 1.0)
        pitch = orientation_noise * (2.0 * torch.rand(count, device=self.device) - 1.0)
        zero = torch.zeros(count, device=self.device)
        tilt_rotation = math_utils.quat_from_euler_xyz(roll, pitch, zero)
        yaw_rotation = self._random_yaw(count)
        root_pose[:, 3:] = quat_multiply(yaw_rotation, quat_multiply(tilt_rotation, state.root_rotation))

        velocity_scale = self.cfg.amp_canonical_velocity_scale
        root_velocity = torch.cat(
            (
                quat_rotate(yaw_rotation, state.root_linear_velocity),
                quat_rotate(yaw_rotation, state.root_angular_velocity),
            ),
            dim=-1,
        )
        self._write_state(
            env_ids,
            joint_position,
            state.joint_velocity * velocity_scale,
            root_pose,
            root_velocity * velocity_scale,
        )

    def _prepare_reset_joint_position(
        self,
        joint_position: torch.Tensor,
        ankle_neutral_fraction: float | None = None,
        position_minimum: torch.Tensor | None = None,
        position_maximum: torch.Tensor | None = None,
        parallel_motor_margin: float = 0.0,
    ) -> torch.Tensor:
        minimum = (
            self._safe_joint_minimum
            if position_minimum is None
            else position_minimum
        )
        maximum = (
            self._safe_joint_maximum
            if position_maximum is None
            else position_maximum
        )
        joint_position = torch.clamp(joint_position, minimum, maximum)
        neutral_fraction = (
            self.cfg.amp_reset_ankle_neutral_fraction
            if ankle_neutral_fraction is None
            else ankle_neutral_fraction
        )
        for foot, indexes in enumerate(((14, 15), (20, 21))):
            pair = torch.tensor(indexes, dtype=torch.long, device=self.device)
            neutral = self._parallel.serial_zero[2 * foot : 2 * foot + 2].to(torch.float32).expand(
                joint_position.shape[0], -1
            )
            projected, _, feasible = self._parallel.project(
                joint_position[:, pair],
                neutral,
                foot,
                minimum_motor_margin=parallel_motor_margin,
            )
            if not torch.all(feasible):
                raise RuntimeError(f"AMP reset ankle projection failed for foot {foot}")
            interior = (1.0 - neutral_fraction) * projected + neutral_fraction * neutral
            projected, _, feasible = self._parallel.project(
                interior,
                neutral,
                foot,
                minimum_motor_margin=parallel_motor_margin,
            )
            if not torch.all(feasible):
                raise RuntimeError(f"AMP reset ankle interior projection failed for foot {foot}")
            joint_position[:, pair] = projected
        return joint_position

    def _write_state(
        self,
        env_ids: torch.Tensor,
        joint_position: torch.Tensor,
        joint_velocity: torch.Tensor,
        root_pose: torch.Tensor,
        root_velocity: torch.Tensor,
    ) -> None:
        self.robot.write_joint_state_to_sim(
            joint_position,
            joint_velocity,
            joint_ids=self._joint_ids,
            env_ids=env_ids,
        )
        self.robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root_velocity, env_ids=env_ids)

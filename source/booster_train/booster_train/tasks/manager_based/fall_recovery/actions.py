from __future__ import annotations

from dataclasses import MISSING
from typing import Any

import torch
from isaaclab.envs.mdp.actions.actions_cfg import JointPositionActionCfg
from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction
from isaaclab.managers.action_manager import ActionTerm
from isaaclab.utils import configclass

from .joint_order import deployment_joint_ids
from .parallel_ankle import K1ParallelAnkleKinematics, select_safe_projected_target
from .recovery_math import (
    limit_joint_position_target,
    sanitize_nonfinite_action_inputs,
    scaled_joint_position_delta,
)


class K1ParallelJointPositionAction(JointPositionAction):
    """Normalized K1 joint targets with optional parallel-ankle constraints."""

    cfg: "K1ParallelJointPositionActionCfg"

    def __init__(self, cfg: "K1ParallelJointPositionActionCfg", env) -> None:
        super().__init__(cfg, env)
        self._env = env
        expected_ids = deployment_joint_ids(self._asset.joint_names, cfg.expected_joint_names)
        if self._joint_names != cfg.expected_joint_names or self._joint_ids != expected_ids:
            raise ValueError(
                "K1 recovery action order does not match the deployment order:\n"
                f"resolved={self._joint_names} {self._joint_ids}\n"
                f"expected={cfg.expected_joint_names} {expected_ids}"
            )
        if len(self._joint_names) != 22:
            raise ValueError("K1 recovery action must own all 22 joints")

        def tensor(values: list[float]) -> torch.Tensor:
            return torch.tensor(values, dtype=torch.float32, device=self.device).unsqueeze(0)

        self._minimum = tensor(cfg.position_minimum)
        self._maximum = tensor(cfg.position_maximum)
        if cfg.position_target_margin < 0.0 or cfg.position_braking_horizon_s < 0.0:
            raise ValueError("K1 joint target margin and braking horizon must be non-negative")
        self._target_minimum = self._minimum + cfg.position_target_margin
        self._target_maximum = self._maximum - cfg.position_target_margin
        if torch.any(self._target_minimum > self._target_maximum):
            raise ValueError("K1 joint target margin collapses a logical joint range")
        if cfg.position_center is None:
            self._center = 0.5 * (self._minimum + self._maximum)
        else:
            self._center = tensor(cfg.position_center)
        if cfg.position_scale is None:
            self._scale = 0.5 * (self._maximum - self._minimum)
        else:
            self._scale = tensor(cfg.position_scale)
        limit_tolerance = 1.0e-6
        if cfg.reference_command_name is None and not cfg.relative_to_current and (
            torch.any(self._center - self._scale < self._minimum - limit_tolerance)
            or torch.any(self._center + self._scale > self._maximum + limit_tolerance)
        ):
            raise ValueError("K1 action center/scale exceeds a logical joint limit")
        if torch.any(self._scale <= 0.0):
            raise ValueError("K1 action scale must be positive")
        if not cfg.normalize_input and cfg.clip is None:
            raise ValueError(
                "unbounded K1 policy actions require processed joint-delta limits"
            )
        if not torch.all(torch.isfinite(self._center)):
            raise ValueError("K1 action fallback center must be finite")
        self._stiffness = tensor(cfg.stiffness)
        self._damping = tensor(cfg.damping)
        self._torque_limit = tensor(cfg.command_torque_limit)
        self._parallel = (
            K1ParallelAnkleKinematics(cfg.parallel_ankle, device=self.device, dtype=torch.float32)
            if cfg.parallel_ankle is not None
            else None
        )
        self._parallel_target_margin = (
            float(cfg.parallel_ankle.get("motor_target_margin", 0.0))
            if cfg.parallel_ankle is not None
            else 0.0
        )
        self._ankle_pairs = ((14, 15), (20, 21))

        self._target_reduction = torch.zeros(self.num_envs, device=self.device)
        self._parallel_motor_utilization = torch.zeros(self.num_envs, 4, device=self.device)
        self._parallel_motor_velocity_ratio = torch.zeros_like(self._parallel_motor_utilization)
        self._parallel_motor_margin = torch.zeros_like(self._parallel_motor_utilization)
        neutral = (
            self._parallel.serial_zero.to(dtype=torch.float32).unsqueeze(0)
            if self._parallel is not None
            else torch.zeros(1, 4, dtype=torch.float32, device=self.device)
        )
        self._last_feasible_ankle_target = neutral.repeat(self.num_envs, 1)
        if self._parallel is not None:
            for foot in range(len(self._ankle_pairs)):
                motor_slice = slice(2 * foot, 2 * foot + 2)
                if not torch.all(
                    self._parallel.is_feasible(
                        self._last_feasible_ankle_target[:, motor_slice],
                        foot,
                        self._parallel_target_margin,
                    )
                ):
                    raise ValueError(f"K1 neutral ankle target is infeasible for foot {foot}")
        self._parallel_target_fallback = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._nonfinite_action = torch.zeros(
            self.num_envs,
            dtype=torch.bool,
            device=self.device,
        )
        self._joint_limit_braking = torch.zeros(
            (self.num_envs, len(self._joint_names)),
            dtype=torch.bool,
            device=self.device,
        )
        self._joint_limit_target_reduction = torch.zeros(
            self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self._pending_nonfinite_policy_input = torch.zeros_like(self._nonfinite_action)

    @property
    def target_reduction(self) -> torch.Tensor:
        return self._target_reduction

    @property
    def parallel_motor_utilization(self) -> torch.Tensor:
        return self._parallel_motor_utilization

    @property
    def parallel_motor_velocity_ratio(self) -> torch.Tensor:
        return self._parallel_motor_velocity_ratio

    @property
    def parallel_motor_margin(self) -> torch.Tensor:
        return self._parallel_motor_margin

    @property
    def parallel_target_fallback(self) -> torch.Tensor:
        return self._parallel_target_fallback

    @property
    def nonfinite_action(self) -> torch.Tensor:
        return self._nonfinite_action

    @property
    def joint_limit_braking(self) -> torch.Tensor:
        return self._joint_limit_braking

    @property
    def joint_limit_target_reduction(self) -> torch.Tensor:
        return self._joint_limit_target_reduction

    def mark_nonfinite_policy_input(self, failed: torch.Tensor) -> None:
        if failed.shape != (self.num_envs,) or failed.dtype != torch.bool:
            raise ValueError("non-finite policy mask must be one boolean per environment")
        self._pending_nonfinite_policy_input |= failed.to(device=self.device)

    @property
    def deployment_joint_position(self) -> torch.Tensor:
        return self._asset.data.joint_pos[:, self._joint_ids]

    def set_runtime_impedance(
        self,
        stiffness: list[float] | torch.Tensor,
        damping: list[float] | torch.Tensor,
        torque_limit: list[float] | torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> None:
        """Switch per-environment gains when recovery hands off to locomotion."""

        if self._stiffness.shape[0] == 1:
            self._stiffness = self._stiffness.expand(self.num_envs, -1).clone()
            self._damping = self._damping.expand(self.num_envs, -1).clone()
            self._torque_limit = self._torque_limit.expand(self.num_envs, -1).clone()
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = env_ids.to(device=self.device, dtype=torch.long)

        def values_for_envs(values: list[float] | torch.Tensor) -> torch.Tensor:
            result = torch.as_tensor(
                values,
                dtype=torch.float32,
                device=self.device,
            )
            if result.shape == (22,):
                return result.unsqueeze(0).expand(env_ids.numel(), -1)
            if result.shape != (env_ids.numel(), 22):
                raise ValueError(
                    "runtime impedance must contain 22 values or one row per environment"
                )
            return result

        runtime_stiffness = values_for_envs(stiffness)
        runtime_damping = values_for_envs(damping)
        runtime_torque_limit = values_for_envs(torque_limit)
        if (
            torch.any(runtime_stiffness <= 0.0)
            or torch.any(runtime_damping < 0.0)
            or torch.any(runtime_torque_limit <= 0.0)
        ):
            raise ValueError("runtime impedance values are invalid")
        self._stiffness[env_ids] = runtime_stiffness
        self._damping[env_ids] = runtime_damping
        self._torque_limit[env_ids] = runtime_torque_limit

        deployment_index = {
            asset_joint_id: index
            for index, asset_joint_id in enumerate(self._joint_ids)
        }
        for actuator in self._asset.actuators.values():
            if isinstance(actuator.joint_indices, slice):
                asset_joint_ids = list(range(self._asset.num_joints))
            else:
                asset_joint_ids = actuator.joint_indices.tolist()
            source_columns = torch.tensor(
                [deployment_index[index] for index in asset_joint_ids],
                dtype=torch.long,
                device=self.device,
            )
            actuator_columns = torch.arange(
                len(asset_joint_ids),
                device=self.device,
            )
            actuator.stiffness[
                env_ids[:, None],
                actuator_columns,
            ] = runtime_stiffness[:, source_columns]
            actuator.damping[
                env_ids[:, None],
                actuator_columns,
            ] = runtime_damping[:, source_columns]
            if actuator.is_implicit_model:
                self._asset.write_joint_stiffness_to_sim(
                    runtime_stiffness[:, source_columns],
                    joint_ids=asset_joint_ids,
                    env_ids=env_ids,
                )
                self._asset.write_joint_damping_to_sim(
                    runtime_damping[:, source_columns],
                    joint_ids=asset_joint_ids,
                    env_ids=env_ids,
                )

    @staticmethod
    def _solve_2x2(matrix: torch.Tensor, vector: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a, b = matrix[:, 0, 0], matrix[:, 0, 1]
        c, d = matrix[:, 1, 0], matrix[:, 1, 1]
        determinant = a * d - b * c
        nonsingular = torch.abs(determinant) > 1e-5
        safe_determinant = torch.where(nonsingular, determinant, torch.ones_like(determinant))
        result = torch.stack(
            (
                (d * vector[:, 0] - b * vector[:, 1]) / safe_determinant,
                (-c * vector[:, 0] + a * vector[:, 1]) / safe_determinant,
            ),
            dim=-1,
        )
        return torch.where(nonsingular.unsqueeze(-1), result, torch.zeros_like(result)), nonsingular

    def process_actions(self, actions: torch.Tensor) -> None:
        self._parallel_target_fallback.zero_()
        current = self._asset.data.joint_pos[:, self._joint_ids]
        velocity = self._asset.data.joint_vel[:, self._joint_ids]
        if self.cfg.relative_to_current:
            center = current
        elif self.cfg.reference_command_name is None:
            center = self._center
        else:
            command = self._env.command_manager.get_term(self.cfg.reference_command_name)
            center = command.joint_pos[:, self._joint_ids]
        actions, current, velocity, center, nonfinite = sanitize_nonfinite_action_inputs(
            actions,
            current,
            velocity,
            center,
            self._center,
        )
        nonfinite |= self._pending_nonfinite_policy_input
        self._pending_nonfinite_policy_input.zero_()
        actions = torch.where(
            nonfinite.unsqueeze(-1),
            torch.zeros_like(actions),
            actions,
        )
        self._nonfinite_action.copy_(nonfinite)
        self._raw_actions[:] = actions
        center = torch.clamp(center, self._minimum, self._maximum)
        delta_minimum = self._clip[:, :, 0] if self.cfg.clip is not None else None
        delta_maximum = self._clip[:, :, 1] if self.cfg.clip is not None else None
        delta = scaled_joint_position_delta(
            actions,
            self._scale,
            normalize_input=self.cfg.normalize_input,
            delta_minimum=delta_minimum,
            delta_maximum=delta_maximum,
        )
        target = torch.clamp(
            center + delta,
            self._minimum,
            self._maximum,
        )
        target = torch.where(
            nonfinite.unsqueeze(-1),
            torch.clamp(current, self._minimum, self._maximum),
            target,
        )

        requested_target = target.clone()
        target, braking = limit_joint_position_target(
            target,
            current,
            velocity,
            self._target_minimum,
            self._target_maximum,
            self.cfg.position_braking_horizon_s,
        )
        self._joint_limit_braking.copy_(braking)
        self._joint_limit_target_reduction.copy_(
            torch.linalg.vector_norm(requested_target - target, dim=-1)
        )
        projection_reduction = torch.zeros(self.num_envs, device=self.device)

        # Match motion FDR's per-cycle position-error clamp before the low-level PD loop.
        error_limit = self._torque_limit / self._stiffness
        target = current + torch.clamp(target - current, -error_limit, error_limit)

        if self._parallel is None:
            self._processed_actions = target
            self._target_reduction = torch.linalg.vector_norm(requested_target - target, dim=-1)
            return

        for foot, indexes in enumerate(self._ankle_pairs):
            pair = torch.tensor(indexes, dtype=torch.long, device=self.device)
            pair_current = current[:, pair]
            pair_velocity = velocity[:, pair]
            pair_target = target[:, pair]
            motor_slice = slice(2 * foot, 2 * foot + 2)
            fallback_target = self._last_feasible_ankle_target[:, motor_slice]
            finite_state = torch.all(torch.isfinite(pair_current), dim=-1) & torch.all(
                torch.isfinite(pair_velocity), dim=-1
            )
            jacobian_input = torch.where(
                finite_state.unsqueeze(-1),
                pair_current,
                fallback_target,
            )
            jacobian, reachable = self._parallel.jacobian(jacobian_input, foot)
            fallback_jacobian, fallback_reachable = self._parallel.jacobian(
                fallback_target,
                foot,
            )
            neutral_target = self._parallel.serial_zero[
                2 * foot : 2 * foot + 2
            ].to(dtype=torch.float32).expand_as(fallback_target)
            neutral_jacobian, neutral_reachable = self._parallel.jacobian(
                neutral_target,
                foot,
            )
            if not torch.all(neutral_reachable):
                raise RuntimeError(f"K1 neutral ankle target is unreachable for foot {foot}")
            fallback_target = torch.where(
                fallback_reachable.unsqueeze(-1),
                fallback_target,
                neutral_target,
            )
            fallback_jacobian = torch.where(
                fallback_reachable[:, None, None],
                fallback_jacobian,
                neutral_jacobian,
            )
            self._last_feasible_ankle_target[:, motor_slice] = fallback_target
            usable_state = finite_state & reachable
            pair_current_model = torch.where(
                usable_state.unsqueeze(-1),
                pair_current,
                fallback_target,
            )
            pair_velocity_model = torch.where(
                usable_state.unsqueeze(-1),
                pair_velocity,
                torch.zeros_like(pair_velocity),
            )
            jacobian = torch.where(
                usable_state[:, None, None],
                jacobian,
                fallback_jacobian,
            ).to(dtype=torch.float32)

            serial_torque = (
                self._stiffness[:, pair] * (pair_target - pair_current_model)
                - self._damping[:, pair] * pair_velocity_model
            )
            motor_torque, nonsingular = self._solve_2x2(jacobian.transpose(-1, -2), serial_torque)
            motor_velocity = torch.einsum("bij,bj->bi", jacobian, pair_velocity_model)

            effort = float(self.cfg.parallel_ankle["motor_effort_limit"])
            knee_velocity = float(self.cfg.parallel_ankle["motor_knee_velocity"])
            velocity_limit = float(self.cfg.parallel_ankle["motor_velocity_limit"])
            command_limit = float(self.cfg.parallel_ankle["motor_command_torque_limit"])
            speed = torch.abs(motor_velocity)
            linear_limit = effort * (velocity_limit - speed) / (velocity_limit - knee_velocity)
            speed_limit = torch.where(
                speed <= knee_velocity,
                effort,
                torch.clamp(linear_limit, min=0.0, max=effort),
            )
            available_torque = torch.clamp(speed_limit, max=command_limit)
            scale = torch.min(
                torch.clamp(available_torque / (torch.abs(motor_torque) + 1e-6), max=1.0),
                dim=-1,
            ).values
            scale = torch.where(nonsingular, scale, torch.zeros_like(scale))

            limited_serial_torque = serial_torque * scale.unsqueeze(-1)
            limited_target = (
                pair_current_model
                + (limited_serial_torque + self._damping[:, pair] * pair_velocity_model)
                / self._stiffness[:, pair]
            )
            projected, reduction, feasible = self._parallel.project(
                limited_target,
                pair_current_model,
                foot,
                minimum_motor_margin=self._parallel_target_margin,
            )
            numerically_valid = (
                usable_state
                & nonsingular
                & torch.all(torch.isfinite(serial_torque), dim=-1)
                & torch.all(torch.isfinite(motor_torque), dim=-1)
            )
            projected, reduction, accepted = select_safe_projected_target(
                limited_target,
                projected,
                reduction,
                feasible & numerically_valid,
                fallback_target,
            )
            target[:, pair] = projected
            projection_reduction += reduction
            self._parallel_target_fallback |= ~accepted
            self._last_feasible_ankle_target[:, motor_slice] = torch.where(
                accepted.unsqueeze(-1),
                projected,
                fallback_target,
            )

            applied_motor_torque = torch.where(
                accepted.unsqueeze(-1),
                motor_torque * scale.unsqueeze(-1),
                torch.zeros_like(motor_torque),
            )
            self._parallel_motor_utilization[:, motor_slice] = torch.abs(applied_motor_torque) / torch.clamp(
                available_torque, min=1e-6
            )
            self._parallel_motor_velocity_ratio[:, motor_slice] = speed / velocity_limit
            self._parallel_motor_margin[:, motor_slice] = self._parallel.motor_margin(projected, foot).to(
                dtype=torch.float32
            )

        self._processed_actions = target
        self._target_reduction = projection_reduction + torch.linalg.vector_norm(requested_target - target, dim=-1)

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._target_reduction[env_ids] = 0.0
        self._parallel_motor_utilization[env_ids] = 0.0
        self._parallel_motor_velocity_ratio[env_ids] = 0.0
        self._parallel_motor_margin[env_ids] = 0.0
        self._parallel_target_fallback[env_ids] = False
        self._nonfinite_action[env_ids] = False
        self._joint_limit_braking[env_ids] = False
        self._joint_limit_target_reduction[env_ids] = 0.0
        self._pending_nonfinite_policy_input[env_ids] = False
        if self._parallel is not None:
            self._last_feasible_ankle_target[env_ids] = self._parallel.serial_zero.to(
                dtype=torch.float32
            )
        else:
            self._last_feasible_ankle_target[env_ids] = 0.0


@configclass
class K1ParallelJointPositionActionCfg(JointPositionActionCfg):
    class_type: type[ActionTerm] = K1ParallelJointPositionAction

    expected_joint_names: list[str] = MISSING
    position_minimum: list[float] = MISSING
    position_maximum: list[float] = MISSING
    position_center: list[float] | None = None
    position_scale: list[float] | None = None
    stiffness: list[float] = MISSING
    damping: list[float] = MISSING
    command_torque_limit: list[float] = MISSING
    parallel_ankle: dict[str, Any] | None = MISSING
    reference_command_name: str | None = None
    relative_to_current: bool = False
    normalize_input: bool = True
    position_target_margin: float = 0.0
    position_braking_horizon_s: float = 0.0


@configclass
class K1SerialJointPositionActionCfg(K1ParallelJointPositionActionCfg):
    """K1 serial-URDF action without physical parallel-ankle constraints."""

    parallel_ankle: dict[str, Any] | None = None

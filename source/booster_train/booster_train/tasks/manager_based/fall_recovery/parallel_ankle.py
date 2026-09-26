from __future__ import annotations

import math
from typing import Any

import torch


def _angle_difference(value: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return torch.remainder(value - reference + math.pi, 2.0 * math.pi) - math.pi


def select_safe_projected_target(
    desired: torch.Tensor,
    projected: torch.Tensor,
    reduction: torch.Tensor,
    feasible: torch.Tensor,
    fallback: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace a numerically invalid projection with a known feasible target."""

    if desired.shape != projected.shape or desired.shape != fallback.shape:
        raise ValueError("parallel target tensors must have matching shapes")
    if reduction.shape != desired.shape[:-1] or feasible.shape != reduction.shape:
        raise ValueError("parallel projection status has an invalid shape")
    if not torch.all(torch.isfinite(fallback)):
        raise ValueError("parallel projection fallback must be finite")

    accepted = (
        feasible
        & torch.all(torch.isfinite(desired), dim=-1)
        & torch.all(torch.isfinite(projected), dim=-1)
        & torch.isfinite(reduction)
    )
    safe_target = torch.where(accepted.unsqueeze(-1), projected, fallback)
    finite_desired = torch.where(torch.isfinite(desired), desired, fallback)
    fallback_reduction = torch.linalg.vector_norm(finite_desired - fallback, dim=-1)
    safe_reduction = torch.where(accepted, reduction, fallback_reduction)
    return safe_target, safe_reduction, accepted


class K1ParallelAnkleKinematics:
    """Batched K1 serial-foot to physical-crank kinematics."""

    def __init__(
        self,
        config: dict[str, Any],
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        self.config = config
        self.device = torch.device(device)
        self.dtype = dtype
        self.clockwise = bool(config["ik_clockwise"])
        self.dist_joint_lr = float(config["dist_joint_lr"])
        self.crank_radius = float(config["crank_radius"])
        self.heel_width = float(config["heel_width"])
        self.jacobian_step = float(config["jacobian_step"])
        self.projection_iterations = int(config["projection_iterations"])

        def tensor(key: str) -> torch.Tensor:
            return torch.tensor(config[key], dtype=self.dtype, device=self.device)

        self.link_lengths = tensor("link_lengths")
        self.ankle_in_limb = tensor("ankle_in_limb")
        self.ankle_shift = tensor("ankle_shift")
        self.heel_in_foot = tensor("heel_in_foot")
        self.parallel_zero = tensor("joint_parallel_zero_position")
        self.serial_zero = tensor("joint_serial_zero_position")
        self.mirror_sign = torch.tensor(
            [-1.0 if value else 1.0 for value in config["mirror"]],
            dtype=self.dtype,
            device=self.device,
        )
        self.motor_minimum = tensor("motor_position_minimum")
        self.motor_maximum = tensor("motor_position_maximum")

    def _as_geometry_dtype(self, value: torch.Tensor) -> torch.Tensor:
        return value.to(device=self.device, dtype=self.dtype)

    def raw_inverse(self, orientation: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        orientation = self._as_geometry_dtype(orientation)
        if orientation.shape[-1] != 2:
            raise ValueError("parallel ankle orientation must end in pitch, roll")
        pitch, roll = orientation.unbind(dim=-1)
        cp, sp = torch.cos(pitch), torch.sin(pitch)
        cr, sr = torch.cos(roll), torch.sin(roll)
        rotation = torch.stack(
            (
                cp,
                sp * sr,
                sp * cr,
                torch.zeros_like(cp),
                cr,
                -sr,
                -sp,
                cp * sr,
                cp * cr,
            ),
            dim=-1,
        ).reshape(*orientation.shape[:-1], 3, 3)

        result = []
        reachable = torch.ones(orientation.shape[:-1], dtype=torch.bool, device=self.device)
        for motor, (side, vertical_offset) in enumerate(((1.0, 0.0), (-1.0, -self.dist_joint_lr))):
            lateral = torch.tensor(
                (0.0, side * 0.5 * self.heel_width, 0.0),
                dtype=self.dtype,
                device=self.device,
            )
            motor_axis = -self.ankle_in_limb + lateral
            motor_axis = motor_axis.clone()
            motor_axis[2] += vertical_offset
            foot_anchor = self.heel_in_foot + lateral
            heel_anchor = self.ankle_shift + torch.matmul(rotation, foot_anchor)
            vector = motor_axis - heel_anchor
            x, z = vector[..., 0], vector[..., 2]
            radius = self.crank_radius
            projection = (self.link_lengths[motor].square() - radius * radius - vector.square().sum(dim=-1)) / (
                2.0 * radius
            )
            xz_squared = x.square() + z.square()
            radicand = xz_squared - projection.square()
            tolerance = 1e-12 * torch.maximum(torch.ones_like(xz_squared), xz_squared)
            motor_reachable = (xz_squared > 1e-16) & (radicand >= -tolerance)
            reachable &= motor_reachable
            root = torch.sqrt(torch.clamp(radicand, min=0.0))
            if self.clockwise:
                root = -root
            denominator = torch.clamp(xz_squared, min=1e-16)
            sin_angle = (projection * x + z * root) / denominator
            cos_angle = (projection * z - x * root) / denominator
            result.append(torch.atan2(sin_angle, cos_angle))
        return torch.stack(result, dim=-1), reachable

    def serial_to_motor(self, serial: torch.Tensor, foot: int) -> tuple[torch.Tensor, torch.Tensor]:
        if foot not in (0, 1):
            raise ValueError("foot must be 0 (left) or 1 (right)")
        serial = self._as_geometry_dtype(serial)
        offset = self.serial_zero[2 * foot : 2 * foot + 2]
        internal = serial - offset
        internal = torch.stack((internal[..., 0], internal[..., 1] * self.mirror_sign[foot]), dim=-1)
        raw, reachable = self.raw_inverse(internal)
        return raw - self.parallel_zero[2 * foot : 2 * foot + 2], reachable

    def jacobian(self, serial: torch.Tensor, foot: int) -> tuple[torch.Tensor, torch.Tensor]:
        serial = self._as_geometry_dtype(serial)
        columns = []
        reachable = torch.ones(serial.shape[:-1], dtype=torch.bool, device=self.device)
        for column in range(2):
            delta = torch.zeros_like(serial)
            delta[..., column] = self.jacobian_step
            high, high_ok = self.serial_to_motor(serial + delta, foot)
            low, low_ok = self.serial_to_motor(serial - delta, foot)
            columns.append(_angle_difference(high, low) / (2.0 * self.jacobian_step))
            reachable &= high_ok & low_ok
        return torch.stack(columns, dim=-1), reachable

    def is_feasible(
        self,
        serial: torch.Tensor,
        foot: int,
        minimum_motor_margin: float = 0.0,
    ) -> torch.Tensor:
        if minimum_motor_margin < 0.0:
            raise ValueError("minimum_motor_margin must be non-negative")
        motor, reachable = self.serial_to_motor(serial, foot)
        minimum = self.motor_minimum[2 * foot : 2 * foot + 2] + minimum_motor_margin
        maximum = self.motor_maximum[2 * foot : 2 * foot + 2] - minimum_motor_margin
        return reachable & torch.all(motor >= minimum, dim=-1) & torch.all(motor <= maximum, dim=-1)

    def project(
        self,
        desired: torch.Tensor,
        current: torch.Tensor,
        foot: int,
        minimum_motor_margin: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        original_dtype = desired.dtype
        desired = self._as_geometry_dtype(desired)
        current = self._as_geometry_dtype(current)
        neutral = self.serial_zero[2 * foot : 2 * foot + 2].expand_as(current)
        current_ok = self.is_feasible(current, foot, minimum_motor_margin)
        start = torch.where(current_ok.unsqueeze(-1), current, neutral)
        desired_ok = self.is_feasible(desired, foot, minimum_motor_margin)

        low = torch.zeros(desired.shape[:-1], dtype=self.dtype, device=self.device)
        high = torch.ones_like(low)
        for _ in range(self.projection_iterations):
            middle = 0.5 * (low + high)
            candidate = start + middle.unsqueeze(-1) * (desired - start)
            candidate_ok = self.is_feasible(candidate, foot, minimum_motor_margin)
            low = torch.where(candidate_ok, middle, low)
            high = torch.where(candidate_ok, high, middle)
        projected = torch.where(
            desired_ok.unsqueeze(-1),
            desired,
            start + low.unsqueeze(-1) * (desired - start),
        )
        reduction = torch.linalg.vector_norm(projected - desired, dim=-1)
        final_ok = self.is_feasible(projected, foot, minimum_motor_margin)
        return projected.to(original_dtype), reduction.to(original_dtype), final_ok

    def motor_margin(self, serial: torch.Tensor, foot: int) -> torch.Tensor:
        motor, reachable = self.serial_to_motor(serial, foot)
        minimum = self.motor_minimum[2 * foot : 2 * foot + 2]
        maximum = self.motor_maximum[2 * foot : 2 * foot + 2]
        margin = torch.minimum(motor - minimum, maximum - motor)
        return torch.where(reachable.unsqueeze(-1), margin, torch.full_like(margin, -1.0))

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch


def quat_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def quat_multiply(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lw, lx, ly, lz = lhs.unbind(-1)
    rw, rx, ry, rz = rhs.unbind(-1)
    return torch.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        dim=-1,
    )


def quat_rotate(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    axis = quaternion[..., 1:]
    twice_cross = 2.0 * torch.linalg.cross(axis, vector, dim=-1)
    return vector + quaternion[..., :1] * twice_cross + torch.linalg.cross(axis, twice_cross, dim=-1)


def quat_slerp_batch(
    start: torch.Tensor,
    end: torch.Tensor,
    blend: float | torch.Tensor,
) -> torch.Tensor:
    """Spherically interpolate equally shaped batches of scalar-first quaternions."""
    if start.shape != end.shape or start.shape[-1] != 4:
        raise ValueError("quaternion batches must have matching (..., 4) shapes")
    blend_tensor = torch.as_tensor(blend, dtype=start.dtype, device=start.device)
    if torch.any((blend_tensor < 0.0) | (blend_tensor > 1.0)):
        raise ValueError("quaternion blend must be in [0, 1]")
    while blend_tensor.ndim < start.ndim:
        blend_tensor = blend_tensor.unsqueeze(-1)

    start = torch.nn.functional.normalize(start, dim=-1)
    end = torch.nn.functional.normalize(end, dim=-1)
    dot = torch.sum(start * end, dim=-1, keepdim=True)
    end = torch.where(dot < 0.0, -end, end)
    dot = torch.abs(dot).clamp(max=1.0)
    linear = dot > 0.9995
    angle = torch.acos(dot)
    sine = torch.sin(angle)
    spherical = (
        torch.sin((1.0 - blend_tensor) * angle) / torch.clamp(sine, min=1.0e-7) * start
        + torch.sin(blend_tensor * angle) / torch.clamp(sine, min=1.0e-7) * end
    )
    interpolated = torch.where(
        linear,
        (1.0 - blend_tensor) * start + blend_tensor * end,
        spherical,
    )
    return torch.nn.functional.normalize(interpolated, dim=-1)


def yaw_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quaternion.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    result = torch.zeros_like(quaternion)
    result[..., 0] = torch.cos(0.5 * yaw)
    result[..., 3] = torch.sin(0.5 * yaw)
    return result


def quaternion_to_tangent_and_normal(quaternion: torch.Tensor) -> torch.Tensor:
    tangent = torch.zeros_like(quaternion[..., :3])
    normal = torch.zeros_like(tangent)
    tangent[..., 0] = 1.0
    normal[..., 2] = 1.0
    return torch.cat((quat_rotate(quaternion, tangent), quat_rotate(quaternion, normal)), dim=-1)


def compute_amp_frame(
    joint_position: torch.Tensor,
    joint_velocity: torch.Tensor,
    root_position: torch.Tensor,
    root_rotation: torch.Tensor,
    root_linear_velocity: torch.Tensor,
    root_angular_velocity: torch.Tensor,
    key_body_position: torch.Tensor,
) -> torch.Tensor:
    """Build one heading-invariant AMP frame in the configured joint order."""
    inverse_heading = quat_conjugate(yaw_quaternion(root_rotation))
    local_rotation = quat_multiply(inverse_heading, root_rotation)
    local_linear_velocity = quat_rotate(inverse_heading, root_linear_velocity)
    local_angular_velocity = quat_rotate(inverse_heading, root_angular_velocity)
    inverse_key_heading = inverse_heading.unsqueeze(-2).expand(-1, key_body_position.shape[-2], -1)
    local_key_position = quat_rotate(inverse_key_heading, key_body_position - root_position.unsqueeze(-2))
    return torch.cat(
        (
            joint_position,
            joint_velocity,
            root_position[..., 2:3],
            quaternion_to_tangent_and_normal(local_rotation),
            local_linear_velocity,
            local_angular_velocity,
            local_key_position.flatten(start_dim=-2),
        ),
        dim=-1,
    )


class ReferenceState(NamedTuple):
    joint_position: torch.Tensor
    joint_velocity: torch.Tensor
    root_position: torch.Tensor
    root_rotation: torch.Tensor
    root_linear_velocity: torch.Tensor
    root_angular_velocity: torch.Tensor


def sample_reference_clip_ids(
    *,
    clip_count: int,
    sample_count: int,
    device: torch.device | str,
    allowed_clip_ids: list[int] | tuple[int, ...] = (),
    clip_weights: list[float] | tuple[float, ...] = (),
) -> torch.Tensor:
    """Sample clips with optional global weights and an allowed-clip subset."""
    if clip_count < 1 or sample_count < 0:
        raise ValueError("clip count must be positive and sample count non-negative")
    allowed = list(allowed_clip_ids) if allowed_clip_ids else list(range(clip_count))
    if len(set(allowed)) != len(allowed):
        raise ValueError("AMP reference clip indexes must be unique")
    if not allowed or any(index < 0 or index >= clip_count for index in allowed):
        raise ValueError(
            f"AMP reference clip indexes {allowed} are outside [0, {clip_count})"
        )
    allowed_tensor = torch.tensor(allowed, dtype=torch.long, device=device)
    if not clip_weights:
        selected = torch.randint(
            allowed_tensor.numel(),
            (sample_count,),
            device=device,
        )
        return allowed_tensor[selected]
    if len(clip_weights) != clip_count:
        raise ValueError(
            "AMP reference clip weights must match the complete motion pool"
        )
    weights = torch.tensor(clip_weights, dtype=torch.float32, device=device)
    if torch.any(~torch.isfinite(weights)) or torch.any(weights < 0.0):
        raise ValueError("AMP reference clip weights must be finite and non-negative")
    allowed_weights = weights[allowed_tensor]
    if torch.sum(allowed_weights) <= 0.0:
        raise ValueError("allowed AMP reference clips must have positive total weight")
    selected = torch.multinomial(allowed_weights, sample_count, replacement=True)
    return allowed_tensor[selected]


def sample_reference_phase_offsets(
    clip_span: torch.Tensor,
    minimum_phase: float,
    maximum_phase: float,
    phase_bin_edges: list[float] | tuple[float, ...] = (),
    phase_bin_weights: list[float] | tuple[float, ...] = (),
) -> torch.Tensor:
    """Sample discrete frame offsets, optionally weighting contiguous phase bins."""
    if clip_span.ndim != 1 or torch.any(clip_span < 0):
        raise ValueError("clip spans must be a non-negative one-dimensional tensor")
    if not 0.0 <= minimum_phase <= maximum_phase <= 1.0:
        raise ValueError(
            "AMP reference phase range must satisfy 0 <= minimum <= maximum <= 1"
        )
    if bool(phase_bin_edges) != bool(phase_bin_weights):
        raise ValueError("phase-bin edges and weights must be configured together")

    if not phase_bin_edges:
        span = clip_span.to(torch.float32)
        minimum_offset = torch.ceil(span * minimum_phase).to(torch.long)
        maximum_offset = torch.floor(span * maximum_phase).to(torch.long)
    else:
        if len(phase_bin_edges) != len(phase_bin_weights) + 1:
            raise ValueError(
                "phase-bin edges must contain one more value than weights"
            )
        edges = torch.tensor(
            phase_bin_edges,
            dtype=torch.float32,
            device=clip_span.device,
        )
        weights = torch.tensor(
            phase_bin_weights,
            dtype=torch.float32,
            device=clip_span.device,
        )
        if torch.any(~torch.isfinite(edges)) or torch.any(~torch.isfinite(weights)):
            raise ValueError("phase-bin edges and weights must be finite")
        if torch.any(edges[1:] <= edges[:-1]):
            raise ValueError("phase-bin edges must be strictly increasing")
        if (
            abs(float(edges[0].item()) - minimum_phase) > 1.0e-6
            or abs(float(edges[-1].item()) - maximum_phase) > 1.0e-6
        ):
            raise ValueError(
                "phase-bin edges must span the configured reference phase range"
            )
        if torch.any(weights < 0.0) or torch.sum(weights) <= 0.0:
            raise ValueError(
                "phase-bin weights must be non-negative with positive total weight"
            )
        selected_bins = torch.multinomial(
            weights,
            clip_span.numel(),
            replacement=True,
        )
        selected_minimum = edges[selected_bins]
        selected_maximum = edges[selected_bins + 1]
        span = clip_span.to(torch.float32)
        minimum_offset = torch.ceil(span * selected_minimum).to(torch.long)
        maximum_offset = torch.ceil(span * selected_maximum).to(torch.long) - 1
        last_bin = selected_bins == len(phase_bin_weights) - 1
        maximum_offset[last_bin] = torch.floor(
            span[last_bin] * selected_maximum[last_bin]
        ).to(torch.long)

    if torch.any(maximum_offset < minimum_offset):
        raise ValueError("a configured reference phase bin contains no trajectory frames")
    window_size = maximum_offset - minimum_offset + 1
    return minimum_offset + torch.floor(
        torch.rand(clip_span.numel(), device=clip_span.device) * window_size
    ).to(torch.long)


class K1AmpMotionLoader:
    """Load full-body Isaac trajectories and sample AMP frames without crossing clip boundaries."""

    REQUIRED_FIELDS = (
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "joint_names",
        "body_names",
    )

    def __init__(
        self,
        motion_files: list[str],
        joint_names: list[str],
        amp_joint_names: list[str],
        root_body_name: str,
        key_body_names: tuple[str, ...],
        history_length: int,
        expected_fps: float,
        device: torch.device | str,
    ) -> None:
        if history_length < 1:
            raise ValueError("AMP history_length must be positive")
        self.device = torch.device(device)
        self.history_length = history_length
        self.expected_fps = expected_fps

        joint_positions: list[torch.Tensor] = []
        joint_velocities: list[torch.Tensor] = []
        root_positions: list[torch.Tensor] = []
        root_rotations: list[torch.Tensor] = []
        root_linear_velocities: list[torch.Tensor] = []
        root_angular_velocities: list[torch.Tensor] = []
        teacher_actions: list[torch.Tensor] = []
        teacher_action_presence: list[bool] = []
        teacher_targets: list[torch.Tensor] = []
        teacher_target_masks: list[torch.Tensor] = []
        amp_frames: list[torch.Tensor] = []
        valid_indexes: list[torch.Tensor] = []
        frame_clip_starts: list[torch.Tensor] = []
        frame_clip_ends: list[torch.Tensor] = []
        clip_starts: list[int] = []
        clip_ends: list[int] = []
        frame_offset = 0

        for motion_file in motion_files:
            path = Path(motion_file)
            if not path.is_file():
                raise FileNotFoundError(f"AMP motion does not exist: {path}")
            with np.load(path, allow_pickle=False) as data:
                missing = [field for field in self.REQUIRED_FIELDS if field not in data]
                if missing:
                    raise ValueError(f"{path}: missing AMP fields {missing}")
                fps = float(np.asarray(data["fps"]).reshape(-1)[0])
                if not np.isclose(fps, expected_fps):
                    raise ValueError(f"{path}: expected {expected_fps} Hz, got {fps} Hz")

                source_joint_names = data["joint_names"].tolist()
                source_body_names = data["body_names"].tolist()
                joint_indexes = self._resolve_names(path, source_joint_names, joint_names, "joint")
                amp_joint_indexes = self._resolve_names(path, joint_names, amp_joint_names, "AMP joint")
                root_body_index = self._resolve_names(path, source_body_names, [root_body_name], "body")[0]
                key_body_indexes = self._resolve_names(path, source_body_names, list(key_body_names), "body")

                joint_position = self._tensor(data["joint_pos"][:, joint_indexes])
                joint_velocity = self._tensor(data["joint_vel"][:, joint_indexes])
                body_position = self._tensor(data["body_pos_w"])
                body_rotation = self._tensor(data["body_quat_w"])
                body_linear_velocity = self._tensor(data["body_lin_vel_w"])
                body_angular_velocity = self._tensor(data["body_ang_vel_w"])
                has_teacher_action = "teacher_normalized_action" in data
                teacher_action_presence.append(has_teacher_action)
                if has_teacher_action:
                    teacher_action = self._tensor(data["teacher_normalized_action"])
                has_teacher_target = "teacher_target" in data
                if has_teacher_target:
                    teacher_target = self._tensor(data["teacher_target"][:, joint_indexes])

            frame_count = joint_position.shape[0]
            if frame_count < history_length:
                raise ValueError(f"{path}: {frame_count} frames cannot provide history length {history_length}")
            root_position = body_position[:, root_body_index]
            root_rotation = body_rotation[:, root_body_index]
            root_linear_velocity = body_linear_velocity[:, root_body_index]
            root_angular_velocity = body_angular_velocity[:, root_body_index]
            amp_frame = compute_amp_frame(
                joint_position[:, amp_joint_indexes],
                joint_velocity[:, amp_joint_indexes],
                root_position,
                root_rotation,
                root_linear_velocity,
                root_angular_velocity,
                body_position[:, key_body_indexes],
            )

            joint_positions.append(joint_position)
            joint_velocities.append(joint_velocity)
            root_positions.append(root_position)
            root_rotations.append(root_rotation)
            root_linear_velocities.append(root_linear_velocity)
            root_angular_velocities.append(root_angular_velocity)
            if has_teacher_action:
                if teacher_action.shape != (frame_count, len(joint_names)):
                    raise ValueError(
                        f"{path}: teacher_normalized_action must have shape "
                        f"({frame_count}, {len(joint_names)})"
                    )
                teacher_actions.append(teacher_action)
            if has_teacher_target:
                if teacher_target.shape != (frame_count, len(joint_names)):
                    raise ValueError(
                        f"{path}: teacher_target must have shape "
                        f"({frame_count}, {len(joint_names)}) after joint reordering"
                    )
                if not torch.all(torch.isfinite(teacher_target)):
                    raise ValueError(f"{path}: teacher_target contains non-finite values")
                teacher_targets.append(teacher_target)
            else:
                teacher_targets.append(torch.zeros_like(joint_position))
            teacher_target_masks.append(
                torch.full(
                    (frame_count,),
                    has_teacher_target,
                    dtype=torch.bool,
                    device=self.device,
                )
            )
            amp_frames.append(amp_frame)
            valid_indexes.append(torch.arange(history_length - 1, frame_count, device=self.device) + frame_offset)
            clip_start = frame_offset
            clip_end = frame_offset + frame_count - 1
            clip_starts.append(clip_start)
            clip_ends.append(clip_end)
            frame_clip_starts.append(
                torch.full((frame_count,), clip_start, dtype=torch.long, device=self.device)
            )
            frame_clip_ends.append(torch.full((frame_count,), clip_end, dtype=torch.long, device=self.device))
            frame_offset += frame_count
            print(f"[INFO] Loaded AMP motion {path}: {frame_count} frames at {fps:g} Hz")

        self.joint_position = torch.cat(joint_positions)
        self.joint_velocity = torch.cat(joint_velocities)
        self.root_position = torch.cat(root_positions)
        self.root_rotation = torch.cat(root_rotations)
        self.root_linear_velocity = torch.cat(root_linear_velocities)
        self.root_angular_velocity = torch.cat(root_angular_velocities)
        if any(teacher_action_presence) and not all(teacher_action_presence):
            print(
                "[INFO] AMP motion pool has mixed teacher_normalized_action coverage; "
                "stored normalized actions are disabled; physical teacher targets remain available"
            )
        self.teacher_action = (
            torch.cat(teacher_actions) if all(teacher_action_presence) else None
        )
        self.teacher_target = torch.cat(teacher_targets)
        self.teacher_target_mask = torch.cat(teacher_target_masks)
        self.amp_frame = torch.cat(amp_frames)
        self.valid_indexes = torch.cat(valid_indexes)
        self.frame_clip_start = torch.cat(frame_clip_starts)
        self.frame_clip_end = torch.cat(frame_clip_ends)
        self.clip_start_indexes = torch.tensor(clip_starts, dtype=torch.long, device=self.device)
        self.clip_end_indexes = torch.tensor(clip_ends, dtype=torch.long, device=self.device)
        self.amp_frame_size = self.amp_frame.shape[-1]

    @staticmethod
    def _resolve_names(path: Path, source: list[str], requested: list[str], kind: str) -> list[int]:
        missing = [name for name in requested if name not in source]
        if missing:
            raise ValueError(f"{path}: missing AMP {kind} names {missing}; available={source}")
        return [source.index(name) for name in requested]

    def _tensor(self, array: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(np.asarray(array), dtype=torch.float32, device=self.device)

    def sample_reference_observations(self, sample_count: int) -> torch.Tensor:
        selected = torch.randint(self.valid_indexes.numel(), (sample_count,), device=self.device)
        current_indexes = self.valid_indexes[selected]
        history_offsets = torch.arange(self.history_length, device=self.device)
        history_indexes = current_indexes.unsqueeze(-1) - history_offsets.unsqueeze(0)
        return self.amp_frame[history_indexes].flatten(start_dim=1)

    def sample_state_indexes(self, sample_count: int) -> torch.Tensor:
        return torch.randint(self.joint_position.shape[0], (sample_count,), device=self.device)

    def states_at(self, indexes: torch.Tensor) -> ReferenceState:
        return ReferenceState(
            self.joint_position[indexes],
            self.joint_velocity[indexes],
            self.root_position[indexes],
            self.root_rotation[indexes],
            self.root_linear_velocity[indexes],
            self.root_angular_velocity[indexes],
        )

    def sample_states(self, sample_count: int) -> ReferenceState:
        return self.states_at(self.sample_state_indexes(sample_count))

    def advance_indexes(self, indexes: torch.Tensor) -> torch.Tensor:
        return torch.minimum(indexes + 1, self.frame_clip_end[indexes])

    def phase_at(self, indexes: torch.Tensor) -> torch.Tensor:
        start = self.frame_clip_start[indexes]
        end = self.frame_clip_end[indexes]
        return (indexes - start).to(torch.float32) / torch.clamp((end - start).to(torch.float32), min=1.0)

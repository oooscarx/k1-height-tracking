from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional

from .hardware_config import K1_HARDWARE_CONFIG


MOTION_DIR = Path(__file__).with_name("motions")

ACTOR_TENSOR_SHAPES = {
    "obs_encoder.0.weight": (256, 100),
    "obs_encoder.0.bias": (256,),
    "obs_encoder.2.weight": (128, 256),
    "obs_encoder.2.bias": (128,),
    "obs_encoder.4.weight": (128, 128),
    "obs_encoder.4.bias": (128,),
    "actor.0.weight": (128, 128),
    "actor.0.bias": (128,),
    "actor.2.weight": (22, 128),
    "actor.2.bias": (22,),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class NativeTeacherStep:
    trajectory_index: torch.Tensor
    trajectory_phase: torch.Tensor
    observation: torch.Tensor
    action: torch.Tensor
    raw_target: torch.Tensor
    target: torch.Tensor
    normalized_action: torch.Tensor
    training_normalized_action: torch.Tensor
    done: torch.Tensor


class K1NativeRecoveryTeacher:
    """Batched Torch reproduction of the deployed K1 recovery actor."""

    def __init__(self, direction: str, num_envs: int, device: torch.device | str) -> None:
        config = K1_HARDWARE_CONFIG
        native = config["native_teacher"]
        if direction not in native["direction_values"]:
            raise ValueError(f"unsupported native recovery direction: {direction}")

        self.direction = direction
        self.num_envs = num_envs
        self.device = torch.device(device)
        self.period = 1.0 / float(config["policy_rate_hz"])
        self.lead_in = float(native["trajectory_lead_in_s"])
        self.wait_time = float(native["wait_time_s"])
        self.prepare_duration = float(native["prepare_duration_s"])
        self.parallel_ankle_zero_duration = float(native["parallel_ankle_zero_duration_s"])
        self.prepare_velocity_limit = float(native["prepare_velocity_limit"])
        self.velocity_scale = float(native["joint_velocity_scale"])
        self.direction_value = float(native["direction_values"][direction])
        self.action_minimum, self.action_maximum = (float(value) for value in native["action_clip"])
        self.zero_action_ids = torch.tensor(
            native["zero_action_joint_ids"], dtype=torch.long, device=self.device
        )
        self.parallel_ankle_ids = torch.tensor(
            native["parallel_ankle_joint_ids"], dtype=torch.long, device=self.device
        )
        self.reference_position = self._tensor(native["reference_position"]).unsqueeze(0)
        self.position_minimum = self._tensor(config["position_minimum"]).unsqueeze(0)
        self.position_maximum = self._tensor(config["position_maximum"]).unsqueeze(0)
        self.action_center = 0.5 * (self.position_minimum + self.position_maximum)
        self.action_scale = 0.5 * (self.position_maximum - self.position_minimum)
        self.training_action_center = self._tensor(native["training_action_center"]).unsqueeze(0)
        self.training_action_scale = self._tensor(native["training_action_scale"]).unsqueeze(0)

        actor_path = MOTION_DIR / native["actor_file"]
        self.actor = self._load_actor(actor_path, native["actor_sha256"])
        reference_path = MOTION_DIR / native[f"{direction}_reference_file"]
        trajectory = self._load_trajectory(reference_path, native[f"{direction}_reference_sha256"])
        self.trajectory_joint = trajectory["joint"]
        self.trajectory_gravity = trajectory["gravity"]
        self.trajectory_height = trajectory["height"]
        self.frame_count = self.trajectory_joint.shape[0]
        self.trajectory_duration = self.lead_in + self.frame_count * self.period
        self.duration = self.trajectory_duration + self.wait_time

        self.elapsed = torch.zeros(num_envs, dtype=torch.float32, device=self.device)
        self.previous_action = torch.zeros((num_envs, 22), dtype=torch.float32, device=self.device)

    def _tensor(self, value) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _load_actor(self, path: Path, expected_sha256: str) -> dict[str, torch.Tensor]:
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"{path}: SHA256 {actual_sha256} does not match {expected_sha256}")
        with np.load(path, allow_pickle=False) as archive:
            missing = sorted(set(ACTOR_TENSOR_SHAPES) - set(archive.files))
            extra = sorted(set(archive.files) - set(ACTOR_TENSOR_SHAPES))
            if missing or extra:
                raise ValueError(f"{path}: unexpected actor tensors; missing={missing}, extra={extra}")
            result = {name: self._tensor(archive[name]) for name in ACTOR_TENSOR_SHAPES}
        for name, expected_shape in ACTOR_TENSOR_SHAPES.items():
            if result[name].shape != expected_shape:
                raise ValueError(f"{path}: {name} has {result[name].shape}, expected {expected_shape}")
        return result

    def _load_trajectory(self, path: Path, expected_sha256: str) -> dict[str, torch.Tensor]:
        actual_sha256 = _sha256(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"{path}: SHA256 {actual_sha256} does not match {expected_sha256}")
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"joint", "gravity", "height"}:
                raise ValueError(f"{path}: trajectory must contain joint, gravity, and height")
            result = {name: self._tensor(archive[name]) for name in archive.files}
        frame_count = result["joint"].shape[0]
        if result["joint"].shape != (frame_count, 22):
            raise ValueError(f"{path}: joint trajectory must have shape (frames, 22)")
        if result["gravity"].shape != (frame_count, 3) or result["height"].shape != (frame_count, 1):
            raise ValueError(f"{path}: trajectory gravity/height length mismatch")
        return result

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.elapsed.zero_()
            self.previous_action.zero_()
            return
        self.elapsed[env_ids] = 0.0
        self.previous_action[env_ids] = 0.0

    def _actor_forward(self, observation: torch.Tensor) -> torch.Tensor:
        value = observation
        for prefix in ("obs_encoder.0", "obs_encoder.2", "obs_encoder.4"):
            value = functional.elu(
                functional.linear(value, self.actor[f"{prefix}.weight"], self.actor[f"{prefix}.bias"])
            )
        value = functional.elu(
            functional.linear(value, self.actor["actor.0.weight"], self.actor["actor.0.bias"])
        )
        return functional.linear(value, self.actor["actor.2.weight"], self.actor["actor.2.bias"])

    def initial_state(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.trajectory_joint[0].clone(),
            self.trajectory_gravity[0].clone(),
            self.trajectory_height[0].clone(),
        )

    def step(
        self,
        projected_gravity: torch.Tensor,
        angular_velocity: torch.Tensor,
        joint_position: torch.Tensor,
        joint_velocity: torch.Tensor,
    ) -> NativeTeacherStep:
        frame_time = torch.clamp(self.elapsed - self.lead_in, min=0.0) / self.period
        trajectory_index = torch.floor(frame_time + 0.5).to(torch.long)
        trajectory_index = torch.clamp(trajectory_index, max=self.frame_count - 1)
        trajectory_phase = torch.clamp(self.elapsed / self.trajectory_duration, max=1.0)
        joint_command = self.trajectory_joint[trajectory_index]
        gravity_command = self.trajectory_gravity[trajectory_index]
        height_command = self.trajectory_height[trajectory_index, 0]

        observation = torch.zeros((self.num_envs, 100), dtype=torch.float32, device=self.device)
        observation[:, 0] = self.direction_value
        observation[:, 1] = trajectory_phase
        observation[:, 2:5] = gravity_command - projected_gravity
        observation[:, 5] = height_command
        observation[:, 6:28] = joint_command - joint_position
        observation[:, 6:8] = 0.0
        observation[:, 28:31] = projected_gravity
        observation[:, 31:34] = angular_velocity
        observation[:, 34:56] = joint_position - self.reference_position
        observation[:, 34:36] = 0.0
        observation[:, 56:78] = torch.clamp(joint_velocity, -5.0, 5.0) * self.velocity_scale
        observation[:, 56:58] = 0.0
        observation[:, 78:100] = self.previous_action

        action = self._actor_forward(observation)
        action[:, self.zero_action_ids] = 0.0
        action = torch.clamp(action, self.action_minimum, self.action_maximum)
        raw_target = joint_command + action

        target = raw_target.clone()
        preparing = self.elapsed < self.prepare_duration
        ankle_zero = self.elapsed < self.parallel_ankle_zero_duration
        if torch.any(ankle_zero):
            rows = torch.nonzero(ankle_zero, as_tuple=False).squeeze(-1)
            target[rows[:, None], self.parallel_ankle_ids[None, :]] = 0.0
        if torch.any(preparing):
            maximum_step = self.prepare_velocity_limit * self.period
            target[preparing] = joint_position[preparing] + torch.clamp(
                target[preparing] - joint_position[preparing], -maximum_step, maximum_step
            )
        target = torch.clamp(target, self.position_minimum, self.position_maximum)
        normalized_action = torch.clamp(
            (target - self.action_center) / self.action_scale,
            -1.0,
            1.0,
        )
        training_normalized_action = torch.clamp(
            (target - self.training_action_center) / self.training_action_scale,
            -1.0,
            1.0,
        )

        self.previous_action.copy_(action)
        self.elapsed += self.period
        return NativeTeacherStep(
            trajectory_index=trajectory_index,
            trajectory_phase=trajectory_phase,
            observation=observation,
            action=action,
            raw_target=raw_target,
            target=target,
            normalized_action=normalized_action,
            training_normalized_action=training_normalized_action,
            done=self.elapsed + 1.0e-12 >= self.duration,
        )

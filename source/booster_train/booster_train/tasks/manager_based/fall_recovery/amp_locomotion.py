from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional


EXPECTED_TENSORS = {
    "actor.0.weight": (512, 690),
    "actor.0.bias": (512, 1),
    "actor.2.weight": (256, 512),
    "actor.2.bias": (256, 1),
    "actor.4.weight": (128, 256),
    "actor.4.bias": (128, 1),
    "actor.6.weight": (20, 128),
    "actor.6.bias": (20, 1),
    "obs_normalizer._mean": (1, 690),
    "obs_normalizer._var": (1, 690),
    "obs_normalizer._std": (1, 690),
    "obs_normalizer.count": (1, 1),
}


def repository_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").is_file() and (parent / "assets").is_dir():
            return parent
    raise RuntimeError("could not locate the booster_train repository root")


def load_amp_locomotion_config(
    path: Path | None = None,
) -> tuple[dict[str, Any], Path]:
    hardware_path = Path(__file__).with_name("config") / "k1_fall_recovery.json"
    hardware = json.loads(hardware_path.read_text(encoding="utf-8"))
    handoff = hardware["handoff_policy"]
    root = repository_root()
    config_path = (
        (root / handoff["source_config"]).resolve()
        if path is None
        else path.expanduser().resolve()
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    expected = {
        "joint_names": hardware["joint_names"],
        "num_observation_frames": handoff["num_observation_frames"],
        "observation_size_per_frame": handoff["observation_size_per_frame"],
        "action_size": handoff["action_size"],
        "body_joint_indexes": handoff["body_joint_indexes"],
        "model_files": handoff["model_files"],
        "model_sha256": handoff["model_sha256"],
        "policy_rate_hz": handoff["policy_rate_hz"],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(
                f"{config_path}: {key} differs from handoff_policy: "
                f"{config.get(key)!r} != {value!r}"
            )
    if (
        config["num_observation_frames"],
        config["observation_size_per_frame"],
        config["action_size"],
    ) != (10, 69, 20):
        raise ValueError(f"{config_path}: expected a 10x69 observation and 20 actions")
    for name in ("lab_to_webots_index", "webots_to_lab_index"):
        if sorted(config[name]) != list(range(20)):
            raise ValueError(f"{config_path}: {name} must be a permutation of 0..19")
    for name in (
        "default_dof_position",
        "stiffness",
        "damping",
        "maximum_torque",
    ):
        if len(config[name]) != len(config["joint_names"]):
            raise ValueError(
                f"{config_path}: {name} must contain one value per joint"
            )
    velocity = config["velocity_command"]
    for name in (
        "maximum",
        "minimum",
        "dead_zone",
        "dead_zone_start",
        "dead_zone_end",
        "map_start_positive",
        "map_end_positive",
        "map_start_negative",
        "map_end_negative",
        "maximum_increase_per_second",
        "maximum_decrease_per_second",
    ):
        if len(velocity[name]) != 3:
            raise ValueError(
                f"{config_path}: velocity_command.{name} must contain 3 values"
            )
    if any(
        end <= start
        for start, end in zip(
            velocity["dead_zone_start"],
            velocity["dead_zone_end"],
        )
    ):
        raise ValueError(
            f"{config_path}: velocity-command nonlinear map spans must be positive"
        )
    return config, config_path


def read_native_model(
    path: Path,
    expected_sha256: str | None = None,
) -> dict[str, np.ndarray]:
    path = path.expanduser().resolve()
    data = path.read_bytes()
    if data.startswith(b"version https://git-lfs.github.com/spec/v1"):
        raise RuntimeError(f"{path}: model is still a Git LFS pointer; run git lfs pull")
    if expected_sha256 is not None:
        digest = hashlib.sha256(data).hexdigest()
        if digest != expected_sha256:
            raise ValueError(
                f"{path}: SHA-256 {digest} does not match {expected_sha256}"
            )

    offset = 0
    tensors: dict[str, np.ndarray] = {}
    while offset < len(data):
        if offset + 4 > len(data):
            raise ValueError(f"{path}: truncated tensor-name length at {offset}")
        (name_length,) = struct.unpack_from("<I", data, offset)
        offset += 4
        name_end = offset + name_length
        if name_length == 0 or name_end > len(data):
            raise ValueError(f"{path}: invalid tensor name at {offset}")
        name = data[offset:name_end].decode("utf-8")
        offset = name_end
        if offset + 8 > len(data):
            raise ValueError(f"{path}: truncated shape for {name}")
        rows, columns = struct.unpack_from("<II", data, offset)
        offset += 8
        value_count = rows * columns
        value_end = offset + value_count * 8
        if value_end > len(data):
            raise ValueError(f"{path}: truncated values for {name}")
        if name in tensors:
            raise ValueError(f"{path}: duplicate tensor {name}")
        tensors[name] = np.frombuffer(
            data,
            dtype="<f8",
            count=value_count,
            offset=offset,
        ).copy().reshape(rows, columns)
        offset = value_end

    missing = sorted(set(EXPECTED_TENSORS) - set(tensors))
    extra = sorted(set(tensors) - set(EXPECTED_TENSORS))
    if missing or extra:
        raise ValueError(
            f"{path}: unexpected tensor set; missing={missing}, extra={extra}"
        )
    for name, shape in EXPECTED_TENSORS.items():
        if tensors[name].shape != shape:
            raise ValueError(
                f"{path}: {name} has shape {tensors[name].shape}, expected {shape}"
            )
        if not np.all(np.isfinite(tensors[name])):
            raise ValueError(f"{path}: {name} contains non-finite values")
    if not np.all(tensors["obs_normalizer._mean"] == 0.0):
        raise ValueError(f"{path}: non-zero observation mean is unsupported")
    if not np.all(tensors["obs_normalizer._std"] == 1.0):
        raise ValueError(f"{path}: non-unit observation std is unsupported")
    return tensors


class TorchNativeAmpActor:
    """CUDA batch importer for Booster's native float64 AMP actor."""

    def __init__(
        self,
        path: Path,
        expected_sha256: str,
        *,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        tensors = read_native_model(path, expected_sha256)
        self.layers = tuple(
            (
                torch.as_tensor(
                    tensors[f"actor.{index}.weight"],
                    dtype=dtype,
                    device=device,
                ),
                torch.as_tensor(
                    tensors[f"actor.{index}.bias"].reshape(-1),
                    dtype=dtype,
                    device=device,
                ),
            )
            for index in (0, 2, 4, 6)
        )
        self.device = torch.device(device)
        self.dtype = dtype

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        value = observation.to(device=self.device, dtype=self.dtype)
        if value.shape[-1] != 690:
            raise ValueError(
                f"AMP locomotion observation has {value.shape[-1]} values, expected 690"
            )
        for index, (weight, bias) in enumerate(self.layers):
            value = functional.linear(value, weight, bias)
            if index + 1 < len(self.layers):
                value = functional.elu(value)
        return value


class TorchAmpZeroCommandPolicy:
    """Stateful 50 Hz forward actor used for post-recovery handoff validation."""

    def __init__(
        self,
        config: dict[str, Any],
        actor: TorchNativeAmpActor,
        *,
        num_envs: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.config = config
        self.actor = actor
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.dtype = dtype

        def tensor(values: list[float] | list[int], tensor_dtype=dtype) -> torch.Tensor:
            return torch.tensor(values, dtype=tensor_dtype, device=self.device)

        self.frames = int(config["num_observation_frames"])
        self.frame_size = int(config["observation_size_per_frame"])
        self.body_indexes = tensor(config["body_joint_indexes"], torch.long)
        self.lab_to_webots = tensor(config["lab_to_webots_index"], torch.long)
        self.webots_to_lab = tensor(config["webots_to_lab_index"], torch.long)
        self.default_position = tensor(config["default_dof_position"])
        self.velocity_scale = float(config["joint_velocity_scale"])
        self.action_scale = float(config["action_scale"])
        self.action_filter = float(config["action_filter_new_weight"])
        self.arm_fix_offset = float(config["arm_action_fix_offset"])
        self.arm_fix_index = int(config["arm_action_fix_webots_index"])
        self.clip_observation = float(config["clip_observation"])
        self.clip_action = float(config["clip_action"])
        velocity = config["velocity_command"]
        self.command_maximum = tensor(velocity["maximum"])
        self.command_minimum = tensor(velocity["minimum"])
        self.command_dead_zone = tensor(velocity["dead_zone"])
        self.command_dead_zone_start = tensor(velocity["dead_zone_start"])
        self.command_dead_zone_end = tensor(velocity["dead_zone_end"])
        self.command_map_start_positive = tensor(velocity["map_start_positive"])
        self.command_map_end_positive = tensor(velocity["map_end_positive"])
        self.command_map_start_negative = tensor(velocity["map_start_negative"])
        self.command_map_end_negative = tensor(velocity["map_end_negative"])
        self.command_maximum_increase = tensor(
            velocity["maximum_increase_per_second"]
        )
        self.command_maximum_decrease = tensor(
            velocity["maximum_decrease_per_second"]
        )
        turn_isolation = velocity["turn_isolation"]
        self.turn_minimum_abs_yaw = float(turn_isolation["minimum_abs_yaw"])
        self.turn_maximum_abs_x = float(turn_isolation["maximum_abs_x"])
        self.turn_output_maximum_abs_y = float(
            turn_isolation["output_maximum_abs_y"]
        )
        self.forward_mix_limits = tuple(
            (
                float(limit["minimum_x"]),
                float(limit["minimum_y"]),
                float(limit["maximum_y"]),
                float(limit["minimum_yaw"]),
                float(limit["maximum_yaw"]),
            )
            for limit in velocity["forward_mix_limits"]
        )

        self.history = torch.zeros(
            (self.num_envs, self.frames, self.frame_size),
            dtype=dtype,
            device=self.device,
        )
        self.previous_action = torch.zeros(
            (self.num_envs, 20),
            dtype=dtype,
            device=self.device,
        )
        self.command_value = torch.zeros(
            (self.num_envs, 3),
            dtype=dtype,
            device=self.device,
        )
        self.filtered_target = self.default_position[2:].repeat(self.num_envs, 1)
        self.update_count = torch.zeros(
            self.num_envs,
            dtype=torch.long,
            device=self.device,
        )

    def reset(self, env_ids: torch.Tensor) -> None:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        if env_ids.numel() == 0:
            return
        self.history[env_ids] = 0.0
        self.previous_action[env_ids] = 0.0
        self.command_value[env_ids] = 0.0
        self.filtered_target[env_ids] = self.default_position[2:]
        self.update_count[env_ids] = 0

    def _processed_command(
        self,
        env_ids: torch.Tensor,
        requested_command: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        if requested_command.shape != (env_ids.numel(), 3):
            raise ValueError("AMP locomotion velocity command has incompatible shape")
        if not torch.all(torch.isfinite(requested_command)):
            raise ValueError("AMP locomotion velocity command contains non-finite values")
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("AMP locomotion command dt must be positive and finite")

        magnitude = torch.abs(requested_command)
        active = magnitude >= self.command_dead_zone_start
        span = self.command_dead_zone_end - self.command_dead_zone_start
        ratio = torch.clamp(
            (magnitude - self.command_dead_zone_start) / span,
            0.0,
            1.0,
        )
        positive = self.command_map_start_positive + ratio * (
            self.command_map_end_positive - self.command_map_start_positive
        )
        negative = self.command_map_start_negative + ratio * (
            self.command_map_end_negative - self.command_map_start_negative
        )
        mapped_magnitude = torch.where(
            requested_command >= 0.0,
            positive,
            negative,
        )
        target = torch.where(
            active,
            torch.copysign(mapped_magnitude, requested_command),
            torch.zeros_like(requested_command),
        )
        value = self.command_value[env_ids]
        delta = torch.minimum(
            self.command_maximum_increase * dt,
            torch.maximum(
                -self.command_maximum_decrease * dt,
                target - value,
            ),
        )
        value = torch.clamp(
            value + delta,
            min=self.command_minimum,
            max=self.command_maximum,
        )
        output = torch.where(
            torch.abs(value) < self.command_dead_zone,
            torch.zeros_like(value),
            value,
        )

        isolate_value = (
            (torch.abs(value[:, 2]) > self.turn_minimum_abs_yaw)
            & (torch.abs(value[:, 0]) < self.turn_maximum_abs_x)
        )
        value = value.clone()
        value[isolate_value, 0:2] = 0.0
        isolate_output = (
            (torch.abs(output[:, 2]) > self.turn_minimum_abs_yaw)
            & (torch.abs(output[:, 0]) < self.turn_maximum_abs_x)
            & (torch.abs(output[:, 1]) < self.turn_output_maximum_abs_y)
        )
        output = output.clone()
        output[isolate_output, 0:2] = 0.0
        for minimum_x, minimum_y, maximum_y, minimum_yaw, maximum_yaw in (
            self.forward_mix_limits
        ):
            value_active = value[:, 0] > minimum_x
            value[value_active, 1] = torch.clamp(
                value[value_active, 1],
                minimum_y,
                maximum_y,
            )
            value[value_active, 2] = torch.clamp(
                value[value_active, 2],
                minimum_yaw,
                maximum_yaw,
            )
            output_active = output[:, 0] > minimum_x
            output[output_active, 1] = torch.clamp(
                output[output_active, 1],
                minimum_y,
                maximum_y,
            )
            output[output_active, 2] = torch.clamp(
                output[output_active, 2],
                minimum_yaw,
                maximum_yaw,
            )
        self.command_value[env_ids] = value
        return output

    def step(
        self,
        env_ids: torch.Tensor,
        *,
        angular_velocity: torch.Tensor,
        projected_gravity: torch.Tensor,
        joint_position: torch.Tensor,
        joint_velocity: torch.Tensor,
        requested_command: torch.Tensor | None = None,
        dt: float | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        env_ids = env_ids.to(device=self.device, dtype=torch.long)
        count = env_ids.numel()
        if count == 0:
            return (
                torch.empty((0, 20), dtype=self.dtype, device=self.device),
                torch.empty((0, 20), dtype=self.dtype, device=self.device),
            )
        if (
            angular_velocity.shape != (count, 3)
            or projected_gravity.shape != (count, 3)
            or joint_position.shape != (count, 22)
            or joint_velocity.shape != (count, 22)
        ):
            raise ValueError("AMP locomotion batch inputs have incompatible shapes")
        if requested_command is None:
            requested_command = torch.zeros(
                (count, 3),
                dtype=self.dtype,
                device=self.device,
            )
        else:
            requested_command = requested_command.to(
                device=self.device,
                dtype=self.dtype,
            )
        processed_command = self._processed_command(
            env_ids,
            requested_command,
            1.0 / float(self.config["policy_rate_hz"]) if dt is None else dt,
        )

        position_webots = (
            joint_position[:, self.body_indexes]
            - self.default_position[self.body_indexes]
        )
        position_webots = position_webots.clone()
        position_webots[:, self.arm_fix_index] += self.arm_fix_offset
        position_lab = position_webots[:, self.webots_to_lab]
        velocity_lab = (
            joint_velocity[:, self.body_indexes][:, self.webots_to_lab]
            * self.velocity_scale
        )
        frame = torch.cat(
            (
                angular_velocity,
                projected_gravity,
                processed_command,
                position_lab,
                velocity_lab,
                self.previous_action[env_ids],
            ),
            dim=-1,
        )
        frame = torch.clamp(
            frame,
            -self.clip_observation,
            self.clip_observation,
        )

        first = self.update_count[env_ids] == 0
        continuing_ids = env_ids[~first]
        if continuing_ids.numel():
            self.history[continuing_ids, :-1] = self.history[
                continuing_ids, 1:
            ].clone()
            self.history[continuing_ids, -1] = frame[~first]
        first_ids = env_ids[first]
        if first_ids.numel():
            self.history[first_ids] = frame[first].unsqueeze(1).expand(
                -1,
                self.frames,
                -1,
            )

        observation = self.history[env_ids].reshape(count, -1)
        raw_action = self.actor(observation)
        if not torch.all(torch.isfinite(raw_action)):
            raise RuntimeError("AMP locomotion actor returned non-finite actions")
        action = torch.clamp(raw_action, -self.clip_action, self.clip_action)
        action_webots = action[:, self.lab_to_webots].clone()
        action_webots[:, self.arm_fix_index] -= self.arm_fix_offset
        raw_target = (
            self.default_position[2:].unsqueeze(0)
            + self.action_scale * action_webots
        )
        filtered = (
            (1.0 - self.action_filter) * self.filtered_target[env_ids]
            + self.action_filter * raw_target
        )
        self.previous_action[env_ids] = action
        self.filtered_target[env_ids] = filtered
        self.update_count[env_ids] += 1
        return action, filtered

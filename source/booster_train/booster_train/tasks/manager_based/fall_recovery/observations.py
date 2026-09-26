from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg

from .commands import RecoveryDiscoveryCommand, RecoveryReferenceCommand


def _command(env, command_name: str) -> RecoveryReferenceCommand:
    return env.command_manager.get_term(command_name)


def motion_command_deployment(
    env,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    command = env.command_manager.get_term(command_name)
    return torch.cat(
        (
            command.joint_pos[:, asset_cfg.joint_ids],
            command.joint_vel[:, asset_cfg.joint_ids],
        ),
        dim=-1,
    )


def reference_joint_position(env, command_name: str) -> torch.Tensor:
    return _command(env, command_name).reference_joint_pos


def reference_gravity(env, command_name: str) -> torch.Tensor:
    return _command(env, command_name).reference_gravity


def reference_height(env, command_name: str) -> torch.Tensor:
    return _command(env, command_name).reference_height


def reference_phase(env, command_name: str) -> torch.Tensor:
    return _command(env, command_name).reference_phase


def discovery_assistance_force(env, command_name: str) -> torch.Tensor:
    command: RecoveryDiscoveryCommand = env.command_manager.get_term(command_name)
    scale = max(command.cfg.maximum_assistance_force, 1.0)
    return (command.assistance_force / scale).unsqueeze(-1)


def amp_future_reference(env) -> torch.Tensor:
    """Return the next 50 Hz reference frame used by the directional tracker."""
    if not hasattr(env, "_motion_loader"):
        joint_count = len(env.cfg.amp_joint_names)
        frame_size = 2 * joint_count + 13 + 3 * len(env.cfg.amp_key_body_names)
        return torch.zeros((env.num_envs, frame_size), device=env.device)
    return env.future_reference_amp_frame


def amp_future_reference_phase(env) -> torch.Tensor:
    if not hasattr(env, "_motion_loader"):
        return torch.zeros((env.num_envs, 1), device=env.device)
    return env.future_reference_amp_phase


def zero_amp_future_reference(env, frame_size: int | None = None) -> torch.Tensor:
    """Keep checkpoint-compatible dimensions without exposing the motion timeline."""
    if frame_size is not None:
        if frame_size < 1:
            raise ValueError("frame_size must be positive")
        return torch.zeros((env.num_envs, frame_size), device=env.device)
    return torch.zeros_like(amp_future_reference(env))


def zero_amp_future_reference_phase(env) -> torch.Tensor:
    """Hide reference phase while preserving the checkpoint observation layout."""
    return torch.zeros_like(amp_future_reference_phase(env))

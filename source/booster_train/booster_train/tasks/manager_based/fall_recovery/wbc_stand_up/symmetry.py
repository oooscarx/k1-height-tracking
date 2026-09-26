from __future__ import annotations

from functools import lru_cache

import torch

HISTORY_LENGTH = 5
JOINT_MIRROR_INDEX = (
    0,
    1,
    6,
    7,
    8,
    9,
    2,
    3,
    4,
    5,
    16,
    17,
    18,
    19,
    20,
    21,
    10,
    11,
    12,
    13,
    14,
    15,
)
JOINT_MIRROR_SIGN = (
    -1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    -1.0,
    1.0,
    -1.0,
    1.0,
    -1.0,
    1.0,
    -1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
    1.0,
    -1.0,
    -1.0,
    1.0,
    1.0,
    -1.0,
)


def _mirror_joint_values(values: torch.Tensor) -> torch.Tensor:
    index = torch.tensor(JOINT_MIRROR_INDEX, device=values.device)
    sign = torch.tensor(
        JOINT_MIRROR_SIGN,
        dtype=values.dtype,
        device=values.device,
    )
    return values[..., index] * sign


def mirror_joint_values(values: torch.Tensor) -> torch.Tensor:
    """Mirror deployment-order K1 joint values across the sagittal plane."""
    return _mirror_joint_values(values)


def _mirror_polar(values: torch.Tensor) -> torch.Tensor:
    mirrored = values.clone()
    mirrored[..., 1] *= -1.0
    return mirrored


def _mirror_axial(values: torch.Tensor) -> torch.Tensor:
    mirrored = values.clone()
    mirrored[..., 0] *= -1.0
    mirrored[..., 2] *= -1.0
    return mirrored


def _identity(values: torch.Tensor) -> torch.Tensor:
    return values.clone()


@lru_cache(maxsize=4)
def _body_mirror_index(body_names: tuple[str, ...]) -> tuple[int, ...]:
    result = []
    for name in body_names:
        if "Left" in name:
            mirrored = name.replace("Left", "Right")
        elif "Right" in name:
            mirrored = name.replace("Right", "Left")
        elif "left" in name:
            mirrored = name.replace("left", "right")
        elif "right" in name:
            mirrored = name.replace("right", "left")
        else:
            mirrored = name
        if mirrored not in body_names:
            raise ValueError(f"missing mirrored K1 body for {name}: {mirrored}")
        result.append(body_names.index(mirrored))
    return tuple(result)


def _mirror_contact(values: torch.Tensor, env) -> torch.Tensor:
    body_names = tuple(env.unwrapped.scene.sensors["contact_forces"].body_names)
    index = torch.tensor(_body_mirror_index(body_names), device=values.device)
    return values[..., index]


def _mirror_history_term(
    term: torch.Tensor,
    width: int,
    mirror,
) -> torch.Tensor:
    shaped = term.reshape(*term.shape[:-1], HISTORY_LENGTH, width)
    return mirror(shaped).reshape(*term.shape[:-1], HISTORY_LENGTH * width)


def _mirror_flat_observation(obs: torch.Tensor, env, obs_type: str) -> torch.Tensor:
    if obs_type == "policy":
        layout = (
            (3, _mirror_axial),
            (3, _mirror_polar),
            (22, _mirror_joint_values),
            (22, _mirror_joint_values),
            (22, _mirror_joint_values),
        )
    elif obs_type == "critic":
        num_bodies = len(env.unwrapped.scene.sensors["contact_forces"].body_names)
        layout = (
            (3, _mirror_polar),
            (3, _mirror_polar),
            (3, _mirror_axial),
            (22, _mirror_joint_values),
            (22, _mirror_joint_values),
            (22, _mirror_joint_values),
            (1, _identity),
            (num_bodies, lambda value: _mirror_contact(value, env)),
            (1, _identity),
        )
    else:
        raise ValueError(f"unsupported observation group for K1 symmetry: {obs_type}")

    mirrored_terms = []
    cursor = 0
    for width, mirror in layout:
        term_size = HISTORY_LENGTH * width
        mirrored_terms.append(
            _mirror_history_term(
                obs[..., cursor : cursor + term_size],
                width,
                mirror,
            )
        )
        cursor += term_size
    if cursor != obs.shape[-1]:
        raise ValueError(f"K1 {obs_type} symmetry layout expects {cursor} values, got {obs.shape[-1]}")
    return torch.cat(mirrored_terms, dim=-1)


def mirror_flat_observation(obs: torch.Tensor, env, obs_type: str) -> torch.Tensor:
    """Mirror one flattened WBC policy or critic observation."""
    return _mirror_flat_observation(obs, env, obs_type)


def lr_mirror_k1(
    env,
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str = "policy",
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Append a physically mirrored K1 transition for PPO data augmentation."""
    augmented_obs = None
    augmented_actions = None
    if obs is not None:
        augmented_obs = torch.cat(
            (obs, _mirror_flat_observation(obs, env, obs_type)),
            dim=0,
        )
    if actions is not None:
        augmented_actions = torch.cat(
            (actions, _mirror_joint_values(actions)),
            dim=0,
        )
    return augmented_obs, augmented_actions

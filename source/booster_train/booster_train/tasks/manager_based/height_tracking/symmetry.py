from __future__ import annotations

from functools import lru_cache

import torch

HISTORY_LENGTH = 5
JOINT_MIRROR_INDEX = (0, 1, 6, 7, 8, 9, 2, 3, 4, 5, 16, 17, 18, 19, 20, 21, 10, 11, 12, 13, 14, 15)
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


def _mirror_joints(values: torch.Tensor) -> torch.Tensor:
    index = torch.tensor(JOINT_MIRROR_INDEX, device=values.device)
    sign = torch.tensor(JOINT_MIRROR_SIGN, dtype=values.dtype, device=values.device)
    return values[..., index] * sign


def _mirror_polar(values: torch.Tensor) -> torch.Tensor:
    result = values.clone()
    result[..., 1] *= -1.0
    return result


def _mirror_axial(values: torch.Tensor) -> torch.Tensor:
    result = values.clone()
    result[..., 0] *= -1.0
    result[..., 2] *= -1.0
    return result


def _identity(values: torch.Tensor) -> torch.Tensor:
    return values.clone()


@lru_cache(maxsize=4)
def _body_mirror_index(body_names: tuple[str, ...]) -> tuple[int, ...]:
    result = []
    for name in body_names:
        if "Left" in name:
            mirror = name.replace("Left", "Right")
        elif "Right" in name:
            mirror = name.replace("Right", "Left")
        elif "left" in name:
            mirror = name.replace("left", "right")
        elif "right" in name:
            mirror = name.replace("right", "left")
        else:
            mirror = name
        if mirror not in body_names:
            raise ValueError(f"missing mirrored K1 body for {name}: {mirror}")
        result.append(body_names.index(mirror))
    return tuple(result)


def _mirror_contact(values: torch.Tensor, env) -> torch.Tensor:
    names = tuple(env.unwrapped.scene.sensors["contact_forces"].body_names)
    index = torch.tensor(_body_mirror_index(names), device=values.device)
    return values[..., index]


def _mirror_flat(obs: torch.Tensor, env, obs_type: str) -> torch.Tensor:
    if obs_type == "policy":
        layout = (
            (3, _mirror_axial),
            (3, _mirror_polar),
            (22, _mirror_joints),
            (22, _mirror_joints),
            (22, _mirror_joints),
            (1, _identity),
        )
    elif obs_type == "critic":
        num_bodies = len(env.unwrapped.scene.sensors["contact_forces"].body_names)
        layout = (
            (3, _mirror_polar),
            (3, _mirror_polar),
            (3, _mirror_axial),
            (22, _mirror_joints),
            (22, _mirror_joints),
            (22, _mirror_joints),
            (num_bodies, lambda value: _mirror_contact(value, env)),
            (1, _identity),
            (1, _identity),
        )
    else:
        raise ValueError(f"unsupported observation group: {obs_type}")
    mirrored = []
    cursor = 0
    for width, function in layout:
        size = HISTORY_LENGTH * width
        term = obs[..., cursor : cursor + size].reshape(*obs.shape[:-1], HISTORY_LENGTH, width)
        mirrored.append(function(term).reshape(*obs.shape[:-1], size))
        cursor += size
    if cursor != obs.shape[-1]:
        raise ValueError(f"K1 {obs_type} mirror expected {cursor} values, got {obs.shape[-1]}")
    return torch.cat(mirrored, dim=-1)


def lr_mirror_k1_height(
    env,
    obs: torch.Tensor | None = None,
    actions: torch.Tensor | None = None,
    obs_type: str = "policy",
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    mirrored_obs = None if obs is None else torch.cat((obs, _mirror_flat(obs, env, obs_type)), dim=0)
    mirrored_actions = None if actions is None else torch.cat((actions, _mirror_joints(actions)), dim=0)
    return mirrored_obs, mirrored_actions

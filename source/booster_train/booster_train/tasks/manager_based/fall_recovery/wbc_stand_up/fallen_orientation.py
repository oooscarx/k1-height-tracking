from __future__ import annotations

from typing import Literal

import torch

FallenOrientationMode = Literal[
    "random",
    "face_up",
    "face_down",
    "left_side",
    "right_side",
    "other",
]
FallenSamplingMode = Literal[
    "random",
    "balanced",
    "face_up",
    "face_down",
    "left_side",
    "right_side",
    "other",
]
EXCLUSIVE_FALLEN_ORIENTATION_MODES = (
    "face_up",
    "face_down",
    "left_side",
    "right_side",
    "other",
)
FALLEN_ORIENTATION_MODES = ("random", *EXCLUSIVE_FALLEN_ORIENTATION_MODES)
FALLEN_SAMPLING_MODES = (*FALLEN_ORIENTATION_MODES, "balanced")


def fallen_orientation_mask(
    root_quat_w: torch.Tensor,
    mode: FallenOrientationMode,
    threshold: float = 0.55,
) -> torch.Tensor:
    """Select an exclusive fallen-orientation class from WXYZ root quaternions."""
    if root_quat_w.ndim != 2 or root_quat_w.shape[-1] != 4:
        raise ValueError("root quaternions must have shape (N, 4)")
    if mode not in FALLEN_ORIENTATION_MODES:
        raise ValueError(f"unsupported fallen orientation mode: {mode}")
    if not 0.0 < threshold < 1.0:
        raise ValueError("fallen orientation threshold must be between zero and one")
    if mode == "random":
        return torch.ones(root_quat_w.shape[0], dtype=torch.bool, device=root_quat_w.device)

    quat = root_quat_w / torch.linalg.vector_norm(
        root_quat_w,
        dim=-1,
        keepdim=True,
    ).clamp_min(1.0e-8)
    w, x, y, z = quat.unbind(dim=-1)

    # World-z components of the robot's local forward (+x) and left (+y) axes.
    forward_z = 2.0 * (x * z - y * w)
    left_z = 2.0 * (y * z + x * w)
    face_dominant = torch.abs(forward_z) >= torch.abs(left_z)

    masks = {
        "face_up": face_dominant & (forward_z >= threshold),
        "face_down": face_dominant & (forward_z <= -threshold),
        "left_side": ~face_dominant & (left_z <= -threshold),
        "right_side": ~face_dominant & (left_z >= threshold),
    }
    if mode == "other":
        return ~torch.stack(tuple(masks.values()), dim=0).any(dim=0)
    return masks[mode]


def sample_balanced_fallen_orientation_indices(
    root_quat_w: torch.Tensor,
    num_samples: int,
    threshold: float = 0.55,
) -> torch.Tensor:
    """Sample all five exclusive fallen classes as evenly as possible."""
    if num_samples < 0:
        raise ValueError("number of fallen-state samples must be non-negative")
    if num_samples == 0:
        return torch.empty(0, dtype=torch.long, device=root_quat_w.device)

    class_count = len(EXCLUSIVE_FALLEN_ORIENTATION_MODES)
    class_offset = torch.randint(
        class_count,
        (1,),
        device=root_quat_w.device,
    )
    requested_classes = torch.arange(
        num_samples,
        device=root_quat_w.device,
    )
    requested_classes = (requested_classes + class_offset) % class_count
    requested_classes = requested_classes[
        torch.randperm(num_samples, device=root_quat_w.device)
    ]
    sampled_indices = torch.empty(
        num_samples,
        dtype=torch.long,
        device=root_quat_w.device,
    )
    for class_index, mode in enumerate(EXCLUSIVE_FALLEN_ORIENTATION_MODES):
        requested = requested_classes == class_index
        requested_count = int(requested.sum().item())
        if requested_count == 0:
            continue
        available = torch.nonzero(
            fallen_orientation_mask(root_quat_w, mode, threshold),
            as_tuple=False,
        ).squeeze(-1)
        if available.numel() == 0:
            raise RuntimeError(
                f"fallen-state cache has no {mode} samples for balanced sampling"
            )
        offsets = torch.randint(
            available.numel(),
            (requested_count,),
            device=root_quat_w.device,
        )
        sampled_indices[requested] = available[offsets]
    return sampled_indices

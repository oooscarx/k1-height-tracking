from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg


def disable_joints(
    env,
    env_ids: torch.Tensor | None,
    rest_duration_s: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Set simulated actuation to zero while a reset pose settles."""
    del env_ids
    asset: Articulation = env.scene[asset_cfg.name]
    in_rest_phase = env.episode_length_buf < int(rest_duration_s / env.step_dt)
    rest_env_ids = in_rest_phase.nonzero().flatten()
    if rest_env_ids.numel() == 0:
        return

    asset._joint_effort_target_sim[rest_env_ids, :] = 0.0  # type: ignore[attr-defined]
    asset.root_physx_view.set_dof_actuation_forces(
        asset._joint_effort_target_sim,  # type: ignore[attr-defined]
        rest_env_ids,
    )

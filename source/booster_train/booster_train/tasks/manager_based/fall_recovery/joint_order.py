from __future__ import annotations

from collections.abc import Sequence


def deployment_joint_ids(simulation_joint_names: Sequence[str], deployment_joint_names: Sequence[str]) -> list[int]:
    """Return simulation indices arranged in deployment/protocol order."""
    if len(set(simulation_joint_names)) != len(simulation_joint_names):
        raise ValueError("simulation joint names must be unique")
    if len(set(deployment_joint_names)) != len(deployment_joint_names):
        raise ValueError("deployment joint names must be unique")

    simulation_indexes = {name: index for index, name in enumerate(simulation_joint_names)}
    missing = [name for name in deployment_joint_names if name not in simulation_indexes]
    extra = [name for name in simulation_joint_names if name not in set(deployment_joint_names)]
    if missing or extra:
        raise ValueError(
            f"simulation and deployment joints differ: missing_in_simulation={missing}, extra_in_simulation={extra}"
        )
    return [simulation_indexes[name] for name in deployment_joint_names]

#!/usr/bin/env python3
"""Export the K1 height-tracking actor and its deployment contract to ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "source/booster_train/booster_train/tasks/manager_based/fall_recovery/config/k1_fall_recovery.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, help="rsl_rl model_*.pt checkpoint")
    parser.add_argument("output", type=Path, help="destination .onnx path")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--opset", type=int, default=17)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_actor(state: dict[str, torch.Tensor]) -> nn.Sequential:
    layer_ids = sorted(
        int(key.split(".")[1])
        for key in state
        if key.startswith("actor.") and key.endswith(".weight")
    )
    if not layer_ids:
        raise ValueError("checkpoint does not contain actor linear layers")

    modules: list[nn.Module] = []
    for index, layer_id in enumerate(layer_ids):
        weight = state[f"actor.{layer_id}.weight"]
        bias = state[f"actor.{layer_id}.bias"]
        linear = nn.Linear(weight.shape[1], weight.shape[0])
        linear.weight.data.copy_(weight)
        linear.bias.data.copy_(bias)
        modules.append(linear)
        if index + 1 < len(layer_ids):
            modules.append(nn.ELU())
    return nn.Sequential(*modules).eval()


def safe_action_scale(config: dict) -> list[float]:
    margin = config["native_teacher"]["robust_training"]["position_target_margin"]
    raw = config["native_teacher"]["training_action_scale"]
    return [
        min(scale, center - lower - margin, upper - center - margin)
        for scale, center, lower, upper in zip(
            raw,
            config["goal_position"],
            config["position_minimum"],
            config["position_maximum"],
            strict=True,
        )
    ]


def write_manifest(
    output: Path,
    checkpoint: Path,
    checkpoint_data: dict,
    config: dict,
    input_size: int,
    output_size: int,
) -> None:
    manifest = {
        "schema_version": 1,
        "policy": "K1 height tracking",
        "checkpoint": checkpoint.name,
        "checkpoint_iteration": int(checkpoint_data.get("iter", -1)),
        "checkpoint_sha256": sha256(checkpoint),
        "onnx_sha256": sha256(output),
        "input_name": "observation",
        "output_name": "action",
        "input_size": input_size,
        "output_size": output_size,
        "history_length": 5,
        "history_order": "oldest_to_newest",
        "observation_layout": [
            {"name": "base_ang_vel", "width": 3, "scale": 0.2},
            {"name": "projected_gravity", "width": 3, "scale": 1.0},
            {"name": "joint_pos_rel", "width": 22, "scale": 1.0},
            {"name": "joint_vel_rel", "width": 22, "scale": 0.05},
            {"name": "last_action", "width": 22, "scale": 1.0},
            {"name": "height_command", "width": 1, "scale": 1.0},
        ],
        "joint_names": config["joint_names"],
        "action_clip": [-1.0, 1.0],
        "action_center": config["goal_position"],
        "action_scale": safe_action_scale(config),
        "position_minimum": config["position_minimum"],
        "position_maximum": config["position_maximum"],
        "stiffness": config["stiffness"],
        "damping": config["damping"],
        "torque_limit": config["command_torque_limit"],
        "policy_rate_hz": config["policy_rate_hz"],
        "simulation_rate_hz": 200.0,
        "tracked_point_offset_m": 0.2,
        "standing_root_height_m": 0.52,
    }
    output.with_suffix(".json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    checkpoint = args.checkpoint.expanduser().resolve()
    output = args.output.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_data = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = checkpoint_data["model_state_dict"]
    actor = build_actor(state)
    config = json.loads(config_path.read_text(encoding="utf-8"))

    first_linear = next(module for module in actor if isinstance(module, nn.Linear))
    last_linear = next(module for module in reversed(actor) if isinstance(module, nn.Linear))
    input_size = first_linear.in_features
    output_size = last_linear.out_features
    expected_input_size = 5 * (3 + 3 + 22 + 22 + 22 + 1)
    if input_size != expected_input_size or output_size != len(config["joint_names"]):
        raise ValueError(
            f"unexpected actor shape: {input_size} -> {output_size}; "
            f"expected {expected_input_size} -> {len(config['joint_names'])}"
        )

    example = torch.zeros(1, input_size, dtype=torch.float32)
    torch.onnx.export(
        actor,
        example,
        output,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=args.opset,
        do_constant_folding=True,
    )
    write_manifest(output, checkpoint, checkpoint_data, config, input_size, output_size)
    print(f"exported {output}")
    print(f"manifest {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()

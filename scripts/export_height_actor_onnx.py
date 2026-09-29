#!/usr/bin/env python3
"""Export the K1 height-tracking actor and its deployment contract to ONNX."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import torch
from torch import nn


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "source/booster_train/booster_train/tasks/manager_based/fall_recovery/config/k1_fall_recovery.json"
)

MOTOR_LIMITS = {
    "HT4438": (6.0, 7.85, 7.85),
    "R14": (14.0, 33.51, 5.24),
    "E6408": (68.0, 14.66, 1.88),
    "E4315": (76.0, 12.57, 2.62),
    "E4310": (38.3, 17.59, 7.85),
    "E4310-parallel": (38.3, 17.59, 7.85),
    "E6416": (112.0, 12.57, 2.09),
}

MOTOR_ARMATURE = {
    "HT4438": 0.001,
    "R14": 0.001,
    "E6408": 0.0478125,
    "E4315": 0.0339552,
    "E4310": 0.0282528,
    "E4310-parallel": 0.0565056,
    "E6416": 0.095625,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path, help="rsl_rl model_*.pt checkpoint")
    parser.add_argument("output", type=Path, help="destination .onnx path")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--env-cfg",
        type=Path,
        default=None,
        help="training env.yaml; defaults to CHECKPOINT_DIR/params/env.yaml",
    )
    parser.add_argument(
        "--agent-cfg",
        type=Path,
        default=None,
        help="training agent.yaml; defaults to CHECKPOINT_DIR/params/agent.yaml",
    )
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


def motor_limit(config: dict, index: int) -> list[float]:
    model = config["motor_models"][index]
    if model not in MOTOR_LIMITS:
        raise ValueError(f"unsupported K1 motor model: {model}")
    return list(MOTOR_LIMITS[model])


def height_posture_center(config: dict) -> list[float]:
    center = list(config["goal_position"])
    for hip_index, knee_index, ankle_index in ((10, 13, 14), (16, 19, 20)):
        center[hip_index] = -1.135
        center[knee_index] = 1.92
        center[ankle_index] = -0.785
    return center


def deployment_armature(config: dict) -> list[float]:
    result = [MOTOR_ARMATURE[model] for model in config["motor_models"]]
    result[:10] = [0.01] * 10
    return result


def load_run_yaml(path: Path | None, fallback: Path) -> tuple[dict | None, Path | None]:
    selected = path.expanduser().resolve() if path is not None else fallback
    if not selected.exists():
        return None, None
    import yaml

    with selected.open("r", encoding="utf-8") as stream:
        return yaml.load(stream, Loader=yaml.UnsafeLoader), selected


def expand_joint_value(value: float | list[float], size: int) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)] * size
    if len(value) != size:
        raise ValueError(f"joint value has {len(value)} entries, expected {size}")
    return [float(item) for item in value]


def resolved_actuator_values(
    env_cfg: dict,
    joint_names: list[str],
    field: str,
    fallback: list[float],
) -> list[float]:
    """Resolve the effective scene actuator values in deployment joint order."""

    result = list(map(float, fallback))
    actuators = env_cfg.get("scene", {}).get("robot", {}).get("actuators", {})
    for actuator in actuators.values():
        expressions = actuator.get("joint_names_expr", [])
        value = actuator.get(field)
        if value is None:
            continue
        for index, joint_name in enumerate(joint_names):
            if not any(re.fullmatch(expression, joint_name) for expression in expressions):
                continue
            if isinstance(value, dict):
                matches = [
                    candidate
                    for pattern, candidate in value.items()
                    if pattern == joint_name or re.fullmatch(pattern, joint_name)
                ]
                if not matches:
                    continue
                result[index] = float(matches[0])
            else:
                result[index] = float(value)
    return result


def actuator_delay_steps(env_cfg: dict, joint_names: list[str]) -> list[int]:
    result = [5] * len(joint_names)
    actuators = env_cfg.get("scene", {}).get("robot", {}).get("actuators", {})
    for actuator in actuators.values():
        expressions = actuator.get("joint_names_expr", [])
        minimum = int(actuator.get("min_delay", 0))
        maximum = int(actuator.get("max_delay", minimum))
        if minimum != maximum:
            raise ValueError(
                "deployment export requires a fixed actuator delay, "
                f"got [{minimum}, {maximum}]"
            )
        for index, joint_name in enumerate(joint_names):
            if any(re.fullmatch(expression, joint_name) for expression in expressions):
                result[index] = minimum
    return result


def write_manifest(
    output: Path,
    checkpoint: Path,
    checkpoint_data: dict,
    config: dict,
    env_cfg: dict | None,
    env_cfg_path: Path | None,
    agent_cfg: dict | None,
    input_size: int,
    output_size: int,
) -> None:
    action_cfg = env_cfg["actions"]["joint_pos"] if env_cfg is not None else None
    command_cfg = env_cfg["commands"]["height"] if env_cfg is not None else None
    if action_cfg is None:
        action_center = config["goal_position"]
        action_scale = safe_action_scale(config)
        position_delta_clip = [-1.0, 1.0]
        position_target_velocity_limit = [1.0, 1.0, *([2.0] * 8), *([10.0] * 12)]
        posture_center = height_posture_center(config)
        posture_high_center = None
        posture_range = [0.54, 0.72]
        posture_exponent = 0.6
        posture_blend = 0.40
        posture_phase_knots = None
        residual_fade_range = [0.66, 0.72]
        residual_minimum_scale = 0.90
        residual_base_maximum_scale = 1.0
        residual_maximum_scale = 1.0
        residual_handoff_range = None
        residual_handoff_minimum_scale = 1.0
        residual_deep_handoff_range = None
        residual_deep_handoff_minimum_scale = 1.0
        residual_deep_handoff_joint_indices = list(range(10, output_size))
        residual_amplify_joint_indices = [10, 13, 14, 16, 19, 20]
        last_action_scales = [1.0] * output_size
        target_velocity_initialize_from_target = False
        target_velocity_warmup_steps = 0
        stiffness = config["stiffness"]
        damping = config["damping"]
        torque_limit = config["command_torque_limit"]
        delays = [5] * output_size
        simulation_rate_hz = 200.0
        policy_rate_hz = config["policy_rate_hz"]
        tracked_point_offset_m = 0.2
        standing_root_height_m = 0.52
        trained_height_range = posture_range
    else:
        action_center = action_cfg["position_center"]
        action_scale = action_cfg["position_scale"]
        position_delta_clip = list(action_cfg["clip"][".*"])
        position_target_velocity_limit = expand_joint_value(
            action_cfg["position_target_velocity_limit"], output_size
        )
        posture_center = action_cfg.get("height_posture_center")
        posture_high_center = action_cfg.get("height_posture_high_center")
        posture_range = list(action_cfg["height_posture_range"])
        posture_exponent = float(action_cfg["height_posture_exponent"])
        posture_blend = float(action_cfg["height_posture_blend"])
        posture_phase_knots = action_cfg.get("height_posture_phase_knots")
        residual_fade_range = list(action_cfg["height_residual_fade_range"])
        residual_minimum_scale = float(action_cfg["height_residual_minimum_scale"])
        residual_base_maximum_scale = float(
            action_cfg.get("height_residual_base_maximum_scale", 1.0)
        )
        residual_maximum_scale = float(action_cfg.get("height_residual_maximum_scale", 1.0))
        residual_handoff_range = action_cfg.get("height_residual_handoff_range")
        residual_handoff_minimum_scale = float(
            action_cfg.get("height_residual_handoff_minimum_scale", 1.0)
        )
        residual_deep_handoff_range = action_cfg.get("height_residual_deep_handoff_range")
        residual_deep_handoff_minimum_scale = float(
            action_cfg.get("height_residual_deep_handoff_minimum_scale", 1.0)
        )
        residual_amplify_joint_indices = [
            config["joint_names"].index(name)
            for name in action_cfg.get("height_residual_amplify_joint_names", [])
        ]
        deep_joint_names = action_cfg.get("height_residual_deep_handoff_joint_names")
        if deep_joint_names is None:
            deep_joint_names = action_cfg.get("height_residual_fade_joint_names", [])
        residual_deep_handoff_joint_indices = [
            config["joint_names"].index(name) for name in deep_joint_names
        ]
        observation_cfg = env_cfg["observations"]["policy"]["actions"]
        last_action_scales = [1.0] * output_size
        if str(observation_cfg.get("func", "")).endswith(":scaled_last_action"):
            observation_params = observation_cfg.get("params", {})
            for index in observation_params.get("joint_indices", []):
                last_action_scales[int(index)] = float(observation_params["scale"])
        target_velocity_initialize_from_target = bool(
            action_cfg.get("position_target_velocity_limit_initialize_from_target", False)
        )
        target_velocity_warmup_steps = int(
            action_cfg.get("position_target_velocity_limit_warmup_steps", 0)
        )
        stiffness = resolved_actuator_values(
            env_cfg, config["joint_names"], "stiffness", action_cfg["stiffness"]
        )
        damping = resolved_actuator_values(
            env_cfg, config["joint_names"], "damping", action_cfg["damping"]
        )
        torque_limit = action_cfg["command_torque_limit"]
        delays = actuator_delay_steps(env_cfg, config["joint_names"])
        simulation_rate_hz = 1.0 / float(env_cfg["sim"]["dt"])
        policy_rate_hz = simulation_rate_hz / int(env_cfg["decimation"])
        tracked_point_offset_m = float(command_cfg["offset"]["pos"][2])
        standing_root_height_m = float(env_cfg["scene"]["robot"]["init_state"]["pos"][2])
        trained_height_range = list(command_cfg["ranges"]["height"])

    action_clip = float(agent_cfg.get("clip_actions", 4.0)) if agent_cfg is not None else 4.0
    manifest = {
        "schema_version": 5,
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
        # RSL-RL keeps the wrapper-clipped raw action in last_action. The K1
        # action term scales that raw action first, then clips the joint delta.
        "action_clip": [-action_clip, action_clip],
        "position_delta_clip": position_delta_clip,
        "position_target_margin": config["native_teacher"]["robust_training"][
            "position_target_margin"
        ],
        "position_braking_horizon_s": config["native_teacher"]["robust_training"][
            "position_braking_horizon_s"
        ],
        "actuator_delay_steps": delays,
        "position_target_velocity_limit": position_target_velocity_limit,
        "position_target_velocity_limit_initialize_from_target": target_velocity_initialize_from_target,
        "position_target_velocity_limit_warmup_steps": target_velocity_warmup_steps,
        "height_residual_fade_joint_indices": list(range(10, len(config["joint_names"]))),
        "height_residual_amplify_joint_indices": residual_amplify_joint_indices,
        "height_residual_fade_range": residual_fade_range,
        "height_residual_minimum_scale": residual_minimum_scale,
        "height_residual_base_maximum_scale": residual_base_maximum_scale,
        "height_residual_maximum_scale": residual_maximum_scale,
        "height_residual_handoff_range": residual_handoff_range,
        "height_residual_handoff_minimum_scale": residual_handoff_minimum_scale,
        "height_residual_deep_handoff_range": residual_deep_handoff_range,
        "height_residual_deep_handoff_minimum_scale": residual_deep_handoff_minimum_scale,
        "height_residual_deep_handoff_joint_indices": residual_deep_handoff_joint_indices,
        "height_posture_center": posture_center,
        "height_posture_high_center": posture_high_center,
        "height_posture_range": posture_range,
        "height_posture_exponent": posture_exponent,
        "height_posture_blend": posture_blend,
        "height_posture_phase_knots": posture_phase_knots,
        "last_action_scales": last_action_scales,
        "trained_height_range": trained_height_range,
        "action_center": action_center,
        "action_scale": action_scale,
        "position_minimum": config["position_minimum"],
        "position_maximum": config["position_maximum"],
        "stiffness": stiffness,
        "damping": damping,
        "torque_limit": torque_limit,
        "motor_effort_limit": [motor_limit(config, index)[0] for index in range(output_size)],
        "motor_velocity_limit": [motor_limit(config, index)[1] for index in range(output_size)],
        "motor_knee_velocity": [motor_limit(config, index)[2] for index in range(output_size)],
        "joint_armature": deployment_armature(config),
        "policy_rate_hz": policy_rate_hz,
        "simulation_rate_hz": simulation_rate_hz,
        "tracked_point_offset_m": tracked_point_offset_m,
        "standing_root_height_m": standing_root_height_m,
        "source_env_config_sha256": sha256(env_cfg_path) if env_cfg_path is not None else None,
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
    env_cfg, env_cfg_path = load_run_yaml(
        args.env_cfg,
        checkpoint.parent / "params" / "env.yaml",
    )
    agent_cfg, _ = load_run_yaml(
        args.agent_cfg,
        checkpoint.parent / "params" / "agent.yaml",
    )

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
    write_manifest(
        output,
        checkpoint,
        checkpoint_data,
        config,
        env_cfg,
        env_cfg_path,
        agent_cfg,
        input_size,
        output_size,
    )
    print(f"exported {output}")
    print(f"manifest {output.with_suffix('.json')}")


if __name__ == "__main__":
    main()

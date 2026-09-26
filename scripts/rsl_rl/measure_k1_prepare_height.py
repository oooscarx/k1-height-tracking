"""Measure the terrain-relative height of the firmware K1 Prepare pose."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Booster-K1-Height-Tracking-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--settle_seconds", type=float, default=8.0)
parser.add_argument("--sample_seconds", type=float, default=2.0)
parser.add_argument("--initial_root_height", type=float, default=0.55182312536978)
parser.add_argument("--terrain", choices=("training", "plane"), default="training")
parser.add_argument("--ray_origin_height", type=float, default=0.0)
parser.add_argument("--output", type=Path, default=Path("logs/k1_prepare_pose_height.json"))
parser.add_argument("--disable_fabric", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import booster_train.tasks  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
import torch  # noqa: E402
from isaaclab.terrains import TerrainImporterCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


# Firmware config/robots/k1_luz.json Prepare source_qpos, with the parallel
# ankle crank targets converted through booster-firmware's motor-to-serial FK.
PREPARE_POSITION = [
    0.0,
    0.0,
    0.2,
    -1.45,
    0.0,
    -0.5,
    0.2,
    1.45,
    0.0,
    0.5,
    0.0,
    0.0,
    0.0,
    0.105,
    -0.0895806435156909,
    -0.005173957381287652,
    0.0,
    0.0,
    0.0,
    0.105,
    -0.0895806435156909,
    0.005173957381287652,
]


def disable_events(events: object) -> None:
    for name in (
        "randomize_physics_material",
        "randomize_actuator_gains",
        "randomize_joint_friction",
        "randomize_joint_armature",
        "randomize_bodies_mass",
        "randomize_base_mass",
        "randomize_bodies_com",
        "randomize_base_com",
        "push_robot",
        "apply_external_force_torque",
        "apply_external_force_torque_extremities",
        "reset_base",
    ):
        if hasattr(events, name):
            setattr(events, name, None)


def main() -> None:
    if args.num_envs <= 0 or args.settle_seconds <= 0.0 or args.sample_seconds <= 0.0:
        raise ValueError("environment count and durations must be positive")

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    if args.terrain == "plane":
        env_cfg.scene.terrain = TerrainImporterCfg(
            prim_path="/World/ground",
            terrain_type="plane",
            collision_group=-1,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
        )
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, args.initial_root_height)
    env_cfg.scene.height_measurement_sensor.offset.pos = (0.0, 0.0, args.ray_origin_height)
    env_cfg.scene.robot.init_state.joint_pos = dict(
        zip(env_cfg.actions.joint_pos.expected_joint_names, PREPARE_POSITION, strict=True)
    )
    env_cfg.scene.robot.init_state.joint_vel = {".*": 0.0}
    env_cfg.commands.height.debug_vis = False
    env_cfg.episode_length_s = args.settle_seconds + args.sample_seconds + 5.0
    disable_events(env_cfg.events)
    env_cfg.curriculum = None

    env = gym.make(args.task, cfg=env_cfg)
    base_env = env.unwrapped
    lift = base_env.action_manager.get_term("lift")
    lift.scale_forces(0.0)
    joint_action = base_env.action_manager.get_term("joint_pos")
    target = torch.tensor(PREPARE_POSITION, device=base_env.device, dtype=torch.float32).unsqueeze(0)
    action = (target - joint_action._center) / joint_action._scale
    action = action.expand(base_env.num_envs, -1).contiguous()

    env.reset()
    initial_height = base_env.command_manager.get_term("height").measured_height.detach().cpu()
    total_steps = int((args.settle_seconds + args.sample_seconds) / base_env.step_dt)
    sample_start = int(args.settle_seconds / base_env.step_dt)
    heights: list[torch.Tensor] = []
    root_heights: list[torch.Tensor] = []
    joint_errors: list[torch.Tensor] = []
    terminations = 0
    first_below_0_4_s = torch.full((base_env.num_envs,), float("nan"))
    timeline: list[dict[str, float]] = []

    for step in range(total_steps):
        _, _, terminated, truncated, _ = env.step(action)
        terminations += int(torch.count_nonzero(terminated | truncated).item())
        current_height = base_env.command_manager.get_term("height").measured_height.detach().cpu()
        newly_below = torch.isnan(first_below_0_4_s) & (current_height < 0.4)
        first_below_0_4_s[newly_below] = (step + 1) * base_env.step_dt
        if step % max(1, round(0.25 / base_env.step_dt)) == 0:
            root_z = base_env.scene["robot"].data.root_pos_w[:, 2].detach().cpu()
            ground_z = root_z - current_height
            timeline.append(
                {
                    "time_s": (step + 1) * base_env.step_dt,
                    "mean_trunk_height_m": float(current_height.mean()),
                    "minimum_trunk_height_m": float(current_height.min()),
                    "maximum_trunk_height_m": float(current_height.max()),
                    "mean_root_world_z_m": float(root_z.mean()),
                    "mean_ground_world_z_m": float(ground_z.mean()),
                }
            )
        if step >= sample_start:
            heights.append(current_height)
            root_heights.append(base_env.scene["robot"].data.root_pos_w[:, 2].detach().cpu())
            actual = joint_action.deployment_joint_position.detach().cpu()
            joint_errors.append(torch.abs(actual - target.cpu()))

    height = torch.stack(heights)
    root_height = torch.stack(root_heights)
    joint_error = torch.stack(joint_errors)
    final_position = joint_action.deployment_joint_position.detach().cpu()
    mean_joint_error = joint_error.mean(dim=(0, 1))
    maximum_joint_error = joint_error.amax(dim=(0, 1))
    joint_names = env_cfg.actions.joint_pos.expected_joint_names
    result = {
        "task": args.task,
        "terrain": args.terrain,
        "ray_origin_height_m": args.ray_origin_height,
        "num_envs": args.num_envs,
        "lift_scale": lift.force_scale,
        "settle_seconds": args.settle_seconds,
        "sample_seconds": args.sample_seconds,
        "initial_root_height_m": args.initial_root_height,
        "initial_measured_trunk_height_m": {
            "mean": float(initial_height.mean()),
            "minimum": float(initial_height.min()),
            "maximum": float(initial_height.max()),
        },
        "prepare_position_serial_rad": PREPARE_POSITION,
        "measured_trunk_height_m": {
            "mean": float(height.mean()),
            "std": float(height.std()),
            "minimum": float(height.min()),
            "maximum": float(height.max()),
        },
        "root_world_z_m": {
            "mean": float(root_height.mean()),
            "std": float(root_height.std()),
        },
        "joint_abs_error_rad": {
            "mean": float(joint_error.mean()),
            "maximum": float(joint_error.max()),
        },
        "joint_diagnostics": {
            name: {
                "target": PREPARE_POSITION[index],
                "final_mean": float(final_position[:, index].mean()),
                "mean_abs_error": float(mean_joint_error[index]),
                "maximum_abs_error": float(maximum_joint_error[index]),
            }
            for index, name in enumerate(joint_names)
        },
        "final_root_quaternion_wxyz": [
            float(value)
            for value in base_env.scene["robot"].data.root_quat_w.mean(dim=0).detach().cpu()
        ],
        "termination_count": terminations,
        "first_below_0_4_s": [
            None if torch.isnan(value) else float(value) for value in first_below_0_4_s
        ],
        "timeline": timeline,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()

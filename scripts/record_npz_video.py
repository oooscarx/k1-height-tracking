#!/usr/bin/env python3
"""Render one K1 motion NPZ to an MP4 with an Isaac Lab camera sensor."""

from __future__ import annotations

import argparse
import pathlib

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--motion", type=pathlib.Path, required=True)
parser.add_argument("--output", type=pathlib.Path, required=True)
parser.add_argument("--width", type=int, default=960)
parser.add_argument("--height", type=int, default=720)
parser.add_argument(
    "--frame-stride",
    type=int,
    default=1,
    help="Render every Nth motion frame while preserving playback speed.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import imageio.v2 as imageio
import isaaclab.sim as sim_utils
import numpy as np
import torch
from booster_train.assets.robots.booster import BOOSTER_K1_CFG
from booster_train.tasks.manager_based.beyond_mimic.mdp.commands import MotionLoader
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import Camera, CameraCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass


@configclass
class VideoSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(intensity=1800.0, color=(0.85, 0.85, 0.85)),
    )
    robot: ArticulationCfg = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera",
        update_period=0.0,
        height=args_cli.height,
        width=args_cli.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=34.0,
            focus_distance=4.0,
            horizontal_aperture=30.0,
            clipping_range=(0.05, 20.0),
        ),
    )


def _write_robot_state(
    robot: Articulation,
    motion: MotionLoader,
    frame: int,
    scene: InteractiveScene,
) -> torch.Tensor:
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] = motion.body_pos_w[frame, 0] + scene.env_origins
    root_state[:, 3:7] = motion.body_quat_w[frame, 0]
    root_state[:, 7:10] = motion.body_lin_vel_w[frame, 0]
    root_state[:, 10:] = motion.body_ang_vel_w[frame, 0]
    robot.write_root_state_to_sim(root_state)
    robot.write_joint_state_to_sim(
        motion.joint_pos[frame].unsqueeze(0),
        motion.joint_vel[frame].unsqueeze(0),
    )
    scene.write_data_to_sim()
    return root_state


def main() -> None:
    motion_path = args_cli.motion.resolve()
    output_path = args_cli.output.resolve()
    if not motion_path.is_file():
        raise FileNotFoundError(motion_path)
    if args_cli.frame_stride < 1:
        raise ValueError("--frame-stride must be at least 1")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with np.load(motion_path) as data:
        fps = int(np.asarray(data["fps"]).reshape(-1)[0])

    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / fps))
    scene = InteractiveScene(VideoSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()

    robot: Articulation = scene["robot"]
    camera: Camera = scene["camera"]
    motion = MotionLoader(
        str(motion_path),
        ["Trunk"],
        robot.joint_names,
        tail_len=0,
        device=str(sim.device),
    )

    first_root = _write_robot_state(robot, motion, 0, scene)
    camera_target = first_root[:, :3] + torch.tensor((0.0, 0.0, 0.25), device=sim.device)
    camera_eye = camera_target + torch.tensor((1.45, 1.45, 0.55), device=sim.device)
    camera.set_world_poses_from_view(camera_eye, camera_target)
    for _ in range(8):
        sim.render()
        scene.update(sim.get_physics_dt())

    writer = imageio.get_writer(
        output_path,
        fps=fps / args_cli.frame_stride,
        codec="libx264",
        pixelformat="yuv420p",
        quality=8,
        macro_block_size=None,
    )
    frame_variances = []
    try:
        frame_indexes = range(0, motion.time_step_total, args_cli.frame_stride)
        for frame in frame_indexes:
            root_state = _write_robot_state(robot, motion, frame, scene)
            camera_target = root_state[:, :3] + torch.tensor((0.0, 0.0, 0.25), device=sim.device)
            camera_eye = camera_target + torch.tensor((1.45, 1.45, 0.55), device=sim.device)
            camera.set_world_poses_from_view(camera_eye, camera_target)
            sim.render()
            scene.update(sim.get_physics_dt())
            rgb = camera.data.output["rgb"][0, :, :, :3].cpu().numpy().astype(np.uint8)
            frame_variances.append(float(np.var(rgb)))
            writer.append_data(rgb)
    finally:
        writer.close()

    if min(frame_variances) < 1.0 or np.mean(frame_variances) < 10.0:
        output_path.unlink(missing_ok=True)
        raise RuntimeError("camera output is blank or nearly blank")
    print(
        f"saved {output_path} ({len(frame_variances)} frames at "
        f"{fps / args_cli.frame_stride:g} Hz, "
        f"mean pixel variance {np.mean(frame_variances):.1f})"
    )


if __name__ == "__main__":
    main()
    simulation_app.close()

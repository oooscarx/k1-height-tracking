#!/usr/bin/env python3
"""Render an Isaac Lab state trajectory without starting a GPU renderer."""

from __future__ import annotations

import argparse
import math
import subprocess
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


@dataclass
class Visual:
    body_index: int
    points: np.ndarray
    color: tuple[int, int, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", type=Path)
    parser.add_argument("urdf", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--fps", type=int, default=25)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--trim-tail", type=int, default=10)
    parser.add_argument("--max-points", type=int, default=2500)
    return parser.parse_args()


def rotation_from_rpy(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    rx = np.array(((1, 0, 0), (0, cr, -sr), (0, sr, cr)), dtype=np.float32)
    ry = np.array(((cp, 0, sp), (0, 1, 0), (-sp, 0, cp)), dtype=np.float32)
    rz = np.array(((cy, -sy, 0), (sy, cy, 0), (0, 0, 1)), dtype=np.float32)
    return rz @ ry @ rx


def rotation_from_quaternion_wxyz(quaternion: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        (
            (1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)),
            (2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)),
            (2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)),
        ),
        dtype=np.float32,
    )


def parse_vector(value: str | None, default: tuple[float, float, float]) -> np.ndarray:
    if value is None:
        return np.asarray(default, dtype=np.float32)
    return np.asarray([float(item) for item in value.split()], dtype=np.float32)


def load_stl_vertices(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    if len(payload) >= 84:
        triangle_count = int.from_bytes(payload[80:84], "little")
        if 84 + triangle_count * 50 == len(payload):
            dtype = np.dtype(
                [
                    ("normal", "<f4", (3,)),
                    ("vertices", "<f4", (3, 3)),
                    ("attribute", "<u2"),
                ]
            )
            return np.frombuffer(payload, dtype=dtype, count=triangle_count, offset=84)["vertices"].reshape(-1, 3)

    vertices = []
    for line in payload.decode("utf-8", errors="ignore").splitlines():
        fields = line.strip().split()
        if len(fields) == 4 and fields[0].lower() == "vertex":
            vertices.append(tuple(float(field) for field in fields[1:]))
    if not vertices:
        raise ValueError(f"STL contains no vertices: {path}")
    return np.asarray(vertices, dtype=np.float32)


def primitive_points(geometry: ET.Element) -> np.ndarray | None:
    box = geometry.find("box")
    if box is not None:
        half = parse_vector(box.get("size"), (0.1, 0.1, 0.1)) * 0.5
        return np.asarray(
            [(x, y, z) for x in (-half[0], half[0]) for y in (-half[1], half[1]) for z in (-half[2], half[2])],
            dtype=np.float32,
        )
    sphere = geometry.find("sphere")
    if sphere is not None:
        radius = float(sphere.get("radius", "0.05"))
        angles = np.linspace(0.0, 2.0 * math.pi, 24, endpoint=False)
        return np.asarray(
            [(radius * math.cos(a), radius * math.sin(a), z) for z in (-radius, 0.0, radius) for a in angles],
            dtype=np.float32,
        )
    cylinder = geometry.find("cylinder")
    if cylinder is not None:
        radius = float(cylinder.get("radius", "0.05"))
        half_length = float(cylinder.get("length", "0.1")) * 0.5
        angles = np.linspace(0.0, 2.0 * math.pi, 32, endpoint=False)
        return np.asarray(
            [(radius * math.cos(a), radius * math.sin(a), z) for z in (-half_length, half_length) for a in angles],
            dtype=np.float32,
        )
    return None


def tint_for_link(color: np.ndarray, link_name: str) -> np.ndarray:
    if link_name.startswith("Left") or link_name.startswith("left"):
        color *= np.asarray((0.90, 0.97, 1.05), dtype=np.float32)
    elif link_name.startswith("Right") or link_name.startswith("right"):
        color *= np.asarray((1.04, 0.95, 0.90), dtype=np.float32)
    return np.clip(color, 0.0, 1.0)


def load_visuals(urdf_path: Path, body_names: list[str], max_points: int) -> list[Visual]:
    root = ET.parse(urdf_path).getroot()
    body_index = {name: index for index, name in enumerate(body_names)}
    visuals: list[Visual] = []
    for link in root.findall("link"):
        name = link.get("name")
        if name not in body_index:
            continue
        for visual in link.findall("visual"):
            geometry = visual.find("geometry")
            if geometry is None:
                continue
            mesh = geometry.find("mesh")
            if mesh is not None:
                mesh_path = (urdf_path.parent / mesh.get("filename", "")).resolve()
                points = load_stl_vertices(mesh_path)
                points *= parse_vector(mesh.get("scale"), (1.0, 1.0, 1.0))
            else:
                points = primitive_points(geometry)
                if points is None:
                    continue
            if len(points) > max_points:
                indices = np.linspace(0, len(points) - 1, max_points, dtype=np.int64)
                points = points[indices]
            origin = visual.find("origin")
            xyz = parse_vector(origin.get("xyz") if origin is not None else None, (0.0, 0.0, 0.0))
            rpy = parse_vector(origin.get("rpy") if origin is not None else None, (0.0, 0.0, 0.0))
            points = points @ rotation_from_rpy(rpy).T + xyz
            color_node = visual.find("material/color")
            rgba = parse_vector(color_node.get("rgba") if color_node is not None else None, (0.72, 0.75, 0.79))
            rgb = tint_for_link(rgba[:3], name)
            visuals.append(Visual(body_index=body_index[name], points=points, color=tuple((rgb * 255).astype(int))))
    if not visuals:
        raise ValueError(f"No URDF visuals matched trajectory bodies: {urdf_path}")
    return visuals


def convex_hull(points: np.ndarray) -> list[tuple[int, int]]:
    unique = sorted(set(map(tuple, np.rint(points).astype(np.int32))))
    if len(unique) <= 2:
        return unique

    def cross(origin, first, second):
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (first[1] - origin[1]) * (
            second[0] - origin[0]
        )

    lower = []
    for point in unique:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)
    upper = []
    for point in reversed(unique):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)
    return lower[:-1] + upper[:-1]


def camera_axes(eye: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    forward = target - eye
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, np.asarray((0.0, 0.0, 1.0), dtype=np.float32))
    right /= np.linalg.norm(right)
    up = np.cross(right, forward)
    return right, up, forward


def project_points(
    points: np.ndarray,
    eye: np.ndarray,
    axes: tuple[np.ndarray, np.ndarray, np.ndarray],
    focal_length: float,
    principal: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    relative = points - eye
    camera = np.column_stack(tuple(relative @ axis for axis in axes))
    depth = camera[:, 2]
    safe_depth = np.maximum(depth, 0.05)
    projected = np.column_stack(
        (
            principal[0] + focal_length * camera[:, 0] / safe_depth,
            principal[1] - focal_length * camera[:, 1] / safe_depth,
        )
    )
    return projected, depth


def load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = (
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
        if bold
        else "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    )
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    color,
    width: int,
) -> None:
    draw.line((start, end), fill=color, width=width)
    angle = math.atan2(end[1] - start[1], end[0] - start[0])
    head = 10 + width
    for offset in (-0.55, 0.55):
        point = (end[0] - head * math.cos(angle + offset), end[1] - head * math.sin(angle + offset))
        draw.line((end, point), fill=color, width=width)


def draw_chart(
    draw: ImageDraw.ImageDraw,
    command: np.ndarray,
    measured: np.ndarray,
    frame_index: int,
    rect: tuple[int, int, int, int],
    font: ImageFont.ImageFont,
) -> None:
    left, top, right, bottom = rect
    draw.rounded_rectangle(rect, radius=6, fill=(250, 252, 253, 230), outline=(190, 198, 205), width=1)
    pad_left, pad_top, pad_right, pad_bottom = 42, 28, 12, 24
    plot = (left + pad_left, top + pad_top, right - pad_right, bottom - pad_bottom)
    p_left, p_top, p_right, p_bottom = plot
    y_min, y_max = 0.0, 0.75
    for value in (0.0, 0.25, 0.5, 0.75):
        y = p_bottom - (value - y_min) / (y_max - y_min) * (p_bottom - p_top)
        draw.line((p_left, y, p_right, y), fill=(220, 225, 229), width=1)
        draw.text((left + 6, y - 7), f"{value:.2f}", fill=(90, 99, 106), font=font)
    sample_indices = np.linspace(0, len(command) - 1, min(300, len(command)), dtype=np.int64)

    def curve(values: np.ndarray) -> list[tuple[float, float]]:
        return [
            (
                p_left + index / max(1, len(command) - 1) * (p_right - p_left),
                p_bottom - (float(values[index]) - y_min) / (y_max - y_min) * (p_bottom - p_top),
            )
            for index in sample_indices
        ]

    draw.line(curve(command), fill=(36, 151, 92), width=3)
    draw.line(curve(measured), fill=(40, 111, 180), width=3)
    marker_x = p_left + frame_index / max(1, len(command) - 1) * (p_right - p_left)
    draw.line((marker_x, p_top, marker_x, p_bottom), fill=(40, 45, 50), width=2)
    draw.text((left + 8, top + 6), "HEIGHT TRACKING", fill=(35, 41, 46), font=font)
    draw.line((right - 155, top + 14, right - 137, top + 14), fill=(36, 151, 92), width=3)
    draw.text((right - 132, top + 7), "target", fill=(50, 58, 64), font=font)
    draw.line((right - 79, top + 14, right - 61, top + 14), fill=(40, 111, 180), width=3)
    draw.text((right - 56, top + 7), "actual", fill=(50, 58, 64), font=font)


def main() -> None:
    args = parse_args()
    data = np.load(args.trajectory)
    body_pos = data["body_pos_w"]
    body_quat = data["body_quat_w"]
    body_names = data["body_names"].tolist()
    command = data["height_command"]
    measured = data["measured_height"]
    lift_scale = float(data["lift_force_scale"])
    step_dt = float(data["step_dt"])
    visuals = load_visuals(args.urdf, body_names, args.max_points)

    final_frame = max(1, len(body_pos) - args.trim_tail)
    stride = max(1, round(1.0 / (args.fps * step_dt)))
    frame_indices = np.arange(0, final_frame, stride, dtype=np.int64)
    trajectory_center = np.asarray(
        (
            np.mean(body_pos[:final_frame, 0, 0]),
            np.mean(body_pos[:final_frame, 0, 1]),
            0.35,
        ),
        dtype=np.float32,
    )
    eye = trajectory_center + np.asarray((2.2, -2.7, 1.25), dtype=np.float32)
    axes = camera_axes(eye, trajectory_center)
    focal_length = args.width * 0.86
    principal = (args.width * 0.43, args.height * 0.48)

    title_font = load_font(24, bold=True)
    body_font = load_font(18)
    small_font = load_font(13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_command = [
        "ffmpeg",
        "-y",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{args.width}x{args.height}",
        "-r",
        str(args.fps),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "19",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(args.output),
    ]

    with subprocess.Popen(ffmpeg_command, stdin=subprocess.PIPE) as encoder:
        assert encoder.stdin is not None
        for frame_index in frame_indices:
            image = Image.new("RGB", (args.width, args.height), (238, 241, 243))
            draw = ImageDraw.Draw(image, "RGBA")

            grid_center = trajectory_center.copy()
            grid_center[2] = 0.0
            for offset in np.arange(-2.0, 2.01, 0.25):
                intensity = (175, 183, 189, 190) if abs(offset % 1.0) < 1.0e-5 else (202, 208, 212, 155)
                first = np.asarray((grid_center[0] - 2.0, grid_center[1] + offset, 0.0), dtype=np.float32)
                second = np.asarray((grid_center[0] + 2.0, grid_center[1] + offset, 0.0), dtype=np.float32)
                projected, depth = project_points(np.vstack((first, second)), eye, axes, focal_length, principal)
                if np.all(depth > 0.0):
                    draw.line(tuple(map(tuple, projected)), fill=intensity, width=1)
                first = np.asarray((grid_center[0] + offset, grid_center[1] - 2.0, 0.0), dtype=np.float32)
                second = np.asarray((grid_center[0] + offset, grid_center[1] + 2.0, 0.0), dtype=np.float32)
                projected, depth = project_points(np.vstack((first, second)), eye, axes, focal_length, principal)
                if np.all(depth > 0.0):
                    draw.line(tuple(map(tuple, projected)), fill=intensity, width=1)

            polygons = []
            for visual in visuals:
                rotation = rotation_from_quaternion_wxyz(body_quat[frame_index, visual.body_index])
                world_points = visual.points @ rotation.T + body_pos[frame_index, visual.body_index]
                projected, depth = project_points(world_points, eye, axes, focal_length, principal)
                visible = depth > 0.05
                if np.count_nonzero(visible) < 3:
                    continue
                hull = convex_hull(projected[visible])
                if len(hull) < 3:
                    continue
                depth_mean = float(np.mean(depth[visible]))
                height_shade = 0.86 + 0.12 * np.clip(np.mean(world_points[:, 2]), 0.0, 0.8) / 0.8
                color = tuple(np.clip(np.asarray(visual.color) * height_shade, 0, 255).astype(int))
                polygons.append((depth_mean, hull, color))
            for _, hull, color in sorted(polygons, key=lambda item: item[0], reverse=True):
                draw.polygon(hull, fill=(*color, 255), outline=(55, 62, 67, 235))

            trunk_point = body_pos[frame_index, 0]
            arrow_points = []
            for lateral in (-0.15, 0.15):
                start = trunk_point + np.asarray((0.0, lateral, -0.16), dtype=np.float32)
                end = start + np.asarray((0.0, 0.0, 0.28), dtype=np.float32)
                projected, depth = project_points(np.vstack((start, end)), eye, axes, focal_length, principal)
                if np.all(depth > 0.0):
                    arrow_points.append(projected)
            for projected in arrow_points:
                draw_arrow(draw, tuple(projected[0]), tuple(projected[1]), (218, 116, 38, 235), 4)

            panel = (20, 18, 330, 132)
            draw.rounded_rectangle(panel, radius=6, fill=(250, 252, 253, 235), outline=(184, 193, 199), width=1)
            draw.text((35, 30), "K1 HEIGHT TRACKING", fill=(30, 36, 41), font=title_font)
            time_seconds = frame_index * step_dt
            draw.text((35, 66), f"t = {time_seconds:5.2f} s", fill=(65, 73, 79), font=body_font)
            draw.text((35, 94), f"LIFT SCALE  {lift_scale:.4f}", fill=(188, 84, 25), font=body_font)
            draw.text((205, 66), f"target {command[frame_index]:.3f} m", fill=(30, 132, 78), font=body_font)
            draw.text((205, 94), f"actual {measured[frame_index]:.3f} m", fill=(34, 96, 162), font=body_font)
            draw_chart(
                draw,
                command[:final_frame],
                measured[:final_frame],
                int(frame_index),
                (560, 500, 944, 704),
                small_font,
            )
            encoder.stdin.write(np.asarray(image, dtype=np.uint8).tobytes())
        encoder.stdin.close()
        return_code = encoder.wait()
    if return_code != 0:
        raise RuntimeError(f"ffmpeg exited with status {return_code}")
    print(f"Rendered {len(frame_indices)} frames to {args.output}")


if __name__ == "__main__":
    main()

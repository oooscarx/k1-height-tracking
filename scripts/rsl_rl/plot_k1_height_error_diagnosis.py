#!/usr/bin/env python3
"""Plot K1 height-tracking error diagnostics from existing training artifacts."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


ERROR_TAG = "Metrics/height/high_height_error"
SUCCESS_TAG = "Metrics/height/high_height_success"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def rolling_mean(values: np.ndarray, window: int = 50) -> np.ndarray:
    if len(values) < window:
        return values
    kernel = np.ones(window, dtype=np.float64) / window
    smooth = np.convolve(values, kernel, mode="valid")
    return np.concatenate((np.full(window - 1, np.nan), smooth))


def load_training_runs(repo_root: Path) -> list[dict[str, object]]:
    event_root = repo_root / "logs/rsl_rl/height_tracking_k1_mastery"
    candidates = sorted(
        event_root.glob("*/events.out.tfevents.*"),
        key=lambda path: path.stat().st_mtime,
    )
    runs = []
    for event_path in candidates:
        accumulator = EventAccumulator(str(event_path), size_guidance={"scalars": 0})
        accumulator.Reload()
        tags = accumulator.Tags().get("scalars", [])
        if ERROR_TAG not in tags or SUCCESS_TAG not in tags:
            continue
        error_events = accumulator.Scalars(ERROR_TAG)
        success_events = accumulator.Scalars(SUCCESS_TAG)
        if not error_events or max(event.step for event in error_events) < 130_000:
            continue
        runs.append(
            {
                "name": event_path.parent.name,
                "error_steps": np.asarray([event.step for event in error_events]),
                "errors": np.asarray([event.value for event in error_events]),
                "success_steps": np.asarray([event.step for event in success_events]),
                "success": np.asarray([event.value for event in success_events]),
            }
        )
    return runs[-2:]


def load_bins(path: Path) -> dict[str, object]:
    return json.loads(path.read_text())


def terrain_summary(data: dict[str, object]) -> dict[str, tuple[float, float]]:
    grouped: dict[str, dict[str, list[float]]] = defaultdict(lambda: {"error": [], "success": []})
    for height_bin in data["bins"]:
        rows = zip(
            height_bin["environment_terrain_type"],
            height_bin["environment_mean_abs_error_m"],
            height_bin["environment_fraction_within_0_08_m"],
        )
        for terrain_type, error, success in rows:
            if terrain_type <= 2:
                family = "Boxes"
            elif terrain_type <= 5:
                family = "Random rough"
            else:
                family = "Waves"
            grouped[family]["error"].append(error)
            grouped[family]["success"].append(success)
    return {
        family: (float(np.mean(values["error"])), float(np.mean(values["success"])))
        for family, values in grouped.items()
    }


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    artifact_dir = repo_root / "artifacts/height_error_diagnosis_2026-09-01"
    clean = load_bins(artifact_dir / "height_mastery_bins_model_124750_standing_clean_v2.json")
    randomized = load_bins(artifact_dir / "height_mastery_bins_model_124750_standing_randomized_v3.json")
    runs = load_training_runs(repo_root)

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.facecolor": "#f5f6f8",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.color": "#d9dde3",
            "grid.alpha": 0.75,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=True)
    fig.suptitle("K1 Height-Tracking Error Diagnosis", fontsize=18, fontweight="bold")

    ax = axes[0, 0]
    ax_success = ax.twinx()
    colors = ["#8b95a5", "#1565c0"]
    for index, run in enumerate(runs):
        label = "pre-fix segment" if index == 0 else "settled-metric restart"
        error_steps = run["error_steps"]
        errors = run["errors"]
        success_steps = run["success_steps"]
        success = run["success"]
        ax.plot(error_steps, errors * 100, color=colors[index], alpha=0.22, linewidth=0.7)
        ax.plot(
            error_steps,
            rolling_mean(errors) * 100,
            color=colors[index],
            linewidth=2.0,
            label=f"error: {label}",
        )
        ax_success.plot(
            success_steps,
            rolling_mean(success) * 100,
            color=colors[index],
            linestyle="--",
            linewidth=1.4,
            label=f"success: {label}",
        )
    ax.axhline(8.0, color="#d32f2f", linestyle=":", linewidth=2, label="K1 mastery gate: 8 cm")
    ax.axhline(10.0, color="#f57c00", linestyle=":", linewidth=1.6, label="WBC-AGILE lift gate: 10 cm")
    ax.set_title("A. Current mastery training (TensorBoard)")
    ax.set_xlabel("iteration")
    ax.set_ylabel("settled high-height MAE (cm)")
    ax_success.set_ylabel("within 8 cm (%)")
    ax.set_ylim(6, 14)
    ax_success.set_ylim(55, 100)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax_success.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", ncol=2)

    ax = axes[0, 1]
    target = np.asarray([row["target_height_m"] for row in clean["bins"]])
    clean_error = np.asarray([row["mean_abs_error_m"] for row in clean["bins"]]) * 100
    random_error = np.asarray([row["mean_abs_error_m"] for row in randomized["bins"]]) * 100
    clean_p95 = np.asarray([row["p95_abs_error_m"] for row in clean["bins"]]) * 100
    random_p95 = np.asarray([row["p95_abs_error_m"] for row in randomized["bins"]]) * 100
    x = np.arange(len(target))
    width = 0.36
    ax.bar(x - width / 2, clean_error, width, color="#2e7d32", label="clean standing: mean")
    ax.bar(x + width / 2, random_error, width, color="#c62828", label="randomized standing: mean")
    ax.scatter(x - width / 2, clean_p95, color="#153b18", marker="_", s=180, label="clean p95")
    ax.scatter(x + width / 2, random_p95, color="#641313", marker="_", s=180, label="randomized p95")
    ax.axhline(8.0, color="#1565c0", linestyle=":", linewidth=2, label="8 cm gate")
    ax.set_xticks(x, [f"{value:.2f}" for value in target])
    ax.set_xlabel("commanded height (m)")
    ax.set_ylabel("absolute height error (cm)")
    ax.set_title("B. Error by commanded height (model 124750, lift 0.1427)")
    ax.legend(loc="upper left", ncol=2)

    ax = axes[1, 0]
    clean_terrain = terrain_summary(clean)
    random_terrain = terrain_summary(randomized)
    families = ["Boxes", "Random rough", "Waves"]
    clean_values = np.asarray([clean_terrain[name][0] for name in families]) * 100
    random_values = np.asarray([random_terrain[name][0] for name in families]) * 100
    clean_success = np.asarray([clean_terrain[name][1] for name in families]) * 100
    random_success = np.asarray([random_terrain[name][1] for name in families]) * 100
    x = np.arange(len(families))
    ax.bar(x - width / 2, clean_values, width, color="#2e7d32", label="clean error")
    ax.bar(x + width / 2, random_values, width, color="#c62828", label="randomized error")
    ax.axhline(8.0, color="#1565c0", linestyle=":", linewidth=2, label="8 cm gate")
    ax.set_xticks(x, families)
    ax.set_ylabel("mean absolute error (cm)")
    ax.set_title("C. Error location by terrain family")
    ax2 = ax.twinx()
    ax2.plot(x, clean_success, color="#1b5e20", marker="o", linewidth=2, label="clean success")
    ax2.plot(x, random_success, color="#8e0000", marker="o", linewidth=2, label="randomized success")
    ax2.set_ylabel("within 8 cm (%)")
    ax2.set_ylim(0, 100)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", ncol=2)

    ax = axes[1, 1]
    time = np.linspace(-0.5, 4.0, 451)
    command = np.where(time < 0, 0.60, 0.72)
    metric_enabled = np.where(time <= 2.0, 0.0, 1.0)
    ax.step(time, command, where="post", color="#1565c0", linewidth=3, label="height command")
    ax.fill_between(time, 0, metric_enabled * 0.18 + 0.48, color="#2e7d32", alpha=0.16, step="post")
    ax.axvspan(0, 2.0, color="#f9a825", alpha=0.18, label="2 s settle: metric excluded")
    ax.axvline(0, color="#4b5563", linewidth=1)
    ax.axvline(2.0, color="#2e7d32", linestyle="--", linewidth=2)
    ax.text(0.08, 0.705, "instant jump", color="#1565c0", fontweight="bold")
    ax.text(2.08, 0.505, "height_error starts accumulating", color="#1b5e20", fontweight="bold")
    ax.text(
        0.03,
        0.04,
        "Official G1 config: velocity_range=(1000,1000), settle=2 s, resample=1-7 s\n"
        "adaptive lift decays at EMA height_error < 0.10 m.\n"
        "This is the code-defined accounting window, not an official training curve.",
        transform=ax.transAxes,
        va="bottom",
        bbox={"boxstyle": "round,pad=0.45", "facecolor": "white", "edgecolor": "#b8bec8"},
    )
    ax.set_xlim(-0.5, 4.0)
    ax.set_ylim(0.45, 0.76)
    ax.set_xlabel("seconds after command resample")
    ax.set_ylabel("commanded height (m)")
    ax.set_title("D. What WBC-AGILE actually measures (source-code schematic)")
    ax.legend(loc="upper right")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()

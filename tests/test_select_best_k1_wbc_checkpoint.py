from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "rsl_rl"
    / "select_best_k1_wbc_checkpoint.py"
)
SPEC = importlib.util.spec_from_file_location("select_best_k1_wbc_checkpoint", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SELECTOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SELECTOR)


def test_default_modes_are_mutually_exclusive() -> None:
    assert SELECTOR.DEFAULT_MODES == (
        "face_up",
        "face_down",
        "left_side",
        "right_side",
        "other",
    )


def _write_candidate(
    run_dir: Path,
    iteration: int,
    successes: list[int],
    *,
    joint_limit_braking_count: int = 0,
    motor_velocity_limit_exceeded_count: int = 0,
    parallel_fallback_count: int = 0,
    raw_action_clip_exceeded_count: int = 0,
) -> Path:
    modes = {}
    for mode, success_count in zip(SELECTOR.DEFAULT_MODES, successes, strict=True):
        modes[mode] = {
            "initial_mode_match_rate": 1.0,
            "lift_force_scale": 0.0,
            "num_envs": 128,
            "success_count": success_count,
            "success_rate": success_count / 128,
        }
    total_success = sum(successes)
    summary = {
        "aggregate": {
            "incomplete_count": 0,
            "joint_limit_braking_count": joint_limit_braking_count,
            "mean_recovery_time": 6.0,
            "motor_velocity_limit_exceeded_count": (
                motor_velocity_limit_exceeded_count
            ),
            "nonfinite_action_count": 0,
            "num_envs": 640,
            "parallel_fallback_count": parallel_fallback_count,
            "raw_action_clip_exceeded_count": raw_action_clip_exceeded_count,
            "success_count": total_success,
            "success_rate": total_success / 640,
        },
        "lift_force_scale": 0.0,
        "modes": modes,
    }
    checkpoint = run_dir / f"model_{iteration}.pt"
    checkpoint.write_bytes(b"checkpoint")
    evaluation_dir = run_dir / "evaluations" / f"model_{iteration}"
    evaluation_dir.mkdir(parents=True)
    summary_path = evaluation_dir / "summary_lift_0_terrain_0.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    return summary_path


def test_selector_prefers_balanced_aggregate_performance(tmp_path: Path) -> None:
    first = _write_candidate(tmp_path, 10_000, [81, 73, 101, 100, 79])
    second = _write_candidate(tmp_path, 12_500, [77, 74, 96, 99, 80])
    candidates = [
        SELECTOR.assess_candidate(first, tmp_path),
        SELECTOR.assess_candidate(second, tmp_path),
    ]
    selected = SELECTOR.select_best(candidates)
    assert selected["iteration"] == 10_000
    assert selected["eligible"]


def test_selector_rejects_unsafe_higher_success_candidate(tmp_path: Path) -> None:
    safe = _write_candidate(tmp_path, 10_000, [80, 80, 80, 80, 80])
    unsafe = _write_candidate(
        tmp_path,
        12_500,
        [110, 110, 110, 110, 110],
        parallel_fallback_count=1,
    )
    candidates = [
        SELECTOR.assess_candidate(safe, tmp_path),
        SELECTOR.assess_candidate(unsafe, tmp_path),
    ]
    selected = SELECTOR.select_best(candidates)
    assert selected["iteration"] == 10_000
    assert candidates[1]["reasons"] == ["parallel_fallback_count=1"]


def test_selector_reports_non_disqualifying_hardware_diagnostics(
    tmp_path: Path,
) -> None:
    summary = _write_candidate(
        tmp_path,
        10_000,
        [80, 80, 80, 80, 80],
        joint_limit_braking_count=640,
        motor_velocity_limit_exceeded_count=32,
        raw_action_clip_exceeded_count=16,
    )
    candidate = SELECTOR.assess_candidate(summary, tmp_path)

    assert candidate["eligible"]
    assert candidate["reasons"] == []
    assert candidate["evaluation_diagnostics"] == {
        "critic_value_abs_above_50_count": 0,
        "critic_value_abs_above_50_rate": 0.0,
        "joint_limit_braking_count": 640,
        "joint_limit_braking_rate": 1.0,
        "maximum_contact_force_norm": 0.0,
        "maximum_joint_velocity_abs": 0.0,
        "maximum_motor_utilization": 0.0,
        "maximum_motor_velocity_ratio": 0.0,
        "maximum_raw_action_abs": 0.0,
        "motor_velocity_limit_exceeded_count": 32,
        "motor_velocity_limit_exceeded_rate": 0.05,
        "raw_action_clip_exceeded_count": 16,
        "raw_action_clip_exceeded_rate": 0.025,
    }


def test_summary_discovery_ignores_noncanonical_model_directories(
    tmp_path: Path,
) -> None:
    summary = _write_candidate(tmp_path, 10_000, [80, 80, 80, 80, 80])
    ad_hoc_dir = tmp_path / "evaluations" / "model_20750_current"
    ad_hoc_dir.mkdir()
    (ad_hoc_dir / summary.name).write_text("{}", encoding="utf-8")

    assert SELECTOR.find_evaluation_summaries(tmp_path, summary.name) == [summary]


def test_external_baseline_can_defeat_all_finetune_candidates(
    tmp_path: Path,
) -> None:
    mastery_run = tmp_path / "mastery"
    mastery_run.mkdir()
    finetuned = _write_candidate(
        mastery_run,
        500,
        [90, 90, 90, 90, 70],
    )
    baseline_run = tmp_path / "legacy"
    baseline_run.mkdir()
    baseline = _write_candidate(
        baseline_run,
        17_500,
        [90, 90, 90, 90, 90],
    )
    baseline_checkpoint = baseline_run / "model_17500.pt"

    candidates = [
        SELECTOR.assess_candidate(finetuned, mastery_run),
        SELECTOR.assess_candidate(
            baseline,
            mastery_run,
            checkpoint_override=baseline_checkpoint,
        ),
    ]
    selected = SELECTOR.select_best(candidates)
    assert selected["iteration"] == 17_500
    assert selected["checkpoint"] == str(baseline_checkpoint.resolve())

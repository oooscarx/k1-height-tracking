from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "source/booster_train/booster_train/tasks/manager_based/height_tracking"
HARDWARE = (
    ROOT
    / "source/booster_train/booster_train/tasks/manager_based/fall_recovery/config/k1_fall_recovery.json"
)


def _load_k1_scaling_module():
    path = TASK / "k1_scaling.py"
    spec = importlib.util.spec_from_file_location("k1_height_scaling_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_height_tracking_matches_upstream_rates_and_schedule() -> None:
    source = (TASK / "robots/k1/env_cfg.py").read_text()
    assert "self.decimation = 4" in source
    assert "self.sim.dt = 1.0 / 200.0" in source
    assert "self.episode_length_s = 32.0" in source
    assert "resampling_time_range=(1.0, 7.0)" in source
    assert "velocity_range=(1000.0, 1000.0)" in source
    assert "standing_ratio=0.15" in source
    assert "flat_ratio=0.2" in source
    assert "settle_time_s=2.0" in source


def test_g1_absolute_height_quantities_are_scaled_for_k1() -> None:
    scaling = _load_k1_scaling_module()
    assert scaling.K1_LENGTH_SCALE == pytest.approx(0.52 / 0.72)
    assert scaling.K1_TRACKED_HEIGHT_SCALE == pytest.approx(0.72 / 0.92)
    assert scaling.K1_FULL_BODY_STANDING_HEIGHT == pytest.approx(0.3611111111)
    assert scaling.K1_MODERATE_STANDING_HEIGHT == pytest.approx(0.2888888889)
    assert scaling.K1_FULL_BODY_TRACKED_HEIGHT == pytest.approx(0.5611111111)
    assert scaling.K1_MODERATE_TRACKED_HEIGHT == pytest.approx(0.4888888889)
    assert scaling.K1_FULL_BODY_HEIGHT_GATE_RANGE == pytest.approx((0.5111111111, 0.6111111111))
    assert scaling.K1_MODERATE_HEIGHT_GATE_RANGE == pytest.approx((0.4388888889, 0.5388888889))
    assert scaling.K1_HEIGHT_GATE_AGE_RANGE == pytest.approx((0.5, 1.5))
    assert scaling.K1_TRACKING_ERROR_THRESHOLD == pytest.approx(0.10)
    assert scaling.K1_REPORTING_SUCCESS_ERROR_THRESHOLD == pytest.approx(0.08)
    assert scaling.K1_STILLNESS_ZERO_PENALTY_ERROR_THRESHOLD == pytest.approx(0.08)
    assert scaling.K1_STILLNESS_FULL_PENALTY_ERROR_THRESHOLD == pytest.approx(0.06)
    assert scaling.K1_HEIGHT_REWARD_ROUGH_STD == pytest.approx(0.3913043478)
    assert scaling.K1_HEIGHT_REWARD_MEDIUM_STD == pytest.approx(0.2347826087)
    assert scaling.K1_HEIGHT_REWARD_FINE_STD == pytest.approx(0.1565217391)

    env_source = (TASK / "robots/k1/env_cfg.py").read_text()
    assert '"standing_height_threshold": 0.5' not in env_source
    assert '"standing_height_threshold": 0.4' not in env_source
    assert '"threshold": 0.1,' not in env_source
    assert '"error_threshold": 0.1' not in env_source
    assert '"std": 0.5' not in env_source
    assert '"std": 0.3' not in env_source
    assert '"std": 0.2' not in env_source
    assert '"std": K1_HEIGHT_REWARD_ROUGH_STD' in env_source
    assert '"std": K1_HEIGHT_REWARD_MEDIUM_STD' in env_source
    assert '"std": K1_HEIGHT_REWARD_FINE_STD' in env_source


def test_g1_dimensional_disturbances_are_scaled_for_k1() -> None:
    scaling = _load_k1_scaling_module()
    assert scaling.K1_MASS_SCALE == pytest.approx(19.666 / 34.13385728)
    assert scaling.K1_TRUNK_MASS_SCALE == pytest.approx(6.50 / 8.562)
    assert scaling.K1_FORCE_SCALE == pytest.approx(scaling.K1_MASS_SCALE)
    assert scaling.K1_TORQUE_SCALE == pytest.approx(scaling.K1_MASS_SCALE * scaling.K1_LENGTH_SCALE)
    assert scaling.K1_PUSH_LINEAR_VELOCITY_SCALE == pytest.approx(scaling.K1_LENGTH_SCALE**0.5)

    source = (TASK / "robots/k1/env_cfg.py").read_text()
    assert "K1_TRUNK_MASS_SCALE" in source
    assert source.count("K1_LENGTH_SCALE") >= 7
    assert source.count("K1_PUSH_LINEAR_VELOCITY_SCALE") >= 5
    assert source.count("K1_FORCE_SCALE") >= 5
    assert source.count("K1_TORQUE_SCALE") >= 5


def test_height_regularizers_use_smooth_command_gates() -> None:
    env_source = (TASK / "robots/k1/env_cfg.py").read_text()
    reward_source = (TASK / "rewards.py").read_text()
    command_source = (TASK / "commands.py").read_text()

    assert "joint_deviation_for_height_command" in env_source
    assert "upright_orientation_for_height_command" in env_source
    assert "feet_distance_from_ref_for_height_command" in env_source
    assert "joint_deviation_if_standing" not in env_source
    assert "upright_orientation_after_standing" not in env_source
    assert "feet_distance_from_ref_if_standing" not in env_source
    assert "standing_height_threshold" not in env_source
    assert env_source.count('"command_name": "height"') >= 7
    assert "K1_FULL_BODY_HEIGHT_GATE_RANGE" in env_source
    assert "K1_MODERATE_HEIGHT_GATE_RANGE" in env_source
    assert "K1_HEIGHT_GATE_AGE_RANGE" in env_source

    assert "command.target_height" in reward_source
    assert "command.command_age_s" in reward_source
    assert "phase * phase * (3.0 - 2.0 * phase)" in reward_source
    assert "def command_age_s" in command_source


def test_stillness_penalty_is_smooth_below_lift_threshold() -> None:
    env_source = (TASK / "robots/k1/env_cfg.py").read_text()
    reward_source = (TASK / "rewards.py").read_text()

    assert "K1_STILLNESS_ZERO_PENALTY_ERROR_THRESHOLD" in env_source
    assert '"error_threshold": K1_STILLNESS_ZERO_PENALTY_ERROR_THRESHOLD' in env_source
    assert "K1_STILLNESS_FULL_PENALTY_ERROR_THRESHOLD" in env_source
    assert '"full_penalty_error_threshold": K1_STILLNESS_FULL_PENALTY_ERROR_THRESHOLD' in env_source
    assert '"settle_gate": True' in env_source
    assert "tracking_weight = 1.0 - _smoothstep(" in reward_source
    assert "command.settled" in reward_source


def test_mastery_inherits_upstream_multiscale_height_reward() -> None:
    source = (TASK / "robots/k1/mastery_env_cfg.py").read_text()
    assert "settle_gate" not in source
    assert "base_height_fine.weight" not in source
    assert "base_height_fine.params" not in source


def test_error_recovery_focuses_observed_bottleneck_without_disabling_randomization() -> None:
    source = (TASK / "robots/k1/error_recovery_env_cfg.py").read_text()
    registration = (TASK / "robots/k1/__init__.py").read_text()
    assert "command.focus_ratio = 0.65" in source
    assert "command.focus_height_range = (0.30, 0.72)" in source
    assert "command.resampling_time_range = (4.0, 7.0)" in source
    assert 'self.events.reset_base.params["standing_ratio"] = 0.75' in source
    assert "randomize_physics_material" not in source
    assert "apply_external_force_torque" not in source
    assert 'id="Booster-K1-Height-Tracking-ErrorRecovery-v0"' in registration


def test_progressive_robustness_gates_external_force_growth() -> None:
    source = (TASK / "robots/k1/progressive_robustness_env_cfg.py").read_text()
    registration = (TASK / "robots/k1/__init__.py").read_text()
    assert "self.scene.terrain.max_init_terrain_level = 7" in source
    assert '"error_metric_name": "high_height_error"' in source
    assert '"jump_error_threshold": 0.18' in source
    assert '"governed_error_threshold": 0.14' in source
    assert '"initial_scale": 1.0' in source
    assert '"maximum_scale": 1.5' in source
    assert 'id="Booster-K1-Height-Tracking-ProgressiveRobustness-v0"' in registration


def test_lift_curriculum_has_a_settled_positive_height_signal() -> None:
    source = (TASK / "robots/k1/env_cfg.py").read_text()
    assert "base_height_settled_fine = RewTerm(" in source
    assert 'weight=8.0' in source
    assert '"std": K1_HEIGHT_REWARD_FINE_STD, "settle_gate": True' in source


def test_k1_terrain_scales_only_vertical_g1_geometry() -> None:
    source = (TASK / "terrains.py").read_text()
    assert "vertical_scale=0.01 * K1_LENGTH_SCALE" in source
    assert "grid_height_range=(0.01 * K1_LENGTH_SCALE, 0.075 * K1_LENGTH_SCALE)" in source
    assert "noise_range=(0.01 * K1_LENGTH_SCALE, 0.1 * K1_LENGTH_SCALE)" in source
    assert "amplitude_range=(0.01 * K1_LENGTH_SCALE, 0.15 * K1_LENGTH_SCALE)" in source
    assert "grid_width=0.45" in source


def test_resume_keeps_checkpoint_action_semantics() -> None:
    source = (TASK / "robots/k1/env_cfg.py").read_text()
    assert "K1_ACTION_SCALE" not in source
    assert "result.append(min(raw," in source


def test_random_fallen_curriculum_is_not_registered() -> None:
    source = (TASK / "robots/k1/env_cfg.py").read_text()
    assert '"random_fallen_ratio": 0.0' in source
    assert "random_fallen_states = CurrTerm" not in source
    assert "fallen_state_dataset_secondary_cfg" not in (TASK / "robots/k1/ppo_cfg.py").read_text()


def test_absolute_action_range_stays_inside_hardware_limits() -> None:
    config = json.loads(HARDWARE.read_text())
    margin = config["native_teacher"]["robust_training"]["position_target_margin"]
    for raw, center, minimum, maximum in zip(
        config["native_teacher"]["training_action_scale"],
        config["goal_position"],
        config["position_minimum"],
        config["position_maximum"],
    ):
        scale = min(raw, center - minimum - margin, maximum - center - margin)
        assert scale > 0.0
        assert center - scale >= minimum + margin - 1e-9
        assert center + scale <= maximum - margin + 1e-9


def test_height_tracking_uses_serial_ankles_without_parallel_constraints() -> None:
    source = (TASK / "robots/k1/env_cfg.py").read_text()
    action_source = (
        ROOT
        / "source/booster_train/booster_train/tasks/manager_based/fall_recovery/actions.py"
    ).read_text()
    ppo_source = (TASK / "robots/k1/ppo_cfg.py").read_text()
    assert "K1SerialJointPositionActionCfg(" in source
    assert 'parallel_ankle=CONFIG["parallel_ankle"]' not in source
    assert "parallel_ankle = DoneTerm(" not in source
    assert "parallel_ankle_termination = RewTerm(" not in source
    assert "parallel_state_limit = RewTerm(" not in source
    assert "parallel_stress = RewTerm(" not in source
    assert "class K1SerialJointPositionActionCfg" in action_source
    assert "if self._parallel is None:" in action_source
    assert "expected_joint_names=None" in ppo_source
    assert "position_minimum=None" in ppo_source
    assert "position_maximum=None" in ppo_source
    assert "parallel_ankle=None" in ppo_source


def test_only_height_tracking_task_is_registered() -> None:
    registration = (TASK / "robots/k1/__init__.py").read_text()
    legacy = (
        ROOT
        / "source/booster_train/booster_train/tasks/manager_based/fall_recovery/robots/k1/__init__.py"
    ).read_text()
    assert 'id="Booster-K1-Height-Tracking-v0"' in registration
    assert "gym.register" not in legacy


def test_height_tracking_ppo_guards_return_scale_and_critic_outliers() -> None:
    source = (TASK / "robots/k1/ppo_cfg.py").read_text()
    train_source = (ROOT / "scripts/rsl_rl/train.py").read_text()
    assert "clip_actions = 4.0" in source
    assert "value_loss_huber_delta=10.0" in source
    assert "max_actor_observation_delta=10.0" in source
    assert "max_critic_observation_delta=50.0" in source
    assert "return_scale_decay=0.999" in source
    assert "outlier_threshold=10.0" in source
    assert "minimum_action_std: float | None = 0.04" in source
    assert "register_step_post_hook(clamp_action_std)" in train_source


def test_height_tracking_resume_accepts_explicit_lift_state() -> None:
    train_source = (ROOT / "scripts/rsl_rl/train.py").read_text()
    curriculum_source = (TASK / "curriculums.py").read_text()
    command_source = (TASK / "commands.py").read_text()
    assert "--wbc_initial_lift_scale" in train_source
    assert "--wbc_initial_lift_ema" in train_source
    assert 'cfg.params.get("initial_force_scale", 1.0)' in curriculum_source
    assert 'cfg.params.get("initial_ema", default_ema)' in curriculum_source
    assert "uses_instantaneous_height = (" in curriculum_source
    assert 'not hasattr(self._command, "settled")' in curriculum_source
    assert "def base_height" not in command_source


def test_resume_synchronizes_adaptive_learning_rate_from_optimizer() -> None:
    runner_source = (
        ROOT
        / "third_party/wbc_agile_rsl_rl/rsl_rl/runners/on_policy_runner.py"
    ).read_text()
    assert "self.alg.optimizer.load_state_dict" in runner_source
    assert (
        'self.alg.learning_rate = float(self.alg.optimizer.param_groups[0]["lr"])'
        in runner_source
    )

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


def test_stability_refinement_is_no_lift_smooth_and_strict() -> None:
    source = (TASK / "robots/k1/stability_env_cfg.py").read_text()
    registration = (TASK / "robots/k1/__init__.py").read_text()
    reward_source = (TASK / "rewards.py").read_text()

    assert "command.ranges.height = (0.54, 0.72)" in source
    assert "command.velocity_range = (0.04, 0.08)" in source
    assert "command.resampling_time_range = (10.0, 14.0)" in source
    assert "command.standing_ratio = 0.15" in source
    assert "command.flat_ratio = 0.15" in source
    assert "command.focus_ratio = 0.25" in source
    assert "command.high_height_threshold = 0.675" in source
    assert "command.success_error_threshold = 0.03" in source
    assert "command.initialize_from_measured_height = True" in source
    assert "if self.cfg.initialize_from_measured_height:" in (
        ROOT
        / "source/booster_train/booster_train/tasks/manager_based/height_tracking/commands.py"
    ).read_text()
    assert 'self.curriculum.adaptive_lift.params["initial_force_scale"] = 0.0' in source
    assert 'self.scene.terrain.terrain_type = "plane"' in source
    assert "self.scene.terrain.terrain_generator = None" in source
    assert '"push_robot"' in source
    assert '"randomize_physics_material"' in source
    assert "setattr(self.events, event_name, None)" in source
    assert '"apply_external_force_torque"' in source
    assert '"randomize_actuator_gains"' in source
    assert "self.observations.policy.enable_corruption = False" in source
    assert "self.rewards.not_moving = None" in source
    assert "base_height_tight" in source
    assert "base_height_l1" in source
    assert "base_height_stationary_tight" in source
    assert "stationary_base_motion" in source
    assert "upper_body_joint_velocity" in source
    assert "weight=-40.0" in source
    assert "self.rewards.feet_slide.weight = -5.0" in source
    assert "self.rewards.body_velocity.weight = -0.5" in source
    assert "def track_height_command_l1" in reward_source
    assert "def stationary_base_motion_l2" in reward_source
    assert "def positive_height_base_motion_l2" in reward_source
    assert "func=mdp.positive_height_base_motion_l2" in source
    assert "self.rewards.joint_deviation_l1 = None" not in source
    assert "DoneTermCfg as WbcDoneTerm" in source
    assert 'termination_type="bad"' in source
    assert "sigma=5.0" in source
    assert "height_posture_center" in source
    assert "height_posture_exponent = 0.6" in source
    assert "height_posture_blend = 0.40" in source
    assert "position_target_velocity_limit = [" in source
    assert "*([10.0] * 12)" in source
    assert 'for actuator_name in ("head", "arms")' in source
    assert "actuator.armature = 0.01" in source
    assert 'id="Booster-K1-Height-Tracking-Stability-v0"' in registration
    assert 'id=f"Booster-K1-Height-Tracking-Stability-{stage}-v0"' in registration
    assert "class K1HeightTrackingStabilityHighEnvCfg" in source
    assert "command.ranges.height = (0.68, 0.72)" in source
    assert "command.standing_ratio = 0.45" in source
    assert "command.flat_ratio = 0.45" in source
    assert "command.focus_ratio = 0.0" in source
    assert "posture_range: tuple[float, float] = (0.68, 0.72)" in source
    assert "action_cfg.height_posture_range = posture_range" in source
    assert "action_cfg.height_posture_blend = 1.0" in source
    assert "class K1HeightTrackingStabilityMidEnvCfg" in source
    assert "command.ranges.height = (0.60, 0.72)" in source
    assert "posture_range=(0.60, 0.72)" in source
    assert "self.actions.joint_pos.height_posture_blend = 0.16" in source
    assert "class K1HeightTrackingStabilityLowEnvCfg" in source
    assert "class K1HeightTrackingStabilityLowBlend17EnvCfg" in source
    assert "class K1HeightTrackingStabilityLowBlend17RateLimitedEnvCfg" in source
    assert "class K1HeightTrackingStabilityLowBlend17Damping075EnvCfg" in source
    assert "class K1HeightTrackingStabilityLowBlend17CorrelatedDelayEnvCfg" in source
    assert "class K1HeightTrackingStabilityLowBlend21CorrelatedDelayEnvCfg" in source
    assert "class K1HeightTrackingStabilityLowBlend21Residual100EnvCfg" in source
    assert "class K1HeightTrackingStabilityLowBlend22Residual100EnvCfg" in source
    assert "class K1HeightTrackingStabilityLowBlend23Residual100EnvCfg" in source
    assert "class K1HeightTrackingStabilityLowBlend21FailureResetEnvCfg" in source
    assert "joint_deviation_from_action_center" in source
    assert "position_target_velocity_limit_initialize_from_target = True" in source
    assert "position_target_velocity_limit_warmup_steps = 50" in source
    assert "class K1HeightTrackingStabilityLowDelay5EnvCfg" in source
    assert "class K1HeightTrackingStabilityLowTightDelay5EnvCfg" in source
    assert "class K1HeightTrackingStabilityLowTightDelay5Knee102EnvCfg" in source
    assert "class K1HeightTrackingStabilityLowTightDelay5Knee104EnvCfg" in source
    assert "class K1HeightTrackingStabilityLowTightDelay5Residual108EnvCfg" in source
    assert "class K1HeightTrackingStabilityLowTightDelay5Residual110EnvCfg" in source
    assert "class K1HeightTrackingStabilityLowTightDelay5Residual112EnvCfg" in source
    assert "class K1HeightTrackingStabilityLowTightDelay5Residual114EnvCfg" in source
    assert "class K1HeightTrackingStabilityLowTightDelay5Residual110KneeScale2EnvCfg" in source
    assert "class K1HeightTrackingStabilityLowTightDelay5Residual110Blend41EnvCfg" in source
    assert "class K1HeightTrackingStabilityLowTightDelay5Residual110Blend42EnvCfg" in source
    assert "class K1HeightTrackingStabilityLowTightDelay5Feedforward445EnvCfg" in source
    assert "class K1HeightTrackingStabilityLowTightDelay5Feedforward460EnvCfg" in source
    assert "class K1HeightTrackingStabilityLowTightDelay5Feedforward490Residual100EnvCfg" in source
    assert "class K1HeightTrackingStabilityLowTightDelay5Feedforward505Residual100EnvCfg" in source
    assert "class K1HeightTrackingStabilityLowTightDelay5Feedforward490Sagittal100EnvCfg" in source
    assert "class K1HeightTrackingStabilityLowTightDelay5Feedforward505Sagittal100EnvCfg" in source
    assert "class K1HeightTrackingStabilityLowTightDelay5Feedforward495Sagittal100EnvCfg" in source
    assert "class K1HeightTrackingStabilityLowTightDelay5Feedforward500Sagittal100EnvCfg" in source
    assert "class K1HeightTrackingStabilityOpenLoopFullRangeEnvCfg" in source
    assert "self.actions.joint_pos.height_posture_exponent = 1.4" in source
    assert "position_scale[knee_index] *= 2.0" in source
    assert "allow_position_scale_beyond_static_center = True" in source
    assert "_set_knee_delta_upper_bound(self.actions.joint_pos, 1.02)" in source
    assert "_set_knee_delta_upper_bound(self.actions.joint_pos, 1.04)" in source
    assert 'weight=-250.0' in source
    assert "class K1HeightTrackingStabilityLowDelay8EnvCfg" in source
    assert "self.commands.height.standing_ratio = 0.0" in source
    assert "self.commands.height.flat_ratio = 1.0" in source
    assert "self.actions.joint_pos.height_residual_base_maximum_scale = 1.05" in source
    assert "self.actions.joint_pos.height_residual_maximum_scale = 1.05" in source
    assert '("Low", "K1HeightTrackingStabilityLowEnvCfg")' in registration
    assert '("LowBlend17", "K1HeightTrackingStabilityLowBlend17EnvCfg")' in registration
    assert '"LowBlend17RateLimited"' in registration
    assert '"LowBlend17Damping075"' in registration
    assert '"LowBlend17CorrelatedDelay"' in registration
    assert '"LowBlend21CorrelatedDelay"' in registration
    assert '"LowBlend21Residual100"' in registration
    assert '"LowBlend22Residual100"' in registration
    assert '"LowBlend23Residual100"' in registration
    assert '"LowBlend21FailureReset"' in registration
    assert '("LowDelay5", "K1HeightTrackingStabilityLowDelay5EnvCfg")' in registration
    assert '("LowTightDelay5", "K1HeightTrackingStabilityLowTightDelay5EnvCfg")' in registration
    assert '"LowTightDelay5Knee102"' in registration
    assert '"LowTightDelay5Knee104"' in registration
    assert '"LowTightDelay5Residual108"' in registration
    assert '"LowTightDelay5Residual110"' in registration
    assert '"LowTightDelay5Residual112"' in registration
    assert '"LowTightDelay5Residual114"' in registration
    assert '"LowTightDelay5Residual110KneeScale2"' in registration
    assert '"LowTightDelay5Residual110Blend41"' in registration
    assert '"LowTightDelay5Residual110Blend42"' in registration
    assert '"LowTightDelay5Feedforward445"' in registration
    assert '"LowTightDelay5Feedforward460"' in registration
    assert '"LowTightDelay5Feedforward490Residual100"' in registration
    assert '"LowTightDelay5Feedforward505Residual100"' in registration
    assert '"LowTightDelay5Feedforward490Sagittal100"' in registration
    assert '"LowTightDelay5Feedforward505Sagittal100"' in registration
    assert '"LowTightDelay5Feedforward495Sagittal100"' in registration
    assert '"LowTightDelay5Feedforward500Sagittal100"' in registration
    assert '("OpenLoopFullRange", "K1HeightTrackingStabilityOpenLoopFullRangeEnvCfg")' in registration
    assert '("Handoff59", "K1HeightTrackingStabilityHandoff59EnvCfg")' in registration
    assert '("Handoff59Gain085", "K1HeightTrackingStabilityHandoff59Gain085EnvCfg")' in registration
    assert '"Handoff59Gain085Origin"' in registration
    assert "self.scene.env_spacing = 0.0" in source
    assert "self.actions.joint_pos.height_posture_exponent = 0.316" in source
    assert 'for actuator_name in ("legs", "feet")' in source
    assert "2.0 * value" in source
    assert '"FullRangeOrigin"' in registration
    assert "class K1HeightTrackingStabilityFullRangeOriginEnvCfg" in source
    assert '("FullRangeHandoff", "K1HeightTrackingStabilityFullRangeHandoffEnvCfg")' in registration
    assert '("LowDelay8", "K1HeightTrackingStabilityLowDelay8EnvCfg")' in registration

def test_web_export_preserves_the_stability_action_contract() -> None:
    exporter = (ROOT / "scripts/export_height_actor_onnx.py").read_text()
    browser_path = ROOT / "web/mujoco_wasm/src/main.ts"
    if not browser_path.exists():
        pytest.skip("the standalone browser demo is not installed on this training host")
    browser = browser_path.read_text()

    assert 'checkpoint.parent / "params" / "env.yaml"' in exporter
    assert '"height_posture_center": posture_center' in exporter
    assert '"height_posture_high_center": posture_high_center' in exporter
    assert '"height_posture_blend": posture_blend' in exporter
    assert '"height_residual_maximum_scale": residual_maximum_scale' in exporter
    assert '"height_residual_base_maximum_scale": residual_base_maximum_scale' in exporter
    assert '"height_residual_handoff_range": residual_handoff_range' in exporter
    assert '"height_residual_handoff_minimum_scale": residual_handoff_minimum_scale' in exporter
    assert '"height_residual_deep_handoff_range": residual_deep_handoff_range' in exporter
    assert '"height_residual_deep_handoff_minimum_scale": residual_deep_handoff_minimum_scale' in exporter
    assert '"height_residual_deep_handoff_joint_indices": residual_deep_handoff_joint_indices' in exporter
    assert '"height_residual_amplify_joint_indices": residual_amplify_joint_indices' in exporter
    assert '"height_posture_phase_knots": posture_phase_knots' in exporter
    assert '"last_action_scales": last_action_scales' in exporter
    assert '"position_target_velocity_limit": position_target_velocity_limit' in exporter
    assert '"actuator_delay_steps": delays' in exporter
    assert '"joint_armature": deployment_armature(config)' in exporter
    assert "private policyCenter(index: number)" in browser
    assert "this.manifest.height_posture_high_center" in browser
    assert "this.manifest.height_posture_blend" in browser
    assert "this.manifest.height_residual_maximum_scale ?? 1" in browser
    assert "this.manifest.height_residual_amplify_joint_indices" in browser
    assert "this.manifest.height_residual_handoff_minimum_scale" in browser
    assert "this.manifest.height_residual_handoff_range" in browser
    assert "this.manifest.height_residual_deep_handoff_range" in browser
    assert "this.manifest.height_residual_deep_handoff_joint_indices" in browser
    assert "this.manifest.height_posture_phase_knots" in browser
    assert "this.manifest.last_action_scales" in browser
    assert "this.model.dof_armature[6 + index]" in browser
    assert "this.manifest.position_target_velocity_limit[index]" in browser
    assert "this.targetDelayHistory.length - 1 - delay" in browser
    assert "MathUtils.clamp(rawTorque, -speedLimit, speedLimit)" in browser


def test_k1_height_actions_use_a_smooth_high_posture_curriculum() -> None:
    env_source = (TASK / "robots/k1/env_cfg.py").read_text()
    action_source = (
        ROOT
        / "source/booster_train/booster_train/tasks/manager_based/fall_recovery/actions.py"
    ).read_text()

    assert 'height_command_name="height"' in env_source
    assert "height_residual_fade_joint_names=JOINT_NAMES[10:]" in env_source
    assert "height_residual_amplify_joint_names=[" in env_source
    assert "height_residual_fade_range=(0.66, 0.72)" in env_source
    assert "height_residual_minimum_scale=0.90" in env_source
    assert "height_residual_base_maximum_scale=1.0" in env_source
    assert "height_residual_maximum_scale=1.0" in env_source
    assert "position_target_velocity_limit=10.0" in env_source
    assert "smooth_phase = phase * phase * (3.0 - 2.0 * phase)" in action_source
    assert "amplified_residual_scale = self.cfg.height_residual_minimum_scale" in action_source
    assert "self.cfg.height_residual_handoff_range" in action_source
    assert "self._env.step_dt" in action_source


def test_height_tracking_penalizes_forward_and_backward_trunk_pitch() -> None:
    env_source = (TASK / "robots/k1/env_cfg.py").read_text()
    assert '"axis": "pitch", "direction": "both", "kernel": "l2"' in env_source


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


def test_actor_sanitization_uses_the_configured_action_clip() -> None:
    source = (ROOT / "scripts/rsl_rl/play_height_diagnostic.py").read_text()
    assert "action_clip = float(agent_cfg.clip_actions)" in source
    assert "torch.clamp(teacher_raw_actions, -action_clip, action_clip)" in source
    assert "teacher_history[:, -1] = teacher_targets" in source
    assert "actor sanitization requires clip_actions=1.0" not in source


def test_height_response_distillation_anchors_low_and_shapes_high() -> None:
    source = (ROOT / "scripts/rsl_rl/play_height_diagnostic.py").read_text()
    assert '"--distill_height_response_output"' in source
    assert '"--distill_height_response_anchor"' in source
    assert "for parameter in output_layer.parameters()" in source
    assert "teacher_low[:, response_indices] + response_target" in source
    assert "teacher_high[:, response_indices] - response_target" in source
    assert 'args_cli.distill_height_response_anchor == "low"' in source
    assert 'infos["height_response_distillation"]' in source


def test_height_transition_diagnostic_accepts_a_custom_sequence() -> None:
    source = (ROOT / "scripts/rsl_rl/play_height_diagnostic.py").read_text()
    assert '"--height_transition_sequence"' in source
    assert '"--height_transition_zero_policy_actions"' in source
    assert '"zero_policy_actions"' in source
    assert "sequence = list(args_cli.height_transition_sequence)" in source
    assert "len(args_cli.height_transition_sequence) * args_cli.height_transition_hold_s" in source
    assert '"raw_action_mean"' in source
    assert '"executed_action_mean"' in source
    assert '"initial_root_yaw_rad"' in source
    assert '"--height_transition_initial_yaw_rad"' in source
    assert '"initial_root_yaw_override_rad"' in source
    assert '"--height_transition_residual_base_maximum_scale"' in source
    assert '"--height_transition_residual_handoff_scale"' in source
    assert '"residual_base_maximum_scale"' in source
    assert '"residual_handoff_scale"' in source
    assert '"--height_transition_leg_damping_multiplier"' in source
    assert '"leg_damping_multiplier"' in source
    assert '"--height_transition_bad_orientation_limit_rad"' in source
    assert '"maximum_torso_tilt_rad"' in source
    assert '"--height_transition_env_spacing"' in source
    assert '"env_spacing_m"' in source
    assert '"--height_transition_posture_depth_scale"' in source
    assert '"posture_depth_scale"' in source
    assert "for joint_index in (10, 13, 16, 19):" in source
    assert "action_cfg.position_target_margin" in source
    assert '"--height_transition_posture_exponent"' in source
    assert '"posture_exponent"' in source
    assert '"--height_transition_posture_low_exponent"' in source
    assert '"posture_low_exponent"' in source
    assert '"--height_transition_posture_minimum"' in source
    assert '"posture_range"' in source
    assert '"--height_policy_command_minimum_m"' in source
    assert '"height_policy_command_minimum_m"' in source
    assert "torch.clamp_min(" in source
    assert '"termination_terms"' in source
    assert "termination_manager.get_term(name)" in source
    assert '"steady_reward_terms_per_s"' in source
    assert "base_env.reward_manager._step_reward" in source


def test_full_range_stability_uses_a_calibrated_posture_curve() -> None:
    source = (TASK / "robots/k1/stability_env_cfg.py").read_text()
    assert "action.height_posture_phase_knots = (" in source
    assert "(0.585, 0.530)" in source
    assert "(0.59, 0.610)" in source
    assert "(0.60, 0.655)" in source
    assert "action.height_posture_low_handoff_height = None" in source
    assert "action.height_residual_deep_handoff_range = (0.56, 0.60)" in source
    assert "action.height_residual_deep_handoff_minimum_scale = 0.20" in source
    assert 'self.terminations.base_too_low.params["minimum_height"] = 0.24' in source
    full_range_source = source.split(
        "class K1HeightTrackingStabilityFullRangeOriginEnvCfg", 1
    )[1].split("class K1HeightTrackingStabilityLow57OriginEnvCfg", 1)[0]
    assert "ankle_roll_scale" not in full_range_source
    assert "position_target_velocity_limit =" not in full_range_source
    assert "class K1HeightTrackingStabilityLow57OriginEnvCfg" in source
    assert "command.ranges.height = (0.57, 0.585)" in source
    assert "command.focus_height_range = (0.57, 0.585)" in source
    assert "command.local_target_ratio = 1.00" in source
    assert "command.local_target_delta_range = (-0.02, 0.02)" in source
    assert "command.velocity_range = (0.08, 0.08)" in source
    assert "command.resampling_time_range = (3.0, 4.0)" in source
    assert 'id="Booster-K1-Height-Tracking-Stability-Low57Origin-v0"' in (
        TASK / "robots/k1/__init__.py"
    ).read_text()
    ppo_source = (TASK / "robots/k1/stability_ppo_cfg.py").read_text()
    assert "class K1HeightTrackingStabilityLow57PpoRunnerCfg" in ppo_source
    assert "self.freeze_resumed_action_std = True" in ppo_source
    assert "self.freeze_action_std = False" in ppo_source
    assert "self.algorithm.learning_rate = 2.0e-6" in ppo_source
    assert "self.algorithm.clip_param = 0.05" in ppo_source
    assert 'self.algorithm.schedule = "fixed"' in ppo_source
    assert "self.algorithm.policy_kl_coefficient = 0.0" in ppo_source
    assert "self.algorithm.policy_kl_max = 1.0e-3" in ppo_source


def test_transition_stage_adds_controlled_low_range_exploration() -> None:
    env_source = (TASK / "robots/k1/stability_env_cfg.py").read_text()
    registration = (TASK / "robots/k1/__init__.py").read_text()
    ppo_source = (TASK / "robots/k1/stability_ppo_cfg.py").read_text()
    assert "class K1HeightTrackingStabilityTransitionOriginEnvCfg" in env_source
    assert "command.ranges.height = (0.54, 0.72)" in env_source
    assert "command.local_target_ratio = 0.50" in env_source
    assert "command.local_target_delta_range = (-0.035, 0.035)" in env_source
    assert "command.focus_height_range = (0.575, 0.60)" in env_source
    assert "command.focus_ratio = 0.75" in env_source
    assert 'id="Booster-K1-Height-Tracking-Stability-TransitionOrigin-v0"' in registration
    assert "class K1HeightTrackingStabilityTransitionPpoRunnerCfg" in ppo_source
    assert "self.freeze_resumed_action_std = False" in ppo_source
    assert "self.freeze_action_std = True" in ppo_source
    assert "self.algorithm.learning_rate = 1.0e-5" in ppo_source
    assert "self.algorithm.policy_kl_max = 1.0e-2" in ppo_source


def test_height_command_gain_remap_preserves_the_pivot() -> None:
    source = (ROOT / "scripts/remap_height_command_gain.py").read_text()
    assert "weight[rows, -history:] = original * gain" in source
    assert "pivot * (1.0 - gain) * original.sum(dim=1)" in source
    assert 'state["actor.0.weight"]' in source
    assert 'state["critic.0.weight"]' in source
    assert "return slice(0, size)" in source


def test_actor_action_bias_recenter_only_changes_the_output_bias() -> None:
    source = (ROOT / "scripts/recenter_actor_action_bias.py").read_text()
    assert 'key.startswith("actor.")' in source
    assert 'key.endswith(".bias")' in source
    assert "bias[index] += delta" in source
    assert "actor_action_bias_recenter" in source


def test_stability_refinement_bounds_preclip_head_and_knee_means() -> None:
    ppo_source = (ROOT / "third_party/wbc_agile_rsl_rl/rsl_rl/algorithms/ppo.py").read_text()
    config_source = (TASK / "robots/k1/stability_ppo_cfg.py").read_text()
    assert "actor_mean_bound_cfg: dict | None = None" in ppo_source
    assert 'loss_dict["actor_mean_bound"]' in ppo_source
    assert "torch.abs(selected_mean) - self.actor_mean_soft_limit" in ppo_source
    assert "action_indices=(13, 19)" in config_source
    assert "soft_limit=3.75" in config_source
    assert "actor_height_response_cfg: dict | None = None" in ppo_source
    assert 'loss_dict["actor_height_response"]' in ppo_source
    assert "high_center[hip_index] += 0.08 * scale" in (
        TASK / "robots/k1/stability_env_cfg.py"
    ).read_text()
    assert "height_posture_high_center" in (
        ROOT
        / "source/booster_train/booster_train/tasks/manager_based/fall_recovery/actions.py"
    ).read_text()
    assert "self._height_posture_high_center is not None" in (
        ROOT
        / "source/booster_train/booster_train/tasks/manager_based/fall_recovery/actions.py"
    ).read_text()
    assert "preserve_low_center=True" in (
        TASK / "robots/k1/stability_env_cfg.py"
    ).read_text()
    assert "class K1HeightTrackingStabilityTopEnvCfg" in (
        TASK / "robots/k1/stability_env_cfg.py"
    ).read_text()
    assert '("Top", "K1HeightTrackingStabilityTopEnvCfg")' in (
        TASK / "robots/k1/__init__.py"
    ).read_text()

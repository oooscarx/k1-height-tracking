from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).with_name("config") / "k1_fall_recovery.json"


@lru_cache(maxsize=1)
def load_k1_hardware_config() -> dict[str, Any]:
    with CONFIG_PATH.open("r", encoding="utf-8") as stream:
        config = json.load(stream)

    vector_keys = (
        "joint_names",
        "motor_models",
        "position_minimum",
        "position_maximum",
        "goal_position",
        "stiffness",
        "damping",
        "command_torque_limit",
    )
    for key in vector_keys:
        if len(config[key]) != 22:
            raise ValueError(f"{CONFIG_PATH}: {key} must contain 22 values")
    if len(set(config["joint_names"])) != 22:
        raise ValueError(f"{CONFIG_PATH}: joint_names must be unique")
    if any(low >= high for low, high in zip(config["position_minimum"], config["position_maximum"])):
        raise ValueError(f"{CONFIG_PATH}: invalid logical joint limits")
    if config["parallel_ankle"]["joint_indexes"] != [14, 15, 20, 21]:
        raise ValueError(f"{CONFIG_PATH}: parallel ankle indexes must be [14, 15, 20, 21]")
    if not 0.0 < float(config["motor_limit_ratio"]) <= 1.0:
        raise ValueError(f"{CONFIG_PATH}: motor_limit_ratio must be in (0, 1]")
    handoff = config["handoff_policy"]
    if handoff["body_joint_indexes"] != list(range(2, 22)):
        raise ValueError(f"{CONFIG_PATH}: AMP handoff must own body joints 2..21")
    if handoff["action_size"] != 20:
        raise ValueError(f"{CONFIG_PATH}: AMP handoff action_size must be 20")
    if float(handoff["policy_rate_hz"]) != float(config["policy_rate_hz"]):
        raise ValueError(f"{CONFIG_PATH}: AMP handoff policy rate must match recovery")
    if float(handoff["maximum_zero_command_xy_drift_m"]) <= 0.0:
        raise ValueError(
            f"{CONFIG_PATH}: AMP handoff maximum zero-command drift must be positive"
        )
    for key in (
        "maximum_start_linear_speed_mps",
        "maximum_start_angular_speed_radps",
        "maximum_start_body_joint_speed_radps",
        "commanded_forward_velocity_mps",
        "commanded_validation_s",
        "minimum_commanded_forward_progress_m",
        "maximum_commanded_lateral_drift_m",
        "minimum_commanded_height_m",
    ):
        if float(handoff[key]) <= 0.0:
            raise ValueError(f"{CONFIG_PATH}: AMP handoff {key} must be positive")
    matching_handoff_gates = (
        ("success_linear_velocity", "maximum_start_linear_speed_mps"),
        ("success_angular_velocity", "maximum_start_angular_speed_radps"),
        ("success_body_joint_velocity", "maximum_start_body_joint_speed_radps"),
        ("success_body_pose_error", "maximum_start_pose_error"),
    )
    for training_key, handoff_key in matching_handoff_gates:
        training_value = float(config["training"][training_key])
        if training_value <= 0.0:
            raise ValueError(f"{CONFIG_PATH}: recovery {training_key} must be positive")
        if training_value != float(handoff[handoff_key]):
            raise ValueError(
                f"{CONFIG_PATH}: recovery {training_key} and handoff "
                f"{handoff_key} gates must match"
            )
    if not -1.0 <= float(handoff["maximum_commanded_gravity_z"]) < 0.0:
        raise ValueError(
            f"{CONFIG_PATH}: AMP handoff maximum commanded gravity z must be in [-1, 0)"
        )
    amp = config["training"]["amp"]
    for prefix in (
        "random_fall_linear_velocity",
        "random_fall_angular_velocity",
        "random_joint_noise",
        "random_ankle_joint_noise",
        "random_joint_velocity",
        "random_ankle_joint_velocity",
    ):
        start = float(amp[f"{prefix}_start"])
        end = float(amp[f"{prefix}_end"])
        if start < 0.0 or end < start:
            raise ValueError(
                f"{CONFIG_PATH}: {prefix} curriculum must be non-negative and increasing"
            )
    uniform_start = float(amp["random_joint_uniform_blend_start"])
    uniform_end = float(amp["random_joint_uniform_blend_end"])
    if not 0.0 <= uniform_start <= uniform_end <= 1.0:
        raise ValueError(
            f"{CONFIG_PATH}: random joint uniform blend must increase inside [0, 1]"
        )
    difficulty_start = float(amp["random_fall_difficulty_start"])
    difficulty_end = float(amp["random_fall_difficulty_end"])
    if not 0.0 <= difficulty_start <= difficulty_end <= 1.0:
        raise ValueError(
            f"{CONFIG_PATH}: random fall difficulty must increase inside [0, 1]"
        )
    native_teacher = config["native_teacher"]
    if native_teacher["direction_values"] != {"facedown": 0, "faceup": 1}:
        raise ValueError(f"{CONFIG_PATH}: native teacher direction values must match deployment")
    if native_teacher["zero_action_joint_ids"] != [0, 1, 12, 15, 18, 21]:
        raise ValueError(f"{CONFIG_PATH}: native teacher zero-action joints differ from deployment")
    if native_teacher["parallel_ankle_joint_ids"] != [14, 15, 20, 21]:
        raise ValueError(f"{CONFIG_PATH}: native teacher parallel-ankle joints differ from deployment")
    if len(native_teacher["reference_position"]) != 22:
        raise ValueError(f"{CONFIG_PATH}: native teacher reference position must contain 22 values")
    if native_teacher["action_clip"] != [-1.0, 1.0]:
        raise ValueError(f"{CONFIG_PATH}: native teacher action clip must remain [-1, 1]")
    for key in (
        "actor_sha256",
        "faceup_reference_sha256",
        "facedown_reference_sha256",
        "faceup_teacher_rollout_sha256",
    ):
        if len(native_teacher[key]) != 64:
            raise ValueError(f"{CONFIG_PATH}: native teacher {key} is invalid")
    for key in ("training_action_center", "training_action_scale"):
        if len(native_teacher[key]) != 22:
            raise ValueError(f"{CONFIG_PATH}: native teacher {key} must contain 22 values")
    if any(float(value) <= 0.0 for value in native_teacher["training_action_scale"]):
        raise ValueError(f"{CONFIG_PATH}: native teacher training action scale must be positive")
    for key in ("action_tracking_weight", "action_tracking_std"):
        if float(native_teacher[key]) <= 0.0:
            raise ValueError(f"{CONFIG_PATH}: native teacher {key} must be positive")
    robust_training = native_teacher["robust_training"]
    if not 0.0 <= float(robust_training["reference_reset_probability"]) <= 1.0:
        raise ValueError(f"{CONFIG_PATH}: native robust reference probability must be in [0, 1]")
    for stage in ("start", "end"):
        failure_probability = float(robust_training[f"failure_reset_probability_{stage}"])
        if not 0.0 <= failure_probability <= 1.0:
            raise ValueError(f"{CONFIG_PATH}: native robust failure probability must be in [0, 1]")
        if float(robust_training["reference_reset_probability"]) + failure_probability > 1.0:
            raise ValueError(f"{CONFIG_PATH}: native robust {stage} reset probabilities exceed one")
        failure_blend = float(robust_training[f"failure_state_blend_{stage}"])
        if not 0.0 <= failure_blend <= 1.0:
            raise ValueError(f"{CONFIG_PATH}: native robust failure blend must be in [0, 1]")
        noise_scale = float(robust_training[f"canonical_noise_scale_{stage}"])
        if not 0.0 <= noise_scale <= 1.0:
            raise ValueError(f"{CONFIG_PATH}: native robust canonical noise scale must be in [0, 1]")
    if int(robust_training["reset_curriculum_steps"]) < 1:
        raise ValueError(f"{CONFIG_PATH}: native robust reset curriculum must be positive")
    if not robust_training["failure_state_file"]:
        raise ValueError(f"{CONFIG_PATH}: native robust failure-state file must be configured")
    if len(robust_training["failure_state_sha256"]) != 64:
        raise ValueError(f"{CONFIG_PATH}: native robust failure-state SHA256 is invalid")
    for key in (
        "canonical_joint_noise",
        "canonical_ankle_joint_noise",
        "canonical_root_xy_noise",
        "canonical_height_noise",
        "canonical_orientation_noise",
        "canonical_velocity_scale",
        "reset_ankle_neutral_fraction",
        "joint_state_tolerance_start",
        "joint_state_tolerance_end",
        "parallel_state_tolerance_start",
        "parallel_state_tolerance_end",
        "position_target_margin",
        "position_braking_horizon_s",
    ):
        if float(robust_training[key]) < 0.0:
            raise ValueError(f"{CONFIG_PATH}: native robust {key} must be non-negative")
    if (
        float(robust_training["joint_state_tolerance_start"])
        < float(robust_training["joint_state_tolerance_end"])
    ):
        raise ValueError(f"{CONFIG_PATH}: native robust joint tolerance must tighten over training")
    if (
        float(robust_training["parallel_state_tolerance_start"])
        < float(robust_training["parallel_state_tolerance_end"])
    ):
        raise ValueError(f"{CONFIG_PATH}: native robust parallel tolerance must tighten over training")
    if float(robust_training["reset_ankle_neutral_fraction"]) > 1.0:
        raise ValueError(f"{CONFIG_PATH}: native robust reset_ankle_neutral_fraction must be at most one")
    for key in (
        "action_tracking_weight",
        "parallel_state_limit_weight",
        "parallel_state_soft_margin",
        "parallel_stress_weight",
        "parallel_stress_margin",
        "unsafe_termination_weight",
    ):
        if float(robust_training[key]) <= 0.0:
            raise ValueError(f"{CONFIG_PATH}: native robust {key} must be positive")
    task_driven = native_teacher["task_driven_training"]
    if not task_driven["amp_style_motion_file"]:
        raise ValueError(f"{CONFIG_PATH}: task-driven AMP style motion must be configured")
    if float(task_driven["episode_length_s"]) <= 0.0:
        raise ValueError(f"{CONFIG_PATH}: task-driven episode length must be positive")
    reset_total = (
        float(task_driven["reference_reset_probability"])
        + float(task_driven["failure_reset_probability"])
    )
    if not 0.0 <= reset_total <= 1.0:
        raise ValueError(f"{CONFIG_PATH}: task-driven reset probabilities exceed one")
    for key in (
        "failure_state_blend_start",
        "failure_state_blend_end",
        "canonical_noise_scale",
        "height_fraction",
        "coupling_fraction",
        "adaptive_success_threshold",
        "adaptive_demotion_threshold",
    ):
        if not 0.0 <= float(task_driven[key]) <= 1.0:
            raise ValueError(f"{CONFIG_PATH}: task-driven {key} must be in [0, 1]")
    if float(task_driven["failure_state_blend_start"]) > float(
        task_driven["failure_state_blend_end"]
    ):
        raise ValueError(f"{CONFIG_PATH}: task-driven failure blend must increase")
    if float(task_driven["adaptive_demotion_threshold"]) >= float(
        task_driven["adaptive_success_threshold"]
    ):
        raise ValueError(f"{CONFIG_PATH}: task-driven curriculum thresholds overlap")
    if int(task_driven["adaptive_min_trials"]) < 1:
        raise ValueError(f"{CONFIG_PATH}: task-driven adaptive_min_trials must be positive")
    if int(task_driven["adaptive_promotion_windows"]) < 2:
        raise ValueError(f"{CONFIG_PATH}: task-driven curriculum must wait for repeated windows")
    if int(task_driven["handoff_curriculum_steps"]) < 1:
        raise ValueError(f"{CONFIG_PATH}: task-driven handoff curriculum must be positive")
    for key in (
        "handoff_curriculum_window_steps",
        "handoff_curriculum_minimum_trials",
        "handoff_curriculum_promotion_windows",
        "handoff_curriculum_demotion_windows",
    ):
        if int(task_driven[key]) < 1:
            raise ValueError(f"{CONFIG_PATH}: task-driven {key} must be positive")
    for key in (
        "handoff_curriculum_promotion_step",
        "handoff_curriculum_demotion_step",
    ):
        if not 0.0 < float(task_driven[key]) <= 1.0:
            raise ValueError(f"{CONFIG_PATH}: task-driven {key} must be in (0, 1]")
    handoff_demotion = float(task_driven["handoff_curriculum_demotion_threshold"])
    handoff_promotion = float(task_driven["handoff_curriculum_success_threshold"])
    if not 0.0 <= handoff_demotion < handoff_promotion <= 1.0:
        raise ValueError(
            f"{CONFIG_PATH}: task-driven handoff curriculum thresholds overlap"
        )
    for key in (
        "handoff_curriculum_maximum_parallel_state_violation_fraction",
        "handoff_curriculum_maximum_hard_joint_limit_fraction",
        "handoff_curriculum_maximum_joint_limit_termination_rate",
        "handoff_curriculum_maximum_parallel_ankle_termination_rate",
    ):
        if not 0.0 <= float(task_driven[key]) <= 1.0:
            raise ValueError(f"{CONFIG_PATH}: task-driven {key} must be in [0, 1]")
    handoff_curriculum_gates = (
        ("success_angular_velocity_start", "success_angular_velocity"),
        ("success_linear_velocity_start", "success_linear_velocity"),
        ("success_body_pose_error_start", "success_body_pose_error"),
        ("success_body_joint_velocity_start", "success_body_joint_velocity"),
    )
    for start_key, strict_key in handoff_curriculum_gates:
        start = float(task_driven[start_key])
        strict = float(config["training"][strict_key])
        if start <= 0.0:
            raise ValueError(f"{CONFIG_PATH}: task-driven {start_key} must be positive")
        if start < strict:
            raise ValueError(
                f"{CONFIG_PATH}: task-driven {start_key} must not be stricter than {strict_key}"
            )
    for key in (
        "maximum_progress_rate",
        "progress_weight",
        "time_cost_weight",
        "target_height_weight",
        "standing_height_weight",
        "upright_weight",
        "feet_support_weight",
        "standing_pose_weight",
        "standing_pose_max_error_std",
        "standing_pose_max_error_hold_weight",
        "standing_pose_mean_error_weight",
        "standing_pose_mean_error_std",
        "standing_pose_progress_maximum_rate",
        "stillness_weight",
        "joint_limit_weight",
        "joint_limit_soft_margin",
        "joint_limit_target_reduction_weight",
        "handoff_quality_weight",
        "stability_hold_progress_weight",
        "early_success_weight",
        "adaptive_promotion_step",
        "adaptive_demotion_step",
    ):
        if float(task_driven[key]) <= 0.0:
            raise ValueError(f"{CONFIG_PATH}: task-driven {key} must be positive")
    strict_quality_minimum_scale = float(
        task_driven["early_success_strict_quality_minimum_scale"]
    )
    strict_quality_maximum_scale = float(
        task_driven["early_success_strict_quality_maximum_scale"]
    )
    if strict_quality_minimum_scale < 1.0:
        raise ValueError(
            f"{CONFIG_PATH}: task-driven early success strict quality minimum scale must preserve completion"
        )
    if strict_quality_maximum_scale <= strict_quality_minimum_scale:
        raise ValueError(
            f"{CONFIG_PATH}: task-driven early success strict quality scales must increase"
        )
    if float(task_driven["height_target"]) <= float(task_driven["height_minimum"]):
        raise ValueError(f"{CONFIG_PATH}: task-driven height target must exceed its minimum")
    amp = config["training"]["amp"]
    if len(amp["joint_names"]) != len(set(amp["joint_names"])):
        raise ValueError(f"{CONFIG_PATH}: training.amp.joint_names must be unique")
    if not set(amp["joint_names"]).issubset(config["joint_names"]):
        raise ValueError(f"{CONFIG_PATH}: training.amp.joint_names must use deployment names")
    if int(amp["history_length"]) < 1:
        raise ValueError(f"{CONFIG_PATH}: training.amp.history_length must be positive")
    for stage in ("start", "end"):
        total = (
            float(amp[f"reference_reset_probability_{stage}"])
            + float(amp[f"standing_reset_probability_{stage}"])
            + float(amp[f"random_fall_probability_{stage}"])
        )
        if not 0.0 <= total <= 1.0:
            raise ValueError(f"{CONFIG_PATH}: AMP {stage} reset probabilities must sum to [0, 1]")
    if float(amp["reset_joint_margin"]) < 0.0:
        raise ValueError(f"{CONFIG_PATH}: training.amp.reset_joint_margin must be nonnegative")
    if float(amp["random_reset_joint_margin"]) < float(amp["reset_joint_margin"]):
        raise ValueError(
            f"{CONFIG_PATH}: random reset joint margin must not be below the reference reset margin"
        )
    reset_parallel_margin = float(amp["random_reset_parallel_motor_margin"])
    command_parallel_margin = float(config["parallel_ankle"]["motor_target_margin"])
    if reset_parallel_margin < command_parallel_margin:
        raise ValueError(
            f"{CONFIG_PATH}: reset parallel motor margin must not be below the command margin"
        )
    velocity_safety_horizon = float(
        amp["random_joint_velocity_safety_horizon_s"]
    )
    if velocity_safety_horizon < 1.0 / float(config["policy_rate_hz"]):
        raise ValueError(
            f"{CONFIG_PATH}: random joint velocity safety horizon must cover at least one policy step"
        )
    if not 0.0 <= float(amp["reference_reset_joint_velocity_scale"]) <= 1.0:
        raise ValueError(f"{CONFIG_PATH}: training.amp.reference_reset_joint_velocity_scale must be in [0, 1]")
    if not 0.0 <= float(amp["reset_ankle_neutral_fraction"]) <= 1.0:
        raise ValueError(f"{CONFIG_PATH}: training.amp.reset_ankle_neutral_fraction must be in [0, 1]")
    difficulty_start = float(amp["random_fall_difficulty_start"])
    difficulty_end = float(amp["random_fall_difficulty_end"])
    if not 0.0 <= difficulty_start <= difficulty_end <= 1.0:
        raise ValueError(
            f"{CONFIG_PATH}: training.amp random-fall difficulty must increase inside [0, 1]"
        )
    minimum_height, maximum_height = (
        float(value) for value in amp["random_fall_height_range"]
    )
    if minimum_height <= 0.0 or maximum_height <= minimum_height:
        raise ValueError(
            f"{CONFIG_PATH}: training.amp random-fall height range must be positive and increasing"
        )
    for prefix in (
        "random_fall_linear_velocity",
        "random_fall_angular_velocity",
        "random_joint_noise",
        "random_ankle_joint_noise",
        "random_joint_velocity",
        "random_ankle_joint_velocity",
    ):
        start = float(amp[f"{prefix}_start"])
        end = float(amp[f"{prefix}_end"])
        if start < 0.0 or end < start:
            raise ValueError(
                f"{CONFIG_PATH}: training.amp {prefix} must be nonnegative and increasing"
            )
    blend_start = float(amp["random_joint_uniform_blend_start"])
    blend_end = float(amp["random_joint_uniform_blend_end"])
    if not 0.0 <= blend_start <= blend_end <= 1.0:
        raise ValueError(
            f"{CONFIG_PATH}: training.amp random joint uniform blend must increase inside [0, 1]"
        )
    directional = amp["directional"]
    if directional["joint_names"] != amp["joint_names"]:
        raise ValueError(f"{CONFIG_PATH}: directional AMP must keep the reviewed lower-body feature set")
    if len(directional["key_body_names"]) != len(set(directional["key_body_names"])):
        raise ValueError(f"{CONFIG_PATH}: directional AMP key bodies must be unique")
    for key in ("faceup_motion_file", "facedown_motion_file"):
        if directional[key] not in amp["motion_files"]:
            raise ValueError(f"{CONFIG_PATH}: directional AMP motion must be listed in training.amp.motion_files")
    for stage in ("start", "end"):
        total = (
            float(directional[f"reference_reset_probability_{stage}"])
            + float(directional[f"standing_reset_probability_{stage}"])
            + float(directional[f"random_fall_probability_{stage}"])
        )
        if not 0.0 <= total <= 1.0:
            raise ValueError(f"{CONFIG_PATH}: directional AMP {stage} reset probabilities must sum to [0, 1]")
    if int(directional["reset_curriculum_steps"]) < 1:
        raise ValueError(f"{CONFIG_PATH}: directional AMP reset curriculum must be positive")
    if not 0.0 <= float(directional["reference_reset_joint_velocity_scale"]) <= 1.0:
        raise ValueError(
            f"{CONFIG_PATH}: directional AMP reference_reset_joint_velocity_scale must be in [0, 1]"
        )
    if not 0.0 <= float(directional["terminal_reset_probability"]) <= 1.0:
        raise ValueError(f"{CONFIG_PATH}: directional AMP terminal_reset_probability must be in [0, 1]")
    if float(directional["terminal_reset_window_s"]) <= 0.0:
        raise ValueError(f"{CONFIG_PATH}: directional AMP terminal_reset_window_s must be positive")
    if not isinstance(directional["canonical_from_reference"], bool):
        raise ValueError(f"{CONFIG_PATH}: directional AMP canonical_from_reference must be boolean")
    for key in (
        "canonical_joint_noise",
        "canonical_root_xy_noise",
        "canonical_height_noise",
        "canonical_orientation_noise",
    ):
        if float(directional[key]) < 0.0:
            raise ValueError(f"{CONFIG_PATH}: directional AMP {key} must be nonnegative")
    if not 0.0 <= float(directional["canonical_velocity_scale"]) <= 1.0:
        raise ValueError(f"{CONFIG_PATH}: directional AMP canonical_velocity_scale must be in [0, 1]")
    for key in (
        "standing_pose_weight",
        "stillness_weight",
        "early_success_weight",
        "parallel_stress_weight",
        "unsafe_termination_weight",
    ):
        if float(directional[key]) <= 0.0:
            raise ValueError(f"{CONFIG_PATH}: directional AMP {key} must be positive")
    if any(float(value) <= 0.0 for value in directional["tracking_reward"].values()):
        raise ValueError(f"{CONFIG_PATH}: directional AMP tracking rewards must be positive")
    for key in ("motor_position_minimum", "motor_position_maximum"):
        if len(config["parallel_ankle"][key]) != 4:
            raise ValueError(f"{CONFIG_PATH}: parallel_ankle.{key} must contain four values")
    motor_target_margin = float(config["parallel_ankle"]["motor_target_margin"])
    if motor_target_margin < 0.0:
        raise ValueError(f"{CONFIG_PATH}: parallel_ankle.motor_target_margin must be non-negative")
    if any(
        high - low <= 2.0 * motor_target_margin
        for low, high in zip(
            config["parallel_ankle"]["motor_position_minimum"],
            config["parallel_ankle"]["motor_position_maximum"],
        )
    ):
        raise ValueError(f"{CONFIG_PATH}: parallel_ankle.motor_target_margin leaves no command range")
    return config


K1_HARDWARE_CONFIG = load_k1_hardware_config()

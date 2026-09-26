from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as UniformNoise

from booster_train.tasks.manager_based.fall_recovery.hardware_config import K1_HARDWARE_CONFIG
from booster_train.tasks.manager_based.fall_recovery.robots.k1.env_cfg import recovery_robot_cfg
from booster_train.tasks.manager_based.height_tracking import mdp
from booster_train.tasks.manager_based.height_tracking.k1_scaling import (
    K1_FORCE_SCALE,
    K1_FULL_BODY_HEIGHT_GATE_RANGE,
    K1_HEIGHT_GATE_AGE_RANGE,
    K1_HEIGHT_REWARD_FINE_STD,
    K1_HEIGHT_REWARD_MEDIUM_STD,
    K1_HEIGHT_REWARD_ROUGH_STD,
    K1_MODERATE_HEIGHT_GATE_RANGE,
    K1_PUSH_LINEAR_VELOCITY_SCALE,
    K1_REPORTING_SUCCESS_ERROR_THRESHOLD,
    K1_RESET_HEIGHT_OFFSET,
    K1_ROOT_HEIGHT,
    K1_STILLNESS_FULL_PENALTY_ERROR_THRESHOLD,
    K1_STILLNESS_ZERO_PENALTY_ERROR_THRESHOLD,
    K1_TORQUE_SCALE,
    K1_TRACKED_POINT_OFFSET,
    K1_TRACKING_ERROR_THRESHOLD,
    K1_TRUNK_MASS_SCALE,
    K1_LENGTH_SCALE,
)
from booster_train.tasks.manager_based.height_tracking.terrains import HEIGHT_TRACKING_ROUGH_TERRAIN_CFG

CONFIG = K1_HARDWARE_CONFIG
JOINT_NAMES = CONFIG["joint_names"]
GOAL_POSITION = CONFIG["goal_position"]
ROBUST = CONFIG["native_teacher"]["robust_training"]
RAW_ACTION_SCALE = CONFIG["native_teacher"]["training_action_scale"]
TRUNK_HEIGHT = K1_ROOT_HEIGHT
FOOT_BODIES = ["left_foot_link", "right_foot_link"]
ARM_JOINTS = JOINT_NAMES[2:10]
ANKLE_JOINTS = ["Left_Ankle_Pitch", "Left_Ankle_Roll", "Right_Ankle_Pitch", "Right_Ankle_Roll"]
ILLEGAL_CONTACTS = ["left_hand_link", "right_hand_link"]
VELOCITY_BODIES = [
    "Trunk",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Left_Shank",
    "Right_Shank",
    "left_hand_link",
    "right_hand_link",
]
SLAM_BODIES = [
    "Trunk",
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Left_Shank",
    "Right_Shank",
    "Left_Ankle_Cross",
    "Right_Ankle_Cross",
]


def _deployment_asset_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)


def _safe_action_scale() -> list[float]:
    margin = ROBUST["position_target_margin"]
    result = []
    for raw, center, minimum, maximum in zip(
        RAW_ACTION_SCALE,
        GOAL_POSITION,
        CONFIG["position_minimum"],
        CONFIG["position_maximum"],
    ):
        result.append(min(raw, center - minimum - margin, maximum - center - margin))
    if min(result) <= 0.0:
        raise ValueError("K1 default pose is too close to a hardware limit for height tracking")
    return result


ACTION_SCALE = _safe_action_scale()


@configclass
class K1HeightTrackingSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=HEIGHT_TRACKING_ROUGH_TERRAIN_CFG,
        max_init_terrain_level=0,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=(
                f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/"
                "TilesMarbleSpiderWhiteBrickBondHoned.mdl"
            ),
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot = recovery_robot_cfg()
    robot.soft_joint_pos_limit_factor = 0.9
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )
    height_measurement_sensor = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Trunk",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.05, size=(0.0, 0.0)),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=5.0,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class CommandsCfg:
    height = mdp.SmoothHeightCommandCfg(
        asset_name="robot",
        body_name="Trunk",
        offset=mdp.SmoothHeightCommandCfg.OffsetCfg(pos=(0.0, 0.0, K1_TRACKED_POINT_OFFSET)),
        height_sensor="height_measurement_sensor",
        resampling_time_range=(1.0, 7.0),
        ranges=mdp.SmoothHeightCommandCfg.Ranges(height=(-0.5, TRUNK_HEIGHT + 0.2)),
        velocity_range=(1000.0, 1000.0),
        debug_vis=True,
        # The WBC G1 distribution puts 30% of all commands at the maximum
        # height.  At low lift this made K1 trade away the sparsely sampled
        # intermediate heights, so retain a standing bias while giving the
        # continuous height range enough samples to remain learnable.
        standing_ratio=0.15,
        flat_ratio=0.2,
        governed_commands=True,
        governor_step_m=0.12,
        governor_hold_s=4.0,
        settle_time_s=2.0,
        success_error_threshold=K1_REPORTING_SUCCESS_ERROR_THRESHOLD,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.2, noise=UniformNoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=UniformNoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _deployment_asset_cfg()},
            noise=UniformNoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _deployment_asset_cfg()},
            scale=0.05,
            noise=UniformNoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))
        height_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "height"})

        def __post_init__(self) -> None:
            self.history_length = 5
            self.enable_corruption = True
            self.concatenate_terms = True
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _deployment_asset_cfg()})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _deployment_asset_cfg()}, scale=0.05)
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))
        contact_forces = ObsTerm(
            func=mdp.contact_force_norm,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*")},
            scale=5.0e-3,
            clip=(-25_000.0, 25_000.0),
        )
        base_height = ObsTerm(
            func=mdp.base_height_from_sensor,
            params={"sensor_cfg": SceneEntityCfg("height_measurement_sensor")},
            clip=(-2.0, 2.0),
        )
        height_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "height"})

        def __post_init__(self) -> None:
            self.history_length = 5
            self.enable_corruption = False
            self.concatenate_terms = True
            self.flatten_history_dim = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ActionsCfg:
    joint_pos = mdp.K1SerialJointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        preserve_order=True,
        use_default_offset=False,
        expected_joint_names=JOINT_NAMES,
        position_minimum=CONFIG["position_minimum"],
        position_maximum=CONFIG["position_maximum"],
        position_center=GOAL_POSITION,
        position_scale=ACTION_SCALE,
        clip={".*": (-1.0, 1.0)},
        stiffness=CONFIG["stiffness"],
        damping=CONFIG["damping"],
        command_torque_limit=CONFIG["command_torque_limit"],
        relative_to_current=False,
        normalize_input=False,
        position_target_margin=ROBUST["position_target_margin"],
        position_braking_horizon_s=ROBUST["position_braking_horizon_s"],
    )
    lift = mdp.HeightLiftActionCfg(
        asset_name="robot",
        link_to_lift="Trunk",
        stiffness_forces=5000.0,
        damping_forces=2500.0,
        force_limit_weight_fraction=0.9,
        damping_torques=100.0,
        torque_limit=250.0,
        height_sensor="height_measurement_sensor",
        target_height=TRUNK_HEIGHT,
        height_command="height",
        force_offset=(0.0, 0.0, K1_TRACKED_POINT_OFFSET),
        allow_push_down=True,
    )


@configclass
class RewardsCfg:
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-0.5e-4)
    torque_limits = RewTerm(func=mdp.applied_torque_limits, weight=-0.01)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-8)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.1)
    joint_vel_limits = RewTerm(func=mdp.joint_vel_limits, weight=-0.01, params={"soft_ratio": 0.8})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0)
    action_rate_rate = RewTerm(
        func=mdp.action_rate_rate_l2,
        weight=-0.15,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-3)
    joint_tracking_error = RewTerm(func=mdp.joint_pos_tracking_error_l2, weight=-0.5)
    base_height_rough = RewTerm(
        func=mdp.track_height_command_exp,
        weight=4.0,
        params={"command_name": "height", "std": K1_HEIGHT_REWARD_ROUGH_STD},
    )
    base_height_medium = RewTerm(
        func=mdp.track_height_command_exp,
        weight=8.0,
        params={"command_name": "height", "std": K1_HEIGHT_REWARD_MEDIUM_STD},
    )
    base_height_fine = RewTerm(
        func=mdp.track_height_command_exp,
        weight=16.0,
        params={"command_name": "height", "std": K1_HEIGHT_REWARD_FINE_STD},
    )
    # The upstream terms also reward negative commands and the unavoidable
    # transient after an instant command jump. Reinforce the exact settled,
    # non-negative samples used by the lift curriculum without removing those
    # transition-learning signals.
    base_height_settled_fine = RewTerm(
        func=mdp.track_height_command_exp,
        weight=8.0,
        params={"command_name": "height", "std": K1_HEIGHT_REWARD_FINE_STD, "settle_gate": True},
    )
    relaxation = RewTerm(
        func=mdp.relaxation_penalty,
        weight=-1.0,
        params={"command_name": "height", "asset_cfg": _deployment_asset_cfg(), "pos_weight": 1.0, "torque_weight": 1e-3},
    )
    joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_for_height_command,
        weight=-0.5,
        params={
            "command_name": "height",
            "height_gate_range": K1_FULL_BODY_HEIGHT_GATE_RANGE,
            "age_gate_range": K1_HEIGHT_GATE_AGE_RANGE,
            "asset_cfg": _deployment_asset_cfg(),
            "mode": "l1",
        },
    )
    joint_deviation_l1_arms = RewTerm(
        func=mdp.joint_deviation_for_height_command,
        weight=-0.5,
        params={
            "command_name": "height",
            "height_gate_range": K1_MODERATE_HEIGHT_GATE_RANGE,
            "age_gate_range": K1_HEIGHT_GATE_AGE_RANGE,
            "asset_cfg": SceneEntityCfg("robot", joint_names=ARM_JOINTS, preserve_order=True),
            "mode": "l1",
        },
    )
    ankle_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=ANKLE_JOINTS, preserve_order=True)},
    )
    not_moving = RewTerm(
        func=mdp.moving_if_tracking,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "height",
            "error_threshold": K1_STILLNESS_ZERO_PENALTY_ERROR_THRESHOLD,
            "full_penalty_error_threshold": K1_STILLNESS_FULL_PENALTY_ERROR_THRESHOLD,
            "settle_gate": True,
        },
    )
    torso_upright = RewTerm(
        func=mdp.upright_orientation_for_height_command,
        weight=-1.0,
        params={
            "command_name": "height",
            "height_gate_range": K1_MODERATE_HEIGHT_GATE_RANGE,
            "age_gate_range": K1_HEIGHT_GATE_AGE_RANGE,
            "asset_cfg": SceneEntityCfg("robot", body_names=["Trunk"]),
            "norm": "l1",
        },
    )
    severely_tilted = RewTerm(
        func=mdp.severely_tilted_penalty,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["Trunk"]), "threshold_rad": math.radians(135)},
    )
    torso_roll = RewTerm(
        func=mdp.body_orientation_penalty,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["Trunk"]), "axis": "roll", "direction": "both", "kernel": "l1"},
    )
    forward_pitch = RewTerm(
        func=mdp.body_orientation_penalty,
        weight=-10.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["Trunk"]), "axis": "pitch", "direction": "forward", "kernel": "l2"},
    )
    illegal_contacts = RewTerm(
        func=mdp.illegal_contact,
        weight=-2.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=ILLEGAL_CONTACTS), "threshold": 1.0},
    )
    feet_distance = RewTerm(
        func=mdp.feet_distance_from_ref,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
            "ref_distance": 0.2,
            "norm": "l2",
            "error_threshold": 0.2,
            "distance_mode": "absolute",
            "close_multiplier": 5.0,
            "episode_delay_s": 1.0,
            "episode_ramp_s": 2.0,
        },
    )
    feet_distance_standing = RewTerm(
        func=mdp.feet_distance_from_ref_for_height_command,
        weight=-10.0,
        params={
            "command_name": "height",
            "height_gate_range": K1_MODERATE_HEIGHT_GATE_RANGE,
            "age_gate_range": K1_HEIGHT_GATE_AGE_RANGE,
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
            "ref_distance": 0.2,
            "norm": "l2",
            "error_threshold": 0.2,
            "distance_mode": "absolute",
            "close_multiplier": 5.0,
            "episode_delay_s": 1.0,
            "episode_ramp_s": 2.0,
        },
    )
    ground_unloaded = RewTerm(
        func=mdp.ground_unloaded,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "height",
        },
    )
    flat_feet = RewTerm(
        func=mdp.foot_orientation_l1,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
            "roll_weight": 1.0,
            "pitch_weight": 2.0,
            "yaw_weight": 0.0,
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODIES, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
        },
    )
    feet_yaw_mean = RewTerm(
        func=mdp.feet_yaw_mean_vs_base,
        weight=-2.0,
        params={
            "feet_asset_cfg": SceneEntityCfg("robot", body_names=FOOT_BODIES, preserve_order=True),
            "base_body_cfg": SceneEntityCfg("robot", body_names=["Trunk"]),
        },
    )
    completely_airborne = RewTerm(
        func=mdp.completely_airborne,
        weight=-50.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*"), "threshold": 1.0},
    )
    body_velocity = RewTerm(
        func=mdp.bodies_lin_vel_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=VELOCITY_BODIES), "threshold": 0.3},
    )
    ground_slam = RewTerm(
        func=mdp.impact_velocity,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=SLAM_BODIES, preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=SLAM_BODIES, preserve_order=True),
            "force_threshold": 10.0,
            "kernel": "l2",
        },
    )
    torso_slam = RewTerm(
        func=mdp.impact_velocity,
        weight=-5.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Trunk"]),
            "asset_cfg": SceneEntityCfg("robot", body_names=["Trunk"]),
            "force_threshold": 10.0,
            "kernel": "l2",
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    invalid_state = DoneTerm(
        func=mdp.invalid_state,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_joint_vel": 100.0,
            "max_root_height": 5.0,
            "max_root_xy_distance": 200.0,
            "max_lin_vel": 20.0,
            "max_ang_vel": 50.0,
        },
    )


@configclass
class EventsCfg:
    randomize_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.2, 1.5),
            "dynamic_friction_range": (0.2, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    randomize_actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "stiffness_distribution_params": (0.9, 1.1),
            "damping_distribution_params": (0.8, 2.0),
            "operation": "scale",
        },
    )
    randomize_bodies_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"), "mass_distribution_params": (0.95, 1.05), "operation": "scale"},
    )
    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "mass_distribution_params": (-1.0 * K1_TRUNK_MASS_SCALE, 3.0 * K1_TRUNK_MASS_SCALE),
            "operation": "add",
        },
    )
    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "com_range": {
                "x": (-0.1 * K1_LENGTH_SCALE, 0.1 * K1_LENGTH_SCALE),
                "y": (-0.05 * K1_LENGTH_SCALE, 0.05 * K1_LENGTH_SCALE),
                "z": (-0.1 * K1_LENGTH_SCALE, 0.1 * K1_LENGTH_SCALE),
            },
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(0.0, 10.0),
        params={
            "velocity_range": {
                "x": (-1.0 * K1_PUSH_LINEAR_VELOCITY_SCALE, 1.0 * K1_PUSH_LINEAR_VELOCITY_SCALE),
                "y": (-1.0 * K1_PUSH_LINEAR_VELOCITY_SCALE, 1.0 * K1_PUSH_LINEAR_VELOCITY_SCALE),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            }
        },
    )
    apply_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["Trunk"]),
            "force_range": (-20.0 * K1_FORCE_SCALE, 20.0 * K1_FORCE_SCALE),
            "torque_range": (-10.0 * K1_TORQUE_SCALE, 10.0 * K1_TORQUE_SCALE),
        },
    )
    apply_external_force_torque_extremities = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_hand_link", "right_hand_link", *FOOT_BODIES]),
            "force_range": (-5.0 * K1_FORCE_SCALE, 5.0 * K1_FORCE_SCALE),
            "torque_range": (-0.5 * K1_TORQUE_SCALE, 0.5 * K1_TORQUE_SCALE),
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_from_fallen_dataset,
        mode="reset",
        params={
            "standing_ratio": 0.5,
            "height_offset": K1_RESET_HEIGHT_OFFSET,
            "random_fallen_ratio": 0.0,
        },
    )


@configclass
class CurriculumCfg:
    adaptive_lift = CurrTerm(
        func=mdp.adaptive_force_decay,
        params={
            "action_name": "lift",
            "metric_name": "height_error",
            "decay_when": "below",
            "threshold": 0.12,
            "governed_threshold": 0.08,
            "command_name": "height",
            "ema_alpha": 0.05,
            "decay": 0.9999,
            "disable_threshold": 0.01,
        },
    )
    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_tracking_at_timeout,
        params={
            "command_name": "height",
            "error_threshold": K1_TRACKING_ERROR_THRESHOLD,
            "n_successes": 5,
            "n_failures": 5,
            "prerequisite_curriculum": "adaptive_lift",
            "prerequisite_threshold": 0.01,
            "prerequisite_direction": "below",
        },
    )
    polish_action_rate_rate = CurrTerm(
        func=mdp.update_reward_weight_after_curriculum,
        params={"reward_name": "action_rate_rate", "terminal_weight": -0.5, "prerequisite_curriculum": "terrain_levels", "prerequisite_threshold": 4.0, "delay_steps": 1750, "num_steps": 24000},
    )
    polish_ground_slam = CurrTerm(
        func=mdp.update_reward_weight_after_curriculum,
        params={"reward_name": "ground_slam", "terminal_weight": -3.0, "prerequisite_curriculum": "terrain_levels", "prerequisite_threshold": 4.0, "delay_steps": 1500, "num_steps": 24000},
    )
    polish_torso_slam = CurrTerm(
        func=mdp.update_reward_weight_after_curriculum,
        params={"reward_name": "torso_slam", "terminal_weight": -250.0, "prerequisite_curriculum": "terrain_levels", "prerequisite_threshold": 4.0, "delay_steps": 1250, "num_steps": 24000},
    )
    polish_not_moving = CurrTerm(
        func=mdp.update_reward_weight_after_curriculum,
        params={"reward_name": "not_moving", "terminal_weight": -2.5, "prerequisite_curriculum": "terrain_levels", "prerequisite_threshold": 4.0, "delay_steps": 1750, "num_steps": 24000},
    )
    polish_joint_vel_l2 = CurrTerm(
        func=mdp.update_reward_weight_after_curriculum,
        params={"reward_name": "joint_vel_l2", "terminal_weight": -0.5e-1, "prerequisite_curriculum": "terrain_levels", "prerequisite_threshold": 4.0, "delay_steps": 1500, "num_steps": 24000},
    )


@configclass
class K1HeightTrackingEnvCfg(ManagerBasedRLEnvCfg):
    scene: K1HeightTrackingSceneCfg = K1HeightTrackingSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.decimation = 4
        self.episode_length_s = 32.0
        self.sim.dt = 1.0 / 200.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_measurement_sensor.update_period = self.sim.dt
        self.sim.physx.gpu_max_rigid_patch_count = 2**20
        self.sim.physx.gpu_collision_stack_size = 2**27
        self.scene.terrain.terrain_generator.curriculum = True
        self.viewer.eye = (0.0, -3.0, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"

from __future__ import annotations

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

from booster_train.tasks.manager_based.fall_recovery import mdp
from booster_train.tasks.manager_based.fall_recovery.hardware_config import (
    K1_HARDWARE_CONFIG,
)
from booster_train.tasks.manager_based.fall_recovery.wbc_stand_up.terrains import (
    STAND_UP_ROUGH_TERRAIN_CFG,
)

from .env_cfg import recovery_robot_cfg

CONFIG = K1_HARDWARE_CONFIG
JOINT_NAMES = CONFIG["joint_names"]
GOAL_POSITION = CONFIG["goal_position"]
ROBUST = CONFIG["native_teacher"]["robust_training"]
REST_DURATION_S = 2.0
TRUNK_HEIGHT = 0.52
STANDING_HEIGHT = TRUNK_HEIGHT * 0.8
FOOT_BODIES = ["left_foot_link", "right_foot_link"]
UNDESIRED_CONTACTS = [
    "Trunk",
    "Head_.*",
    "Left_Arm_.*",
    "Right_Arm_.*",
    ".*hand_link",
    "Left_Hip_.*",
    "Right_Hip_.*",
]


def _deployment_asset_cfg() -> SceneEntityCfg:
    return SceneEntityCfg(
        "robot",
        joint_names=JOINT_NAMES,
        preserve_order=True,
    )


@configclass
class K1WbcStandUpSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=STAND_UP_ROUGH_TERRAIN_CFG,
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
        pattern_cfg=patterns.GridPatternCfg(
            resolution=0.05,
            size=(0.0, 0.0),
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
        max_distance=5.0,
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=(
                f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr"
            ),
        ),
    )


@configclass
class CommandsCfg:
    pass


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.2,
            noise=UniformNoise(n_min=-0.2, n_max=0.2),
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=UniformNoise(n_min=-0.05, n_max=0.05),
        )
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
        actions = ObsTerm(func=mdp.last_action, clip=(-10.0, 10.0))

        def __post_init__(self) -> None:
            self.history_length = 5
            self.flatten_history_dim = True
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _deployment_asset_cfg()},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _deployment_asset_cfg()},
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-10.0, 10.0))
        is_env_inactive = ObsTerm(
            func=mdp.is_env_inactive,
            params={"rest_duration_s": REST_DURATION_S},
        )
        contact_forces = ObsTerm(
            func=mdp.contact_force_norm,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=".*",
                )
            },
            scale=5.0e-3,
            clip=(-25_000.0, 25_000.0),
        )
        base_height = ObsTerm(
            func=mdp.base_height_from_sensor,
            params={"sensor_cfg": SceneEntityCfg("height_measurement_sensor")},
            clip=(-2.0, 2.0),
        )

        def __post_init__(self) -> None:
            self.history_length = 5
            self.flatten_history_dim = True
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ActionsCfg:
    joint_pos = mdp.K1ParallelJointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        preserve_order=True,
        use_default_offset=False,
        expected_joint_names=JOINT_NAMES,
        position_minimum=CONFIG["position_minimum"],
        position_maximum=CONFIG["position_maximum"],
        position_center=GOAL_POSITION,
        position_scale=[0.1] * 22,
        clip={".*": (-1.0, 1.0)},
        stiffness=CONFIG["stiffness"],
        damping=CONFIG["damping"],
        command_torque_limit=CONFIG["command_torque_limit"],
        parallel_ankle=CONFIG["parallel_ankle"],
        relative_to_current=True,
        normalize_input=False,
        position_target_margin=ROBUST["position_target_margin"],
        position_braking_horizon_s=ROBUST["position_braking_horizon_s"],
    )
    lift = mdp.LiftActionCfg(
        asset_name="robot",
        link_to_lift="Head_2",
        stiffness_forces=5000.0,
        damping_forces=500.0,
        force_limit=300.0,
        height_sensor="height_measurement_sensor",
        target_height=TRUNK_HEIGHT,
        start_lifting_time_s=3.0,
        lifting_duration_s=10.0,
    )


@configclass
class RewardsCfg:
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    torque_limits = RewTerm(func=mdp.applied_torque_limits, weight=-0.001)
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-8)
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-0.01)
    joint_vel_limits = RewTerm(
        func=mdp.joint_vel_limits,
        weight=-0.01,
        params={"soft_ratio": 0.8},
    )
    action_rate = RewTerm(
        func=mdp.action_rate_l2_if_actor_active,
        weight=-0.01,
        params={"rest_duration_s": REST_DURATION_S},
    )
    action_rate_rate = RewTerm(
        func=mdp.action_rate_rate_l2_if_actor_is_active,
        weight=-0.0001,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "rest_duration_s": REST_DURATION_S,
        },
    )
    action_l2 = RewTerm(
        func=mdp.action_l2_if_actor_active,
        weight=-0.05,
        params={"rest_duration_s": REST_DURATION_S},
    )
    incoming_forces_penalty = RewTerm(
        func=mdp.max_incoming_forces_penalty,
        weight=-5.0e-7,
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=".*"),
        },
    )

    base_height_rough = RewTerm(
        func=mdp.base_height_exp,
        weight=2.0,
        params={
            "target_height": TRUNK_HEIGHT,
            "std": 0.5,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )
    base_height_medium = RewTerm(
        func=mdp.base_height_exp,
        weight=8.0,
        params={
            "target_height": TRUNK_HEIGHT,
            "std": 0.25,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )
    base_height_fine = RewTerm(
        func=mdp.base_height_exp,
        weight=16.0,
        params={
            "target_height": TRUNK_HEIGHT,
            "std": 0.1,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )
    joint_deviation_l1 = RewTerm(
        func=mdp.joint_deviation_exp_if_standing,
        weight=0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "standing_height_threshold": STANDING_HEIGHT,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
            "std": 0.1,
        },
    )
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-5.0)
    not_moving = RewTerm(
        func=mdp.moving_if_standing,
        weight=-0.05,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "weight_lin": 1.0,
            "weight_ang": 1.0,
            "standing_height_threshold": STANDING_HEIGHT,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )
    equal_foot_force = RewTerm(
        func=mdp.equal_foot_force_if_standing,
        weight=2.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=FOOT_BODIES,
                preserve_order=True,
            ),
            "asset_cfg": SceneEntityCfg("robot"),
            "standing_height_threshold": STANDING_HEIGHT,
            "height_measurement_sensor": SceneEntityCfg("height_measurement_sensor"),
        },
    )
    illegal_contacts = RewTerm(
        func=mdp.illegal_contact,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=UNDESIRED_CONTACTS,
            ),
            "threshold": 1.0,
        },
    )
    feet_distance = RewTerm(
        func=mdp.feet_distance_from_ref_if_standing,
        weight=-50.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=FOOT_BODIES,
                preserve_order=True,
            ),
            "ref_distance": 0.2,
            "standing_height_threshold": STANDING_HEIGHT,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
            "norm": "l2",
        },
    )
    feet_yaw_mean = RewTerm(
        func=mdp.feet_yaw_mean_vs_base_if_standing,
        weight=-5.0,
        params={
            "feet_asset_cfg": SceneEntityCfg(
                "robot",
                body_names=FOOT_BODIES,
                preserve_order=True,
            ),
            "base_body_cfg": SceneEntityCfg(
                "robot",
                body_names=["Trunk"],
            ),
            "standing_height_threshold": STANDING_HEIGHT,
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
        },
    )
    root_acc = RewTerm(
        func=mdp.body_acc_l2,
        weight=-5.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    stand_up_termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=10.0,
        params={"term_keys": "standing"},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    invalid_state = DoneTerm(
        func=mdp.invalid_state,
        params={
            "max_joint_vel": 100.0,
            "max_root_height": 10.0,
            "max_root_xy_distance": 200.0,
            "max_lin_vel": 20.0,
            "max_ang_vel": 50.0,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    standing = DoneTerm(
        func=mdp.standing,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_measurement_sensor"),
            "min_height": STANDING_HEIGHT,
            "duration_s": 5.0,
        },
    )


@configclass
class EventsCfg:
    disable_robot_joint_actions = EventTerm(
        func=mdp.disable_joints,
        mode="pre_sim_step",
        params={"rest_duration_s": REST_DURATION_S},
    )

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
    randomize_joint_friction = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": _deployment_asset_cfg(),
            "friction_distribution_params": (0.0, 0.005),
            "operation": "abs",
            "distribution": "uniform",
        },
    )
    randomize_joint_armature = EventTerm(
        func=mdp.randomize_joint_parameters,
        mode="startup",
        params={
            "asset_cfg": _deployment_asset_cfg(),
            "armature_distribution_params": (0.0, 2.0),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    randomize_bodies_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
        },
    )
    randomize_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    randomize_bodies_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "com_range": {
                "x": (-0.01, 0.01),
                "y": (-0.01, 0.01),
                "z": (-0.01, 0.01),
            },
        },
    )
    randomize_base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "com_range": {
                "x": (-0.15, 0.15),
                "y": (-0.05, 0.05),
                "z": (-0.15, 0.15),
            },
        },
    )
    apply_external_force_torque = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "force_range": (-10.0, 10.0),
            "torque_range": (-5.0, 5.0),
        },
    )
    apply_external_force_torque_extremities = EventTerm(
        func=mdp.apply_external_force_torque,
        mode="interval",
        interval_range_s=(0.0, 10.0),
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                body_names=[".*hand_link", ".*foot_link"],
            ),
            "force_range": (-5.0, 5.0),
            "torque_range": (-0.5, 0.5),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "roll": (-0.25, 0.25),
                "pitch": (-0.25, 0.25),
                "yaw": (-0.25, 0.25),
            }
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_from_fallen_dataset,
        mode="reset",
        params={"standing_ratio": 0.1},
    )


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(
        func=mdp.terrain_levels_successful_termination,
        params={
            "successful_termination_term": "standing",
            "n_successes": 5,
            "n_failures": 5,
        },
    )
    remove_lift = CurrTerm(
        func=mdp.remove_harness,
        params={
            "harness_action_name": "lift",
            "start": 5_000,
            "num_steps": 200_000,
            "linear": False,
        },
    )
    action_limit_successful_termination = CurrTerm(
        func=mdp.action_limit_successful_termination,
        params={
            "successful_termination_term": "standing",
            "activate_after_steps": 100_000,
            "action_name": "joint_pos",
            "update_rate": 0.001,
            "move_up_ratio": 0.95,
            "move_down_ratio": 0.8,
            "max_action_limit": 1.0,
        },
    )
    increase_action_regularization = CurrTerm(
        func=mdp.update_reward_weight_step,
        params={
            "reward_name": "action_l2",
            "start_step": 70_000,
            "num_steps": 150_000,
            "terminal_weight": -0.25,
            "use_log_space": True,
        },
    )
    increase_action_rate_regularization = CurrTerm(
        func=mdp.update_reward_weight_step,
        params={
            "reward_name": "action_rate",
            "start_step": 100_000,
            "num_steps": 150_000,
            "terminal_weight": -0.1,
            "use_log_space": True,
        },
    )
    increase_action_rate_rate_regularization = CurrTerm(
        func=mdp.update_reward_weight_step,
        params={
            "reward_name": "action_rate_rate",
            "start_step": 150_000,
            "num_steps": 150_000,
            "terminal_weight": -0.1,
            "use_log_space": False,
        },
    )
    increase_joint_deviation_regularization = CurrTerm(
        func=mdp.update_reward_weight_step,
        params={
            "reward_name": "joint_deviation_l1",
            "start_step": 100_000,
            "num_steps": 150_000,
            "terminal_weight": 10.0,
            "use_log_space": False,
        },
    )
    increase_incoming_forces_penalty = CurrTerm(
        func=mdp.update_reward_weight_step,
        params={
            "reward_name": "incoming_forces_penalty",
            "start_step": 120_000,
            "num_steps": 150_000,
            "terminal_weight": -1.0e-5,
            "use_log_space": True,
        },
    )


@configclass
class K1WbcStandUpEnvCfg(ManagerBasedRLEnvCfg):
    scene: K1WbcStandUpSceneCfg = K1WbcStandUpSceneCfg(
        num_envs=4096,
        env_spacing=2.5,
    )
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        self.decimation = 10
        self.episode_length_s = 20.0
        self.sim.dt = 1.0 / 500.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.height_measurement_sensor.update_period = self.sim.dt
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.terrain.terrain_generator.curriculum = True
        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"

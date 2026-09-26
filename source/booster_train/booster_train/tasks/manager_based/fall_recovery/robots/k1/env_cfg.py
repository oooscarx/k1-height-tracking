from __future__ import annotations

import copy
import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as UniformNoise

from booster_train.assets.robots.booster import BOOSTER_K1_CFG
from booster_train.tasks.manager_based.fall_recovery import mdp
from booster_train.tasks.manager_based.fall_recovery.hardware_config import (
    K1_HARDWARE_CONFIG,
)

CONFIG = K1_HARDWARE_CONFIG
MOTION_DIR = Path(__file__).resolve().parents[2] / "motions"
ASSET_DIR = Path(__file__).resolve().parents[2] / "assets"
JOINT_NAMES = CONFIG["joint_names"]
GOAL_POSITION = CONFIG["goal_position"]
TRAINING = CONFIG["training"]
AMP_TRAINING = TRAINING["amp"]
AMP_CANONICAL = AMP_TRAINING["canonical"]
AMP_DIRECTIONAL = AMP_TRAINING["directional"]
AMP_TRACKING = AMP_DIRECTIONAL["tracking_reward"]
NATIVE_TEACHER = CONFIG["native_teacher"]
NATIVE_ROBUST = NATIVE_TEACHER["robust_training"]
NATIVE_TASK = NATIVE_TEACHER["task_driven_training"]
TASK_HANDOFF_LIMITS = {
    "maximum_angular_velocity_start": NATIVE_TASK["success_angular_velocity_start"],
    "maximum_angular_velocity_end": TRAINING["success_angular_velocity"],
    "maximum_linear_velocity_start": NATIVE_TASK["success_linear_velocity_start"],
    "maximum_linear_velocity_end": TRAINING["success_linear_velocity"],
    "maximum_body_pose_error_start": NATIVE_TASK["success_body_pose_error_start"],
    "maximum_body_pose_error_end": TRAINING["success_body_pose_error"],
    "maximum_body_joint_velocity_start": NATIVE_TASK[
        "success_body_joint_velocity_start"
    ],
    "maximum_body_joint_velocity_end": TRAINING["success_body_joint_velocity"],
}
TASK_HANDOFF_CURRICULUM = {
    **TASK_HANDOFF_LIMITS,
    "curriculum_steps": NATIVE_TASK["handoff_curriculum_steps"],
}
TASK_POLICY_REFERENCE_FRAME_SIZE = (
    2 * len(AMP_CANONICAL["joint_names"])
    + 13
    + 3 * len(AMP_CANONICAL["key_body_names"])
)


def _deployment_asset_cfg() -> SceneEntityCfg:
    return SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)


def _gain_map(indexes: tuple[int, ...], values: list[float]) -> dict[str, float]:
    return {JOINT_NAMES[index]: values[index] for index in indexes}


def recovery_robot_cfg() -> ArticulationCfg:
    robot = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.0, 0.0, 0.52)
    robot.init_state.joint_pos = dict(zip(JOINT_NAMES, GOAL_POSITION))
    robot.init_state.joint_vel = {".*": 0.0}
    robot.soft_joint_pos_limit_factor = 1.0

    robot.actuators["head"].stiffness = _gain_map((0, 1), CONFIG["stiffness"])
    robot.actuators["head"].damping = _gain_map((0, 1), CONFIG["damping"])
    robot.actuators["arms"].stiffness = _gain_map(tuple(range(2, 10)), CONFIG["stiffness"])
    robot.actuators["arms"].damping = _gain_map(tuple(range(2, 10)), CONFIG["damping"])
    robot.actuators["legs"].stiffness = _gain_map((10, 11, 12, 13, 16, 17, 18, 19), CONFIG["stiffness"])
    robot.actuators["legs"].damping = _gain_map((10, 11, 12, 13, 16, 17, 18, 19), CONFIG["damping"])
    robot.actuators["feet"].stiffness = _gain_map((14, 15, 20, 21), CONFIG["stiffness"])
    robot.actuators["feet"].damping = _gain_map((14, 15, 20, 21), CONFIG["damping"])
    return robot


@configclass
class RecoverySceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="usd",
        usd_path=str(ASSET_DIR / "k1_flat_ground.usda"),
        collision_group=-1,
    )
    robot: ArticulationCfg = recovery_robot_cfg()
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=False,
        force_threshold=1.0,
        debug_vis=False,
    )
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.8, 0.8, 0.8), intensity=3000.0),
    )


@configclass
class CommandsCfg:
    recovery = mdp.RecoveryReferenceCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        reference_files=[
            str(MOTION_DIR / "faceup_reference.npz"),
            str(MOTION_DIR / "facedown_reference.npz"),
        ],
        reference_fps=CONFIG["policy_rate_hz"],
        joint_names=JOINT_NAMES,
        position_minimum=CONFIG["position_minimum"],
        position_maximum=CONFIG["position_maximum"],
        parallel_ankle=CONFIG["parallel_ankle"],
        curriculum_steps=TRAINING["curriculum_steps"],
        reference_reset_probability_start=TRAINING["reference_reset_probability_start"],
        reference_reset_probability_end=TRAINING["reference_reset_probability_end"],
        reference_phase_speed=tuple(TRAINING["reference_phase_speed"]),
    )


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
        stiffness=CONFIG["stiffness"],
        damping=CONFIG["damping"],
        command_torque_limit=CONFIG["command_torque_limit"],
        parallel_ankle=CONFIG["parallel_ankle"],
        position_target_margin=NATIVE_ROBUST["position_target_margin"],
        position_braking_horizon_s=NATIVE_ROBUST["position_braking_horizon_s"],
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, noise=UniformNoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=UniformNoise(n_min=-0.025, n_max=0.025))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _deployment_asset_cfg()},
            noise=UniformNoise(n_min=-0.015, n_max=0.015),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _deployment_asset_cfg()},
            scale=0.05,
            noise=UniformNoise(n_min=-0.5, n_max=0.5),
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_height = ObsTerm(func=mdp.base_pos_z)
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _deployment_asset_cfg()},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _deployment_asset_cfg()},
        )
        last_action = ObsTerm(func=mdp.last_action)
        reference_joint_pos = ObsTerm(func=mdp.reference_joint_position, params={"command_name": "recovery"})
        reference_gravity = ObsTerm(func=mdp.reference_gravity, params={"command_name": "recovery"})
        reference_height = ObsTerm(func=mdp.reference_height, params={"command_name": "recovery"})
        reference_phase = ObsTerm(func=mdp.reference_phase, params={"command_name": "recovery"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.45, 1.20),
            "dynamic_friction_range": (0.35, 1.05),
            "restitution_range": (0.0, 0.12),
            "num_buckets": 64,
        },
    )
    body_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.90, 1.10),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )
    trunk_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Trunk"),
            "com_range": {"x": (-0.02, 0.02), "y": (-0.025, 0.025), "z": (-0.02, 0.02)},
        },
    )
    actuator_gains = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.85, 1.15),
            "damping_distribution_params": (0.80, 1.20),
            "operation": "scale",
        },
    )
    push = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(0.8, 2.0),
        params={
            "velocity_range": {
                "x": (-0.45, 0.45),
                "y": (-0.45, 0.45),
                "z": (-0.2, 0.2),
                "roll": (-0.8, 0.8),
                "pitch": (-0.8, 0.8),
                "yaw": (-0.5, 0.5),
            }
        },
    )


def configure_training_randomization(
    env_cfg: ManagerBasedRLEnvCfg,
    *,
    domain_scale: float,
    push_scale: float,
    observation_noise_scale: float,
    push_interval_range_s: tuple[float, float] = (0.8, 2.0),
) -> None:
    """Restore disabled AMP randomization terms at a controlled strength."""

    scales = {
        "domain_scale": domain_scale,
        "push_scale": push_scale,
        "observation_noise_scale": observation_noise_scale,
    }
    for name, value in scales.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]")
    interval_minimum, interval_maximum = push_interval_range_s
    if interval_minimum <= 0.0 or interval_minimum > interval_maximum:
        raise ValueError("push interval must be positive and ordered")

    templates = EventsCfg()
    if domain_scale > 0.0:
        env_cfg.events.material = copy.deepcopy(templates.material)
        material = env_cfg.events.material.params
        material["static_friction_range"] = (
            0.8 + domain_scale * (0.45 - 0.8),
            0.8 + domain_scale * (1.20 - 0.8),
        )
        material["dynamic_friction_range"] = (
            0.7 + domain_scale * (0.35 - 0.7),
            0.7 + domain_scale * (1.05 - 0.7),
        )
        material["restitution_range"] = (0.0, 0.12 * domain_scale)

        env_cfg.events.body_mass = copy.deepcopy(templates.body_mass)
        env_cfg.events.body_mass.params["mass_distribution_params"] = (
            1.0 - 0.10 * domain_scale,
            1.0 + 0.10 * domain_scale,
        )

        env_cfg.events.trunk_com = copy.deepcopy(templates.trunk_com)
        env_cfg.events.trunk_com.params["com_range"] = {
            axis: (minimum * domain_scale, maximum * domain_scale)
            for axis, (minimum, maximum) in templates.trunk_com.params[
                "com_range"
            ].items()
        }

        env_cfg.events.actuator_gains = copy.deepcopy(templates.actuator_gains)
        gains = env_cfg.events.actuator_gains.params
        gains["stiffness_distribution_params"] = (
            1.0 - 0.15 * domain_scale,
            1.0 + 0.15 * domain_scale,
        )
        gains["damping_distribution_params"] = (
            1.0 - 0.20 * domain_scale,
            1.0 + 0.20 * domain_scale,
        )

    if push_scale > 0.0:
        env_cfg.events.push = copy.deepcopy(templates.push)
        env_cfg.events.push.interval_range_s = push_interval_range_s
        velocity_range = env_cfg.events.push.params["velocity_range"]
        env_cfg.events.push.params["velocity_range"] = {
            axis: (minimum * push_scale, maximum * push_scale)
            for axis, (minimum, maximum) in velocity_range.items()
        }

    if observation_noise_scale > 0.0:
        policy = env_cfg.observations.policy
        policy.enable_corruption = True
        for term_name in (
            "base_ang_vel",
            "projected_gravity",
            "joint_pos",
            "joint_vel",
        ):
            noise = getattr(policy, term_name).noise
            if noise is None:
                continue
            noise.n_min *= observation_noise_scale
            noise.n_max *= observation_noise_scale


@configclass
class RewardsCfg:
    upright = RewTerm(func=mdp.upright_exp, weight=2.0, params={"std": 0.45})
    height = RewTerm(
        func=mdp.trunk_height_progress,
        weight=1.5,
        params={"minimum_height": 0.06, "target_height": 0.52},
    )
    standing_pose = RewTerm(
        func=mdp.standing_pose_exp,
        weight=1.0,
        params={
            "goal_position": GOAL_POSITION,
            "std": 0.45,
            "asset_cfg": _deployment_asset_cfg(),
        },
    )
    stillness = RewTerm(func=mdp.body_stillness_exp, weight=0.5, params={"std": 1.0})
    early_success = RewTerm(
        func=mdp.early_recovery_bonus,
        weight=8.0,
        params={
            "hold_steps": round(TRAINING["success_hold_s"] * CONFIG["policy_rate_hz"]),
            "minimum_height": TRAINING["success_height"],
            "maximum_gravity_z": TRAINING["success_gravity_z"],
            "maximum_angular_velocity": TRAINING["success_angular_velocity"],
            "maximum_linear_velocity": TRAINING["success_linear_velocity"],
            "goal_position": GOAL_POSITION,
            "maximum_body_pose_error": TRAINING["success_body_pose_error"],
            "maximum_body_joint_velocity": TRAINING["success_body_joint_velocity"],
            "asset_cfg": _deployment_asset_cfg(),
        },
    )
    reference_joint = RewTerm(
        func=mdp.reference_joint_pose_exp,
        weight=0.8,
        params={
            "command_name": "recovery",
            "std": 0.5,
            "asset_cfg": _deployment_asset_cfg(),
        },
    )
    reference_gravity = RewTerm(
        func=mdp.reference_gravity_exp,
        weight=0.4,
        params={"command_name": "recovery", "std": 0.5},
    )
    reference_height = RewTerm(
        func=mdp.reference_height_exp,
        weight=0.4,
        params={"command_name": "recovery", "std": 0.12},
    )
    feet_support = RewTerm(
        func=mdp.feet_support,
        weight=0.25,
        params={
            "threshold": 20.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["left_foot_link", "right_foot_link"]),
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    joint_acceleration = RewTerm(func=mdp.joint_acc_l2, weight=-2.0e-7)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    parallel_projection = RewTerm(
        func=mdp.parallel_target_reduction,
        weight=-1.0,
        params={"action_name": "joint_pos"},
    )
    parallel_stress = RewTerm(
        func=mdp.parallel_motor_stress,
        weight=-0.2,
        params={"action_name": "joint_pos", "margin": 0.04, "velocity_ratio": 0.85},
    )
    head_trunk_impact = RewTerm(
        func=mdp.capped_contact_force_penalty,
        weight=-0.5,
        params={
            "threshold": 180.0,
            "maximum_force": 800.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Trunk", "Head_1", "Head_2"]),
        },
    )
@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    recovered = DoneTerm(
        func=mdp.recovered,
        params={
            "hold_steps": round(TRAINING["success_hold_s"] * CONFIG["policy_rate_hz"]),
            "minimum_height": TRAINING["success_height"],
            "maximum_gravity_z": TRAINING["success_gravity_z"],
            "maximum_angular_velocity": TRAINING["success_angular_velocity"],
            "maximum_linear_velocity": TRAINING["success_linear_velocity"],
            "goal_position": GOAL_POSITION,
            "maximum_body_pose_error": TRAINING["success_body_pose_error"],
            "maximum_body_joint_velocity": TRAINING["success_body_joint_velocity"],
            "asset_cfg": _deployment_asset_cfg(),
        },
    )
    joint_limit = DoneTerm(
        func=mdp.hard_joint_limit,
        params={
            "position_minimum": CONFIG["position_minimum"],
            "position_maximum": CONFIG["position_maximum"],
            "margin": 0.05,
            "asset_cfg": _deployment_asset_cfg(),
        },
    )
    parallel_ankle = DoneTerm(func=mdp.parallel_ankle_infeasible, params={"action_name": "joint_pos"})
    nonfinite_action = DoneTerm(
        func=mdp.nonfinite_action_or_state,
        params={"action_name": "joint_pos"},
    )
    excessive_impact = DoneTerm(
        func=mdp.excessive_impact,
        params={
            "threshold": 550.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Trunk", "Head_1", "Head_2"]),
        },
    )


@configclass
class K1FallRecoveryEnvCfg(ManagerBasedRLEnvCfg):
    scene: RecoverySceneCfg = RecoverySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = TRAINING["episode_length_s"]
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.contact_forces.update_period = self.sim.dt
        self.viewer.eye = (1.5, 1.5, 1.2)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"


@configclass
class K1FallRecoveryPlayEnvCfg(K1FallRecoveryEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.commands.recovery.play = True
        self.commands.recovery.play_all_falls = True
        self.observations.policy.enable_corruption = False
        self.events.material = None
        self.events.body_mass = None
        self.events.trunk_com = None
        self.events.actuator_gains = None
        self.events.push = None


DISCOVERY = TRAINING["discovery"]


@configclass
class DiscoveryCommandsCfg:
    recovery = mdp.RecoveryDiscoveryCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        joint_names=JOINT_NAMES,
        position_minimum=CONFIG["position_minimum"],
        position_maximum=CONFIG["position_maximum"],
        initial_joint_position=DISCOVERY["fallen_joint_position"],
        standing_joint_position=GOAL_POSITION,
        parallel_ankle=CONFIG["parallel_ankle"],
        initial_pitch=-0.5 * math.pi,
        initial_height=DISCOVERY["initial_height"],
        standing_height=DISCOVERY["standing_height"],
        standing_reset_fraction_start=DISCOVERY["standing_reset_fraction_start"],
        standing_reset_fraction_end=DISCOVERY["standing_reset_fraction_end"],
        standing_reset_curriculum_steps=DISCOVERY["standing_reset_curriculum_steps"],
        joint_noise=DISCOVERY["joint_noise"],
        root_position_noise=DISCOVERY["root_position_noise"],
        orientation_noise=DISCOVERY["orientation_noise"],
        assistance_body_name=DISCOVERY["assistance_body_name"],
        maximum_assistance_force=DISCOVERY["maximum_assistance_force"],
        assistance_target_height=DISCOVERY["assistance_target_height"],
        assistance_interval_steps=DISCOVERY["assistance_interval_steps"],
    )


@configclass
class DiscoveryActionsCfg:
    joint_pos = mdp.K1ParallelJointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        preserve_order=True,
        use_default_offset=False,
        expected_joint_names=JOINT_NAMES,
        position_minimum=CONFIG["position_minimum"],
        position_maximum=CONFIG["position_maximum"],
        position_center=GOAL_POSITION,
        position_scale=DISCOVERY["action_scale"],
        stiffness=CONFIG["stiffness"],
        damping=CONFIG["damping"],
        command_torque_limit=CONFIG["command_torque_limit"],
        parallel_ankle=CONFIG["parallel_ankle"],
        position_target_margin=NATIVE_ROBUST["position_target_margin"],
        position_braking_horizon_s=NATIVE_ROBUST["position_braking_horizon_s"],
    )


@configclass
class NativeTeacherActionsCfg:
    joint_pos = mdp.K1ParallelJointPositionActionCfg(
        asset_name="robot",
        joint_names=JOINT_NAMES,
        preserve_order=True,
        use_default_offset=False,
        expected_joint_names=JOINT_NAMES,
        position_minimum=CONFIG["position_minimum"],
        position_maximum=CONFIG["position_maximum"],
        position_center=NATIVE_TEACHER["training_action_center"],
        position_scale=NATIVE_TEACHER["training_action_scale"],
        stiffness=CONFIG["stiffness"],
        damping=CONFIG["damping"],
        command_torque_limit=CONFIG["command_torque_limit"],
        parallel_ankle=CONFIG["parallel_ankle"],
    )


@configclass
class DiscoveryObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, scale=0.25, noise=UniformNoise(n_min=-0.1, n_max=0.1))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=UniformNoise(n_min=-0.025, n_max=0.025))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _deployment_asset_cfg()},
            noise=UniformNoise(n_min=-0.015, n_max=0.015),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _deployment_asset_cfg()},
            scale=0.05,
            noise=UniformNoise(n_min=-0.5, n_max=0.5),
        )
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True
            self.history_length = DISCOVERY["history_length"]
            self.flatten_history_dim = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_height = ObsTerm(func=mdp.base_pos_z)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _deployment_asset_cfg()})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _deployment_asset_cfg()})
        last_action = ObsTerm(func=mdp.last_action)
        assistance_force = ObsTerm(func=mdp.discovery_assistance_force, params={"command_name": "recovery"})

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class DiscoveryRewardsCfg:
    height = RewTerm(
        func=mdp.trunk_height_exp,
        weight=5.0,
        params={"target_height": 0.52},
    )
    head_height = RewTerm(
        func=mdp.body_height_exp,
        weight=DISCOVERY["head_height_weight"],
        params={
            "target_height": DISCOVERY["head_height_target"],
            "asset_cfg": SceneEntityCfg("robot", body_names="Head_2"),
        },
    )
    height_increase = RewTerm(func=mdp.trunk_height_increase, weight=1.0)
    body_up = RewTerm(func=mdp.body_up_exp, weight=DISCOVERY["body_up_weight"])
    body_orientation = RewTerm(func=mdp.body_orientation_l2, weight=DISCOVERY["body_orientation_weight"])
    upright = RewTerm(func=mdp.upright_exp, weight=2.5, params={"std": 0.45})
    feet_support = RewTerm(
        func=mdp.feet_support,
        weight=2.5,
        params={
            "threshold": 20.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_foot_link", "right_foot_link"],
            ),
        },
    )
    feet_contact_increase = RewTerm(
        func=mdp.feet_contact_force_increase,
        weight=DISCOVERY["feet_contact_increase_weight"],
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_foot_link", "right_foot_link"],
            ),
        },
    )
    feet_height = RewTerm(
        func=mdp.feet_height_exp,
        weight=DISCOVERY["feet_height_weight"],
        params={
            "scale": DISCOVERY["feet_height_scale"],
            "asset_cfg": SceneEntityCfg("robot", body_names=["left_foot_link", "right_foot_link"]),
        },
    )
    symmetry = RewTerm(
        func=mdp.bilateral_action_symmetry,
        weight=DISCOVERY["symmetry_weight"],
        params={
            "action_name": "joint_pos",
            "left_indices": DISCOVERY["symmetry_left_indices"],
            "right_indices": DISCOVERY["symmetry_right_indices"],
            "mirror_signs": DISCOVERY["symmetry_mirror_signs"],
        },
    )
    standing_pose = RewTerm(
        func=mdp.standing_pose_after_height_exp,
        weight=1.0,
        params={
            "goal_position": GOAL_POSITION,
            "std": 0.45,
            "minimum_height": DISCOVERY["standing_reward_height"],
            "asset_cfg": _deployment_asset_cfg(),
        },
    )
    stillness = RewTerm(
        func=mdp.body_stillness_after_height_exp,
        weight=0.5,
        params={"std": 1.0, "minimum_height": DISCOVERY["standing_reward_height"]},
    )
    early_success = RewTerm(
        func=mdp.early_recovery_bonus,
        weight=10.0,
        params={
            "hold_steps": round(TRAINING["success_hold_s"] * CONFIG["policy_rate_hz"]),
            "minimum_height": TRAINING["success_height"],
            "maximum_gravity_z": TRAINING["success_gravity_z"],
            "maximum_angular_velocity": TRAINING["success_angular_velocity"],
            "maximum_linear_velocity": TRAINING["success_linear_velocity"],
            "goal_position": GOAL_POSITION,
            "maximum_body_pose_error": TRAINING["success_body_pose_error"],
            "maximum_body_joint_velocity": TRAINING["success_body_joint_velocity"],
            "asset_cfg": _deployment_asset_cfg(),
        },
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.02)
    joint_acceleration = RewTerm(
        func=mdp.joint_acc_l2,
        weight=DISCOVERY["joint_acceleration_weight"],
    )
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-5)
    joint_limit = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*")},
    )
    parallel_projection = RewTerm(
        func=mdp.parallel_target_reduction,
        weight=-1.0,
        params={"action_name": "joint_pos"},
    )
    parallel_stress = RewTerm(
        func=mdp.parallel_motor_stress,
        weight=-0.2,
        params={"action_name": "joint_pos", "margin": 0.04, "velocity_ratio": 0.85},
    )
    head_trunk_impact = RewTerm(
        func=mdp.capped_contact_force_penalty,
        weight=-0.5,
        params={
            "threshold": 180.0,
            "maximum_force": 800.0,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=["Trunk", "Head_1", "Head_2"]),
        },
    )


@configclass
class DiscoveryTerminationsCfg(TerminationsCfg):
    excessive_impact = None


@configclass
class K1FallRecoveryDiscoveryFaceUpEnvCfg(K1FallRecoveryEnvCfg):
    observations: DiscoveryObservationsCfg = DiscoveryObservationsCfg()
    actions: DiscoveryActionsCfg = DiscoveryActionsCfg()
    commands: DiscoveryCommandsCfg = DiscoveryCommandsCfg()
    rewards: DiscoveryRewardsCfg = DiscoveryRewardsCfg()
    terminations: DiscoveryTerminationsCfg = DiscoveryTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = DISCOVERY["episode_length_s"]
        self.decimation = 20
        self.sim.dt = 0.001
        self.sim.render_interval = self.decimation
        self.scene.contact_forces.update_period = self.sim.dt
        self.events.material = None
        self.events.body_mass = None
        self.events.trunk_com = None
        self.events.actuator_gains = None
        self.events.push = None


@configclass
class K1FallRecoveryDiscoveryFaceDownEnvCfg(K1FallRecoveryDiscoveryFaceUpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.recovery.initial_pitch = 0.5 * math.pi


@configclass
class K1FallRecoveryNativeReplayEnvCfg(K1FallRecoveryDiscoveryFaceUpEnvCfg):
    """Deterministic 50 Hz environment for replaying the deployed K1 teacher."""

    actions: ActionsCfg = ActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 7.0


@configclass
class AmpRewardsCfg(DiscoveryRewardsCfg):
    height = RewTerm(
        func=mdp.trunk_target_height_exp,
        weight=AMP_TRAINING["target_height_weight"],
        params={
            "target_height": AMP_TRAINING["target_height"],
            "sigma": AMP_TRAINING["target_height_sigma"],
        },
    )
    stand = RewTerm(
        func=mdp.standing_height,
        weight=AMP_TRAINING["standing_height_weight"],
        params={"minimum_height": AMP_TRAINING["standing_height"]},
    )
    head_height = None
    height_increase = None
    body_up = RewTerm(func=mdp.body_up_exp, weight=AMP_TRAINING["body_up_weight"])
    body_orientation = RewTerm(func=mdp.body_orientation_l2, weight=AMP_TRAINING["orientation_weight"])
    upright = None
    feet_support = RewTerm(
        func=mdp.upright_feet_support,
        weight=AMP_TRAINING["feet_support_weight"],
        params={
            "threshold": 20.0,
            "minimum_height": AMP_TRAINING["feet_support_minimum_height"],
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_foot_link", "right_foot_link"],
            ),
        },
    )
    feet_contact_increase = None
    feet_height = None
    symmetry = None
    standing_pose = RewTerm(
        func=mdp.standing_pose_after_height_exp,
        weight=AMP_TRAINING["standing_pose_weight"],
        params={
            "goal_position": GOAL_POSITION,
            "std": 0.45,
            "minimum_height": DISCOVERY["standing_reward_height"],
            "asset_cfg": _deployment_asset_cfg(),
        },
    )
    stillness = RewTerm(
        func=mdp.body_stillness_after_height_exp,
        weight=AMP_TRAINING["stillness_weight"],
        params={"std": 1.0, "minimum_height": DISCOVERY["standing_reward_height"]},
    )
    early_success = RewTerm(
        func=mdp.early_recovery_bonus,
        weight=AMP_TRAINING["early_success_weight"],
        params={
            "hold_steps": round(TRAINING["success_hold_s"] * CONFIG["policy_rate_hz"]),
            "minimum_height": TRAINING["success_height"],
            "maximum_gravity_z": TRAINING["success_gravity_z"],
            "maximum_angular_velocity": TRAINING["success_angular_velocity"],
            "maximum_linear_velocity": TRAINING["success_linear_velocity"],
            "goal_position": GOAL_POSITION,
            "maximum_body_pose_error": TRAINING["success_body_pose_error"],
            "maximum_body_joint_velocity": TRAINING["success_body_joint_velocity"],
            "asset_cfg": _deployment_asset_cfg(),
        },
    )
    parallel_state_limit = RewTerm(
        func=mdp.parallel_state_limit_penalty,
        weight=-0.2,
        params={"action_name": "joint_pos", "soft_margin": 0.03},
    )


@configclass
class AmpCanonicalRewardsCfg(AmpRewardsCfg):
    height = RewTerm(
        func=mdp.trunk_height_progress,
        weight=AMP_CANONICAL["height_progress_weight"],
        params={
            "minimum_height": AMP_CANONICAL["height_minimum"],
            "target_height": AMP_TRAINING["target_height"],
        },
    )
    body_up = None
    body_orientation = RewTerm(
        func=mdp.gravity_z_progress,
        weight=AMP_CANONICAL["orientation_progress_weight"],
        params={"initial_gravity_z": AMP_CANONICAL["initial_gravity_z"]},
    )
    unsafe_termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=-1000.0,
        params={"term_keys": ["joint_limit", "nonfinite_action"]},
    )


@configclass
class AmpDirectionalRewardsCfg(AmpRewardsCfg):
    standing_pose = RewTerm(
        func=mdp.standing_pose_after_height_exp,
        weight=AMP_DIRECTIONAL["standing_pose_weight"],
        params={
            "goal_position": GOAL_POSITION,
            "std": 0.45,
            "minimum_height": DISCOVERY["standing_reward_height"],
            "asset_cfg": _deployment_asset_cfg(),
        },
    )
    stillness = RewTerm(
        func=mdp.body_stillness_after_height_exp,
        weight=AMP_DIRECTIONAL["stillness_weight"],
        params={"std": 1.0, "minimum_height": DISCOVERY["standing_reward_height"]},
    )
    early_success = RewTerm(
        func=mdp.early_recovery_bonus,
        weight=AMP_DIRECTIONAL["early_success_weight"],
        params={
            "hold_steps": round(TRAINING["success_hold_s"] * CONFIG["policy_rate_hz"]),
            "minimum_height": TRAINING["success_height"],
            "maximum_gravity_z": TRAINING["success_gravity_z"],
            "maximum_angular_velocity": TRAINING["success_angular_velocity"],
            "maximum_linear_velocity": TRAINING["success_linear_velocity"],
            "goal_position": GOAL_POSITION,
            "maximum_body_pose_error": TRAINING["success_body_pose_error"],
            "maximum_body_joint_velocity": TRAINING["success_body_joint_velocity"],
            "asset_cfg": _deployment_asset_cfg(),
        },
    )
    parallel_stress = RewTerm(
        func=mdp.parallel_motor_stress,
        weight=-AMP_DIRECTIONAL["parallel_stress_weight"],
        params={"action_name": "joint_pos", "margin": 0.04, "velocity_ratio": 0.85},
    )
    unsafe_termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=-AMP_DIRECTIONAL["unsafe_termination_weight"],
        params={"term_keys": ["joint_limit", "parallel_ankle", "nonfinite_action"]},
    )
    reference_max_joint_position = RewTerm(
        func=mdp.amp_reference_max_joint_position_exp,
        weight=AMP_TRACKING["max_joint_position_weight"],
        params={"std": AMP_TRACKING["max_joint_position_std"]},
    )
    reference_joint_position = RewTerm(
        func=mdp.amp_reference_joint_position_exp,
        weight=AMP_TRACKING["joint_position_weight"],
        params={"std": AMP_TRACKING["joint_position_std"]},
    )
    reference_joint_velocity = RewTerm(
        func=mdp.amp_reference_joint_velocity_exp,
        weight=AMP_TRACKING["joint_velocity_weight"],
        params={"std": AMP_TRACKING["joint_velocity_std"]},
    )
    reference_root_height = RewTerm(
        func=mdp.amp_reference_root_height_exp,
        weight=AMP_TRACKING["root_height_weight"],
        params={"std": AMP_TRACKING["root_height_std"]},
    )
    reference_root_orientation = RewTerm(
        func=mdp.amp_reference_root_orientation_exp,
        weight=AMP_TRACKING["root_orientation_weight"],
        params={"std": AMP_TRACKING["root_orientation_std"]},
    )
    reference_root_linear_velocity = RewTerm(
        func=mdp.amp_reference_root_linear_velocity_exp,
        weight=AMP_TRACKING["root_linear_velocity_weight"],
        params={"std": AMP_TRACKING["root_linear_velocity_std"]},
    )
    reference_root_angular_velocity = RewTerm(
        func=mdp.amp_reference_root_angular_velocity_exp,
        weight=AMP_TRACKING["root_angular_velocity_weight"],
        params={"std": AMP_TRACKING["root_angular_velocity_std"]},
    )
    reference_feet_position = RewTerm(
        func=mdp.amp_reference_key_body_position_exp,
        weight=AMP_TRACKING["feet_position_weight"],
        params={"std": AMP_TRACKING["feet_position_std"]},
    )


@configclass
class AmpDirectionalObservationsCfg(DiscoveryObservationsCfg):
    @configclass
    class PolicyCfg(DiscoveryObservationsCfg.PolicyCfg):
        future_reference = ObsTerm(func=mdp.amp_future_reference)
        future_reference_phase = ObsTerm(func=mdp.amp_future_reference_phase)

    @configclass
    class CriticCfg(DiscoveryObservationsCfg.CriticCfg):
        future_reference = ObsTerm(func=mdp.amp_future_reference)
        future_reference_phase = ObsTerm(func=mdp.amp_future_reference_phase)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class AmpTaskDrivenObservationsCfg(AmpDirectionalObservationsCfg):
    """Checkpoint-compatible observations with no reference trajectory information."""

    @configclass
    class PolicyCfg(AmpDirectionalObservationsCfg.PolicyCfg):
        future_reference = ObsTerm(
            func=mdp.zero_amp_future_reference,
            params={"frame_size": TASK_POLICY_REFERENCE_FRAME_SIZE},
        )
        future_reference_phase = ObsTerm(func=mdp.zero_amp_future_reference_phase)

    @configclass
    class CriticCfg(AmpDirectionalObservationsCfg.CriticCfg):
        future_reference = ObsTerm(
            func=mdp.zero_amp_future_reference,
            params={"frame_size": TASK_POLICY_REFERENCE_FRAME_SIZE},
        )
        future_reference_phase = ObsTerm(func=mdp.zero_amp_future_reference_phase)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class K1FallRecoveryAmpEnvCfg(K1FallRecoveryDiscoveryFaceUpEnvCfg):
    actions: ActionsCfg = ActionsCfg()
    rewards: AmpRewardsCfg = AmpRewardsCfg()

    amp_motion_files: list[str] = [
        str(MOTION_DIR / "deepmimic" / filename) for filename in AMP_TRAINING["motion_files"]
    ]
    amp_style_motion_files: list[str] = []
    amp_joint_names: list[str] = AMP_TRAINING["joint_names"]
    amp_root_body_name: str = AMP_TRAINING["root_body_name"]
    amp_key_body_names: list[str] = AMP_TRAINING["key_body_names"]
    amp_history_length: int = AMP_TRAINING["history_length"]
    amp_reset_mode: str = "train"
    amp_canonical_reset_mode: str = "canonical"
    amp_canonical_from_reference: bool = False
    amp_canonical_joint_noise: float = 0.0
    amp_canonical_ankle_joint_noise: float = 0.0
    amp_canonical_root_xy_noise: float = 0.0
    amp_canonical_height_noise: float = 0.0
    amp_canonical_orientation_noise: float = 0.0
    amp_canonical_velocity_scale: float = 0.0
    amp_canonical_side_height: float = 0.30
    amp_canonical_random_height: float = 0.35
    amp_bilateral_left_indices: list[int] = DISCOVERY["symmetry_left_indices"]
    amp_bilateral_right_indices: list[int] = DISCOVERY["symmetry_right_indices"]
    amp_bilateral_mirror_signs: list[float] = DISCOVERY["symmetry_mirror_signs"]
    amp_failure_state_file: str = ""
    amp_failure_state_sha256: str = ""
    amp_failure_reset_probability_start: float = 0.0
    amp_failure_reset_probability_end: float = 0.0
    amp_failure_state_blend_start: float = 0.0
    amp_failure_state_blend_end: float = 0.0
    amp_failure_adaptive_curriculum: bool = False
    amp_failure_adaptive_blend_start: float = 0.75
    amp_failure_adaptive_blend_end: float = 1.0
    amp_failure_adaptive_promotion: float = 0.025
    amp_failure_adaptive_demotion: float = 0.002
    amp_failure_adaptive_success_threshold: float = 0.5
    amp_failure_adaptive_demotion_threshold: float = 0.2
    amp_failure_adaptive_min_trials: int = 32
    amp_failure_adaptive_promotion_windows: int = 1
    amp_failure_adaptive_state_output: str = ""
    amp_failure_adaptive_state_resume: str = ""
    amp_failure_adaptive_reset_statistics: bool = False
    amp_failure_adaptive_state_interval: int = 480
    amp_handoff_curriculum_adaptive: bool = False
    amp_handoff_curriculum_initial_progress: float = 0.0
    amp_handoff_curriculum_window_steps: int = 1000
    amp_handoff_curriculum_minimum_trials: int = 1024
    amp_handoff_curriculum_promotion_windows: int = 2
    amp_handoff_curriculum_demotion_windows: int = 3
    amp_handoff_curriculum_promotion_step: float = 0.05
    amp_handoff_curriculum_demotion_step: float = 0.025
    amp_handoff_curriculum_success_threshold: float = 0.85
    amp_handoff_curriculum_demotion_threshold: float = 0.50
    amp_handoff_curriculum_maximum_parallel_state_violation_fraction: float = 0.001
    amp_handoff_curriculum_maximum_hard_joint_limit_fraction: float = 0.0
    amp_handoff_curriculum_maximum_joint_limit_termination_rate: float = 0.001
    amp_handoff_curriculum_maximum_parallel_ankle_termination_rate: float = 0.001
    amp_canonical_noise_scale_start: float = 1.0
    amp_canonical_noise_scale_end: float = 1.0
    amp_robust_reset_curriculum_steps: int = 1
    amp_reference_reset_probability_start: float = AMP_TRAINING["reference_reset_probability_start"]
    amp_reference_reset_probability_end: float = AMP_TRAINING["reference_reset_probability_end"]
    amp_standing_reset_probability_start: float = AMP_TRAINING["standing_reset_probability_start"]
    amp_standing_reset_probability_end: float = AMP_TRAINING["standing_reset_probability_end"]
    amp_random_fall_probability_start: float = AMP_TRAINING["random_fall_probability_start"]
    amp_random_fall_probability_end: float = AMP_TRAINING["random_fall_probability_end"]
    amp_reference_reset_curriculum_steps: int = AMP_TRAINING["reset_curriculum_steps"]
    amp_random_fall_difficulty_start: float = AMP_TRAINING[
        "random_fall_difficulty_start"
    ]
    amp_random_fall_difficulty_end: float = AMP_TRAINING[
        "random_fall_difficulty_end"
    ]
    amp_reference_reset_height_offset: float = AMP_TRAINING["reference_reset_height_offset"]
    amp_reference_reset_joint_velocity_scale: float = AMP_TRAINING["reference_reset_joint_velocity_scale"]
    amp_reference_terminal_reset_probability: float = 0.0
    amp_reference_terminal_reset_window_s: float = 1.0
    amp_reference_clip_indices: list[int] = []
    amp_reference_clip_weights: list[float] = []
    amp_reference_phase_minimum: float = 0.0
    amp_reference_phase_maximum: float = 1.0
    amp_reference_phase_bin_edges: list[float] = []
    amp_reference_phase_bin_weights: list[float] = []
    amp_reset_joint_margin: float = AMP_TRAINING["reset_joint_margin"]
    amp_random_reset_joint_margin: float = AMP_TRAINING[
        "random_reset_joint_margin"
    ]
    amp_random_reset_parallel_motor_margin: float = AMP_TRAINING[
        "random_reset_parallel_motor_margin"
    ]
    amp_reset_ankle_neutral_fraction: float = AMP_TRAINING["reset_ankle_neutral_fraction"]
    amp_random_fall_height_range: tuple[float, float] = tuple(AMP_TRAINING["random_fall_height_range"])
    amp_random_fall_linear_velocity_start: float = AMP_TRAINING[
        "random_fall_linear_velocity_start"
    ]
    amp_random_fall_linear_velocity_end: float = AMP_TRAINING[
        "random_fall_linear_velocity_end"
    ]
    amp_random_fall_angular_velocity_start: float = AMP_TRAINING[
        "random_fall_angular_velocity_start"
    ]
    amp_random_fall_angular_velocity_end: float = AMP_TRAINING[
        "random_fall_angular_velocity_end"
    ]
    amp_random_joint_noise_start: float = AMP_TRAINING["random_joint_noise_start"]
    amp_random_joint_noise_end: float = AMP_TRAINING["random_joint_noise_end"]
    amp_random_ankle_joint_noise_start: float = AMP_TRAINING[
        "random_ankle_joint_noise_start"
    ]
    amp_random_ankle_joint_noise_end: float = AMP_TRAINING[
        "random_ankle_joint_noise_end"
    ]
    amp_random_joint_uniform_blend_start: float = AMP_TRAINING[
        "random_joint_uniform_blend_start"
    ]
    amp_random_joint_uniform_blend_end: float = AMP_TRAINING[
        "random_joint_uniform_blend_end"
    ]
    amp_random_joint_velocity_start: float = AMP_TRAINING[
        "random_joint_velocity_start"
    ]
    amp_random_joint_velocity_end: float = AMP_TRAINING[
        "random_joint_velocity_end"
    ]
    amp_random_ankle_joint_velocity_start: float = AMP_TRAINING[
        "random_ankle_joint_velocity_start"
    ]
    amp_random_ankle_joint_velocity_end: float = AMP_TRAINING[
        "random_ankle_joint_velocity_end"
    ]
    amp_random_joint_velocity_safety_horizon_s: float = AMP_TRAINING[
        "random_joint_velocity_safety_horizon_s"
    ]

    def __post_init__(self):
        super().__post_init__()
        self.commands.recovery.standing_reset_fraction_start = 0.0
        self.commands.recovery.standing_reset_fraction_end = 0.0


@configclass
class AmpCanonicalTerminationsCfg(DiscoveryTerminationsCfg):
    # The equivalent serial URDF has no physical coupled hard-stop. Stage one
    # uses a motor-space penalty and restores strict termination later.
    parallel_ankle = None


@configclass
class K1FallRecoveryAmpFaceUpCanonicalEnvCfg(K1FallRecoveryAmpEnvCfg):
    """Phase-free AMP curriculum from one stationary face-up pose."""

    actions: DiscoveryActionsCfg = DiscoveryActionsCfg()
    amp_motion_files: list[str] = [str(MOTION_DIR / "deepmimic" / AMP_DIRECTIONAL["faceup_motion_file"])]
    amp_joint_names: list[str] = AMP_CANONICAL["joint_names"]
    amp_key_body_names: list[str] = AMP_CANONICAL["key_body_names"]
    amp_reset_mode: str = "faceup"
    amp_canonical_reset_mode: str = "faceup"
    amp_canonical_from_reference: bool = True
    amp_canonical_joint_noise: float = 0.0
    amp_canonical_root_xy_noise: float = 0.0
    amp_canonical_height_noise: float = 0.0
    amp_canonical_orientation_noise: float = 0.0
    amp_canonical_velocity_scale: float = 0.0
    rewards: AmpCanonicalRewardsCfg = AmpCanonicalRewardsCfg()
    terminations: AmpCanonicalTerminationsCfg = AmpCanonicalTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False


@configclass
class K1FallRecoveryAmpDirectionalEnvCfg(K1FallRecoveryAmpEnvCfg):
    observations: AmpDirectionalObservationsCfg = AmpDirectionalObservationsCfg()
    rewards: AmpDirectionalRewardsCfg = AmpDirectionalRewardsCfg()
    amp_joint_names: list[str] = AMP_DIRECTIONAL["joint_names"]
    amp_key_body_names: list[str] = AMP_DIRECTIONAL["key_body_names"]
    amp_reference_reset_probability_start: float = AMP_DIRECTIONAL["reference_reset_probability_start"]
    amp_reference_reset_probability_end: float = AMP_DIRECTIONAL["reference_reset_probability_end"]
    amp_standing_reset_probability_start: float = AMP_DIRECTIONAL["standing_reset_probability_start"]
    amp_standing_reset_probability_end: float = AMP_DIRECTIONAL["standing_reset_probability_end"]
    amp_random_fall_probability_start: float = AMP_DIRECTIONAL["random_fall_probability_start"]
    amp_random_fall_probability_end: float = AMP_DIRECTIONAL["random_fall_probability_end"]
    amp_reference_reset_curriculum_steps: int = AMP_DIRECTIONAL["reset_curriculum_steps"]
    amp_reference_reset_height_offset: float = AMP_DIRECTIONAL["reference_reset_height_offset"]
    amp_reference_reset_joint_velocity_scale: float = AMP_DIRECTIONAL["reference_reset_joint_velocity_scale"]
    amp_reference_terminal_reset_probability: float = AMP_DIRECTIONAL["terminal_reset_probability"]
    amp_reference_terminal_reset_window_s: float = AMP_DIRECTIONAL["terminal_reset_window_s"]
    amp_canonical_from_reference: bool = AMP_DIRECTIONAL["canonical_from_reference"]
    amp_canonical_joint_noise: float = AMP_DIRECTIONAL["canonical_joint_noise"]
    amp_canonical_ankle_joint_noise: float = AMP_DIRECTIONAL["canonical_joint_noise"]
    amp_canonical_root_xy_noise: float = AMP_DIRECTIONAL["canonical_root_xy_noise"]
    amp_canonical_height_noise: float = AMP_DIRECTIONAL["canonical_height_noise"]
    amp_canonical_orientation_noise: float = AMP_DIRECTIONAL["canonical_orientation_noise"]
    amp_canonical_velocity_scale: float = AMP_DIRECTIONAL["canonical_velocity_scale"]


@configclass
class K1FallRecoveryAmpFaceUpEnvCfg(K1FallRecoveryAmpDirectionalEnvCfg):
    amp_motion_files: list[str] = [str(MOTION_DIR / "deepmimic" / AMP_DIRECTIONAL["faceup_motion_file"])]
    amp_canonical_reset_mode: str = "faceup"


@configclass
class K1FallRecoveryAmpFaceDownEnvCfg(K1FallRecoveryAmpDirectionalEnvCfg):
    amp_motion_files: list[str] = [str(MOTION_DIR / "deepmimic" / AMP_DIRECTIONAL["facedown_motion_file"])]
    amp_canonical_reset_mode: str = "facedown"


@configclass
class AmpNativeTrackingRewardsCfg(AmpDirectionalRewardsCfg):
    reference_action = RewTerm(
        func=mdp.amp_reference_action_exp,
        weight=NATIVE_TEACHER["action_tracking_weight"],
        params={
            "action_name": "joint_pos",
            "std": NATIVE_TEACHER["action_tracking_std"],
        },
    )


@configclass
class AmpNativeRobustRewardsCfg(AmpNativeTrackingRewardsCfg):
    reference_action = RewTerm(
        func=mdp.amp_reference_action_exp,
        weight=NATIVE_ROBUST["action_tracking_weight"],
        params={
            "action_name": "joint_pos",
            "std": NATIVE_TEACHER["action_tracking_std"],
        },
    )
    parallel_state_limit = RewTerm(
        func=mdp.parallel_state_limit_penalty,
        weight=-NATIVE_ROBUST["parallel_state_limit_weight"],
        params={
            "action_name": "joint_pos",
            "soft_margin": NATIVE_ROBUST["parallel_state_soft_margin"],
        },
    )
    parallel_stress = RewTerm(
        func=mdp.parallel_motor_stress,
        weight=-NATIVE_ROBUST["parallel_stress_weight"],
        params={
            "action_name": "joint_pos",
            "margin": NATIVE_ROBUST["parallel_stress_margin"],
            "velocity_ratio": 0.85,
        },
    )
    unsafe_termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=-NATIVE_ROBUST["unsafe_termination_weight"],
        params={"term_keys": ["joint_limit", "parallel_ankle", "nonfinite_action"]},
    )


@configclass
class AmpNativeTaskDrivenRewardsCfg(AmpNativeRobustRewardsCfg):
    """Optimize recovery outcome and duration while AMP supplies a style prior."""

    reference_action = None
    reference_max_joint_position = None
    reference_joint_position = None
    reference_joint_velocity = None
    reference_root_height = None
    reference_root_orientation = None
    reference_root_linear_velocity = None
    reference_root_angular_velocity = None
    reference_feet_position = None

    joint_limit = RewTerm(
        func=mdp.hardware_joint_limit_violation,
        weight=-NATIVE_TASK["joint_limit_weight"],
        params={
            "position_minimum": CONFIG["position_minimum"],
            "position_maximum": CONFIG["position_maximum"],
            "soft_margin": NATIVE_TASK["joint_limit_soft_margin"],
            "asset_cfg": _deployment_asset_cfg(),
        },
    )
    joint_limit_target_reduction = RewTerm(
        func=mdp.joint_limit_target_reduction,
        weight=-NATIVE_TASK["joint_limit_target_reduction_weight"],
        params={"action_name": "joint_pos"},
    )
    height = RewTerm(
        func=mdp.trunk_target_height_upright_exp,
        weight=NATIVE_TASK["target_height_weight"],
        params={
            "target_height": NATIVE_TASK["height_target"],
            "sigma": AMP_TRAINING["target_height_sigma"],
        },
    )
    stand = RewTerm(
        func=mdp.standing_height_upright,
        weight=NATIVE_TASK["standing_height_weight"],
        params={"minimum_height": AMP_TRAINING["standing_height"]},
    )
    body_up = None
    body_orientation = None
    upright = RewTerm(
        func=mdp.upright_exp,
        weight=NATIVE_TASK["upright_weight"],
        params={"std": 0.45},
    )
    feet_support = RewTerm(
        func=mdp.upright_feet_support,
        weight=NATIVE_TASK["feet_support_weight"],
        params={
            "threshold": 20.0,
            "minimum_height": AMP_TRAINING["feet_support_minimum_height"],
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_foot_link", "right_foot_link"],
            ),
        },
    )
    standing_pose = RewTerm(
        func=mdp.standing_pose_max_error_progress_after_height_upright_exp,
        weight=NATIVE_TASK["standing_pose_weight"],
        params={
            "goal_position": GOAL_POSITION,
            "std": NATIVE_TASK["standing_pose_max_error_std"],
            "minimum_height": TRAINING["success_height"],
            "maximum_rate": NATIVE_TASK["standing_pose_progress_maximum_rate"],
            "asset_cfg": _deployment_asset_cfg(),
        },
    )
    standing_pose_max_error_hold = RewTerm(
        func=mdp.standing_pose_max_error_after_height_upright_exp,
        weight=NATIVE_TASK["standing_pose_max_error_hold_weight"],
        params={
            "goal_position": GOAL_POSITION,
            "std": NATIVE_TASK["standing_pose_max_error_std"],
            "minimum_height": TRAINING["success_height"],
            "asset_cfg": _deployment_asset_cfg(),
        },
    )
    standing_pose_all_joints = RewTerm(
        func=mdp.standing_pose_progress_after_height_upright_exp,
        weight=NATIVE_TASK["standing_pose_mean_error_weight"],
        params={
            "goal_position": GOAL_POSITION,
            "std": NATIVE_TASK["standing_pose_mean_error_std"],
            "minimum_height": TRAINING["success_height"],
            "maximum_rate": NATIVE_TASK["standing_pose_progress_maximum_rate"],
            "asset_cfg": _deployment_asset_cfg(),
        },
    )
    stillness = RewTerm(
        func=mdp.body_stillness_after_height_upright_exp,
        weight=NATIVE_TASK["stillness_weight"],
        params={
            "std": 1.0,
            "minimum_height": DISCOVERY["standing_reward_height"],
        },
    )
    handoff_quality = RewTerm(
        func=mdp.strict_handoff_quality,
        weight=NATIVE_TASK["handoff_quality_weight"],
        params={
            "minimum_height": TRAINING["success_height"],
            "maximum_gravity_z": TRAINING["success_gravity_z"],
            "goal_position": GOAL_POSITION,
            "asset_cfg": _deployment_asset_cfg(),
            **TASK_HANDOFF_LIMITS,
        },
    )
    recovery_progress = RewTerm(
        func=mdp.recovery_potential_progress,
        weight=NATIVE_TASK["progress_weight"],
        params={
            "minimum_height": NATIVE_TASK["height_minimum"],
            "target_height": NATIVE_TASK["height_target"],
            "height_fraction": NATIVE_TASK["height_fraction"],
            "coupling_fraction": NATIVE_TASK["coupling_fraction"],
            "maximum_rate": NATIVE_TASK["maximum_progress_rate"],
            "asset_cfg": _deployment_asset_cfg(),
        },
    )
    recovery_time = RewTerm(
        func=mdp.recovery_time_cost_curriculum,
        weight=-NATIVE_TASK["time_cost_weight"],
        params={
            "minimum_height": TRAINING["success_height"],
            "maximum_gravity_z": TRAINING["success_gravity_z"],
            "goal_position": GOAL_POSITION,
            "asset_cfg": _deployment_asset_cfg(),
            **TASK_HANDOFF_CURRICULUM,
        },
    )
    stability_hold = RewTerm(
        func=mdp.recovery_stability_hold_progress_curriculum,
        weight=NATIVE_TASK["stability_hold_progress_weight"],
        params={
            "hold_steps": round(TRAINING["success_hold_s"] * CONFIG["policy_rate_hz"]),
            "minimum_height": TRAINING["success_height"],
            "maximum_gravity_z": TRAINING["success_gravity_z"],
            "goal_position": GOAL_POSITION,
            "asset_cfg": _deployment_asset_cfg(),
            **TASK_HANDOFF_CURRICULUM,
        },
    )
    early_success = RewTerm(
        func=mdp.early_recovery_bonus_curriculum,
        weight=NATIVE_TASK["early_success_weight"],
        params={
            "hold_steps": round(TRAINING["success_hold_s"] * CONFIG["policy_rate_hz"]),
            "minimum_height": TRAINING["success_height"],
            "maximum_gravity_z": TRAINING["success_gravity_z"],
            "goal_position": GOAL_POSITION,
            "asset_cfg": _deployment_asset_cfg(),
            "minimum_strict_quality_scale": NATIVE_TASK[
                "early_success_strict_quality_minimum_scale"
            ],
            "maximum_strict_quality_scale": NATIVE_TASK[
                "early_success_strict_quality_maximum_scale"
            ],
            **TASK_HANDOFF_CURRICULUM,
        },
    )


@configclass
class AmpNativeRobustTerminationsCfg(DiscoveryTerminationsCfg):
    joint_limit = DoneTerm(
        func=mdp.hard_joint_limit_curriculum,
        params={
            "position_minimum": CONFIG["position_minimum"],
            "position_maximum": CONFIG["position_maximum"],
            "margin_start": NATIVE_ROBUST["joint_state_tolerance_start"],
            "margin_end": NATIVE_ROBUST["joint_state_tolerance_end"],
            "curriculum_steps": NATIVE_ROBUST["reset_curriculum_steps"],
            "asset_cfg": _deployment_asset_cfg(),
        },
    )
    parallel_ankle = DoneTerm(
        func=mdp.parallel_ankle_infeasible_curriculum,
        params={
            "action_name": "joint_pos",
            "motor_tolerance_start": NATIVE_ROBUST["parallel_state_tolerance_start"],
            "motor_tolerance_end": NATIVE_ROBUST["parallel_state_tolerance_end"],
            "curriculum_steps": NATIVE_ROBUST["reset_curriculum_steps"],
        },
    )


@configclass
class AmpNativeTaskDrivenTerminationsCfg(AmpNativeRobustTerminationsCfg):
    recovered = DoneTerm(
        func=mdp.recovered_curriculum,
        params={
            "hold_steps": round(TRAINING["success_hold_s"] * CONFIG["policy_rate_hz"]),
            "minimum_height": TRAINING["success_height"],
            "maximum_gravity_z": TRAINING["success_gravity_z"],
            "goal_position": GOAL_POSITION,
            "asset_cfg": _deployment_asset_cfg(),
            **TASK_HANDOFF_CURRICULUM,
        },
    )


@configclass
class K1FallRecoveryAmpFaceUpNativeTrackingEnvCfg(K1FallRecoveryAmpDirectionalEnvCfg):
    """Reference-conditioned first-stage imitation of the deployed K1 recovery."""

    actions: NativeTeacherActionsCfg = NativeTeacherActionsCfg()
    rewards: AmpNativeTrackingRewardsCfg = AmpNativeTrackingRewardsCfg()
    amp_motion_files: list[str] = [str(MOTION_DIR / NATIVE_TEACHER["faceup_teacher_rollout_file"])]
    amp_joint_names: list[str] = AMP_CANONICAL["joint_names"]
    amp_key_body_names: list[str] = AMP_CANONICAL["key_body_names"]
    amp_reset_mode: str = "reference"
    amp_canonical_reset_mode: str = "faceup"
    amp_reference_reset_probability_start: float = 1.0
    amp_reference_reset_probability_end: float = 1.0
    amp_standing_reset_probability_start: float = 0.0
    amp_standing_reset_probability_end: float = 0.0
    amp_random_fall_probability_start: float = 0.0
    amp_random_fall_probability_end: float = 0.0
    amp_reference_reset_joint_velocity_scale: float = 1.0
    amp_reference_terminal_reset_probability: float = 0.0
    amp_reset_ankle_neutral_fraction: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.enable_corruption = False


@configclass
class K1FallRecoveryAmpFaceUpNativeRobustEnvCfg(K1FallRecoveryAmpFaceUpNativeTrackingEnvCfg):
    """Skill-preserving AMP stage for perturbed face-up starts."""

    rewards: AmpNativeRobustRewardsCfg = AmpNativeRobustRewardsCfg()
    terminations: AmpNativeRobustTerminationsCfg = AmpNativeRobustTerminationsCfg()
    amp_reset_mode: str = "train"
    amp_reference_reset_probability_start: float = NATIVE_ROBUST["reference_reset_probability"]
    amp_reference_reset_probability_end: float = NATIVE_ROBUST["reference_reset_probability"]
    amp_standing_reset_probability_start: float = 0.0
    amp_standing_reset_probability_end: float = 0.0
    amp_random_fall_probability_start: float = 0.0
    amp_random_fall_probability_end: float = 0.0
    amp_failure_state_file: str = str(MOTION_DIR / NATIVE_ROBUST["failure_state_file"])
    amp_failure_state_sha256: str = NATIVE_ROBUST["failure_state_sha256"]
    amp_failure_reset_probability_start: float = NATIVE_ROBUST["failure_reset_probability_start"]
    amp_failure_reset_probability_end: float = NATIVE_ROBUST["failure_reset_probability_end"]
    amp_failure_state_blend_start: float = NATIVE_ROBUST["failure_state_blend_start"]
    amp_failure_state_blend_end: float = NATIVE_ROBUST["failure_state_blend_end"]
    amp_canonical_noise_scale_start: float = NATIVE_ROBUST["canonical_noise_scale_start"]
    amp_canonical_noise_scale_end: float = NATIVE_ROBUST["canonical_noise_scale_end"]
    amp_robust_reset_curriculum_steps: int = NATIVE_ROBUST["reset_curriculum_steps"]
    amp_canonical_joint_noise: float = NATIVE_ROBUST["canonical_joint_noise"]
    amp_canonical_ankle_joint_noise: float = NATIVE_ROBUST["canonical_ankle_joint_noise"]
    amp_canonical_root_xy_noise: float = NATIVE_ROBUST["canonical_root_xy_noise"]
    amp_canonical_height_noise: float = NATIVE_ROBUST["canonical_height_noise"]
    amp_canonical_orientation_noise: float = NATIVE_ROBUST["canonical_orientation_noise"]
    amp_canonical_velocity_scale: float = NATIVE_ROBUST["canonical_velocity_scale"]
    amp_reset_ankle_neutral_fraction: float = NATIVE_ROBUST["reset_ankle_neutral_fraction"]


@configclass
class K1FallRecoveryAmpFaceUpNativeTaskDrivenEnvCfg(K1FallRecoveryAmpFaceUpNativeRobustEnvCfg):
    """Phase-independent face-up recovery with the native motion used only by AMP."""

    observations: AmpTaskDrivenObservationsCfg = AmpTaskDrivenObservationsCfg()
    rewards: AmpNativeTaskDrivenRewardsCfg = AmpNativeTaskDrivenRewardsCfg()
    terminations: AmpNativeTaskDrivenTerminationsCfg = AmpNativeTaskDrivenTerminationsCfg()
    amp_handoff_curriculum_adaptive: bool = True
    amp_handoff_curriculum_window_steps: int = NATIVE_TASK[
        "handoff_curriculum_window_steps"
    ]
    amp_handoff_curriculum_minimum_trials: int = NATIVE_TASK[
        "handoff_curriculum_minimum_trials"
    ]
    amp_handoff_curriculum_promotion_windows: int = NATIVE_TASK[
        "handoff_curriculum_promotion_windows"
    ]
    amp_handoff_curriculum_demotion_windows: int = NATIVE_TASK[
        "handoff_curriculum_demotion_windows"
    ]
    amp_handoff_curriculum_promotion_step: float = NATIVE_TASK[
        "handoff_curriculum_promotion_step"
    ]
    amp_handoff_curriculum_demotion_step: float = NATIVE_TASK[
        "handoff_curriculum_demotion_step"
    ]
    amp_handoff_curriculum_success_threshold: float = NATIVE_TASK[
        "handoff_curriculum_success_threshold"
    ]
    amp_handoff_curriculum_demotion_threshold: float = NATIVE_TASK[
        "handoff_curriculum_demotion_threshold"
    ]
    amp_handoff_curriculum_maximum_parallel_state_violation_fraction: float = NATIVE_TASK[
        "handoff_curriculum_maximum_parallel_state_violation_fraction"
    ]
    amp_handoff_curriculum_maximum_hard_joint_limit_fraction: float = NATIVE_TASK[
        "handoff_curriculum_maximum_hard_joint_limit_fraction"
    ]
    amp_handoff_curriculum_maximum_joint_limit_termination_rate: float = NATIVE_TASK[
        "handoff_curriculum_maximum_joint_limit_termination_rate"
    ]
    amp_handoff_curriculum_maximum_parallel_ankle_termination_rate: float = NATIVE_TASK[
        "handoff_curriculum_maximum_parallel_ankle_termination_rate"
    ]
    amp_style_motion_files: list[str] = [
        str(MOTION_DIR / NATIVE_TASK["amp_style_motion_file"])
    ]
    amp_reference_reset_probability_start: float = NATIVE_TASK["reference_reset_probability"]
    amp_reference_reset_probability_end: float = NATIVE_TASK["reference_reset_probability"]
    amp_failure_reset_probability_start: float = NATIVE_TASK["failure_reset_probability"]
    amp_failure_reset_probability_end: float = NATIVE_TASK["failure_reset_probability"]
    amp_failure_state_blend_start: float = NATIVE_TASK["failure_state_blend_start"]
    amp_failure_state_blend_end: float = NATIVE_TASK["failure_state_blend_end"]
    amp_canonical_noise_scale_start: float = NATIVE_TASK["canonical_noise_scale"]
    amp_canonical_noise_scale_end: float = NATIVE_TASK["canonical_noise_scale"]
    amp_failure_adaptive_blend_start: float = NATIVE_TASK["failure_state_blend_start"]
    amp_failure_adaptive_blend_end: float = NATIVE_TASK["failure_state_blend_end"]
    amp_failure_adaptive_promotion: float = NATIVE_TASK["adaptive_promotion_step"]
    amp_failure_adaptive_demotion: float = NATIVE_TASK["adaptive_demotion_step"]
    amp_failure_adaptive_success_threshold: float = NATIVE_TASK["adaptive_success_threshold"]
    amp_failure_adaptive_demotion_threshold: float = NATIVE_TASK["adaptive_demotion_threshold"]
    amp_failure_adaptive_min_trials: int = NATIVE_TASK["adaptive_min_trials"]
    amp_failure_adaptive_promotion_windows: int = NATIVE_TASK["adaptive_promotion_windows"]

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = NATIVE_TASK["episode_length_s"]


@configclass
class K1FallRecoveryAmpFaceDownTaskDrivenEnvCfg(K1FallRecoveryAmpFaceDownEnvCfg):
    """Phase-independent face-down recovery with DeepMimic used only by AMP."""

    actions: ActionsCfg = ActionsCfg()
    observations: AmpTaskDrivenObservationsCfg = AmpTaskDrivenObservationsCfg()
    rewards: AmpNativeTaskDrivenRewardsCfg = AmpNativeTaskDrivenRewardsCfg()
    terminations: AmpNativeTaskDrivenTerminationsCfg = AmpNativeTaskDrivenTerminationsCfg()
    amp_handoff_curriculum_adaptive: bool = True
    amp_handoff_curriculum_window_steps: int = NATIVE_TASK[
        "handoff_curriculum_window_steps"
    ]
    amp_handoff_curriculum_minimum_trials: int = NATIVE_TASK[
        "handoff_curriculum_minimum_trials"
    ]
    amp_handoff_curriculum_promotion_windows: int = NATIVE_TASK[
        "handoff_curriculum_promotion_windows"
    ]
    amp_handoff_curriculum_demotion_windows: int = NATIVE_TASK[
        "handoff_curriculum_demotion_windows"
    ]
    amp_handoff_curriculum_promotion_step: float = NATIVE_TASK[
        "handoff_curriculum_promotion_step"
    ]
    amp_handoff_curriculum_demotion_step: float = NATIVE_TASK[
        "handoff_curriculum_demotion_step"
    ]
    amp_handoff_curriculum_success_threshold: float = NATIVE_TASK[
        "handoff_curriculum_success_threshold"
    ]
    amp_handoff_curriculum_demotion_threshold: float = NATIVE_TASK[
        "handoff_curriculum_demotion_threshold"
    ]
    amp_handoff_curriculum_maximum_parallel_state_violation_fraction: float = NATIVE_TASK[
        "handoff_curriculum_maximum_parallel_state_violation_fraction"
    ]
    amp_handoff_curriculum_maximum_hard_joint_limit_fraction: float = NATIVE_TASK[
        "handoff_curriculum_maximum_hard_joint_limit_fraction"
    ]
    amp_handoff_curriculum_maximum_joint_limit_termination_rate: float = NATIVE_TASK[
        "handoff_curriculum_maximum_joint_limit_termination_rate"
    ]
    amp_handoff_curriculum_maximum_parallel_ankle_termination_rate: float = NATIVE_TASK[
        "handoff_curriculum_maximum_parallel_ankle_termination_rate"
    ]
    amp_reset_mode: str = "reference"
    amp_reference_reset_probability_start: float = 1.0
    amp_reference_reset_probability_end: float = 1.0
    amp_standing_reset_probability_start: float = 0.0
    amp_standing_reset_probability_end: float = 0.0
    amp_random_fall_probability_start: float = 0.0
    amp_random_fall_probability_end: float = 0.0
    amp_failure_reset_probability_start: float = 0.0
    amp_failure_reset_probability_end: float = 0.0
    amp_reference_reset_joint_velocity_scale: float = 1.0
    amp_reset_ankle_neutral_fraction: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = NATIVE_TASK["episode_length_s"]


@configclass
class K1FallRecoveryAmpTaskDrivenEnvCfg(K1FallRecoveryAmpEnvCfg):
    """One phase-free policy for face-up, face-down, side, and random recovery."""

    actions: ActionsCfg = ActionsCfg()
    observations: AmpTaskDrivenObservationsCfg = AmpTaskDrivenObservationsCfg()
    rewards: AmpNativeTaskDrivenRewardsCfg = AmpNativeTaskDrivenRewardsCfg()
    terminations: AmpNativeTaskDrivenTerminationsCfg = AmpNativeTaskDrivenTerminationsCfg()
    amp_handoff_curriculum_adaptive: bool = True
    amp_handoff_curriculum_window_steps: int = NATIVE_TASK[
        "handoff_curriculum_window_steps"
    ]
    amp_handoff_curriculum_minimum_trials: int = NATIVE_TASK[
        "handoff_curriculum_minimum_trials"
    ]
    amp_handoff_curriculum_promotion_windows: int = NATIVE_TASK[
        "handoff_curriculum_promotion_windows"
    ]
    amp_handoff_curriculum_demotion_windows: int = NATIVE_TASK[
        "handoff_curriculum_demotion_windows"
    ]
    amp_handoff_curriculum_promotion_step: float = NATIVE_TASK[
        "handoff_curriculum_promotion_step"
    ]
    amp_handoff_curriculum_demotion_step: float = NATIVE_TASK[
        "handoff_curriculum_demotion_step"
    ]
    amp_handoff_curriculum_success_threshold: float = NATIVE_TASK[
        "handoff_curriculum_success_threshold"
    ]
    amp_handoff_curriculum_demotion_threshold: float = NATIVE_TASK[
        "handoff_curriculum_demotion_threshold"
    ]
    amp_handoff_curriculum_maximum_parallel_state_violation_fraction: float = NATIVE_TASK[
        "handoff_curriculum_maximum_parallel_state_violation_fraction"
    ]
    amp_handoff_curriculum_maximum_hard_joint_limit_fraction: float = NATIVE_TASK[
        "handoff_curriculum_maximum_hard_joint_limit_fraction"
    ]
    amp_handoff_curriculum_maximum_joint_limit_termination_rate: float = NATIVE_TASK[
        "handoff_curriculum_maximum_joint_limit_termination_rate"
    ]
    amp_handoff_curriculum_maximum_parallel_ankle_termination_rate: float = NATIVE_TASK[
        "handoff_curriculum_maximum_parallel_ankle_termination_rate"
    ]
    amp_motion_files: list[str] = [
        str(MOTION_DIR / NATIVE_TEACHER["faceup_teacher_rollout_file"]),
        str(MOTION_DIR / "deepmimic" / AMP_DIRECTIONAL["facedown_motion_file"]),
    ]
    amp_style_motion_files: list[str] = [
        str(MOTION_DIR / NATIVE_TASK["amp_style_motion_file"]),
        str(MOTION_DIR / "deepmimic" / AMP_DIRECTIONAL["facedown_motion_file"]),
    ]
    amp_reset_mode: str = "train"
    amp_reference_reset_probability_start: float = 1.0
    amp_reference_reset_probability_end: float = 1.0
    amp_standing_reset_probability_start: float = 0.0
    amp_standing_reset_probability_end: float = 0.0
    amp_random_fall_probability_start: float = 0.0
    amp_random_fall_probability_end: float = 0.0
    amp_failure_reset_probability_start: float = 0.0
    amp_failure_reset_probability_end: float = 0.0
    amp_reference_reset_joint_velocity_scale: float = 1.0
    amp_reference_terminal_reset_probability: float = 0.0
    amp_reset_ankle_neutral_fraction: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = NATIVE_TASK["episode_length_s"]

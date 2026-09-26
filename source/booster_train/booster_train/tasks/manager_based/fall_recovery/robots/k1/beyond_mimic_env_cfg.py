from __future__ import annotations

from pathlib import Path

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as UniformNoise

import booster_train.tasks.manager_based.beyond_mimic.mdp as bm_mdp
from booster_train.tasks.manager_based.fall_recovery import mdp
from booster_train.tasks.manager_based.fall_recovery.hardware_config import K1_HARDWARE_CONFIG

from .env_cfg import GOAL_POSITION, JOINT_NAMES, EventsCfg, RecoverySceneCfg, _deployment_asset_cfg

CONFIG = K1_HARDWARE_CONFIG
TRAINING = CONFIG["training"]
DISCOVERY = TRAINING["discovery"]
MOTION_DIR = Path(__file__).resolve().parents[2] / "motions" / "deepmimic"

TRACKED_BODY_NAMES = [
    "Trunk",
    "Left_Hip_Roll",
    "Left_Shank",
    "left_foot_link",
    "Right_Hip_Roll",
    "Right_Shank",
    "right_foot_link",
    "left_hand_link",
    "right_hand_link",
]

FAILURE_TERMS = [
    "anchor_pos",
    "anchor_ori",
    "end_effector_pos",
]


@configclass
class BeyondMimicCommandsCfg:
    motion = bm_mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        motion_file=str(MOTION_DIR / "k1_getup_faceup_50hz.npz"),
        anchor_body_name="Trunk",
        body_names=TRACKED_BODY_NAMES,
        pose_range={
            "x": (-0.03, 0.03),
            "y": (-0.03, 0.03),
            "z": (-0.01, 0.01),
            "roll": (-0.08, 0.08),
            "pitch": (-0.08, 0.08),
            "yaw": (-0.20, 0.20),
        },
        velocity_range={
            "x": (-0.15, 0.15),
            "y": (-0.15, 0.15),
            "z": (-0.10, 0.10),
            "roll": (-0.30, 0.30),
            "pitch": (-0.30, 0.30),
            "yaw": (-0.30, 0.30),
        },
        joint_position_range=(0.0, 0.0),
        joint_velocity_scale=1.0,
        static_start_probability=0.25,
        adaptive_kernel_size=3,
        adaptive_lambda=0.8,
        adaptive_uniform_ratio=0.1,
        adaptive_alpha=0.001,
        failure_term_names=FAILURE_TERMS,
    )


@configclass
class BeyondMimicActionsCfg:
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
        reference_command_name="motion",
    )


@configclass
class BeyondMimicObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        reference_joint_state = ObsTerm(
            func=mdp.motion_command_deployment,
            params={"command_name": "motion", "asset_cfg": _deployment_asset_cfg()},
        )
        reference_anchor_orientation = ObsTerm(
            func=bm_mdp.motion_anchor_ori_b,
            params={"command_name": "motion"},
            noise=UniformNoise(n_min=-0.03, n_max=0.03),
        )
        base_angular_velocity = ObsTerm(
            func=mdp.base_ang_vel,
            scale=0.25,
            noise=UniformNoise(n_min=-0.1, n_max=0.1),
        )
        joint_position = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _deployment_asset_cfg()},
            noise=UniformNoise(n_min=-0.015, n_max=0.015),
        )
        joint_velocity = ObsTerm(
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
        reference_joint_state = ObsTerm(
            func=mdp.motion_command_deployment,
            params={"command_name": "motion", "asset_cfg": _deployment_asset_cfg()},
        )
        reference_anchor_position = ObsTerm(
            func=bm_mdp.motion_anchor_pos_b,
            params={"command_name": "motion"},
        )
        reference_anchor_orientation = ObsTerm(
            func=bm_mdp.motion_anchor_ori_b,
            params={"command_name": "motion"},
        )
        body_position = ObsTerm(func=bm_mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_orientation = ObsTerm(func=bm_mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_linear_velocity = ObsTerm(func=mdp.base_lin_vel)
        base_angular_velocity = ObsTerm(func=mdp.base_ang_vel)
        joint_position = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _deployment_asset_cfg()})
        joint_velocity = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _deployment_asset_cfg()})
        last_action = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class BeyondMimicRewardsCfg:
    height = RewTerm(func=mdp.trunk_height_exp, weight=5.0, params={"target_height": 0.52})
    height_increase = RewTerm(func=mdp.trunk_height_increase, weight=1.0)
    upright = RewTerm(func=mdp.upright_exp, weight=2.5, params={"std": 0.45})
    body_up = RewTerm(func=mdp.body_up_exp, weight=0.5)
    body_orientation = RewTerm(func=mdp.body_orientation_l2, weight=-1.0)
    feet_support = RewTerm(
        func=mdp.upright_feet_support,
        weight=2.0,
        params={
            "threshold": 20.0,
            "minimum_height": 0.3,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["left_foot_link", "right_foot_link"],
            ),
        },
    )
    reference_anchor_position = RewTerm(
        func=bm_mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    reference_anchor_orientation = RewTerm(
        func=bm_mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    reference_body_position = RewTerm(
        func=bm_mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    reference_body_orientation = RewTerm(
        func=bm_mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    reference_body_linear_velocity = RewTerm(
        func=bm_mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    reference_body_angular_velocity = RewTerm(
        func=bm_mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )
    reference_foot_orientation = RewTerm(
        func=bm_mdp.motion_relative_body_orientation_error_exp,
        weight=15.0,
        params={
            "command_name": "motion",
            "std": 0.2,
            "body_names": ["left_foot_link", "right_foot_link"],
        },
    )
    reference_hand_orientation = RewTerm(
        func=bm_mdp.motion_relative_body_orientation_error_exp,
        weight=3.0,
        params={
            "command_name": "motion",
            "std": 0.2,
            "body_names": ["left_hand_link", "right_hand_link"],
        },
    )
    reference_foot_position = RewTerm(
        func=bm_mdp.motion_relative_body_position_error_exp,
        weight=15.0,
        params={
            "command_name": "motion",
            "std": 0.2,
            "body_names": ["left_foot_link", "right_foot_link"],
        },
    )
    reference_hand_position = RewTerm(
        func=bm_mdp.motion_relative_body_position_error_exp,
        weight=8.0,
        params={
            "command_name": "motion",
            "std": 0.2,
            "body_names": ["left_hand_link", "right_hand_link"],
        },
    )
    reference_trunk_orientation = RewTerm(
        func=bm_mdp.motion_relative_body_orientation_error_exp,
        weight=10.0,
        params={"command_name": "motion", "std": 0.2, "body_names": ["Trunk"]},
    )
    reference_trunk_position = RewTerm(
        func=bm_mdp.motion_relative_body_position_error_exp,
        weight=20.0,
        params={"command_name": "motion", "std": 0.2, "body_names": ["Trunk"]},
    )
    standing_pose = RewTerm(
        func=mdp.standing_pose_after_height_exp,
        weight=5.0,
        params={
            "goal_position": GOAL_POSITION,
            "std": 0.45,
            "minimum_height": DISCOVERY["standing_reward_height"],
            "asset_cfg": _deployment_asset_cfg(),
        },
    )
    stillness = RewTerm(
        func=mdp.body_stillness_after_height_exp,
        weight=2.0,
        params={"std": 1.0, "minimum_height": DISCOVERY["standing_reward_height"]},
    )
    early_success = RewTerm(
        func=mdp.early_recovery_bonus,
        weight=40.0,
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
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_acceleration = RewTerm(func=mdp.joint_acc_l2, weight=-1.0e-8)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6)
    joint_limit = RewTerm(
        func=mdp.hardware_joint_limit_violation,
        weight=-20.0,
        params={
            "position_minimum": CONFIG["position_minimum"],
            "position_maximum": CONFIG["position_maximum"],
            "soft_margin": 0.02,
            "asset_cfg": _deployment_asset_cfg(),
        },
    )
    parallel_projection = RewTerm(
        func=mdp.parallel_target_reduction,
        weight=-1.0,
        params={"action_name": "joint_pos"},
    )
    parallel_stress = RewTerm(
        func=mdp.parallel_motor_stress,
        weight=-1.0,
        params={"action_name": "joint_pos", "margin": 0.04, "velocity_ratio": 0.85},
    )
    unsafe_termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=-100.0,
        params={"term_keys": FAILURE_TERMS},
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
class BeyondMimicTerminationsCfg:
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
    anchor_pos = DoneTerm(
        func=bm_mdp.bad_anchor_pos_z_only,
        params={"command_name": "motion", "threshold": 0.6},
    )
    anchor_ori = DoneTerm(
        func=bm_mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 1.8},
    )
    end_effector_pos = DoneTerm(
        func=bm_mdp.bad_motion_body_pos_z_only,
        params={
            "command_name": "motion",
            "threshold": 0.8,
            "body_names": ["left_hand_link", "right_hand_link", "left_foot_link", "right_foot_link"],
        },
    )
    # Hardware targets remain clamped/projected by the action term. Training
    # uses continuous penalties here because impacts can push the serial
    # simulation briefly beyond a target even for a valid reference motion.
    joint_limit = None
    parallel_ankle = None


@configclass
class K1FallRecoveryBeyondMimicEnvCfg(ManagerBasedRLEnvCfg):
    scene: RecoverySceneCfg = RecoverySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: BeyondMimicObservationsCfg = BeyondMimicObservationsCfg()
    actions: BeyondMimicActionsCfg = BeyondMimicActionsCfg()
    commands: BeyondMimicCommandsCfg = BeyondMimicCommandsCfg()
    rewards: BeyondMimicRewardsCfg = BeyondMimicRewardsCfg()
    terminations: BeyondMimicTerminationsCfg = BeyondMimicTerminationsCfg()
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
        self.events.material = None
        self.events.body_mass = None
        self.events.trunk_com = None
        self.events.actuator_gains = None
        self.events.push = None


@configclass
class K1FallRecoveryBeyondMimicFaceUpEnvCfg(K1FallRecoveryBeyondMimicEnvCfg):
    pass


@configclass
class K1FallRecoveryBeyondMimicFaceUpCanonicalEnvCfg(K1FallRecoveryBeyondMimicFaceUpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.static_start_probability = 1.0
        self.commands.motion.pose_range = {}
        self.commands.motion.velocity_range = {}
        self.observations.policy.enable_corruption = False


@configclass
class K1FallRecoveryBeyondMimicFaceDownEnvCfg(K1FallRecoveryBeyondMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = str(MOTION_DIR / "k1_getup_facedown_50hz.npz")

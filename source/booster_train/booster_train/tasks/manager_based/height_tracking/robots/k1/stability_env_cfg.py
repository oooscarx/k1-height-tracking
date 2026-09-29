"""Nominal no-lift refinement for accurate, quiet K1 height transitions."""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from booster_train.tasks.manager_based.height_tracking import mdp
from booster_train.tasks.manager_based.fall_recovery.wbc_stand_up import DoneTermCfg as WbcDoneTerm

from .env_cfg import JOINT_NAMES, K1HeightTrackingEnvCfg


def _set_high_posture_scaffold(
    action_cfg,
    scale: float,
    *,
    preserve_low_center: bool = False,
    posture_range: tuple[float, float] = (0.68, 0.72),
) -> None:
    high_center = list(action_cfg.position_center)
    for hip_index, knee_index, ankle_index in ((10, 13, 14), (16, 19, 20)):
        high_center[hip_index] += 0.08 * scale
        high_center[knee_index] -= 0.15 * scale
        high_center[ankle_index] += 0.06 * scale
    if not preserve_low_center:
        action_cfg.height_posture_center = None
    action_cfg.height_posture_high_center = high_center
    action_cfg.height_posture_range = posture_range
    action_cfg.height_posture_exponent = 1.0
    action_cfg.height_posture_blend = 1.0


@configclass
class K1HeightTrackingStabilityEnvCfg(K1HeightTrackingEnvCfg):
    """Refine positive-height tracking before terrain and pull curricula resume."""

    def __post_init__(self) -> None:
        super().__post_init__()

        command = self.commands.height
        command.ranges.height = (0.54, 0.72)
        command.velocity_range = (0.04, 0.08)
        command.resampling_time_range = (10.0, 14.0)
        command.standing_ratio = 0.15
        command.flat_ratio = 0.15
        command.focus_ratio = 0.25
        command.focus_height_range = (0.64, 0.72)
        command.high_height_threshold = 0.675
        command.success_error_threshold = 0.03
        command.governed_commands = False
        command.initialize_from_measured_height = True
        command.settle_time_s = 1.0

        # A fixed action center cannot reach 0.54 m upright: the wrapper-clipped
        # knee residual tops out near 1.1 rad, while K1 needs roughly 1.9 rad.
        # Introduce the kinematic squat center gradually so checkpoint behavior
        # stays continuous while the policy learns residual balance control.
        low_posture = list(self.actions.joint_pos.position_center)
        for hip_index, knee_index, ankle_index in ((10, 13, 14), (16, 19, 20)):
            low_posture[hip_index] = -1.135
            low_posture[knee_index] = 1.92
            low_posture[ankle_index] = -0.785
        self.actions.joint_pos.height_posture_center = low_posture
        self.actions.joint_pos.height_posture_range = (0.54, 0.72)
        self.actions.joint_pos.height_posture_exponent = 0.6
        self.actions.joint_pos.height_posture_blend = 0.40
        self.actions.joint_pos.position_target_velocity_limit = [
            1.0,
            1.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            2.0,
            *([10.0] * 12),
        ]
        for actuator_name in ("head", "arms"):
            actuator = self.scene.robot.actuators[actuator_name]
            actuator.armature = 0.01

        self.events.reset_base.params["standing_ratio"] = 1.0
        self.events.reset_base.params["random_fallen_ratio"] = 0.0
        self.observations.policy.enable_corruption = False
        for event_name in (
            "push_robot",
            "apply_external_force_torque",
            "apply_external_force_torque_extremities",
            "randomize_physics_material",
            "randomize_actuator_gains",
            "randomize_joint_friction",
            "randomize_joint_armature",
            "randomize_bodies_mass",
            "randomize_base_mass",
            "randomize_bodies_com",
            "randomize_base_com",
        ):
            setattr(self.events, event_name, None)

        # Terrain level zero still contains boxes, random roughness, and waves.
        # Use a true plane while learning the nominal height-control manifold;
        # robustness is added only after unaided flat-ground tracking converges.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.adaptive_lift.params["initial_force_scale"] = 0.0
        self.curriculum.adaptive_lift.params.pop("governed_threshold", None)
        self.curriculum.terrain_levels = None
        self.curriculum.polish_action_rate_rate = None
        self.curriculum.polish_ground_slam = None
        self.curriculum.polish_torso_slam = None
        self.curriculum.polish_not_moving = None
        self.curriculum.polish_joint_vel_l2 = None

        # Keep centimeter-scale height feedback, but do not let it dominate
        # quiet contact.  The earlier 16/24/-40 combination reduced the height
        # error by making the policy walk and pump its arms in place.
        self.rewards.not_moving = None
        self.rewards.base_height_tight = RewTerm(
            func=mdp.track_height_command_exp,
            weight=8.0,
            params={"command_name": "height", "std": 0.04},
        )
        self.rewards.base_height_l1 = RewTerm(
            func=mdp.track_height_command_l1,
            weight=-10.0,
            params={"command_name": "height"},
        )
        self.rewards.base_height_stationary_tight = RewTerm(
            func=mdp.track_height_command_exp,
            weight=8.0,
            params={"command_name": "height", "std": 0.03, "stationary_gate": True},
        )
        self.rewards.stationary_base_motion = RewTerm(
            func=mdp.positive_height_base_motion_l2,
            weight=-40.0,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "height",
                "linear_weight": 1.0,
                "angular_weight": 1.0,
            },
        )
        self.rewards.upper_body_joint_velocity = RewTerm(
            func=mdp.joint_vel_l2,
            weight=-0.20,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=JOINT_NAMES[:10],
                    preserve_order=True,
                )
            },
        )
        self.rewards.joint_deviation_l1.func = mdp.joint_deviation_from_action_center
        self.rewards.joint_deviation_l1.params = {
            "action_name": "joint_pos",
            "mode": "l1",
        }
        self.rewards.torso_upright.weight = -5.0
        self.rewards.torso_roll.weight = -15.0
        self.rewards.forward_pitch.weight = -30.0
        self.rewards.joint_deviation_l1_arms.weight = -2.0
        self.rewards.feet_slide.weight = -5.0
        self.rewards.body_velocity.weight = -0.5
        self.rewards.action_rate.weight = -4.0
        self.rewards.action_rate_rate.weight = -2.0
        self.rewards.joint_vel_l2.weight = -3.0e-2


@configclass
class K1HeightTrackingStabilityHighEnvCfg(K1HeightTrackingStabilityEnvCfg):
    """First refinement stage: close the high-pose error without losing stillness."""

    def __post_init__(self) -> None:
        super().__post_init__()
        command = self.commands.height
        command.ranges.height = (0.68, 0.72)
        command.focus_height_range = (0.68, 0.72)
        # Most samples are the two endpoints.  A continuous-only distribution
        # let the resumed policy improve reward by learning one compromise pose
        # while continuing to ignore the height command.
        command.standing_ratio = 0.45
        command.flat_ratio = 0.45
        command.focus_ratio = 0.0
        # Preserve the source action coordinates exactly at 0.68 m, then move
        # the nominal sagittal leg pose toward extension as the command rises.
        # The policy continues to provide balance residuals; this only removes
        # the saturated-knee dead zone from the high-height command response.
        _set_high_posture_scaffold(self.actions.joint_pos, scale=0.50)


@configclass
class K1HeightTrackingStabilityTopEnvCfg(K1HeightTrackingStabilityHighEnvCfg):
    """Master the maximum unaided height before mixing lower commands back in."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.height.standing_ratio = 1.0
        self.commands.height.flat_ratio = 0.0
        self.commands.height.focus_ratio = 0.0
        _set_high_posture_scaffold(self.actions.joint_pos, scale=0.50)


@configclass
class K1HeightTrackingStabilityMidEnvCfg(K1HeightTrackingStabilityEnvCfg):
    """First downward expansion after the high-pose acceptance gate passes."""

    def __post_init__(self) -> None:
        super().__post_init__()
        command = self.commands.height
        command.ranges.height = (0.60, 0.72)
        command.focus_height_range = (0.60, 0.72)
        command.standing_ratio = 0.45
        command.flat_ratio = 0.45
        command.focus_ratio = 0.0
        _set_high_posture_scaffold(
            self.actions.joint_pos,
            scale=0.50,
            preserve_low_center=True,
            posture_range=(0.60, 0.72),
        )
        self.actions.joint_pos.height_posture_exponent = 0.6
        # The robust source checkpoint was trained in this action coordinate
        # system.  Advance it only after a staged candidate passes evaluation.
        self.actions.joint_pos.height_posture_blend = 0.16


@configclass
class K1HeightTrackingStabilityLowEnvCfg(K1HeightTrackingStabilityMidEnvCfg):
    """Master the low endpoint before restoring the mixed height range."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.height.standing_ratio = 0.0
        self.commands.height.flat_ratio = 1.0
        self.commands.height.focus_ratio = 0.0
        # Preserve the 1.05 all-leg envelope learned by the source checkpoint,
        # then expand only sagittal hip-knee-ankle authority in small stages.
        self.actions.joint_pos.height_residual_base_maximum_scale = 1.05
        self.actions.joint_pos.height_residual_maximum_scale = 1.05


@configclass
class K1HeightTrackingStabilityLowBlend17EnvCfg(K1HeightTrackingStabilityLowEnvCfg):
    """Adapt the delayed closed loop to the first low-posture scaffold step."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos.height_posture_blend = 0.17


@configclass
class K1HeightTrackingStabilityLowBlend17RateLimitedEnvCfg(
    K1HeightTrackingStabilityLowBlend17EnvCfg
):
    """Reject delayed-policy target spikes while preserving nominal motion."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos.position_target_velocity_limit = [
            1.0,
            1.0,
            *([2.0] * 8),
            *([1.0] * 12),
        ]
        self.actions.joint_pos.position_target_velocity_limit_initialize_from_target = True
        self.actions.joint_pos.position_target_velocity_limit_warmup_steps = 50


@configclass
class K1HeightTrackingStabilityLowBlend17Damping075EnvCfg(
    K1HeightTrackingStabilityLowBlend17EnvCfg
):
    """Reduce delayed derivative feedback at the lower posture."""

    def __post_init__(self) -> None:
        super().__post_init__()
        for actuator_name in ("legs", "feet"):
            actuator = self.scene.robot.actuators[actuator_name]
            actuator.damping = {
                pattern: value * 0.75
                for pattern, value in actuator.damping.items()
            }


@configclass
class K1HeightTrackingStabilityLowBlend17CorrelatedDelayEnvCfg(
    K1HeightTrackingStabilityLowBlend17EnvCfg
):
    """Randomize one communication delay per robot instead of per body group."""

    def __post_init__(self) -> None:
        super().__post_init__()
        for actuator_name, actuator in self.scene.robot.actuators.items():
            actuator.min_delay = 2
            actuator.max_delay = 7
            actuator.synchronized_delay_group = "k1_height_stability"
            actuator.synchronized_delay_master = actuator_name == "legs"


@configclass
class K1HeightTrackingStabilityLowBlend21CorrelatedDelayEnvCfg(
    K1HeightTrackingStabilityLowBlend17CorrelatedDelayEnvCfg
):
    """Final low-posture scaffold under robot-level delay randomization."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos.height_posture_blend = 0.21


@configclass
class K1HeightTrackingStabilityLowBlend21Residual100EnvCfg(
    K1HeightTrackingStabilityLowBlend21CorrelatedDelayEnvCfg
):
    """Keep the new posture scaffold without amplifying delayed residuals."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos.height_residual_base_maximum_scale = 1.0
        self.actions.joint_pos.height_residual_maximum_scale = 1.0


@configclass
class K1HeightTrackingStabilityLowBlend22Residual100EnvCfg(
    K1HeightTrackingStabilityLowBlend21Residual100EnvCfg
):
    """Probe the feed-forward stability boundary without changing residual gain."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos.height_posture_blend = 0.22


@configclass
class K1HeightTrackingStabilityLowBlend23Residual100EnvCfg(
    K1HeightTrackingStabilityLowBlend21Residual100EnvCfg
):
    """Close the remaining low-height bias with feed-forward posture only."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos.height_posture_blend = 0.23


@configclass
class K1HeightTrackingStabilityLowBlend21FailureResetEnvCfg(
    K1HeightTrackingStabilityLowBlend21CorrelatedDelayEnvCfg
):
    """Give PPO a short, explicit signal when the low-posture loop diverges."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.terminations.bad_orientation = WbcDoneTerm(
            func=mdp.bad_orientation,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "limit_angle": 0.75,
            },
            termination_type="bad",
            sigma=5.0,
        )
        self.terminations.base_too_low = WbcDoneTerm(
            func=mdp.root_height_below_minimum,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "minimum_height": 0.32,
            },
            termination_type="bad",
            sigma=5.0,
        )
        self.rewards.stability_failure = RewTerm(
            func=mdp.is_terminated_term,
            weight=-2500.0,
            params={"term_keys": ["bad_orientation", "base_too_low"]},
        )


@configclass
class K1HeightTrackingStabilityLowDelay8EnvCfg(K1HeightTrackingStabilityLowEnvCfg):
    """Harden the low posture against the worst configured leg-command delay."""

    def __post_init__(self) -> None:
        super().__post_init__()
        legs = self.scene.robot.actuators["legs"]
        legs.min_delay = 8
        legs.max_delay = 8


@configclass
class K1HeightTrackingStabilityLowDelay5EnvCfg(K1HeightTrackingStabilityLowEnvCfg):
    """Master nominal deployment dynamics before worst-delay hardening."""

    def __post_init__(self) -> None:
        super().__post_init__()
        for actuator in self.scene.robot.actuators.values():
            actuator.min_delay = 5
            actuator.max_delay = 5


@configclass
class K1HeightTrackingStabilityLowTightDelay5EnvCfg(
    K1HeightTrackingStabilityLowDelay5EnvCfg
):
    """Learn the first full-range low posture with centimeter-scale feedback."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.height.ranges.height = (0.59, 0.72)
        self.actions.joint_pos.height_posture_blend = 0.40
        self.actions.joint_pos.height_posture_range = (0.54, 0.72)
        self.rewards.base_height_tight.weight = 32.0
        self.rewards.base_height_tight.params["std"] = 0.03
        self.rewards.base_height_stationary_tight.weight = 32.0
        self.rewards.base_height_stationary_tight.params["std"] = 0.02
        self.rewards.base_height_l1.weight = -30.0
        self.terminations.bad_orientation = WbcDoneTerm(
            func=mdp.bad_orientation,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "limit_angle": 0.75,
            },
            termination_type="bad",
            sigma=5.0,
        )
        self.terminations.base_too_low = WbcDoneTerm(
            func=mdp.root_height_below_minimum,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "minimum_height": 0.32,
            },
            termination_type="bad",
            sigma=5.0,
        )
        self.rewards.stability_failure = RewTerm(
            func=mdp.is_terminated_term,
            weight=-250.0,
            params={"term_keys": ["bad_orientation", "base_too_low"]},
        )


def _set_knee_delta_upper_bound(action_cfg, upper_bound: float) -> None:
    """Increase only knee flexion authority while preserving every other action bound."""
    action_cfg.clip = {
        joint_name: (-1.0, upper_bound if "Knee_Pitch" in joint_name else 1.0)
        for joint_name in JOINT_NAMES
    }


@configclass
class K1HeightTrackingStabilityLowTightDelay5Knee102EnvCfg(
    K1HeightTrackingStabilityLowTightDelay5EnvCfg
):
    """Probe two additional degrees of knee target authority at the low endpoint."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _set_knee_delta_upper_bound(self.actions.joint_pos, 1.02)


@configclass
class K1HeightTrackingStabilityLowTightDelay5Knee104EnvCfg(
    K1HeightTrackingStabilityLowTightDelay5EnvCfg
):
    """Probe four additional degrees of knee target authority at the low endpoint."""

    def __post_init__(self) -> None:
        super().__post_init__()
        _set_knee_delta_upper_bound(self.actions.joint_pos, 1.04)


@configclass
class K1HeightTrackingStabilityLowTightDelay5Residual108EnvCfg(
    K1HeightTrackingStabilityLowTightDelay5EnvCfg
):
    """Expand low-height sagittal residual authority by three percent."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos.height_residual_maximum_scale = 1.08


@configclass
class K1HeightTrackingStabilityLowTightDelay5Residual110EnvCfg(
    K1HeightTrackingStabilityLowTightDelay5EnvCfg
):
    """Expand low-height sagittal residual authority by five percent."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos.height_residual_maximum_scale = 1.10


@configclass
class K1HeightTrackingStabilityLowTightDelay5Residual112EnvCfg(
    K1HeightTrackingStabilityLowTightDelay5EnvCfg
):
    """Probe the next low-height sagittal residual step."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos.height_residual_maximum_scale = 1.12


@configclass
class K1HeightTrackingStabilityLowTightDelay5Residual114EnvCfg(
    K1HeightTrackingStabilityLowTightDelay5EnvCfg
):
    """Probe the predicted three-centimeter low-height tracking boundary."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos.height_residual_maximum_scale = 1.14


@configclass
class K1HeightTrackingStabilityLowTightDelay5Residual110KneeScale2EnvCfg(
    K1HeightTrackingStabilityLowTightDelay5Residual110EnvCfg
):
    """Use a wider knee action coordinate without changing physical joint limits."""

    def __post_init__(self) -> None:
        super().__post_init__()
        position_scale = list(self.actions.joint_pos.position_scale)
        for knee_index in (13, 19):
            position_scale[knee_index] *= 2.0
        self.actions.joint_pos.position_scale = position_scale
        self.actions.joint_pos.allow_position_scale_beyond_static_center = True


@configclass
class K1HeightTrackingStabilityLowTightDelay5Residual110Blend41EnvCfg(
    K1HeightTrackingStabilityLowTightDelay5Residual110EnvCfg
):
    """Advance the coordinated low-posture scaffold by one small step."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos.height_posture_blend = 0.41


@configclass
class K1HeightTrackingStabilityLowTightDelay5Residual110Blend42EnvCfg(
    K1HeightTrackingStabilityLowTightDelay5Residual110EnvCfg
):
    """Probe the next coordinated low-posture scaffold step."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos.height_posture_blend = 0.42


@configclass
class K1HeightTrackingStabilityLowTightDelay5Feedforward445EnvCfg(
    K1HeightTrackingStabilityLowTightDelay5EnvCfg
):
    """Move the low target from delayed residual feedback into the posture scaffold."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos.height_posture_blend = 0.445


@configclass
class K1HeightTrackingStabilityLowTightDelay5Feedforward460EnvCfg(
    K1HeightTrackingStabilityLowTightDelay5EnvCfg
):
    """Probe a lower feed-forward posture with the original residual gain."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos.height_posture_blend = 0.46


@configclass
class K1HeightTrackingStabilityLowTightDelay5Feedforward490Residual100EnvCfg(
    K1HeightTrackingStabilityLowTightDelay5EnvCfg
):
    """Preserve the low target while removing another delayed residual step."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos.height_posture_blend = 0.49
        self.actions.joint_pos.height_residual_base_maximum_scale = 1.0
        self.actions.joint_pos.height_residual_maximum_scale = 1.0


@configclass
class K1HeightTrackingStabilityLowTightDelay5Feedforward505Residual100EnvCfg(
    K1HeightTrackingStabilityLowTightDelay5EnvCfg
):
    """Probe a lower posture with reduced delayed residual feedback."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos.height_posture_blend = 0.505
        self.actions.joint_pos.height_residual_base_maximum_scale = 1.0
        self.actions.joint_pos.height_residual_maximum_scale = 1.0


@configclass
class K1HeightTrackingStabilityLowTightDelay5Feedforward490Sagittal100EnvCfg(
    K1HeightTrackingStabilityLowTightDelay5EnvCfg
):
    """Reduce delayed sagittal feedback while retaining lateral balance authority."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos.height_posture_blend = 0.49
        self.actions.joint_pos.height_residual_base_maximum_scale = 1.05
        self.actions.joint_pos.height_residual_maximum_scale = 1.0


@configclass
class K1HeightTrackingStabilityLowTightDelay5Feedforward505Sagittal100EnvCfg(
    K1HeightTrackingStabilityLowTightDelay5EnvCfg
):
    """Probe a lower feed-forward posture without weakening lateral feedback."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos.height_posture_blend = 0.505
        self.actions.joint_pos.height_residual_base_maximum_scale = 1.05
        self.actions.joint_pos.height_residual_maximum_scale = 1.0


@configclass
class K1HeightTrackingStabilityLowTightDelay5Feedforward495Sagittal100EnvCfg(
    K1HeightTrackingStabilityLowTightDelay5EnvCfg
):
    """First guarded low-posture stage beyond the stable feed-forward anchor."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos.height_posture_blend = 0.495
        self.actions.joint_pos.height_residual_base_maximum_scale = 1.05
        self.actions.joint_pos.height_residual_maximum_scale = 1.0
        self.rewards.stability_failure.weight = -2500.0


@configclass
class K1HeightTrackingStabilityLowTightDelay5Feedforward500Sagittal100EnvCfg(
    K1HeightTrackingStabilityLowTightDelay5EnvCfg
):
    """Second guarded low-posture stage beyond the stable feed-forward anchor."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.actions.joint_pos.height_posture_blend = 0.50
        self.actions.joint_pos.height_residual_base_maximum_scale = 1.05
        self.actions.joint_pos.height_residual_maximum_scale = 1.0
        self.rewards.stability_failure.weight = -2500.0


@configclass
class K1HeightTrackingStabilityOpenLoopFullRangeEnvCfg(
    K1HeightTrackingStabilityLowTightDelay5EnvCfg
):
    """Diagnose the K1 squat manifold with feed-forward sagittal targets only."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.height.ranges.height = (0.54, 0.72)
        self.actions.joint_pos.height_posture_blend = 1.0
        self.actions.joint_pos.height_posture_exponent = 1.4


@configclass
class K1HeightTrackingStabilityHandoff59EnvCfg(
    K1HeightTrackingStabilityLowTightDelay5EnvCfg
):
    """Adapt the learned controller to the low-height feed-forward handoff."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.height.standing_ratio = 0.45
        self.commands.height.flat_ratio = 0.45
        self.commands.height.focus_ratio = 0.0
        self.commands.height.resampling_time_range = (5.0, 8.0)
        # This pair is fitted to the stable checkpoint's sagittal targets at
        # 0.59 and 0.68 m.  Below 0.59 m the geometric center takes over from
        # delayed policy feedback, while the lateral feedback map is unchanged.
        self.actions.joint_pos.height_posture_blend = 1.0
        self.actions.joint_pos.height_posture_exponent = 0.29623
        self.actions.joint_pos.height_residual_base_maximum_scale = 1.05
        self.actions.joint_pos.height_residual_maximum_scale = 1.0
        self.actions.joint_pos.height_residual_handoff_range = (0.59, 0.66)
        self.actions.joint_pos.height_residual_handoff_minimum_scale = 0.91108
        self.rewards.stability_failure.weight = -2500.0


@configclass
class K1HeightTrackingStabilityHandoff59Gain085EnvCfg(
    K1HeightTrackingStabilityHandoff59EnvCfg
):
    """Lower low-pose feedback gain while preserving safe learned targets."""

    def __post_init__(self) -> None:
        super().__post_init__()
        action = self.actions.joint_pos
        low_center = list(action.height_posture_center)
        high_center = list(action.height_posture_high_center)
        # Symmetric hip and knee corrections are fitted from the stable 303000
        # policy.  The ankle center stays untouched because it already sits at
        # the hardware-safe low limit.
        for left, right, low_delta, high_delta in (
            (10, 16, -0.102882482, 0.007951592),
            (13, 19, 0.1815, -0.014082406),
        ):
            low_center[left] += low_delta
            low_center[right] += low_delta
            high_center[left] += high_delta
            high_center[right] += high_delta
        action.height_posture_center = low_center
        action.height_posture_high_center = high_center
        action.height_residual_handoff_minimum_scale = 0.85


@configclass
class K1HeightTrackingStabilityHandoff59Gain085OriginEnvCfg(
    K1HeightTrackingStabilityHandoff59Gain085EnvCfg
):
    """Stable single-origin control contract for flat-ground refinement."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.env_spacing = 0.0
        self.actions.joint_pos.height_posture_exponent = 0.316
        for actuator_name in ("legs", "feet"):
            actuator = self.scene.robot.actuators[actuator_name]
            if isinstance(actuator.damping, dict):
                actuator.damping = {
                    name: 2.0 * value
                    for name, value in actuator.damping.items()
                }
            else:
                actuator.damping *= 2.0


@configclass
class K1HeightTrackingStabilityFullRangeOriginEnvCfg(
    K1HeightTrackingStabilityHandoff59Gain085OriginEnvCfg
):
    """Extend the accepted high controller with a low-only squat segment."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.height.ranges.height = (0.54, 0.72)
        action = self.actions.joint_pos
        # Calibrate the geometric center against the measured tracked-point
        # height.  This keeps the safe low-pose shape while removing the
        # 3--4 cm mid-range height bias of the two-exponent handoff.
        action.height_posture_phase_knots = (
            (0.54, 0.000),
            (0.55, 0.060),
            (0.56, 0.169),
            (0.57, 0.310),
            (0.58, 0.460),
            (0.585, 0.530),
            (0.59, 0.610),
            (0.60, 0.655),
            (0.62, 0.735),
            (0.66, 0.870),
            (0.72, 1.000),
        )
        action.height_posture_low_handoff_height = None
        action.height_residual_deep_handoff_range = (0.56, 0.60)
        action.height_residual_deep_handoff_minimum_scale = 0.20

        # A 0.54 m tracking command corresponds to roughly 0.34 m root height
        # on K1. The inherited 0.32 m threshold leaves only 2 cm of valid
        # tracking margin and resets upright low poses, so retain a 10 cm fall
        # margin below the commanded root height instead.
        self.terminations.base_too_low.params["minimum_height"] = 0.24


@configclass
class K1HeightTrackingStabilityLow57OriginEnvCfg(
    K1HeightTrackingStabilityFullRangeOriginEnvCfg
):
    """Refine the critical 0.57--0.585 m down/up transition band."""

    def __post_init__(self) -> None:
        super().__post_init__()
        command = self.commands.height
        command.ranges.height = (0.57, 0.585)
        command.standing_ratio = 0.15
        command.flat_ratio = 0.15
        command.focus_ratio = 1.0
        command.focus_height_range = (0.57, 0.585)
        command.local_target_ratio = 1.00
        command.local_target_delta_range = (-0.02, 0.02)
        command.velocity_range = (0.08, 0.08)
        command.resampling_time_range = (3.0, 4.0)


@configclass
class K1HeightTrackingStabilityTransitionOriginEnvCfg(
    K1HeightTrackingStabilityFullRangeOriginEnvCfg
):
    """Learn the low posture handoff without changing the accepted high range."""

    def __post_init__(self) -> None:
        super().__post_init__()
        command = self.commands.height
        command.ranges.height = (0.54, 0.72)
        command.standing_ratio = 0.20
        command.flat_ratio = 0.20
        command.focus_ratio = 0.75
        command.focus_height_range = (0.575, 0.60)
        command.local_target_ratio = 0.50
        command.local_target_delta_range = (-0.035, 0.035)
        command.velocity_range = (0.08, 0.08)
        command.resampling_time_range = (4.0, 7.0)


@configclass
class K1HeightTrackingStabilityFullRangeHandoffEnvCfg(
    K1HeightTrackingStabilityHandoff59EnvCfg
):
    """Open the full command range after the handoff endpoint is stable."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.commands.height.ranges.height = (0.54, 0.72)

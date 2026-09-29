"""PPO continuation settings for nominal K1 height stability."""

from isaaclab.utils import configclass

from booster_train.tasks.manager_based.fall_recovery.wbc_stand_up import RslRlActorMeanBoundCfg

from .ppo_cfg import K1HeightTrackingPpoRunnerCfg


@configclass
class K1HeightTrackingStabilityPpoRunnerCfg(K1HeightTrackingPpoRunnerCfg):
    run_name = "height_stability_high_balanced"
    save_interval = 25
    freeze_resumed_action_std: bool = True
    reset_action_std_optimizer_state_on_resume: bool = False
    minimum_action_std: float | None = None
    load_optimizer_on_resume: bool = False
    restore_wbc_curriculum_on_resume: bool = False

    def __post_init__(self) -> None:
        super().__post_init__()
        self.algorithm.learning_rate = 1.0e-6
        self.algorithm.max_learning_rate = 1.0e-6
        self.algorithm.clip_param = 0.05
        self.algorithm.num_learning_epochs = 2
        self.algorithm.desired_kl = 2.0e-4
        self.algorithm.actor_mean_bound_cfg = RslRlActorMeanBoundCfg(
            action_indices=(13, 19),
            soft_limit=3.75,
            loss_coefficient=1.0,
        )


@configclass
class K1HeightTrackingStabilityLow57PpoRunnerCfg(
    K1HeightTrackingStabilityPpoRunnerCfg
):
    run_name = "height_stability_low57_origin"
    save_interval = 10

    def __post_init__(self) -> None:
        super().__post_init__()
        # The source checkpoint has a 0.005 action standard deviation, so tiny
        # unconstrained mean changes can cross the narrow low-pose stability
        # boundary. Preserve the deployment-like exploration and constrain all
        # actor losses against the policy that collected the current rollout.
        self.freeze_resumed_action_std = True
        self.freeze_action_std = False
        self.algorithm.learning_rate = 2.0e-6
        self.algorithm.max_learning_rate = 2.0e-6
        self.algorithm.schedule = "fixed"
        self.algorithm.clip_param = 0.05
        self.algorithm.num_learning_epochs = 1
        self.algorithm.desired_kl = 2.0e-5
        self.algorithm.policy_kl_coefficient = 0.0
        self.algorithm.policy_kl_max = 1.0e-3


@configclass
class K1HeightTrackingStabilityTransitionPpoRunnerCfg(
    K1HeightTrackingStabilityLow57PpoRunnerCfg
):
    """Controlled exploration for learning the 0.59 m low-posture handoff."""

    run_name = "height_stability_transition_origin"

    def __post_init__(self) -> None:
        super().__post_init__()
        self.freeze_resumed_action_std = False
        self.freeze_action_std = True
        self.action_std = [
            0.005,
            0.005,
            *([0.010] * 8),
            0.030,
            0.020,
            0.015,
            0.030,
            0.020,
            0.020,
            0.030,
            0.020,
            0.015,
            0.030,
            0.020,
            0.020,
        ]
        self.algorithm.learning_rate = 1.0e-5
        self.algorithm.max_learning_rate = 1.0e-5
        self.algorithm.num_learning_epochs = 2
        self.algorithm.desired_kl = 5.0e-4
        self.algorithm.policy_kl_max = 1.0e-2

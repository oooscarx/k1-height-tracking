from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlSymmetryCfg

from booster_train.tasks.manager_based.fall_recovery.hardware_config import K1_HARDWARE_CONFIG
from booster_train.tasks.manager_based.fall_recovery.wbc_stand_up import (
    FallenStateDatasetCfg,
    RslRlL2C2Cfg,
    RslRlRewardNormalizationCfg,
    WbcRslRlPpoAlgorithmCfg,
)
from booster_train.tasks.manager_based.height_tracking.symmetry import lr_mirror_k1_height

CONFIG = K1_HARDWARE_CONFIG
ROBUST = CONFIG["native_teacher"]["robust_training"]


@configclass
class K1HeightTrackingPpoRunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 100_000
    save_interval = 250
    experiment_name = "height_tracking_k1"
    run_name = "height_tracking_k1"
    wandb_project = "HeightTracking-K1"
    empirical_normalization = False
    clip_actions = 4.0
    freeze_action_std: bool = False
    freeze_resumed_action_std: bool = False
    reset_action_std_optimizer_state_on_resume: bool = True
    # A small non-zero floor keeps the low-noise hip and ankle policies from
    # making the adaptive KL scheduler collapse the optimizer step size. This
    # remains well below the exploration level that previously destabilized the
    # mature policy when entropy was reintroduced.
    minimum_action_std: float | None = 0.04
    action_std: list[float] = [
        0.10,
        0.10,
        0.20,
        0.20,
        0.20,
        0.20,
        0.20,
        0.20,
        0.20,
        0.20,
        0.30,
        0.30,
        0.30,
        0.35,
        0.20,
        0.20,
        0.30,
        0.30,
        0.30,
        0.35,
        0.20,
        0.20,
    ]

    fallen_state_dataset_cfg: FallenStateDatasetCfg | None = FallenStateDatasetCfg(
        num_spawns_per_level=2,
        fall_duration_s=1.0,
        spawn_height_offset=2.0,
        spawn_xy_range=3.0,
        initial_lin_vel_range=0.0,
        initial_ang_vel_range=0.0,
        spawn_orientation="on_back",
        spawn_pitch_range=(-1.7, -1.4),
        spawn_joint_mode="default",
        cache_enabled=True,
        cache_dir="height_tracking_states_cache",
        expected_joint_names=None,
        position_minimum=None,
        position_maximum=None,
        parallel_ankle=None,
        joint_position_margin=0.1,
        motor_position_margin=0.14,
        random_joint_velocity=0.0,
        joint_velocity_safety_horizon_s=0.1,
    )

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = WbcRslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        value_loss_huber_delta=10.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        # This run resumes a mature low-noise policy. Reintroducing WBC's
        # from-scratch entropy bonus here makes every joint's noise grow at
        # once and degrades height tracking before the lift curriculum adapts.
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        max_learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=RslRlSymmetryCfg(
            use_data_augmentation=True,
            use_mirror_loss=False,
            data_augmentation_func=lr_mirror_k1_height,
        ),
        l2c2_cfg=RslRlL2C2Cfg(
            lambda_actor=1.0,
            lambda_critic=0.1,
            max_actor_observation_delta=10.0,
            max_critic_observation_delta=50.0,
        ),
        reward_normalization_cfg=RslRlRewardNormalizationCfg(
            decay=0.999,
            epsilon=1.0e-2,
            return_scale_decay=0.999,
            outlier_threshold=10.0,
        ),
    )

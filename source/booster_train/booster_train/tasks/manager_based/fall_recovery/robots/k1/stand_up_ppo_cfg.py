from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlSymmetryCfg,
)

from booster_train.tasks.manager_based.fall_recovery.hardware_config import (
    K1_HARDWARE_CONFIG,
)
from booster_train.tasks.manager_based.fall_recovery.wbc_stand_up import (
    FallenStateDatasetCfg,
    RslRlL2C2Cfg,
    RslRlRewardNormalizationCfg,
    WbcRslRlPpoAlgorithmCfg,
    lr_mirror_k1,
)

CONFIG = K1_HARDWARE_CONFIG


@configclass
class K1WbcStandUpPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    seed = 42
    num_steps_per_env = 24
    max_iterations = 100_000
    save_interval = 250
    experiment_name = "k1_wbc_stand_up"
    run_name = "wbc_stand_up"
    wandb_project = "K1-WBC-Stand-Up"
    empirical_normalization = False
    clip_actions = None

    fallen_state_dataset_cfg: FallenStateDatasetCfg | None = FallenStateDatasetCfg(
        num_spawns_per_level=2,
        fall_duration_s=1.0,
        spawn_height_offset=2.0,
        spawn_xy_range=3.0,
        initial_lin_vel_range=1.0,
        initial_ang_vel_range=1.0,
        spawn_orientation="random",
        spawn_joint_mode="random",
        cache_enabled=True,
        cache_dir="fallen_states_cache",
        expected_joint_names=CONFIG["joint_names"],
        position_minimum=CONFIG["position_minimum"],
        position_maximum=CONFIG["position_maximum"],
        parallel_ankle=CONFIG["parallel_ankle"],
        joint_position_margin=0.1,
        motor_position_margin=0.14,
        random_joint_velocity=1.0,
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
        entropy_coef=0.0025,
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
            data_augmentation_func=lr_mirror_k1,
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
            return_scale_decay=None,
            outlier_threshold=3.0,
        ),
    )

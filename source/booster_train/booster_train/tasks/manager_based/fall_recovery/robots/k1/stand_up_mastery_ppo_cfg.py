from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlPpoActorCriticCfg,
    RslRlSymmetryCfg,
)

from booster_train.tasks.manager_based.fall_recovery.wbc_stand_up import (
    RslRlL2C2Cfg,
    RslRlRewardNormalizationCfg,
    WbcRslRlPpoAlgorithmCfg,
    lr_mirror_k1,
)

from .stand_up_ppo_cfg import K1WbcStandUpPPORunnerCfg


@configclass
class K1WbcStandUpMasteryPPORunnerCfg(K1WbcStandUpPPORunnerCfg):
    """Low-step-size PPO continuation that preserves the selected legacy policy."""

    max_iterations = 10_000
    save_interval = 250
    experiment_name = "k1_wbc_stand_up_mastery"
    run_name = "terrain0_mastery"

    load_optimizer_on_resume: bool = False
    reset_learning_iteration_on_resume: bool = True
    restore_wbc_curriculum_on_resume: bool = False
    freeze_action_std: bool = True
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

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.2,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = WbcRslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        value_loss_huber_delta=10.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.0,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-4,
        max_learning_rate=1.0e-4,
        schedule="adaptive",
        gamma=0.995,
        lam=0.95,
        desired_kl=0.005,
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

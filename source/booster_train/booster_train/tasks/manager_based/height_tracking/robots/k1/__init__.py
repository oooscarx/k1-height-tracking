import gymnasium as gym

gym.register(
    id="Booster-K1-Height-Tracking-v0",
    entry_point="booster_train.tasks.manager_based.fall_recovery.wbc_stand_up.env:WbcManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:K1HeightTrackingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.ppo_cfg:K1HeightTrackingPpoRunnerCfg",
        "pre_learn_entry_point": f"{__name__}.pre_learn:pre_learn",
        "wbc_height_tracking": True,
    },
)

gym.register(
    id="Booster-K1-Height-Tracking-Mastery-v0",
    entry_point="booster_train.tasks.manager_based.fall_recovery.wbc_stand_up.env:WbcManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.mastery_env_cfg:K1HeightTrackingMasteryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.mastery_ppo_cfg:K1HeightTrackingMasteryPpoRunnerCfg",
        "pre_learn_entry_point": f"{__name__}.pre_learn:pre_learn",
        "wbc_height_tracking": True,
    },
)

gym.register(
    id="Booster-K1-Height-Tracking-ErrorRecovery-v0",
    entry_point="booster_train.tasks.manager_based.fall_recovery.wbc_stand_up.env:WbcManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.error_recovery_env_cfg:K1HeightTrackingErrorRecoveryEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.error_recovery_ppo_cfg:K1HeightTrackingErrorRecoveryPpoRunnerCfg",
        "pre_learn_entry_point": f"{__name__}.pre_learn:pre_learn",
        "wbc_height_tracking": True,
    },
)

gym.register(
    id="Booster-K1-Height-Tracking-ProgressiveRobustness-v0",
    entry_point="booster_train.tasks.manager_based.fall_recovery.wbc_stand_up.env:WbcManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.progressive_robustness_env_cfg:"
            "K1HeightTrackingProgressiveRobustnessEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.progressive_robustness_ppo_cfg:"
            "K1HeightTrackingProgressiveRobustnessPpoRunnerCfg"
        ),
        "pre_learn_entry_point": f"{__name__}.pre_learn:pre_learn",
        "wbc_height_tracking": True,
    },
)

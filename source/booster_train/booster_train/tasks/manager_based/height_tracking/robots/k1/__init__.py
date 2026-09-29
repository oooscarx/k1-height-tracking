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

gym.register(
    id="Booster-K1-Height-Tracking-Stability-v0",
    entry_point="booster_train.tasks.manager_based.fall_recovery.wbc_stand_up.env:WbcManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.stability_env_cfg:K1HeightTrackingStabilityEnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.stability_ppo_cfg:K1HeightTrackingStabilityPpoRunnerCfg",
        "pre_learn_entry_point": f"{__name__}.pre_learn:pre_learn",
        "wbc_height_tracking": True,
    },
)

for stage, env_cfg in (
    ("Top", "K1HeightTrackingStabilityTopEnvCfg"),
    ("High", "K1HeightTrackingStabilityHighEnvCfg"),
    ("Mid", "K1HeightTrackingStabilityMidEnvCfg"),
    ("Low", "K1HeightTrackingStabilityLowEnvCfg"),
    ("LowBlend17", "K1HeightTrackingStabilityLowBlend17EnvCfg"),
    (
        "LowBlend17RateLimited",
        "K1HeightTrackingStabilityLowBlend17RateLimitedEnvCfg",
    ),
    (
        "LowBlend17Damping075",
        "K1HeightTrackingStabilityLowBlend17Damping075EnvCfg",
    ),
    (
        "LowBlend17CorrelatedDelay",
        "K1HeightTrackingStabilityLowBlend17CorrelatedDelayEnvCfg",
    ),
    (
        "LowBlend21CorrelatedDelay",
        "K1HeightTrackingStabilityLowBlend21CorrelatedDelayEnvCfg",
    ),
    (
        "LowBlend21Residual100",
        "K1HeightTrackingStabilityLowBlend21Residual100EnvCfg",
    ),
    (
        "LowBlend22Residual100",
        "K1HeightTrackingStabilityLowBlend22Residual100EnvCfg",
    ),
    (
        "LowBlend23Residual100",
        "K1HeightTrackingStabilityLowBlend23Residual100EnvCfg",
    ),
    (
        "LowBlend21FailureReset",
        "K1HeightTrackingStabilityLowBlend21FailureResetEnvCfg",
    ),
    ("LowDelay5", "K1HeightTrackingStabilityLowDelay5EnvCfg"),
    ("LowTightDelay5", "K1HeightTrackingStabilityLowTightDelay5EnvCfg"),
    (
        "LowTightDelay5Knee102",
        "K1HeightTrackingStabilityLowTightDelay5Knee102EnvCfg",
    ),
    (
        "LowTightDelay5Knee104",
        "K1HeightTrackingStabilityLowTightDelay5Knee104EnvCfg",
    ),
    (
        "LowTightDelay5Residual108",
        "K1HeightTrackingStabilityLowTightDelay5Residual108EnvCfg",
    ),
    (
        "LowTightDelay5Residual110",
        "K1HeightTrackingStabilityLowTightDelay5Residual110EnvCfg",
    ),
    (
        "LowTightDelay5Residual112",
        "K1HeightTrackingStabilityLowTightDelay5Residual112EnvCfg",
    ),
    (
        "LowTightDelay5Residual114",
        "K1HeightTrackingStabilityLowTightDelay5Residual114EnvCfg",
    ),
    (
        "LowTightDelay5Residual110KneeScale2",
        "K1HeightTrackingStabilityLowTightDelay5Residual110KneeScale2EnvCfg",
    ),
    (
        "LowTightDelay5Residual110Blend41",
        "K1HeightTrackingStabilityLowTightDelay5Residual110Blend41EnvCfg",
    ),
    (
        "LowTightDelay5Residual110Blend42",
        "K1HeightTrackingStabilityLowTightDelay5Residual110Blend42EnvCfg",
    ),
    (
        "LowTightDelay5Feedforward445",
        "K1HeightTrackingStabilityLowTightDelay5Feedforward445EnvCfg",
    ),
    (
        "LowTightDelay5Feedforward460",
        "K1HeightTrackingStabilityLowTightDelay5Feedforward460EnvCfg",
    ),
    (
        "LowTightDelay5Feedforward490Residual100",
        "K1HeightTrackingStabilityLowTightDelay5Feedforward490Residual100EnvCfg",
    ),
    (
        "LowTightDelay5Feedforward505Residual100",
        "K1HeightTrackingStabilityLowTightDelay5Feedforward505Residual100EnvCfg",
    ),
    (
        "LowTightDelay5Feedforward490Sagittal100",
        "K1HeightTrackingStabilityLowTightDelay5Feedforward490Sagittal100EnvCfg",
    ),
    (
        "LowTightDelay5Feedforward505Sagittal100",
        "K1HeightTrackingStabilityLowTightDelay5Feedforward505Sagittal100EnvCfg",
    ),
    (
        "LowTightDelay5Feedforward495Sagittal100",
        "K1HeightTrackingStabilityLowTightDelay5Feedforward495Sagittal100EnvCfg",
    ),
    (
        "LowTightDelay5Feedforward500Sagittal100",
        "K1HeightTrackingStabilityLowTightDelay5Feedforward500Sagittal100EnvCfg",
    ),
    ("OpenLoopFullRange", "K1HeightTrackingStabilityOpenLoopFullRangeEnvCfg"),
    ("Handoff59", "K1HeightTrackingStabilityHandoff59EnvCfg"),
    ("Handoff59Gain085", "K1HeightTrackingStabilityHandoff59Gain085EnvCfg"),
    (
        "Handoff59Gain085Origin",
        "K1HeightTrackingStabilityHandoff59Gain085OriginEnvCfg",
    ),
    (
        "FullRangeOrigin",
        "K1HeightTrackingStabilityFullRangeOriginEnvCfg",
    ),
    ("FullRangeHandoff", "K1HeightTrackingStabilityFullRangeHandoffEnvCfg"),
    ("LowDelay8", "K1HeightTrackingStabilityLowDelay8EnvCfg"),
):
    gym.register(
        id=f"Booster-K1-Height-Tracking-Stability-{stage}-v0",
        entry_point=(
            "booster_train.tasks.manager_based.fall_recovery.wbc_stand_up.env:"
            "WbcManagerBasedRLEnv"
        ),
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.stability_env_cfg:{env_cfg}",
            "rsl_rl_cfg_entry_point": (
                f"{__name__}.stability_ppo_cfg:K1HeightTrackingStabilityPpoRunnerCfg"
            ),
            "pre_learn_entry_point": f"{__name__}.pre_learn:pre_learn",
            "wbc_height_tracking": True,
        },
    )

gym.register(
    id="Booster-K1-Height-Tracking-Stability-Low57Origin-v0",
    entry_point=(
        "booster_train.tasks.manager_based.fall_recovery.wbc_stand_up.env:"
        "WbcManagerBasedRLEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.stability_env_cfg:"
            "K1HeightTrackingStabilityLow57OriginEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.stability_ppo_cfg:"
            "K1HeightTrackingStabilityLow57PpoRunnerCfg"
        ),
        "pre_learn_entry_point": f"{__name__}.pre_learn:pre_learn",
        "wbc_height_tracking": True,
    },
)

gym.register(
    id="Booster-K1-Height-Tracking-Stability-TransitionOrigin-v0",
    entry_point=(
        "booster_train.tasks.manager_based.fall_recovery.wbc_stand_up.env:"
        "WbcManagerBasedRLEnv"
    ),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.stability_env_cfg:"
            "K1HeightTrackingStabilityTransitionOriginEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{__name__}.stability_ppo_cfg:"
            "K1HeightTrackingStabilityTransitionPpoRunnerCfg"
        ),
        "pre_learn_entry_point": f"{__name__}.pre_learn:pre_learn",
        "wbc_height_tracking": True,
    },
)

"""PPO logging configuration for progressive K1 robustness training."""

from isaaclab.utils import configclass

from .error_recovery_ppo_cfg import K1HeightTrackingErrorRecoveryPpoRunnerCfg


@configclass
class K1HeightTrackingProgressiveRobustnessPpoRunnerCfg(K1HeightTrackingErrorRecoveryPpoRunnerCfg):
    run_name = "height_progressive_robustness"

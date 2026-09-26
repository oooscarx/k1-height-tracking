"""PPO logging configuration for K1 height-error recovery."""

from isaaclab.utils import configclass

from .ppo_cfg import K1HeightTrackingPpoRunnerCfg


@configclass
class K1HeightTrackingErrorRecoveryPpoRunnerCfg(K1HeightTrackingPpoRunnerCfg):
    run_name = "height_error_recovery"

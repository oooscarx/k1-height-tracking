"""PPO logging configuration for K1 high-height mastery."""

from isaaclab.utils import configclass

from .ppo_cfg import K1HeightTrackingPpoRunnerCfg


@configclass
class K1HeightTrackingMasteryPpoRunnerCfg(K1HeightTrackingPpoRunnerCfg):
    experiment_name = "height_tracking_k1_mastery"
    run_name = "height_tracking_k1_mastery"

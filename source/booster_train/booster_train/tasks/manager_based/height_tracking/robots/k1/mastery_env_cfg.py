"""High-height mastery phase for the K1 height-tracking policy."""

from isaaclab.utils import configclass

from ...k1_scaling import K1_TRACKING_ERROR_THRESHOLD
from .env_cfg import K1HeightTrackingEnvCfg


@configclass
class K1HeightTrackingMasteryEnvCfg(K1HeightTrackingEnvCfg):
    """Concentrate training on stable tracking near K1's maximum standing height."""

    def __post_init__(self) -> None:
        super().__post_init__()

        command = self.commands.height
        command.ranges.height = (0.60, 0.72)
        command.resampling_time_range = (4.0, 7.0)
        command.standing_ratio = 0.7
        command.flat_ratio = 0.0
        command.high_height_threshold = 0.60
        command.success_error_threshold = K1_TRACKING_ERROR_THRESHOLD

        # Keep recovery experience while spending most resets on height mastery.
        self.events.reset_base.params["standing_ratio"] = 0.75

        # Hold the inherited lift scale fixed until mastery is demonstrated.
        self.curriculum.adaptive_lift.params["threshold"] = -1.0

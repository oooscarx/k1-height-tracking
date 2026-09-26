"""Error-focused continuation phase for the mature K1 height policy."""

from isaaclab.utils import configclass

from .env_cfg import K1HeightTrackingEnvCfg


@configclass
class K1HeightTrackingErrorRecoveryEnvCfg(K1HeightTrackingEnvCfg):
    """Spend more rollout budget on the measured mid/high-height bottleneck."""

    def __post_init__(self) -> None:
        super().__post_init__()

        command = self.commands.height
        command.resampling_time_range = (4.0, 7.0)
        command.standing_ratio = 0.20
        command.flat_ratio = 0.10
        command.focus_ratio = 0.65
        command.focus_height_range = (0.30, 0.72)
        command.high_height_threshold = 0.54

        # Preserve recovery experience while increasing stable tracking updates.
        self.events.reset_base.params["standing_ratio"] = 0.75

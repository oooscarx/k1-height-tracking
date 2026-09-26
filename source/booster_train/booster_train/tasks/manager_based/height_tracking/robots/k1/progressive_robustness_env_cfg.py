"""Progress terrain and pulling disturbances after height tracking stabilizes."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass

from booster_train.tasks.manager_based.height_tracking import mdp

from .env_cfg import CurriculumCfg
from .error_recovery_env_cfg import K1HeightTrackingErrorRecoveryEnvCfg


@configclass
class ProgressiveRobustnessCurriculumCfg(CurriculumCfg):
    disturbance_scale = CurrTerm(
        func=mdp.adaptive_external_force_growth,
        params={
            "event_names": (
                "apply_external_force_torque",
                "apply_external_force_torque_extremities",
            ),
            "command_name": "height",
            "error_metric_name": "high_height_error",
            # Calibrated from four stable 250-iteration windows at terrain
            # level 3.87: jump ~= 0.177 m and governed ~= 0.134 m.
            "jump_error_threshold": 0.18,
            "governed_error_threshold": 0.14,
            "ema_alpha": 0.05,
            "minimum_terrain_level": 3.5,
            "initial_scale": 1.0,
            "maximum_scale": 1.5,
            # A full 1.0 -> 1.5 ramp takes 20k simulator steps, roughly
            # 830 PPO iterations at 24 steps per environment.
            "growth_per_step": 2.5e-5,
        },
    )


@configclass
class K1HeightTrackingProgressiveRobustnessEnvCfg(K1HeightTrackingErrorRecoveryEnvCfg):
    curriculum: ProgressiveRobustnessCurriculumCfg = ProgressiveRobustnessCurriculumCfg()

    def __post_init__(self) -> None:
        super().__post_init__()
        # The preceding run finished near mean terrain level 3.7. Sampling all
        # eight levels starts this continuation at a comparable mean while the
        # existing timeout curriculum keeps advancing successful environments.
        self.scene.terrain.max_init_terrain_level = 7

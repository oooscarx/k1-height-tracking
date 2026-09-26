from isaaclab.utils import configclass

from booster_train.tasks.manager_based.beyond_mimic.agents.rsl_rl_ppo_cfg import (
    BasePPORunnerCfg,
)
from booster_train.tasks.manager_based.fall_recovery.hardware_config import K1_HARDWARE_CONFIG

DISCOVERY = K1_HARDWARE_CONFIG["training"]["discovery"]


@configclass
class K1FallRecoveryPPORunnerCfg(BasePPORunnerCfg):
    num_steps_per_env = 32
    max_iterations = 20_000
    save_interval = 500
    experiment_name = "k1_fall_recovery"
    clip_actions = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.algorithm.entropy_coef = 0.01
        self.algorithm.learning_rate = 5.0e-4
        self.algorithm.desired_kl = 0.015


@configclass
class K1FallRecoveryDiscoveryPPORunnerCfg(K1FallRecoveryPPORunnerCfg):
    max_iterations = 50_000
    save_interval = 100
    freeze_action_std: bool = DISCOVERY["freeze_action_std"]
    action_std: list[float] = DISCOVERY["action_std"]

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.2
        self.policy.noise_std_type = "scalar"
        self.algorithm.entropy_coef = DISCOVERY["entropy_coef"]


@configclass
class K1FallRecoveryDiscoveryFaceUpPPORunnerCfg(K1FallRecoveryDiscoveryPPORunnerCfg):
    experiment_name = "k1_fall_recovery_discovery_faceup"


@configclass
class K1FallRecoveryDiscoveryFaceDownPPORunnerCfg(K1FallRecoveryDiscoveryPPORunnerCfg):
    experiment_name = "k1_fall_recovery_discovery_facedown"


@configclass
class K1FallRecoveryBeyondMimicPPORunnerCfg(BasePPORunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10_000
    save_interval = 100
    experiment_name = "k1_fall_recovery_beyond_mimic"
    empirical_normalization = True
    clip_actions = 1.0

    def __post_init__(self):
        super().__post_init__()
        self.policy.init_noise_std = 0.35
        self.policy.actor_hidden_dims = [512, 256, 128]
        self.policy.critic_hidden_dims = [512, 256, 128]
        self.algorithm.entropy_coef = 0.001
        self.algorithm.learning_rate = 1.0e-3
        self.algorithm.desired_kl = 0.01


@configclass
class K1FallRecoveryBeyondMimicFaceUpPPORunnerCfg(K1FallRecoveryBeyondMimicPPORunnerCfg):
    experiment_name = "k1_fall_recovery_beyond_mimic_faceup"


@configclass
class K1FallRecoveryBeyondMimicFaceUpCanonicalPPORunnerCfg(K1FallRecoveryBeyondMimicPPORunnerCfg):
    experiment_name = "k1_fall_recovery_beyond_mimic_faceup_canonical"


@configclass
class K1FallRecoveryBeyondMimicFaceDownPPORunnerCfg(K1FallRecoveryBeyondMimicPPORunnerCfg):
    experiment_name = "k1_fall_recovery_beyond_mimic_facedown"

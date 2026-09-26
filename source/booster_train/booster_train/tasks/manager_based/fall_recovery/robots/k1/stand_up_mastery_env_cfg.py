from __future__ import annotations

from isaaclab.utils import configclass

from booster_train.tasks.manager_based.fall_recovery import mdp

from .stand_up_env_cfg import K1WbcStandUpEnvCfg


@configclass
class K1WbcStandUpMasteryEnvCfg(K1WbcStandUpEnvCfg):
    """Accuracy-focused continuation after the legacy WBC curriculum."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Begin on the easiest terrain while retaining the full terrain asset so
        # later stages can expose levels through --wbc_max_terrain_level.
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator.curriculum = False
        self.curriculum.terrain_levels = None

        self.events.reset_base.params.update(
            standing_ratio=0.0,
            orientation_mode="balanced",
        )

        # Stage zero isolates recovery accuracy. Robustness randomization is
        # restored only after the flat-ground orientation gates pass.
        self.events.randomize_physics_material = None
        self.events.randomize_actuator_gains = None
        self.events.randomize_joint_friction = None
        self.events.randomize_joint_armature = None
        self.events.randomize_bodies_mass = None
        self.events.randomize_base_mass = None
        self.events.randomize_bodies_com = None
        self.events.randomize_base_com = None
        self.events.apply_external_force_torque = None
        self.events.apply_external_force_torque_extremities = None
        self.events.push_robot = None

        # Match the strict evaluator rather than treating height alone as success.
        self.terminations.standing.params.update(
            duration_s=1.0,
            projected_gravity_z_max=-0.85,
            projected_gravity_xy_norm_max=0.4,
            root_linear_speed_max=0.5,
            root_angular_speed_max=1.0,
        )

        # The selected legacy checkpoint already reached the terminal reward
        # recipe. Keep those weights fixed while resetting optimizer progress.
        self.rewards.action_l2.weight = -0.25
        self.rewards.action_rate.weight = -0.1
        self.rewards.action_rate_rate.weight = -0.1
        self.rewards.joint_deviation_l1.weight = 10.0
        self.rewards.joint_deviation_l1.func = mdp.joint_deviation_strict_success_potential
        self.rewards.joint_deviation_l1.params.update(
            duration_s=1.0,
            gamma=0.995,
            successful_termination_term="standing",
            projected_gravity_z_max=-0.85,
            projected_gravity_xy_norm_max=0.4,
            root_linear_speed_max=0.5,
            root_angular_speed_max=1.0,
        )
        self.rewards.incoming_forces_penalty.weight = -1.0e-5
        self.curriculum.increase_action_regularization = None
        self.curriculum.increase_action_rate_regularization = None
        self.curriculum.increase_action_rate_rate_regularization = None
        self.curriculum.increase_joint_deviation_regularization = None
        self.curriculum.increase_incoming_forces_penalty = None
        self.curriculum.action_limit_successful_termination = None

        # Preserve Curriculum/remove_lift telemetry while forcing zero assistance.
        self.actions.lift.force_limit = 0.0
        self.curriculum.remove_lift.params.update(
            start=-1,
            num_steps=1,
            linear=True,
        )

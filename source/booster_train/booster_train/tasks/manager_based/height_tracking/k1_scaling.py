"""K1 equivalents of WBC-AGILE's G1 dimensional constants."""

import math

WBC_G1_ROOT_HEIGHT = 0.72
WBC_G1_TRACKED_POINT_OFFSET = 0.20
WBC_G1_TOTAL_MASS = 34.13385728
WBC_G1_TORSO_MASS = 8.562

K1_ROOT_HEIGHT = 0.52
K1_TRACKED_POINT_OFFSET = 0.20
K1_TOTAL_MASS = 19.666
K1_TRUNK_MASS = 6.50

K1_LENGTH_SCALE = K1_ROOT_HEIGHT / WBC_G1_ROOT_HEIGHT
K1_TRACKED_HEIGHT_SCALE = (K1_ROOT_HEIGHT + K1_TRACKED_POINT_OFFSET) / (
    WBC_G1_ROOT_HEIGHT + WBC_G1_TRACKED_POINT_OFFSET
)
K1_MASS_SCALE = K1_TOTAL_MASS / WBC_G1_TOTAL_MASS
K1_TRUNK_MASS_SCALE = K1_TRUNK_MASS / WBC_G1_TORSO_MASS
# Preserve comparable linear and angular accelerations instead of applying the
# G1's absolute disturbances to the substantially lighter K1.
K1_FORCE_SCALE = K1_MASS_SCALE
K1_TORQUE_SCALE = K1_MASS_SCALE * K1_LENGTH_SCALE
K1_PUSH_LINEAR_VELOCITY_SCALE = math.sqrt(K1_LENGTH_SCALE)


def scale_root_height(value: float) -> float:
    """Scale a terrain-relative G1 root-height quantity to K1."""

    return value * K1_LENGTH_SCALE


def scale_tracked_height_error(value: float) -> float:
    """Scale a G1 tracked-point error while preserving relative accuracy."""

    return value * K1_TRACKED_HEIGHT_SCALE


K1_FULL_BODY_STANDING_HEIGHT = scale_root_height(0.50)
K1_MODERATE_STANDING_HEIGHT = scale_root_height(0.40)
K1_FULL_BODY_TRACKED_HEIGHT = K1_FULL_BODY_STANDING_HEIGHT + K1_TRACKED_POINT_OFFSET
K1_MODERATE_TRACKED_HEIGHT = K1_MODERATE_STANDING_HEIGHT + K1_TRACKED_POINT_OFFSET
K1_HEIGHT_GATE_HALF_WIDTH = 0.05
K1_FULL_BODY_HEIGHT_GATE_RANGE = (
    K1_FULL_BODY_TRACKED_HEIGHT - K1_HEIGHT_GATE_HALF_WIDTH,
    K1_FULL_BODY_TRACKED_HEIGHT + K1_HEIGHT_GATE_HALF_WIDTH,
)
K1_MODERATE_HEIGHT_GATE_RANGE = (
    K1_MODERATE_TRACKED_HEIGHT - K1_HEIGHT_GATE_HALF_WIDTH,
    K1_MODERATE_TRACKED_HEIGHT + K1_HEIGHT_GATE_HALF_WIDTH,
)
K1_HEIGHT_GATE_AGE_RANGE = (0.5, 1.5)
# This is a task-success tolerance, not a morphology-dependent length. WBC-AGILE
# uses the same 0.10 m threshold for lift decay and terrain progression.
K1_TRACKING_ERROR_THRESHOLD = 0.10
K1_REPORTING_SUCCESS_ERROR_THRESHOLD = 0.08
# Keep the stillness objective entirely below the lift-decay boundary. Otherwise
# improving into the curriculum's success band immediately adds a competing
# movement penalty while the robot is still settling into the commanded height.
K1_STILLNESS_ZERO_PENALTY_ERROR_THRESHOLD = K1_REPORTING_SUCCESS_ERROR_THRESHOLD
K1_STILLNESS_FULL_PENALTY_ERROR_THRESHOLD = 0.06
K1_RESET_HEIGHT_OFFSET = scale_root_height(0.05)
K1_HEIGHT_REWARD_ROUGH_STD = scale_tracked_height_error(0.50)
K1_HEIGHT_REWARD_MEDIUM_STD = scale_tracked_height_error(0.30)
K1_HEIGHT_REWARD_FINE_STD = scale_tracked_height_error(0.20)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

import isaaclab.terrains as terrain_gen
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from booster_train.tasks.manager_based.fall_recovery.wbc_stand_up.hf_terrains_cfg import (
    HfRandomUniformTerrainDifficultyCfg,
)
from booster_train.tasks.manager_based.height_tracking.k1_scaling import K1_LENGTH_SCALE

HEIGHT_TRACKING_ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=100.0,
    num_rows=8,
    num_cols=9,
    horizontal_scale=0.1,
    # Preserve WBC-AGILE's terrain difficulty relative to the robot's leg height.
    vertical_scale=0.01 * K1_LENGTH_SCALE,
    slope_threshold=0.9,
    use_cache=False,
    sub_terrains={
        "boxes": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.2,
            grid_width=0.45,
            grid_height_range=(0.01 * K1_LENGTH_SCALE, 0.075 * K1_LENGTH_SCALE),
            platform_width=0.1,
        ),
        "random_rough_small": HfRandomUniformTerrainDifficultyCfg(
            proportion=0.2,
            noise_range=(0.01 * K1_LENGTH_SCALE, 0.1 * K1_LENGTH_SCALE),
            noise_step=0.01 * K1_LENGTH_SCALE,
            border_width=0.4,
        ),
        "wave_small": terrain_gen.HfWaveTerrainCfg(
            proportion=0.2,
            amplitude_range=(0.01 * K1_LENGTH_SCALE, 0.15 * K1_LENGTH_SCALE),
            num_waves=4,
            border_width=0.2,
        ),
    },
)

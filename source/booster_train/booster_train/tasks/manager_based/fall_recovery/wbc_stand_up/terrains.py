# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg

from .hf_terrains_cfg import HfRandomUniformTerrainDifficultyCfg

STAND_UP_ROUGH_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=42,
    size=(8.0, 8.0),
    border_width=100.0,
    num_rows=20,
    num_cols=16,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "random_rough_small": HfRandomUniformTerrainDifficultyCfg(
            proportion=1.0,
            noise_range=(0.01, 0.05),
            noise_step=0.02,
            border_width=0.4,
        ),
    },
)

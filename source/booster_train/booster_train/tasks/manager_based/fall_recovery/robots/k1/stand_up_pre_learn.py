from __future__ import annotations

import logging
import os

from booster_train.tasks.manager_based.fall_recovery.wbc_stand_up import (
    FallenStateDataset,
    FallenStateDatasetCfg,
    compute_fallen_state_cache_key,
    get_fallen_state_cache_path,
    reset_from_fallen_dataset,
)

logger = logging.getLogger(__name__)


def pre_learn(env, task_name: str, agent_cfg) -> None:
    """Collect or load WBC fallen states before the first training reset."""
    dataset_cfg: FallenStateDatasetCfg | None = getattr(
        agent_cfg,
        "fallen_state_dataset_cfg",
        None,
    )
    if dataset_cfg is None:
        return
    reset_event = _get_reset_event(env)
    dataset = FallenStateDataset(cfg=dataset_cfg)
    if dataset_cfg.cache_path_override is not None:
        cache_path = os.path.abspath(
            os.path.expanduser(dataset_cfg.cache_path_override)
        )
        if not dataset.load(cache_path):
            raise FileNotFoundError(
                f"failed to load explicit K1 fallen-state cache: {cache_path}"
            )
        logger.info("Loaded explicit K1 fallen-state cache: %s", cache_path)
        reset_event.set_dataset(dataset)
        return

    cache_path = _get_cache_path(env, task_name, dataset_cfg)
    if dataset_cfg.cache_enabled and dataset.load(cache_path):
        logger.info("Loaded K1 fallen-state cache: %s", cache_path)
        reset_event.set_dataset(dataset)
        return

    logger.info("Collecting K1 WBC fallen-state dataset.")
    dataset.collect(env, verbose=True)
    if dataset_cfg.cache_enabled:
        dataset.save(cache_path)
        logger.info("Saved K1 fallen-state cache: %s", cache_path)
    reset_event.set_dataset(dataset)


def _get_reset_event(env) -> reset_from_fallen_dataset:
    if "reset" in env.event_manager.active_terms:
        for term_name in env.event_manager.active_terms["reset"]:
            term_cfg = env.event_manager.get_term_cfg(term_name)
            if isinstance(term_cfg.func, reset_from_fallen_dataset):
                return term_cfg.func
    raise AssertionError("K1 WBC task is missing reset_from_fallen_dataset")


def _get_cache_path(
    env,
    task_name: str,
    dataset_cfg: FallenStateDatasetCfg,
) -> str:
    terrain_cfg = None
    if env.scene.terrain.cfg.terrain_generator is not None:
        terrain_cfg = env.scene.terrain.cfg.terrain_generator.to_dict()
        terrain_cfg.pop("class_type", None)
        terrain_cfg.pop("use_cache", None)
        terrain_cfg.pop("cache_dir", None)
        for sub_cfg in terrain_cfg.get("sub_terrains", {}).values():
            if isinstance(sub_cfg, dict):
                sub_cfg.pop("class_type", None)
                sub_cfg.pop("function", None)

    dataset_dict = dataset_cfg.to_dict()
    dataset_dict.pop("cache_enabled", None)
    dataset_dict.pop("cache_dir", None)
    dataset_dict.pop("cache_path_override", None)
    dataset_dict["num_envs"] = env.num_envs
    cache_key = compute_fallen_state_cache_key(
        task_name,
        terrain_cfg,
        dataset_dict,
    )
    return get_fallen_state_cache_path(dataset_cfg.cache_dir, cache_key)

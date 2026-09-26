from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

try:
    from isaaclab.managers import ManagerTermBase  # noqa: F401
except ModuleNotFoundError:
    isaaclab = sys.modules.setdefault("isaaclab", types.ModuleType("isaaclab"))
    managers = types.ModuleType("isaaclab.managers")
    managers.ManagerTermBase = object
    isaaclab.managers = managers
    sys.modules["isaaclab.managers"] = managers


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "source"
    / "booster_train"
    / "booster_train"
    / "tasks"
    / "manager_based"
    / "height_tracking"
    / "curriculums.py"
)
SPEC = importlib.util.spec_from_file_location("height_tracking_curriculums", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
CURRICULUMS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CURRICULUMS)


class Action:
    def __init__(self) -> None:
        self.scale = None

    def scale_forces(self, scale: float) -> None:
        self.scale = scale


class EventManager:
    def __init__(self) -> None:
        self.cfgs = {
            "trunk": SimpleNamespace(
                params={"force_range": (-10.0, 10.0), "torque_range": (-4.0, 4.0)}
            ),
            "extremities": SimpleNamespace(
                params={"force_range": (-2.0, 2.0), "torque_range": (-0.5, 0.5)}
            ),
        }

    def get_term_cfg(self, name):
        return self.cfgs[name]

    def set_term_cfg(self, name, cfg):
        self.cfgs[name] = cfg


def dual_term(values):
    term = object.__new__(CURRICULUMS.adaptive_force_decay)
    term._force_scale = 0.032
    term._ema = 0.13
    term._governed_ema = None
    term._jump_seen = False
    term._resume_warmup_until_step = None
    term._scale_method = "scale_forces"
    term._action = Action()
    term._command = SimpleNamespace(
        settled=True, uses_reference_cohort=True,
        cohort_curriculum_metrics=lambda ids, name: values,
    )
    env = SimpleNamespace(curriculum_manager=SimpleNamespace(_curriculum_state={}),
                          common_step_counter=100)
    return term, env


@pytest.mark.parametrize("managed,jump,decays", [
    (.07, .11, True), (.09, .11, False), (.07, .13, False),
    (.08, .11, False), (.07, .12, False), (.001, .13, False),
    (.09, .001, False), (.07, float("nan"), False),
])
def test_dual_gate_requires_both_strict_thresholds(managed, jump, decays):
    term, env = dual_term({"governed": (managed, .8), "jump": (jump, .2)})
    result = term(env, torch.arange(5), action_name="lift", threshold=.12,
                  governed_threshold=.08, ema_alpha=1.0)
    assert result == pytest.approx(.032 * (.9999 if decays else 1.0))


def test_dual_gate_waits_for_both_groups_and_new_reference_sample():
    values = {"governed": (.07, 1.0)}
    term, env = dual_term(values)
    def update():
        return term(env, torch.arange(5), action_name="lift", threshold=.12,
                    governed_threshold=.08, ema_alpha=1.0)
    assert update() == pytest.approx(.032)
    values.clear()
    values["jump"] = (.11, 1.0)
    assert update() == pytest.approx(.032 * .9999)
    values.clear()
    values["governed"] = (.07, 1.0)
    assert update() == pytest.approx(.032 * .9999)


def test_dual_gate_sample_weighting_warmup_and_disable():
    values = {"governed": (.07, .8), "jump": (.11, .2)}
    term, env = dual_term(values)
    term._governed_ema = .09
    term.arm_resume_warmup(100, 20)
    kwargs = dict(action_name="lift", threshold=.12, governed_threshold=.08)
    term(env, torch.arange(5), **kwargs)
    assert term._ema == .13
    assert term._governed_ema == .09
    env.common_step_counter = 120
    term(env, torch.arange(5), **kwargs)
    assert term._ema == pytest.approx(.13 * .95 ** .2 + .11 * (1 - .95 ** .2))
    assert term._governed_ema == pytest.approx(.09 * .95 ** .8 + .07 * (1 - .95 ** .8))
    assert term._command.lift_gate_metrics["lift_gate_passed"] == 0
    term._force_scale = .0100001
    assert term(env, torch.arange(5), ema_alpha=1.0, **kwargs) == 0.0


@pytest.mark.parametrize("errors, expected", [([0.0, 0.2], 0.1), ([0.05, 0.25], 0.15)])
def test_restored_lift_uses_all_reset_environments(errors, expected):
    term = object.__new__(CURRICULUMS.adaptive_force_decay)
    term._force_scale = 0.22
    term._ema = 0.2
    term._action = Action()
    term._scale_method = "scale_forces"
    term._resume_warmup_until_step = None
    term._command = SimpleNamespace(
        settled=torch.tensor([True, True]),
        metrics={"height_error": torch.tensor(errors)},
    )
    env = SimpleNamespace(curriculum_manager=SimpleNamespace(_curriculum_state={}))
    assert term(env, torch.tensor([0, 1]), action_name="lift", metric_name="height_error",
                threshold=0.1, ema_alpha=1.0, decay=0.5) == pytest.approx(0.22)
    assert term._ema == pytest.approx(expected)


def test_restored_lift_uses_original_ema_alpha_and_decay():
    term = object.__new__(CURRICULUMS.adaptive_force_decay)
    term._force_scale = 0.22
    term._ema = 0.103
    term._action = Action()
    term._scale_method = "scale_forces"
    term._resume_warmup_until_step = None
    term._command = SimpleNamespace(
        settled=torch.tensor([True]),
        metrics={"height_error": torch.tensor([0.0])},
    )
    env = SimpleNamespace(curriculum_manager=SimpleNamespace(_curriculum_state={}))
    result = term(env, torch.tensor([0]), action_name="lift", metric_name="height_error",
                  threshold=0.1, ema_alpha=0.05, decay=0.9999)
    assert term._ema == pytest.approx(0.103 * 0.95)
    assert result == pytest.approx(0.22 * 0.9999)
    term._force_scale = 0.22
    term._ema = 0.09
    result = term(env, torch.tensor([0]), action_name="lift", metric_name="height_error",
                  threshold=0.1, ema_alpha=0.05, decay=0.9999)
    assert result == pytest.approx(0.22 * 0.9999)


def test_instantaneous_height_command_updates_without_reset_envs() -> None:
    action = Action()
    term = object.__new__(CURRICULUMS.adaptive_force_decay)
    term._force_scale = 1.0
    term._ema = 1.0
    term._action = action
    term._scale_method = "scale_forces"
    term._resume_warmup_until_step = None
    term._command = SimpleNamespace(
        base_height=torch.tensor([0.50, 0.40]),
        target_height=torch.tensor([0.55, 0.45]),
        metrics={"height_error": torch.tensor([1.0, 1.0])},
    )
    env = SimpleNamespace(
        curriculum_manager=SimpleNamespace(_curriculum_state={}),
    )

    result = term(
        env,
        torch.empty(0, dtype=torch.long),
        action_name="lift",
        metric_name="height_error",
        threshold=0.1,
        ema_alpha=1.0,
        decay=0.5,
    )

    assert result == pytest.approx(0.5)
    assert term._ema == pytest.approx(0.05)
    assert action.scale == pytest.approx(0.5)


def test_smooth_height_command_uses_settled_episode_metric() -> None:
    action = Action()
    term = object.__new__(CURRICULUMS.adaptive_force_decay)
    term._force_scale = 1.0
    term._ema = 1.0
    term._action = action
    term._scale_method = "scale_forces"
    term._resume_warmup_until_step = None
    term._command = SimpleNamespace(
        settled=torch.tensor([True, True]),
        metrics={"height_error": torch.tensor([0.05, 0.20])},
    )
    env = SimpleNamespace(
        curriculum_manager=SimpleNamespace(_curriculum_state={}),
    )

    result = term(
        env,
        torch.tensor([0], dtype=torch.long),
        action_name="lift",
        metric_name="height_error",
        threshold=0.1,
        ema_alpha=1.0,
        decay=0.5,
    )

    assert result == pytest.approx(0.5)
    assert term._ema == pytest.approx(0.05)
    assert action.scale == pytest.approx(0.5)


def test_smooth_height_command_waits_for_reset_envs() -> None:
    action = Action()
    term = object.__new__(CURRICULUMS.adaptive_force_decay)
    term._force_scale = 1.0
    term._ema = 1.0
    term._action = action
    term._scale_method = "scale_forces"
    term._resume_warmup_until_step = None
    term._command = SimpleNamespace(
        settled=torch.tensor([True]),
        metrics={"height_error": torch.tensor([0.05])},
    )
    env = SimpleNamespace(
        curriculum_manager=SimpleNamespace(_curriculum_state={}),
    )

    result = term(
        env,
        torch.empty(0, dtype=torch.long),
        action_name="lift",
        metric_name="height_error",
        threshold=0.1,
        ema_alpha=1.0,
        decay=0.5,
    )

    assert result == pytest.approx(1.0)
    assert term._ema == pytest.approx(1.0)
    assert action.scale is None


def test_episode_metric_still_waits_for_reset_envs() -> None:
    action = Action()
    term = object.__new__(CURRICULUMS.adaptive_force_decay)
    term._force_scale = 1.0
    term._ema = 0.0
    term._action = action
    term._scale_method = "scale_forces"
    term._resume_warmup_until_step = None
    term._command = SimpleNamespace(metrics={"standing_ratio": torch.tensor([1.0])})
    env = SimpleNamespace(
        curriculum_manager=SimpleNamespace(_curriculum_state={}),
    )

    result = term(
        env,
        torch.empty(0, dtype=torch.long),
        action_name="lift",
        metric_name="standing_ratio",
        threshold=0.8,
        ema_alpha=1.0,
        decay=0.5,
    )

    assert result == pytest.approx(1.0)
    assert action.scale is None


def progressive_disturbance_term(values, terrain_level=4.0):
    term = object.__new__(CURRICULUMS.adaptive_external_force_growth)
    term._scale = 1.0
    term._last_step = 90
    term._jump_ema = None
    term._governed_ema = None
    term._command = SimpleNamespace(
        cohort_curriculum_metrics=lambda ids, name: values,
    )
    term._event_manager = EventManager()
    term._event_ranges = {
        "trunk": {"force_range": (-10.0, 10.0), "torque_range": (-4.0, 4.0)},
        "extremities": {"force_range": (-2.0, 2.0), "torque_range": (-0.5, 0.5)},
    }
    env = SimpleNamespace(
        common_step_counter=100,
        scene=SimpleNamespace(
            terrain=SimpleNamespace(terrain_levels=torch.tensor([terrain_level]))
        ),
    )
    return term, env


def test_external_force_growth_requires_both_cohorts_and_terrain() -> None:
    term, env = progressive_disturbance_term(
        {"jump": (0.13, 0.2), "governed": (0.11, 0.8)}
    )
    result = term(
        env,
        torch.arange(5),
        event_names=("trunk", "extremities"),
        command_name="height",
        ema_alpha=1.0,
        growth_per_step=0.01,
    )
    assert result == pytest.approx(1.1)
    assert term._event_manager.cfgs["trunk"].params["force_range"] == pytest.approx((-11.0, 11.0))
    assert term._event_manager.cfgs["extremities"].params["torque_range"] == pytest.approx((-0.55, 0.55))
    assert term._command.disturbance_gate_metrics["disturbance_gate_passed"] == 1.0

    term, env = progressive_disturbance_term({"jump": (0.13, 1.0)})
    assert term(
        env,
        torch.arange(5),
        event_names=("trunk", "extremities"),
        command_name="height",
        ema_alpha=1.0,
        growth_per_step=0.01,
    ) == pytest.approx(1.0)

    term, env = progressive_disturbance_term(
        {"jump": (0.13, 0.2), "governed": (0.11, 0.8)}, terrain_level=3.0
    )
    assert term(
        env,
        torch.arange(5),
        event_names=("trunk", "extremities"),
        command_name="height",
        ema_alpha=1.0,
        growth_per_step=0.01,
    ) == pytest.approx(1.0)


def test_external_force_growth_pauses_on_error_and_caps_scale() -> None:
    values = {"jump": (0.20, 0.2), "governed": (0.11, 0.8)}
    term, env = progressive_disturbance_term(values)
    term._jump_ema = 0.13
    term._governed_ema = 0.11
    term._scale = 1.49
    assert term(
        env,
        torch.arange(5),
        event_names=("trunk", "extremities"),
        command_name="height",
        ema_alpha=1.0,
        growth_per_step=0.01,
    ) == pytest.approx(1.49)

    values["jump"] = (0.13, 0.2)
    env.common_step_counter = 110
    assert term(
        env,
        torch.arange(5),
        event_names=("trunk", "extremities"),
        command_name="height",
        ema_alpha=1.0,
        growth_per_step=0.01,
    ) == pytest.approx(1.5)

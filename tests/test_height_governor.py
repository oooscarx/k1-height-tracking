import importlib.util
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "source/booster_train/booster_train/tasks/manager_based/height_tracking/governor.py"
spec = importlib.util.spec_from_file_location("height_governor_test", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Governor = module.HeightCommandGovernor


def request(goals, measured, managed, durations=None):
    g = Governor(len(goals))
    durations = [1.0] * len(goals) if durations is None else durations
    budget = g.request(torch.arange(len(goals)), torch.tensor(goals), torch.tensor(measured),
                       torch.tensor(managed), torch.tensor(durations))
    return g, budget


def test_first_upward_step_uses_actual_height_and_keeps_final_goal():
    g, budget = request([.72], [.1], [True])
    assert g.command.item() == pytest.approx(.22)
    assert g.final.item() == pytest.approx(.72)
    assert budget.item() == pytest.approx(24)


def test_fixed_clock_advances_without_success_and_does_not_reset_reward_age():
    g, _ = request([.72], [.1], [True])
    g.step(3.99)
    assert g.command.item() == pytest.approx(.22)
    g.step(.01)
    assert g.command.item() == pytest.approx(.34)
    assert g.elapsed.item() == pytest.approx(4)
    assert g.command_age.item() == pytest.approx(0, abs=1e-5)
    # There is no measured-height or success argument to progression.
    g.step(16)
    assert g.at_final.item()
    assert g.command.item() == pytest.approx(.72)
    g.step(4)
    assert g.command_age.item() == pytest.approx(4)


def test_jump_negative_downward_and_small_commands():
    goals = [.72, -.5, .2, .4]
    g, budget = request(goals, [.1, .4, .7, .35], [False, True, True, True], [2., 2., 2., 6.])
    assert g.command.tolist() == pytest.approx(goals)
    assert budget.tolist() == pytest.approx([2, 2, 4, 6])


def test_fixed_steps_never_exceed_increment_or_reverse_when_body_falls():
    g, budget = request([.72], [-.02], [True], [7.])
    previous = g.command.clone()
    for _ in range(1400):
        current = g.step(.02)
        assert (current >= previous - 1e-6).all()
        assert (current - previous <= .120001).all()
        previous = current.clone()
    assert g.command.item() == pytest.approx(.72)
    assert budget.item() <= 27.001


def test_partial_request_does_not_retime_other_environments():
    g, _ = request([.72, .72], [.1, .2], [True, False])
    g.step(5)
    g.request(torch.tensor([0]), torch.tensor([.5]), torch.tensor([.2]),
              torch.tensor([True]), torch.tensor([4.]))
    assert g.elapsed.tolist() == pytest.approx([0, 5])
    assert g.command.tolist() == pytest.approx([.32, .72])


@pytest.mark.parametrize("goal,measured", [(float('nan'), .2), (.5, float('inf'))])
def test_invalid_measurements_are_rejected(goal, measured):
    with pytest.raises(ValueError):
        request([goal], [measured], [True])


def test_training_uses_same_governor_and_keeps_separate_reward_clock():
    commands = PATH.with_name("commands.py").read_text()
    assert "from .governor import HeightCommandGovernor" in commands
    assert "self._target_height = self.governor.step" in commands
    assert "self.governor.command_age > self.cfg.settle_time_s" in commands
    assert commands.count("self._steps_since_resample[env_ids] = 0") == 1
    assert "self._steps_since_resample[advanced]" not in commands
    assert "self.cfg.focus_ratio" in commands
    assert "self.cfg.focus_height_range" in commands


def test_ppo_remains_full_update_and_config_has_goal_budget():
    folder = PATH.parent / "robots/k1"
    assert "kl_early_stop" not in (folder / "ppo_cfg.py").read_text()
    env = (folder / "env_cfg.py").read_text()
    assert "self.episode_length_s = 32.0" in env
    assert "governed_commands=True" in env


def command_method(name):
    tree = ast.parse(PATH.with_name("commands.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SmoothHeightCommand")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(PATH), "exec"), ns)
    return ns[name]


def test_goal_accounting_counts_final_success_not_intermediate_arrival():
    obj = SimpleNamespace(device="cpu", _env=SimpleNamespace(step_dt=.02),
        _goal_active=torch.ones(4, dtype=torch.bool), _final_count=torch.tensor([50., 0., 80., 70.]),
        _final_good=torch.tensor([49., 0., 79., 30.]), _first_success_s=torch.tensor([8., -1., 6., -1.]),
        _reference=torch.tensor([False, False, True, True]), _goal_totals=torch.zeros(2, 7))
    finish = command_method("_finish_goals")
    finish(obj, torch.arange(4), timeout=True, terminated=torch.tensor([False, False, True, False]))
    assert obj._goal_totals[0].tolist() == pytest.approx([2, 1, 1, 2, 0, 8, 1])
    assert obj._goal_totals[1].tolist() == pytest.approx([2, 2, 0, 1, 1, 0, 0])
    before = obj._goal_totals.clone()
    finish(obj, torch.arange(4))
    assert torch.equal(before, obj._goal_totals)


def test_lift_input_excludes_managed_group():
    obj = SimpleNamespace(_reference=torch.tensor([True, False, False]),
                          metrics={"height_error": torch.tensor([.2, .001, .001])})
    method = command_method("reference_curriculum_metric")
    assert method(obj, torch.arange(3), "height_error").item() == pytest.approx(.2)
    assert obj._reference_curriculum_fraction == pytest.approx(1/3)
    assert method(obj, torch.tensor([1, 2]), "height_error") is None


def test_dual_lift_inputs_separate_groups_and_ignore_unobserved_episodes():
    obj = SimpleNamespace(_reference=torch.tensor([True, False, True, False]),
        _episode_step_count=torch.tensor([50., 100., 0., 0.]),
        metrics={"height_error": torch.tensor([.13, .07, 0., 0.])})
    values = command_method("cohort_curriculum_metrics")(obj, torch.arange(4), "height_error")
    assert values["jump"] == pytest.approx((.13, .25))
    assert values["governed"] == pytest.approx((.07, .25))
    assert command_method("cohort_curriculum_metrics")(obj, torch.tensor([2, 3]), "height_error") == {}


def test_dual_gate_thresholds_are_configured():
    env = (PATH.parent / "robots/k1/env_cfg.py").read_text()
    assert '"threshold": 0.12' in env
    assert '"governed_threshold": 0.08' in env


def test_deployment_uses_identical_schedule_and_repeated_goal_does_not_reset():
    spec = importlib.util.spec_from_file_location("deploy_governor_test", ROOT / "deployment/height_command.py")
    deploy = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(deploy)
    adapter = deploy.DeploymentHeightCommand()
    g, _ = request([.72], [.1], [True])
    for _ in range(500):
        assert adapter.update(.1, .72) == pytest.approx(g.step(.02).item())
    assert adapter.governor.elapsed.item() > 9.9
    assert adapter.update(.05, .72, replan=True) == pytest.approx(.17)

import importlib.util
from pathlib import Path
import torch
import ast
from types import SimpleNamespace, MethodType

ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("delta", ROOT / "source/booster_train/booster_train/tasks/manager_based/height_tracking/delta_curriculum.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
HeightDeltaCurriculum = module.HeightDeltaCurriculum


def test_actual_height_negative_downward_and_reference_preserved():
    c = HeightDeltaCurriculum("cpu")
    targets = torch.tensor([.72, .72, -.5, .2, .4])
    measured = torch.tensor([.1, .1, .4, .6, .35])
    reference = torch.tensor([False, True, False, False, False])
    result, frontier = c.constrain(targets, measured, reference)
    torch.testing.assert_close(result, torch.tensor([.22, .72, -.5, .2, .4]))
    assert frontier.tolist() == [True, False, False, False, False]


def test_same_target_different_actual_origins():
    c = HeightDeltaCurriculum("cpu")
    result, _ = c.constrain(torch.tensor([.6, .6]), torch.tensor([.3, .58]), torch.tensor([False, False]))
    torch.testing.assert_close(result, torch.tensor([.42, .6]))


def record(c, good=True, eligible=True, failure=False):
    c.finish(torch.tensor([100.]), torch.tensor([4. if good else 20.]),
             torch.tensor([100. if good else 20.]), torch.tensor([eligible]), torch.tensor([failure]))


def test_no_promotion_until_window_and_minimum_samples():
    c = HeightDeltaCurriculum("cpu", window_steps=2, minimum_segments=2)
    record(c)
    c.step()
    assert c.limit == .12
    c.step()
    assert c.limit == .12
    record(c)
    record(c)
    c.step()
    c.step()
    assert c.limit == .16 and c.stage == 1


def test_failures_and_censoring_cannot_promote():
    c = HeightDeltaCurriculum("cpu", window_steps=1, minimum_segments=1)
    record(c, eligible=False)
    c.step()
    assert c.last_count == 0 and c.limit == .12
    record(c, eligible=False, failure=True)
    c.step()
    assert c.last_count == 1 and c.last_rate == 0 and c.limit == .12


def test_bad_error_cannot_promote():
    c = HeightDeltaCurriculum("cpu", window_steps=1, minimum_segments=1)
    record(c, good=False)
    c.step()
    assert c.limit == .12


def test_resume_preserves_window_and_stage():
    c = HeightDeltaCurriculum("cpu", window_steps=2, minimum_segments=1)
    record(c)
    c.step()
    state = c.state_dict()
    restored = HeightDeltaCurriculum("cpu", window_steps=2, minimum_segments=1)
    restored.load_state_dict(state)
    restored.step()
    c.step()
    assert c.limit == restored.limit == .16
    assert c.steps == restored.steps == 2


def test_complete_curriculum_restores_full_support():
    c = HeightDeltaCurriculum("cpu", initial=.72)
    targets = torch.tensor([.72])
    output, frontier = c.constrain(targets, torch.tensor([-.1]), torch.tensor([False]))
    torch.testing.assert_close(output, targets)
    assert not frontier.any()


def test_invalid_state_rejected():
    c = HeightDeltaCurriculum("cpu")
    state = c.state_dict()
    state["limit"] = float("nan")
    try:
        c.load_state_dict(state)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted nonfinite curriculum state")


def command_method(name):
    tree = ast.parse((ROOT / "source/booster_train/booster_train/tasks/manager_based/height_tracking/commands.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SmoothHeightCommand")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"torch": torch, "Sequence": list}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "commands.py", "exec"), namespace)
    return namespace[name]


def test_command_sampling_control_rng_and_negative_proportion_unchanged():
    n = 10000
    obj = SimpleNamespace(device="cpu", delta_curriculum=None, progress_sampler=None, staircase=None,
        cfg=SimpleNamespace(standing_ratio=.15, flat_ratio=.2, ranges=SimpleNamespace(height=(-.5, .72)), velocity_range=(1000, 1000)),
        _steps_since_resample=torch.zeros(n), _current_height_cmd=torch.zeros(n),
        _command_origin=torch.zeros(n), _continuous_positive=torch.zeros(n, dtype=torch.bool),
        _reference=torch.arange(n) % 5 == 0, _target_height=torch.zeros(n), _velocity=torch.zeros(n),
        _delta_frontier=torch.zeros(n, dtype=torch.bool), _delta_stage=torch.zeros(n, dtype=torch.long),
        _delta_count=torch.zeros(n), _delta_error=torch.zeros(n), _delta_success=torch.zeros(n),
        _env=SimpleNamespace(step_dt=.02), measured_height=torch.full((n,), .1))
    method = command_method("_resample_command")
    torch.manual_seed(73)
    method(obj, torch.arange(n))
    baseline = obj._target_height.clone()
    obj.delta_curriculum = HeightDeltaCurriculum("cpu")
    obj._finish_delta_commands = MethodType(command_method("_finish_delta_commands"), obj)
    torch.manual_seed(73)
    method(obj, torch.arange(n))
    protected = obj._reference | (baseline < 0) | (baseline <= .22)
    assert torch.equal(obj._target_height[protected], baseline[protected])
    assert torch.equal(obj._target_height < 0, baseline < 0)
    assert (obj._target_height[~obj._reference] <= .22 + 1e-6).all()


def test_checkpoint_attachment_preserves_delta_under_separate_key():
    path = ROOT / "source/booster_train/booster_train/tasks/manager_based/height_tracking/progress_sampling.py"
    s = importlib.util.spec_from_file_location("progress", path)
    progress = importlib.util.module_from_spec(s)
    s.loader.exec_module(progress)
    captured = {}
    runner = SimpleNamespace(save=lambda path, infos=None: captured.update(infos=infos))
    c = HeightDeltaCurriculum("cpu")
    record(c)
    c.step()
    progress.attach_sampler_checkpoint(runner, c, state_key="height_delta_curriculum")
    runner.save("test.pt", {"existing": 42})
    assert captured["infos"]["existing"] == 42
    restored = HeightDeltaCurriculum("cpu")
    progress.attach_sampler_checkpoint(SimpleNamespace(save=lambda *a, **k: None), restored,
                                       captured["infos"], state_key="height_delta_curriculum")
    assert restored.steps == c.steps
    torch.testing.assert_close(restored.totals, c.totals)


def test_old_stage_commands_cannot_promote_next_stage():
    c = HeightDeltaCurriculum("cpu")
    c.stage = 1
    c.limit = .16
    obj = SimpleNamespace(delta_curriculum=c, staircase=None, _env=SimpleNamespace(step_dt=.02),
        _delta_frontier=torch.tensor([True]), _delta_stage=torch.tensor([0]),
        _delta_count=torch.tensor([100.]), _delta_error=torch.tensor([1.]), _delta_success=torch.tensor([100.]))
    command_method("_finish_delta_commands")(obj, torch.tensor([0]))
    assert c.totals.sum() == 0
    assert obj._delta_count.sum() == 0 and not obj._delta_frontier.any()

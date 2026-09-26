import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "source/booster_train/booster_train/tasks/manager_based/height_tracking"
SPEC = importlib.util.spec_from_file_location("progress_sampling", MODULE / "progress_sampling.py")
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def sampler(**kwargs):
    return MOD.HeightProgressSampler(0.72, "cpu", **kwargs)


def test_warmup_and_uniform_floor():
    s = sampler()
    s.seen.fill_(3)
    s.slow[15] = 0.2
    origins = torch.tensor([-0.5, 0.72])
    assert torch.allclose(s.probabilities(origins), torch.full((2, 12), 1 / 12))
    s.steps = 6000
    p = s.probabilities(origins)
    assert p[0, 3] > 0.5
    assert torch.all(p >= 0.5 / 12)
    assert torch.allclose(p.sum(1), torch.ones(2))
    assert torch.allclose(p[1], torch.full((12,), 1 / 12))
    draws = s.sample(torch.zeros(10000))
    assert torch.all((draws >= 0) & (draws < 0.72))


def test_learning_progress_and_invalid_samples():
    s = sampler(update_steps=1, warmup_steps=0)
    heights = torch.full((256,), 0.3)
    errors = torch.full((256,), 0.2)
    errors[0] = float("nan")
    valid = torch.ones(256, dtype=torch.bool)
    s.observe(heights, torch.zeros(256), errors, valid)
    s.observe(heights, torch.zeros(256), torch.full((256,), 0.1), valid)
    assert torch.isfinite(s.fast).all()
    assert s.probabilities(torch.zeros(1))[0, 5] > 1 / 12
    s.success.fill_(1)
    assert torch.allclose(s.probabilities(torch.zeros(1)), torch.full((1, 12), 1 / 12))


def test_checkpoint_roundtrip_and_atomic_validation():
    s = sampler()
    s.steps = 7777
    s.fast.fill_(0.1)
    state = s.state_dict()
    other = sampler()
    other.load_state_dict(state)
    assert other.steps == 7777
    assert torch.equal(other.fast, s.fast)
    torch.manual_seed(42)
    a = s.sample(torch.zeros(100))
    torch.manual_seed(42)
    assert torch.equal(a, other.sample(torch.zeros(100)))
    bad = copy.deepcopy(state)
    bad["success"][0] = float("nan")
    bad["steps"] = 1
    with pytest.raises(ValueError):
        other.load_state_dict(bad)
    assert other.steps == 7777


def test_runner_checkpoint_preserves_other_infos():
    saved = {}
    runner = SimpleNamespace(save=lambda path, infos=None: saved.update(path=path, infos=infos))
    s = sampler()
    s.steps = 999
    MOD.attach_sampler_checkpoint(runner, s)
    runner.save("checkpoint.pt", {"other": 42})
    assert saved["infos"]["other"] == 42
    other = sampler()
    MOD.attach_sampler_checkpoint(SimpleNamespace(save=lambda *a, **kw: None), other, saved["infos"])
    assert other.steps == 999


def command_method(name):
    tree = ast.parse((MODULE / "commands.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SmoothHeightCommand")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = {"torch": torch, "Sequence": list}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "commands.py", "exec"), namespace)
    return namespace[name]


def test_categories_negative_and_reference_are_unchanged():
    n = 50000
    obj = SimpleNamespace(
        device="cpu", cfg=SimpleNamespace(standing_ratio=0.15, flat_ratio=0.2,
        ranges=SimpleNamespace(height=(-0.5, 0.72)), velocity_range=(1000, 1000)),
        _steps_since_resample=torch.zeros(n), _current_height_cmd=torch.zeros(n),
        _command_origin=torch.zeros(n), _continuous_positive=torch.zeros(n, dtype=torch.bool),
        _reference=torch.arange(n) % 5 == 0, _target_height=torch.zeros(n),
        _velocity=torch.zeros(n), progress_sampler=None, delta_curriculum=None)
    method = command_method("_resample_command")
    torch.manual_seed(9)
    method(obj, torch.arange(n))
    baseline = obj._target_height.clone()
    obj.progress_sampler = sampler(warmup_steps=0)
    obj.progress_sampler.seen.fill_(3)
    obj.progress_sampler.slow[15] = 0.2
    torch.manual_seed(9)
    method(obj, torch.arange(n))
    protected = obj._reference | ~obj._continuous_positive
    assert torch.equal(baseline[protected], obj._target_height[protected])
    assert torch.equal(baseline < 0, obj._target_height < 0)
    assert not torch.equal(baseline, obj._target_height)
    assert torch.all(obj._velocity == 1000)


def test_reference_metric_excludes_adaptive_envs():
    obj = SimpleNamespace(device="cpu", progress_sampler=sampler(), uses_reference_cohort=True,
        _reference=torch.tensor([True, False, False]),
        metrics={"height_error": torch.tensor([0.2, 0.0, 0.0])})
    method = command_method("reference_curriculum_metric")
    assert method(obj, [0, 1, 2], "height_error").item() == pytest.approx(0.2)
    assert obj._reference_curriculum_fraction == pytest.approx(1 / 3)
    assert method(obj, [1, 2], "height_error") is None

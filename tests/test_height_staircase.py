import importlib.util
from pathlib import Path
from types import SimpleNamespace, MethodType

import torch
from test_height_delta_curriculum import command_method, HeightDeltaCurriculum


ROOT = Path(__file__).parents[1]
spec = importlib.util.spec_from_file_location("staircase", ROOT / "source/booster_train/booster_train/tasks/manager_based/height_tracking/staircase.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_reaches_original_final_without_resampling():
    stair = module.HeightStaircase(1, "cpu")
    stair.sample(torch.tensor([0]), torch.tensor([.72]), torch.tensor([True]))
    target = torch.tensor([.42])
    for expected in (.54, .66, .72):
        measured = target.clone()
        for i in range(25):
            target, advanced = stair.advance(target, measured, .12, .02)
            assert bool(advanced[0]) == (i == 24)
        torch.testing.assert_close(target, torch.tensor([expected]))
    for _ in range(100):
        target, advanced = stair.advance(target, torch.tensor([.72]), .12, .02)
        assert not advanced.any()
    assert stair.totals.tolist() == [1, 3, 1]


def test_collapse_nonfinite_and_interrupted_stability_do_not_advance():
    stair = module.HeightStaircase(1, "cpu")
    stair.sample(torch.tensor([0]), torch.tensor([.72]), torch.tensor([True]))
    target = torch.tensor([.42])
    for height in [.42] * 24 + [.3, float("nan"), float("inf")] + [.42] * 24:
        target, advance = stair.advance(target, torch.tensor([height]), .12, .02)
        assert not advance.any()
        torch.testing.assert_close(target, torch.tensor([.42]))


def test_reference_negative_downward_and_subset_reset_are_unchanged():
    stair = module.HeightStaircase(4, "cpu")
    target = torch.tensor([.42, .72, -.5, .2])
    stair.sample(torch.arange(4), torch.tensor([.72, .72, -.5, .2]), torch.tensor([True, False, False, False]))
    for _ in range(25):
        target, _ = stair.advance(target, target.clone(), .12, .02)
    torch.testing.assert_close(target, torch.tensor([.54, .72, -.5, .2]))
    stair.sample(torch.tensor([0]), torch.tensor([-.5]), torch.tensor([False]))
    target[0] = -.5
    for _ in range(50):
        updated, advanced = stair.advance(target, target, .12, .02)
        torch.testing.assert_close(updated, target)
        assert not advanced.any()


def test_step_resets_only_advanced_command_age_and_preserves_timer():
    stair = module.HeightStaircase(2, "cpu")
    stair.sample(torch.arange(2), torch.tensor([.72, .72]), torch.tensor([True, False]))
    obj = SimpleNamespace(staircase=stair, _target_height=torch.tensor([.42, .72]),
        measured_height=torch.tensor([.42, .72]), delta_curriculum=HeightDeltaCurriculum("cpu"),
        _env=SimpleNamespace(step_dt=.02), _steps_since_resample=torch.tensor([200., 200.]),
        _delta_count=torch.zeros(2), _delta_stage=torch.zeros(2, dtype=torch.long),
        _velocity=torch.tensor([1000., 1000.]), _current_height_cmd=torch.tensor([.42, .72]),
        time_left=torch.tensor([4., 4.]))
    update = command_method("_update_command")
    for _ in range(25):
        update(obj)
    assert obj._steps_since_resample.tolist() == [0, 200]
    torch.testing.assert_close(obj.time_left, torch.tensor([4., 4.]))
    torch.testing.assert_close(obj._current_height_cmd, torch.tensor([.54, .72]))


def test_failure_categories_partition_failures_and_ignore_censored():
    counts = torch.full((5,), 100.)
    result = module.classify_failures(counts, torch.tensor([4., 12., 5., 1., 50.]),
        torch.tensor([95., 20., 80., 100., 0.]),
        torch.tensor([True, True, True, False, False]),
        torch.tensor([False, False, False, True, False]))
    assert result.tolist() == [4, 1, 1, 1]


def test_resampling_retains_full_goal_but_initial_command_is_limited():
    n = 10
    obj = SimpleNamespace(device="cpu", delta_curriculum=HeightDeltaCurriculum("cpu"), progress_sampler=None,
        staircase=module.HeightStaircase(n, "cpu"),
        cfg=SimpleNamespace(standing_ratio=1., flat_ratio=0., ranges=SimpleNamespace(height=(-.5, .72)), velocity_range=(1000, 1000)),
        _steps_since_resample=torch.zeros(n), _current_height_cmd=torch.zeros(n),
        _command_origin=torch.zeros(n), _continuous_positive=torch.zeros(n, dtype=torch.bool),
        _reference=torch.arange(n) % 5 == 0, _target_height=torch.zeros(n), _velocity=torch.zeros(n),
        _delta_frontier=torch.zeros(n, dtype=torch.bool), _delta_stage=torch.zeros(n, dtype=torch.long),
        _finish_delta_commands=lambda ids: None, measured_height=torch.full((n,), .3))
    command_method("_resample_command")(obj, torch.arange(n))
    torch.testing.assert_close(obj.staircase.final, torch.full((n,), .72))
    torch.testing.assert_close(obj._target_height[obj._reference], torch.full((2,), .72))
    torch.testing.assert_close(obj._target_height[~obj._reference], torch.full((8,), .42))


def test_fast_completed_steps_are_counted_without_two_second_gate():
    stair = module.HeightStaircase(1, "cpu")
    stair.sample(torch.tensor([0]), torch.tensor([.72]), torch.tensor([True]), torch.tensor([.3]))
    obj = SimpleNamespace(staircase=stair, _target_height=torch.tensor([.42]),
        measured_height=torch.tensor([.42]), delta_curriculum=HeightDeltaCurriculum("cpu"),
        _env=SimpleNamespace(step_dt=.02), _steps_since_resample=torch.tensor([0.]),
        _delta_count=torch.zeros(1), _delta_stage=torch.zeros(1, dtype=torch.long),
        _velocity=torch.tensor([1000.]), _current_height_cmd=torch.tensor([.42]))
    for _ in range(25):
        command_method("_update_command")(obj)
    assert obj.delta_curriculum.totals[:2].tolist() == [1., 1.]
    assert obj._steps_since_resample.item() == 0


def test_end_accounting_exhaustive_disjoint_and_idempotent():
    stair = module.HeightStaircase(3, "cpu")
    ids = torch.arange(3)
    stair.sample(ids, torch.full((3,), .72), torch.tensor([True, True, False]), torch.full((3,), .3))
    for _ in range(200):
        stair.advance(torch.full((3,), .42), torch.full((3,), .2), .12, .02)
    failed, _ = stair.finish(torch.tensor([0]), timeout=False)
    assert failed.item()
    stair.finish(torch.tensor([1, 2]), timeout=True)
    metrics = stair.diagnostics()
    assert metrics["stair_run_end_total"] == 2
    assert metrics["stair_run_end_resample"] == metrics["stair_run_end_timeout"] == 1
    assert metrics["stair_run_end_no_advance"] == 2
    before = stair.end_totals.clone()
    stair.finish(ids)
    torch.testing.assert_close(stair.end_totals, before)


def test_short_interruption_is_censored_not_failed_but_still_logged():
    stair = module.HeightStaircase(1, "cpu")
    stair.sample(torch.tensor([0]), torch.tensor([.72]), torch.tensor([True]), torch.tensor([.3]))
    for _ in range(25):
        stair.advance(torch.tensor([.42]), torch.tensor([.2]), .12, .02)
    failed, _ = stair.finish(torch.tensor([0]))
    assert not failed.item()
    assert stair.step_outcomes.tolist() == [0, 0, 1]
    assert stair.end_totals[0].item() == 1


def test_final_reached_is_not_final_stable_success():
    stair = module.HeightStaircase(1, "cpu")
    stair.sample(torch.tensor([0]), torch.tensor([.72]), torch.tensor([True]), torch.tensor([.3]))
    target = torch.tensor([.42])
    for _ in range(100):
        target, _ = stair.advance(target, target.clone(), .12, .02)
    stair.finish(torch.tensor([0]))
    assert stair.end_totals[8].item() == 1  # Reached, but final settled exposure is too short.
    assert stair.end_totals[7].item() == 0


def test_final_success_height_and_delta_bins_and_signed_error():
    stair = module.HeightStaircase(1, "cpu")
    stair.sample(torch.tensor([0]), torch.tensor([.72]), torch.tensor([True]), torch.tensor([.3]))
    target = torch.tensor([.42])
    for _ in range(100):
        target, _ = stair.advance(target, target.clone(), .12, .02)
    for _ in range(175):
        stair.advance(target, target - .02, .12, .02)
    stair.finish(torch.tensor([0]), timeout=True)
    assert stair.end_totals[7].item() == 1
    assert stair.height_bins[3, :3].tolist() == [1, 1, 1]
    assert stair.delta_bins[3].tolist() == [1, 1, 1]
    assert abs((stair.height_bins[3, 5] / stair.height_bins[3, 3]).item() + .02) < .002


def test_terminal_failure_with_short_budget_and_reset_does_not_leak():
    stair = module.HeightStaircase(1, "cpu")
    ids = torch.tensor([0])
    stair.sample(ids, torch.tensor([.72]), torch.tensor([True]), torch.tensor([.3]))
    failed, _ = stair.finish(ids, failures=torch.tensor([True]), timeout=True)
    assert failed.item() and stair.end_totals[3].item() == 1
    assert stair.end_totals[2].item() == 0
    stair.sample(ids, torch.tensor([-.5]), torch.tensor([False]), torch.tensor([.3]))
    for _ in range(100):
        stair.advance(torch.tensor([-.5]), torch.tensor([-.5]), .12, .02)
    stair.finish(ids)
    assert stair.end_totals[0].item() == 1


def test_accounting_checkpoint_version_preserved_but_physics_counters_reset():
    stair = module.HeightStaircase(1, "cpu")
    stair.steps = 150
    restored = module.HeightStaircase(1, "cpu")
    restored.load_state_dict(stair.state_dict())
    assert restored.steps == 150 and restored.end_totals.sum() == 0
    state = stair.state_dict()
    state["version"] = 1
    try:
        restored.load_state_dict(state)
    except ValueError:
        pass
    else:
        raise AssertionError("accepted incompatible accounting version")

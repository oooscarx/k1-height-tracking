import importlib.util
import math
from pathlib import Path
from types import SimpleNamespace

import torch

from test_height_delta_curriculum import HeightDeltaCurriculum, command_method

ROOT = Path(__file__).parents[1]
TASK = ROOT / "source/booster_train/booster_train/tasks/manager_based/height_tracking"
spec = importlib.util.spec_from_file_location("ramp", TASK / "ramp.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def height_reward(error, settled):
    scale = .72 / .92
    return sum(w * math.exp(-(error / (std * scale)) ** 2)
               for w, std in ((4, .5), (8, .3), (16, .2))) + (
                   8 * math.exp(-(error / (.2 * scale)) ** 2) if settled else 0)


def test_old_stair_has_a_reward_drop_for_success_not_for_hovering():
    # This is a height-component counterexample, not a whole-policy return proof.
    hovering = height_reward(.041, True)
    before_success = height_reward(.039, True)
    after_success = height_reward(.039 + .12, False)
    assert before_success > hovering
    assert after_success < hovering * .6


def test_ramp_keeps_final_goal_and_advances_without_tracking_success():
    ramp = module.HeightRamp(1, "cpu")
    ramp.sample(torch.tensor([0]), torch.tensor([.72]), torch.tensor([True]), torch.tensor([.1]), .12)
    previous = .1
    for _ in range(150):
        command = ramp.step(.02).item()
        assert previous <= command + 1e-6
        assert command <= .72 + 1e-6
        assert command - previous <= .0048 + 1e-6
        previous = command
    assert abs(previous - .72) < 1e-6 and ramp.at_final.item()
    # At a fixed clock time, improving tracking does not cause a new target.
    assert height_reward(.039, False) > height_reward(.041, False)


def test_command_integration_protects_reference_negative_and_downward():
    n = 5
    obj = SimpleNamespace(staircase=None, ramp=module.HeightRamp(n, "cpu"),
        delta_curriculum=HeightDeltaCurriculum("cpu"), _target_height=torch.tensor([.72, .72, -.5, .2, .4]),
        _env=SimpleNamespace(step_dt=.02), _steps_since_resample=torch.full((n,), 100.),
        _velocity=torch.full((n,), 1000.), _current_height_cmd=torch.full((n,), .1),
        time_left=torch.full((n,), 4.))
    obj.ramp.sample(torch.arange(n), obj._target_height, torch.tensor([True, False, False, False, False]),
                    torch.full((n,), .1), .12)
    for _ in range(10):
        command_method("_update_command")(obj)
    torch.testing.assert_close(obj._current_height_cmd, torch.tensor([.148, .72, -.5, .2, .4]))
    assert obj._steps_since_resample.tolist() == [0, 100, 100, 100, 100]
    assert obj.time_left.tolist() == [4] * n
    for _ in range(150):
        command_method("_update_command")(obj)
    obj._steps_since_resample += 1
    command_method("_update_command")(obj)
    assert obj._steps_since_resample[0] == 1  # Final-target settle clock can now advance.


def test_resample_final_goal_and_control_rng_unchanged():
    n = 10000
    obj = SimpleNamespace(device="cpu", delta_curriculum=HeightDeltaCurriculum("cpu"),
        progress_sampler=None, staircase=None, ramp=module.HeightRamp(n, "cpu"),
        cfg=SimpleNamespace(standing_ratio=.15, flat_ratio=.2, ranges=SimpleNamespace(height=(-.5, .72)), velocity_range=(1000, 1000)),
        _steps_since_resample=torch.zeros(n), _current_height_cmd=torch.zeros(n),
        _command_origin=torch.zeros(n), _continuous_positive=torch.zeros(n, dtype=torch.bool),
        _reference=torch.arange(n) % 5 == 0, _target_height=torch.zeros(n), _velocity=torch.zeros(n),
        _delta_frontier=torch.zeros(n, dtype=torch.bool), _delta_stage=torch.zeros(n, dtype=torch.long),
        _finish_delta_commands=lambda ids: None, measured_height=torch.full((n,), .1))
    torch.manual_seed(73)
    command_method("_resample_command")(obj, torch.arange(n))
    targets = obj._target_height.clone()
    assert not obj.ramp.active[obj._reference].any()
    assert not obj.ramp.active[targets < 0].any()
    obj.delta_curriculum = None
    torch.manual_seed(73)
    command_method("_resample_command")(obj, torch.arange(n))
    torch.testing.assert_close(obj._target_height, targets)


def test_end_outcomes_count_timeout_when_observed_and_censor_short_commands():
    ramp = module.HeightRamp(3, "cpu")
    ids = torch.arange(3)
    ramp.sample(ids, torch.full((3,), .72), torch.ones(3, dtype=torch.bool), torch.full((3,), .1), .12)
    for _ in range(150):
        ramp.step(.02)
    observed, failed = ramp.finish(ids, torch.tensor([50., 10., 0.]), torch.tensor([1., .2, 0.]),
        torch.tensor([50., 10., 0.]), torch.tensor([False, False, True]), .02, timeout=True)
    assert observed.tolist() == [True, False, False]
    assert failed.tolist() == [False, False, True]
    metrics = ramp.diagnostics()
    assert metrics["ramp_run_ended"] == 3 and metrics["ramp_run_success"] == 1
    assert metrics["ramp_run_censored"] == metrics["ramp_run_terminated"] == 1
    before = ramp.totals.clone()
    ramp.finish(ids, torch.zeros(3), torch.zeros(3), torch.zeros(3), torch.zeros(3, dtype=torch.bool), .02)
    torch.testing.assert_close(ramp.totals, before)


def test_timeout_with_final_observations_updates_curriculum_once():
    n = 2
    ramp = module.HeightRamp(n, "cpu")
    ramp.sample(torch.arange(n), torch.full((n,), .72), torch.ones(n, dtype=torch.bool), torch.full((n,), .1), .12)
    obj = SimpleNamespace(delta_curriculum=HeightDeltaCurriculum("cpu"), staircase=None, ramp=ramp,
        _env=SimpleNamespace(step_dt=.02), _delta_frontier=torch.ones(n, dtype=torch.bool),
        _delta_stage=torch.zeros(n, dtype=torch.long), _delta_count=torch.tensor([50., 10.]),
        _delta_error=torch.tensor([1., .2]), _delta_success=torch.tensor([50., 10.]))
    command_method("_finish_delta_commands")(obj, torch.arange(n), censored=True)
    assert obj.delta_curriculum.totals[0] == obj.delta_curriculum.totals[1] == 1
    command_method("_finish_delta_commands")(obj, torch.arange(n))
    assert obj.delta_curriculum.totals[0] == 1


def test_marker_restore_does_not_restore_physics_state():
    ramp = module.HeightRamp(1, "cpu")
    ramp.steps = 200
    restored = module.HeightRamp(1, "cpu")
    restored.load_state_dict(ramp.state_dict())
    assert restored.steps == 200 and not restored.active.any()

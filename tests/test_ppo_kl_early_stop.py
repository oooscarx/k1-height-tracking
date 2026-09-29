import importlib.util
from pathlib import Path

import pytest
import torch
from rsl_rl.modules import ActorCritic

ROOT = Path(__file__).resolve().parents[1] / "third_party/wbc_agile_rsl_rl/rsl_rl"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PPO = load("kl_ppo_test", ROOT / "algorithms/ppo.py").PPO
Runner = load("kl_runner_test", ROOT / "runners/on_policy_runner.py").OnPolicyRunner


def test_adaptive_schedule_floor_respects_configured_maximum():
    source = (ROOT / "algorithms/ppo.py").read_text()
    assert "minimum_learning_rate = min(1e-5, self.max_learning_rate)" in source


def make(factor=1.5, diagnostic_updates=0, regularization=False):
    torch.manual_seed(12)
    policy = ActorCritic(3, 4, 2, actor_hidden_dims=[8], critic_hidden_dims=[8])
    alg = PPO(policy, device="cpu", num_learning_epochs=5, num_mini_batches=4,
              schedule="adaptive", desired_kl=0.01, kl_early_stop_factor=factor,
              kl_early_stop_diagnostic_updates=diagnostic_updates,
              l2c2_cfg={"lambda_actor": 1.0, "lambda_critic": 0.1} if regularization else None,
              symmetry_cfg={"use_data_augmentation": True, "use_mirror_loss": False,
                            "data_augmentation_func": mirror, "_env": None} if regularization else None)
    alg.init_storage("rl", 16, 4, [3], [4], [2])
    return alg


def mirror(obs=None, actions=None, env=None, obs_type=None):
    return (None if obs is None else torch.cat((obs, -obs)),
            None if actions is None else torch.cat((actions, -actions)))


def fill(alg, high_kl=False):
    torch.manual_seed(34)
    with torch.inference_mode():
        for _ in range(4):
            alg.act(torch.randn(16, 3), torch.randn(16, 4))
            alg.process_env_step(torch.randn(16), torch.zeros(16, dtype=torch.bool), {})
        alg.compute_returns(torch.randn(16, 4))
    if high_kl:
        alg.storage.mu.add_(1.0)


def test_disabled_and_diagnostic_updates_have_identical_parameters():
    baseline = make(factor=None)
    diagnostic = make(diagnostic_updates=50)
    fill(baseline)
    torch.manual_seed(56)
    original = baseline.update()
    fill(diagnostic)
    torch.manual_seed(56)
    observed = diagnostic.update()
    for key, value in baseline.policy.state_dict().items():
        assert torch.equal(value, diagnostic.policy.state_dict()[key]), key
    for key in original:
        assert observed[key] == pytest.approx(original[key])
    assert observed["kl_optimizer_steps"] == 20
    assert observed["kl_early_stop_active"] == 0


def test_high_kl_stops_after_one_epoch_and_uses_actual_loss_denominator():
    alg = make()
    fill(alg, high_kl=True)
    alg._value_error_loss = lambda value, target: value * 0 + 7
    stats = alg.update()
    assert stats["kl_optimizer_steps"] == 4
    assert stats["kl_early_stopped"] == 1
    assert stats["value_function"] == pytest.approx(7)
    assert stats["minibatches_epoch_1"] == 4
    assert stats["minibatches_epoch_2"] == 0
    assert alg.storage.step == 0
    assert alg.kl_update_count == 1


def test_diagnostic_phase_observes_would_stop_without_stopping():
    alg = make(diagnostic_updates=1)
    fill(alg, high_kl=True)
    first = alg.update()
    assert first["kl_optimizer_steps"] == 20
    assert first["kl_would_stop_after"] == 4
    assert first["kl_early_stopped"] == 0
    fill(alg, high_kl=True)
    second = alg.update()
    assert second["kl_optimizer_steps"] == 4
    assert second["kl_early_stop_active"] == 1


def test_below_budget_uses_all_epochs():
    alg = make(factor=1e6)
    fill(alg)
    stats = alg.update()
    assert stats["kl_optimizer_steps"] == 20
    assert stats["kl_early_stopped"] == 0


def test_early_stop_preserves_symmetry_and_l2c2_paths():
    alg = make(regularization=True)
    fill(alg, high_kl=True)
    stats = alg.update()
    assert stats["kl_optimizer_steps"] == 4
    assert torch.isfinite(torch.tensor(list(stats.values()))).all()
    assert stats["l2c2_actor"] >= 0
    assert stats["l2c2_critic"] >= 0


@pytest.mark.parametrize("factor", [0, -1, float("nan"), float("inf")])
def test_invalid_factor_rejected(factor):
    with pytest.raises(ValueError):
        make(factor=factor)


def test_checkpoint_preserves_diagnostic_progress(tmp_path):
    alg = make(diagnostic_updates=50)
    alg.kl_update_count = 61
    runner = object.__new__(Runner)
    runner.alg = alg
    runner.current_learning_iteration = 123
    runner.empirical_normalization = False
    runner.logger_type = "tensorboard"
    runner.disable_logs = True
    checkpoint = str(tmp_path / "model.pt")
    runner.save(checkpoint, infos={"other": 42})
    runner.alg = make(diagnostic_updates=50)
    assert runner.load(checkpoint) == {"other": 42}
    assert runner.alg.kl_update_count == 61
    assert runner.current_learning_iteration == 123

    state = torch.load(checkpoint, weights_only=False)
    del state["kl_update_count"]
    torch.save(state, checkpoint)
    runner.load(checkpoint)
    assert runner.alg.kl_update_count == 0

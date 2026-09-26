from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/rsl_rl/supervise_k1_height_extension.py"
SPEC = importlib.util.spec_from_file_location("supervise_k1_height_extension", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
SUPERVISOR = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SUPERVISOR
SPEC.loader.exec_module(SUPERVISOR)


def test_next_end_reaches_initial_target_then_extends_until_zero() -> None:
    assert SUPERVISOR._next_end(100_000, 700_000, 100_000, 0.7) == 700_000
    assert SUPERVISOR._next_end(700_000, 700_000, 100_000, 0.01) == 800_000
    assert SUPERVISOR._next_end(700_000, 700_000, 100_000, 0.0) is None


def test_training_command_uses_global_target_and_curriculum_state(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    checkpoint = root / "run/model_99999.pt"
    cache = root / "cache.pt"
    command = SUPERVISOR._training_command(
        root,
        checkpoint,
        99_999,
        700_000,
        0.42,
        0.103,
        16_384,
        cache,
    )

    assert command[command.index("--max_iterations") + 1] == "600001"
    assert command[command.index("--num_envs") + 1] == "16384"
    assert command[command.index("--wbc_initial_lift_scale") + 1] == "0.42"
    assert command[command.index("--wbc_initial_lift_ema") + 1] == "0.103"
    assert command[command.index("--resume_path") + 1] == str(checkpoint)


def test_training_command_can_resume_distributed_training(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    checkpoint = root / "run/model_700000.pt"
    cache = root / "cache.pt"
    command = SUPERVISOR._training_command(
        root,
        checkpoint,
        700_000,
        800_000,
        0.02,
        0.11,
        16_384,
        cache,
        distributed=True,
        num_gpus=2,
        num_envs_per_gpu=8192,
    )

    assert command[:4] == [
        "env",
        "NUM_GPUS=2",
        "NUM_ENVS_PER_GPU=8192",
        "MAX_ITERATIONS=100000",
    ]
    assert command[4] == str(root / "scripts/rsl_rl/train_k1_height_tracking_distributed.sh")
    assert command[command.index("--resume_path") + 1] == str(checkpoint)


def test_checkpoint_iteration_rejects_unrelated_files() -> None:
    assert SUPERVISOR._checkpoint_iteration(Path("model_40000.pt")) == 40_000
    try:
        SUPERVISOR._checkpoint_iteration(Path("events.out.tfevents"))
    except ValueError:
        pass
    else:
        raise AssertionError("unrelated file must not be accepted as a checkpoint")

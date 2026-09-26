import ast
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
COMMANDS = (
    ROOT
    / "source"
    / "booster_train"
    / "booster_train"
    / "tasks"
    / "manager_based"
    / "height_tracking"
    / "commands.py"
)


def command_method(name: str):
    tree = ast.parse(COMMANDS.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SmoothHeightCommand")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[method], type_ignores=[]), "commands.py", "exec"), namespace)
    return namespace[name]


def test_governed_height_metrics_ignore_nonfinite_terminated_environment() -> None:
    count = 2
    obj = SimpleNamespace(
        measured_height=torch.tensor([0.5, float("inf")]),
        _current_height_cmd=torch.tensor([0.4, 0.4]),
        _steps_since_resample=torch.zeros(count),
        settled=torch.ones(count, dtype=torch.bool),
        governor=SimpleNamespace(
            command_age=torch.ones(count),
            at_final=torch.ones(count, dtype=torch.bool),
            elapsed=torch.zeros(count),
        ),
        _goal_active=torch.zeros(count, dtype=torch.bool),
        _final_count=torch.zeros(count),
        _final_good=torch.zeros(count),
        _first_success_s=torch.full((count,), -1.0),
        _episode_error_sum=torch.zeros(count),
        _episode_step_count=torch.zeros(count),
        _episode_high_error_sum=torch.zeros(count),
        _episode_high_success_sum=torch.zeros(count),
        _episode_high_step_count=torch.zeros(count),
        metrics={
            "height_error": torch.zeros(count),
            "high_height_error": torch.zeros(count),
            "high_height_success": torch.zeros(count),
        },
        cfg=SimpleNamespace(
            settle_time_s=0.1,
            high_height_threshold=0.3,
            success_error_threshold=0.08,
        ),
        _env=SimpleNamespace(step_dt=0.02),
    )

    command_method("_update_metrics")(obj)

    assert torch.isfinite(obj.metrics["height_error"]).all()
    assert torch.isfinite(obj.metrics["high_height_error"]).all()
    torch.testing.assert_close(obj.metrics["height_error"], torch.tensor([0.1, 0.0]))
    torch.testing.assert_close(obj.metrics["high_height_error"], torch.tensor([0.1, 0.0]))

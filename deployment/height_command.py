"""JSON-lines adapter for the shared governor; does not send motor commands.

Input per policy step: {"height": 0.3, "goal": 0.72, "dt": 0.02}.
Repeated goals do not restart a request. Set replan=true to explicitly start a
new request from measured height after a disturbance. Heights must use the same
terrain-relative tracked point as training, not the root or head height.
"""

import importlib.util
import json
import math
from pathlib import Path
import sys

import torch

path = Path(__file__).resolve().parents[1] / "source/booster_train/booster_train/tasks/manager_based/height_tracking/governor.py"
spec = importlib.util.spec_from_file_location("shared_height_governor", path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DeploymentHeightCommand:
    def __init__(self):
        self.governor = module.HeightCommandGovernor(1)
        self.last_goal = None

    def update(self, height, goal, dt=.02, replan=False):
        if not math.isfinite(goal) or not -.5 <= goal <= .72:
            raise ValueError("goal must be finite and in [-0.5, 0.72] m")
        if not math.isfinite(height) or not math.isfinite(dt) or dt <= 0:
            raise ValueError("height and dt must be finite; dt must be positive")
        if self.last_goal != goal or replan:
            self.governor.request(torch.tensor([0]), torch.tensor([goal]), torch.tensor([height]),
                                  torch.tensor([True]), torch.tensor([4.]))
            self.last_goal = goal
        self.governor.step(dt)
        return float(self.governor.command[0])


if __name__ == "__main__":
    adapter = DeploymentHeightCommand()
    for line in sys.stdin:
        data = json.loads(line)
        value = adapter.update(**data)
        print(json.dumps({"height_command": value}), flush=True)

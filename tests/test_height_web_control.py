from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/rsl_rl"))

from height_web_control import HeightWebControlServer  # noqa: E402


@pytest.fixture
def server():
    instance = HeightWebControlServer("127.0.0.1", 0)
    instance.start()
    try:
        yield instance, f"http://127.0.0.1:{instance._httpd.server_port}"
    finally:
        instance.close()


def _post(url: str, payload: dict):
    request = urllib.request.Request(
        url + "/api/command",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request) as response:
        return json.load(response)


def test_serves_ui_and_state(server):
    instance, url = server
    instance.update_state(measured_height=0.61)
    with urllib.request.urlopen(url + "/") as response:
        assert b"Height Control" in response.read()
    with urllib.request.urlopen(url + "/api/state") as response:
        assert json.load(response)["measured_height"] == pytest.approx(0.61)


def test_clamps_height_and_drains_commands(server):
    instance, url = server
    reply = _post(url, {"action": "set_height", "value": 4.0})
    assert reply["command"]["value"] == pytest.approx(0.72)
    assert instance.drain_commands() == [{"action": "set_height", "value": 0.72}]
    assert instance.drain_commands() == []


def test_rejects_nonfinite_and_unknown_commands(server):
    _, url = server
    for payload in ({"action": "set_height", "value": "nan"}, {"action": "dance"}):
        with pytest.raises(urllib.error.HTTPError) as error:
            _post(url, payload)
        assert error.value.code == 400


def test_command_term_has_manual_override_without_changing_default_sampling():
    source = (
        ROOT / "source/booster_train/booster_train/tasks/manager_based/height_tracking/commands.py"
    ).read_text()
    assert "def set_manual_height" in source
    assert "def clear_manual_height" in source
    assert "self._manual_mask = torch.zeros" in source
    assert "self._apply_manual_height(env_ids)" in source

"""Small local HTTP bridge for interactive Isaac Lab playback."""

from __future__ import annotations

import io
import json
import math
import threading
from collections import deque
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any


class HeightWebControlServer:
    """Serve the control UI and exchange commands with the simulation thread."""

    def __init__(self, host: str, port: int, static_dir: Path | None = None) -> None:
        self.host = host
        self.port = port
        self.static_dir = static_dir or Path(__file__).resolve().parents[2] / "web" / "height_control"
        self._lock = threading.Lock()
        self._commands: deque[dict[str, Any]] = deque()
        self._frame: bytes | None = None
        self._state: dict[str, Any] = {
            "connected": True,
            "target_height": 0.0,
            "measured_height": 0.0,
            "height_error": 0.0,
            "minimum_height": -0.5,
            "maximum_height": 0.72,
            "terrain_level": 0,
            "lift": 0.0,
            "domain_randomization": True,
            "external_disturbances": True,
            "policy_hz": 50.0,
            "frame_hz": 0.0,
        }
        self._httpd = ThreadingHTTPServer((host, port), self._handler_type())
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="height-web-control", daemon=True)

    def _handler_type(self):
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self) -> None:  # noqa: N802
                if self.path == "/api/state":
                    self._send_json(owner.snapshot())
                    return
                if self.path.startswith("/frame.jpg"):
                    frame = owner.frame()
                    if frame is None:
                        self.send_error(HTTPStatus.SERVICE_UNAVAILABLE, "frame not ready")
                        return
                    self._send_bytes(frame, "image/jpeg", cache=False)
                    return
                route = self.path.split("?", 1)[0]
                files = {
                    "/": ("index.html", "text/html; charset=utf-8"),
                    "/index.html": ("index.html", "text/html; charset=utf-8"),
                    "/styles.css": ("styles.css", "text/css; charset=utf-8"),
                    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
                }
                item = files.get(route)
                if item is None:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                filename, content_type = item
                try:
                    content = (owner.static_dir / filename).read_bytes()
                except OSError:
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                self._send_bytes(content, content_type, cache=True)

            def do_POST(self) -> None:  # noqa: N802
                if self.path != "/api/command":
                    self.send_error(HTTPStatus.NOT_FOUND)
                    return
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                    if length <= 0 or length > 8192:
                        raise ValueError("invalid request size")
                    payload = json.loads(self.rfile.read(length))
                    accepted = owner.submit(payload)
                except (ValueError, TypeError, json.JSONDecodeError) as exc:
                    self._send_json({"ok": False, "error": str(exc)}, HTTPStatus.BAD_REQUEST)
                    return
                self._send_json({"ok": True, "command": accepted})

            def _send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
                self._send_bytes(json.dumps(value).encode("utf-8"), "application/json", status, cache=False)

            def _send_bytes(
                self,
                value: bytes,
                content_type: str,
                status: HTTPStatus = HTTPStatus.OK,
                cache: bool = False,
            ) -> None:
                self.send_response(status)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(value)))
                self.send_header("Cache-Control", "public, max-age=3600" if cache else "no-store")
                self.end_headers()
                self.wfile.write(value)

            def log_message(self, _format: str, *_args: Any) -> None:
                return

        return Handler

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        self._thread.join(timeout=2.0)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return dict(self._state)

    def update_state(self, **values: Any) -> None:
        with self._lock:
            self._state.update(values)

    def frame(self) -> bytes | None:
        with self._lock:
            return self._frame

    def update_frame(self, frame: Any, quality: int = 82) -> None:
        import numpy as np
        from PIL import Image

        if hasattr(frame, "detach"):
            frame = frame.detach().cpu().numpy()
        array = np.asarray(frame)
        if array.ndim == 4:
            array = array[0]
        if array.ndim != 3 or array.shape[-1] not in (3, 4):
            raise ValueError(f"expected RGB(A) frame, got shape {array.shape}")
        if array.dtype != np.uint8:
            upper = float(np.nanmax(array)) if array.size else 0.0
            if upper <= 1.0:
                array = array * 255.0
            array = np.clip(array, 0.0, 255.0).astype(np.uint8)
        image = Image.fromarray(array[..., :3], mode="RGB")
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=quality, optimize=True)
        with self._lock:
            self._frame = output.getvalue()

    def submit(self, payload: Any) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise ValueError("command must be a JSON object")
        action = payload.get("action")
        if action == "set_height":
            value = self._finite(payload.get("value"), "height")
            with self._lock:
                value = min(max(value, self._state["minimum_height"]), self._state["maximum_height"])
                command = {"action": action, "value": value}
                self._state["target_height"] = value
                self._commands.append(command)
            return command
        if action == "reset":
            command = {"action": action}
        elif action == "push":
            command = {
                "action": action,
                "x": min(max(self._finite(payload.get("x", 0.0), "push x"), -2.0), 2.0),
                "y": min(max(self._finite(payload.get("y", 0.0), "push y"), -2.0), 2.0),
                "yaw": min(max(self._finite(payload.get("yaw", 0.0), "push yaw"), -3.0), 3.0),
            }
        else:
            raise ValueError(f"unsupported action: {action!r}")
        with self._lock:
            self._commands.append(command)
        return command

    def drain_commands(self) -> list[dict[str, Any]]:
        with self._lock:
            commands = list(self._commands)
            self._commands.clear()
        return commands

    @staticmethod
    def _finite(value: Any, name: str) -> float:
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        return value

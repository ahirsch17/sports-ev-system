"""Browser heartbeat file used by the dashboard runner for idle shutdown."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HEARTBEAT_INTERVAL_SECONDS = 10
DEFAULT_IDLE_SHUTDOWN_SECONDS = 45
TAB_CLOSE_SHUTDOWN_SECONDS = 8


def heartbeat_path(port: int) -> Path:
    base = Path(os.environ.get("TEMP") or os.environ.get("TMP") or "/tmp")
    return base / f"sports_ev_dashboard_{port}.heartbeat"


def bye_path(port: int) -> Path:
    return heartbeat_path(port).with_suffix(".bye")


def focus_path(port: int) -> Path:
    return heartbeat_path(port).with_suffix(".focus")


def heartbeat_listen_port(streamlit_port: int) -> int:
    raw = os.environ.get("SPORTS_EV_HEARTBEAT_PORT")
    if raw:
        try:
            return int(raw)
        except ValueError:
            pass
    return streamlit_port + 1


def dashboard_port() -> int:
    for key in ("SPORTS_EV_DASHBOARD_PORT", "STREAMLIT_SERVER_PORT"):
        raw = os.environ.get(key)
        if raw:
            try:
                return int(raw)
            except ValueError:
                continue
    return 8501


def touch_heartbeat(port: int | None = None) -> None:
    port = port if port is not None else dashboard_port()
    path = heartbeat_path(port)
    path.write_text(f"{time.time():.6f}\n", encoding="utf-8")
    try:
        bye_path(port).unlink(missing_ok=True)
    except OSError:
        pass


def mark_tab_closed(port: int | None = None) -> None:
    """Browser tab/window closing — runner should exit quickly."""
    port = port if port is not None else dashboard_port()
    bye_path(port).write_text(f"{time.time():.6f}\n", encoding="utf-8")


def tab_close_requested(port: int) -> bool:
    return bye_path(port).exists()


def seconds_since_heartbeat(port: int) -> float | None:
    path = heartbeat_path(port)
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip()
        last = float(raw.splitlines()[0])
    except (OSError, ValueError):
        return None
    return max(0.0, time.time() - last)


def request_focus(port: int | None = None) -> float:
    port = port if port is not None else dashboard_port()
    stamp = time.time()
    focus_path(port).write_text(f"{stamp:.6f}\n", encoding="utf-8")
    return stamp


def focus_requested_at(port: int | None = None) -> float:
    port = port if port is not None else dashboard_port()
    path = focus_path(port)
    if not path.exists():
        return 0.0
    try:
        return float(path.read_text(encoding="utf-8").strip().splitlines()[0])
    except (OSError, ValueError):
        return 0.0


def clear_heartbeat(port: int) -> None:
    for path in (heartbeat_path(port), bye_path(port), focus_path(port)):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def count_local_connections(port: int) -> int:
    """Established loopback connections to the Streamlit port (0 after tab close)."""
    try:
        proc = subprocess.run(
            ["netstat", "-an"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return 0

    needle = f":{port}"
    count = 0
    for line in proc.stdout.splitlines():
        if "ESTABLISHED" not in line.upper():
            continue
        if needle not in line:
            continue
        if "127.0.0.1" in line or "[::1]" in line or "::1" in line:
            count += 1
    return count


class _HeartbeatHandler(BaseHTTPRequestHandler):
    """Accept /ping, /bye, /focus from browser JS — not from Streamlit Python reruns."""

    def log_message(self, _format: str, *_args) -> None:
        return

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Cache-Control", "no-store")

    def _handle_ping(self) -> None:
        touch_heartbeat(self.server.streamlit_port)  # type: ignore[attr-defined]
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _handle_bye(self) -> None:
        mark_tab_closed(self.server.streamlit_port)  # type: ignore[attr-defined]
        self.send_response(204)
        self._cors()
        self.end_headers()

    def _handle_focus(self) -> None:
        port = self.server.streamlit_port  # type: ignore[attr-defined]
        if self.command == "POST":
            stamp = request_focus(port)
        else:
            stamp = focus_requested_at(port)
        body = f'{{"at":{stamp:.6f}}}'.encode("utf-8")
        self.send_response(200)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        if self.path.startswith("/ping"):
            self._handle_ping()
            return
        if self.path.startswith("/bye"):
            self._handle_bye()
            return
        if self.path.startswith("/focus"):
            self._handle_focus()
            return
        self.send_response(404)
        self._cors()
        self.end_headers()

    def do_POST(self) -> None:
        if self.path.startswith("/bye"):
            self._handle_bye()
            return
        if self.path.startswith("/ping"):
            self._handle_ping()
            return
        if self.path.startswith("/focus"):
            self._handle_focus()
            return
        self.send_response(404)
        self._cors()
        self.end_headers()


class HeartbeatServer:
    """Tiny localhost server; browser pings here so tab-close stops heartbeats."""

    def __init__(self, streamlit_port: int, listen_port: int | None = None):
        self.streamlit_port = streamlit_port
        self.listen_port = listen_port if listen_port is not None else heartbeat_listen_port(streamlit_port)
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._httpd is not None:
            return
        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.listen_port), _HeartbeatHandler)
        self._httpd.streamlit_port = self.streamlit_port  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if self._httpd is None:
            return
        self._httpd.shutdown()
        self._httpd.server_close()
        self._httpd = None
        if self._thread is not None:
            self._thread.join(timeout=2)
            self._thread = None

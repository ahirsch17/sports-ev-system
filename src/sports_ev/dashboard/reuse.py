"""If SportsPredictor is already up, focus it instead of starting a second instance."""

from __future__ import annotations

import socket
import subprocess
import sys
import urllib.error
import urllib.request

from sports_ev.dashboard.heartbeat import (
    HEARTBEAT_INTERVAL_SECONDS,
    count_local_connections,
    heartbeat_listen_port,
    request_focus,
    seconds_since_heartbeat,
)


def _port_listening(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.4):
            return True
    except OSError:
        return False


def _url_ready(url: str) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=1.5) as resp:
            return 200 <= getattr(resp, "status", 200) < 500
    except urllib.error.HTTPError as exc:
        return 400 <= exc.code < 600
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def _focus_window_by_title(*substrings: str) -> bool:
    if sys.platform != "win32" or not substrings:
        return False
    checks = " -or ".join(f"$_.MainWindowTitle -like '*{s}*'" for s in substrings)
    script = (
        "Add-Type @'\n"
        "using System;\n"
        "using System.Runtime.InteropServices;\n"
        "public class W {\n"
        "  [DllImport(\"user32.dll\")] public static extern bool SetForegroundWindow(IntPtr h);\n"
        "  [DllImport(\"user32.dll\")] public static extern bool ShowWindow(IntPtr h, int n);\n"
        "}\n"
        "'@\n"
        f"$p = Get-Process | Where-Object {{ $_.MainWindowTitle -and ({checks}) }} "
        "| Select-Object -First 1\n"
        "if ($p) { [void][W]::ShowWindow($p.MainWindowHandle, 9); "
        "[void][W]::SetForegroundWindow($p.MainWindowHandle); exit 0 }\n"
        "exit 1\n"
    )
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", script],
            capture_output=True,
            timeout=5,
            check=False,
        )
        return completed.returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def _open_browser(url: str) -> None:
    if sys.platform == "win32":
        subprocess.Popen(f'start "" "{url}"', shell=True)
    else:
        import webbrowser

        webbrowser.open(url)


def _tab_looks_open(port: int) -> bool:
    if count_local_connections(port) > 0:
        return True
    age = seconds_since_heartbeat(port)
    return age is not None and age <= max(45, HEARTBEAT_INTERVAL_SECONDS * 4)


def try_reuse_running(*, port: int) -> bool:
    if not _url_ready(f"http://127.0.0.1:{port}") and not _port_listening(port):
        return False

    print(f"SportsPredictor already running on http://localhost:{port}")
    request_focus(port)
    hb = heartbeat_listen_port(port)
    try:
        req = urllib.request.Request(
            f"http://127.0.0.1:{hb}/focus",
            data=b"",
            method="POST",
        )
        urllib.request.urlopen(req, timeout=1.2)
    except (urllib.error.URLError, TimeoutError, OSError):
        pass

    _focus_window_by_title(
        "SportsPredictor",
        "Paper Bets",
        f"localhost:{port}",
        f"127.0.0.1:{port}",
    )

    if _tab_looks_open(port):
        print("Focusing the open tab (no second instance / no new tab).")
        return True

    print("Server is up but no live tab - opening browser.")
    _open_browser(f"http://localhost:{port}")
    return True

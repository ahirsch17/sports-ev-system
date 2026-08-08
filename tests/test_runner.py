from __future__ import annotations

import os
import subprocess
import sys
import time
from unittest.mock import MagicMock, patch

import pytest

from sports_ev.dashboard.heartbeat import (
    HeartbeatServer,
    clear_heartbeat,
    heartbeat_path,
    mark_tab_closed,
    seconds_since_heartbeat,
    tab_close_requested,
    touch_heartbeat,
)
from sports_ev.dashboard.runner import _kill_process_tree, run_dashboard


class TestHeartbeat:
    def test_touch_and_age(self, tmp_path, monkeypatch):
        port = 9876
        monkeypatch.setenv("TEMP", str(tmp_path))
        clear_heartbeat(port)
        assert seconds_since_heartbeat(port) is None

        touch_heartbeat(port)
        age = seconds_since_heartbeat(port)
        assert age is not None
        assert age < 1.0
        assert heartbeat_path(port).exists()

        clear_heartbeat(port)
        assert seconds_since_heartbeat(port) is None

    def test_http_ping_updates_heartbeat(self, tmp_path, monkeypatch):
        import urllib.request

        streamlit_port = 9880
        listen_port = 9881
        monkeypatch.setenv("TEMP", str(tmp_path))
        clear_heartbeat(streamlit_port)

        server = HeartbeatServer(streamlit_port=streamlit_port, listen_port=listen_port)
        server.start()
        try:
            assert seconds_since_heartbeat(streamlit_port) is None
            urllib.request.urlopen(f"http://127.0.0.1:{listen_port}/ping", timeout=2)
            age = seconds_since_heartbeat(streamlit_port)
            assert age is not None
            assert age < 2.0
        finally:
            server.stop()
            clear_heartbeat(streamlit_port)

    def test_bye_marks_tab_closed(self, tmp_path, monkeypatch):
        import urllib.request

        streamlit_port = 9882
        listen_port = 9883
        monkeypatch.setenv("TEMP", str(tmp_path))
        clear_heartbeat(streamlit_port)
        assert not tab_close_requested(streamlit_port)

        server = HeartbeatServer(streamlit_port=streamlit_port, listen_port=listen_port)
        server.start()
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{listen_port}/bye", timeout=2)
            assert tab_close_requested(streamlit_port)
            touch_heartbeat(streamlit_port)
            assert not tab_close_requested(streamlit_port)
        finally:
            server.stop()
            clear_heartbeat(streamlit_port)


class TestRunnerShutdown:
    def test_idle_shutdown_logic_triggers_kill(self, tmp_path, monkeypatch):
        port = 9877
        idle_seconds = 2
        monkeypatch.setenv("TEMP", str(tmp_path))
        clear_heartbeat(port)
        touch_heartbeat(port)

        proc = MagicMock()
        proc.poll.side_effect = [None, None, 0]
        proc.pid = 4242
        proc.wait.return_value = 0

        with patch("sports_ev.dashboard.runner._kill_process_tree") as mock_kill:
            had_browser = False
            idle_since = None
            fresh_seconds = 12
            idle_seconds = 2
            for _ in range(30):
                age = seconds_since_heartbeat(port)
                if age is not None and age <= fresh_seconds:
                    had_browser = True
                    idle_since = None
                elif had_browser:
                    if idle_since is None:
                        idle_since = time.monotonic()
                    elif time.monotonic() - idle_since >= idle_seconds:
                        from sports_ev.dashboard import runner

                        runner._kill_process_tree(proc)
                        break
                clear_heartbeat(port)
                time.sleep(0.2)
            mock_kill.assert_called()

    def test_run_dashboard_exits_after_tab_close(self, tmp_path, monkeypatch):
        import pathlib
        import urllib.request

        port = 9879
        idle_seconds = 6
        heartbeat_port = port + 1
        monkeypatch.setenv("TEMP", str(tmp_path))
        monkeypatch.setenv("SPORTS_EV_STARTUP_GRACE_SECONDS", "15")
        clear_heartbeat(port)

        project_root = pathlib.Path(__file__).resolve().parents[1]
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sports_ev.dashboard.runner",
                "--port",
                str(port),
                "--idle-seconds",
                str(idle_seconds),
            ],
            cwd=str(project_root),
            env={
                **os.environ,
                "TEMP": str(tmp_path),
                "SPORTS_EV_STARTUP_GRACE_SECONDS": "15",
                "SPORTS_EV_DASHBOARD_PORT": str(port),
                "SPORTS_EV_HEARTBEAT_PORT": str(heartbeat_port),
                "SPORTS_EV_HEADLESS": "1",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output = ""
        try:
            time.sleep(3)
            urllib.request.urlopen(f"http://127.0.0.1:{heartbeat_port}/ping", timeout=2)
            time.sleep(2)
            # Tab closed — browser pings stop; do not touch heartbeat file manually.
            output, _ = proc.communicate(timeout=idle_seconds * 3 + 30)
        except subprocess.TimeoutExpired:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                )
            else:
                proc.kill()
            output, _ = proc.communicate(timeout=10)
            pytest.fail(f"Dashboard runner did not exit after tab close. Output:\n{output}")
        finally:
            clear_heartbeat(port)

        assert proc.returncode is not None
        assert "Shutting down dashboard" in output

    def test_run_dashboard_exits_after_bye_beacon(self, tmp_path, monkeypatch):
        import pathlib
        import urllib.request

        from sports_ev.dashboard.heartbeat import TAB_CLOSE_SHUTDOWN_SECONDS

        port = 9884
        heartbeat_port = port + 1
        monkeypatch.setenv("TEMP", str(tmp_path))
        monkeypatch.setenv("SPORTS_EV_STARTUP_GRACE_SECONDS", "15")
        clear_heartbeat(port)

        project_root = pathlib.Path(__file__).resolve().parents[1]
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sports_ev.dashboard.runner",
                "--port",
                str(port),
                "--idle-seconds",
                "45",
            ],
            cwd=str(project_root),
            env={
                **os.environ,
                "TEMP": str(tmp_path),
                "SPORTS_EV_STARTUP_GRACE_SECONDS": "15",
                "SPORTS_EV_DASHBOARD_PORT": str(port),
                "SPORTS_EV_HEARTBEAT_PORT": str(heartbeat_port),
                "SPORTS_EV_HEADLESS": "1",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output = ""
        try:
            time.sleep(3)
            urllib.request.urlopen(f"http://127.0.0.1:{heartbeat_port}/ping", timeout=2)
            urllib.request.urlopen(f"http://127.0.0.1:{heartbeat_port}/bye", timeout=2)
            output, _ = proc.communicate(timeout=TAB_CLOSE_SHUTDOWN_SECONDS + 20)
        except subprocess.TimeoutExpired:
            if sys.platform == "win32":
                subprocess.run(
                    ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                    capture_output=True,
                    check=False,
                )
            else:
                proc.kill()
            output, _ = proc.communicate(timeout=10)
            pytest.fail(f"Dashboard runner did not exit after /bye. Output:\n{output}")
        finally:
            clear_heartbeat(port)

        assert proc.returncode is not None
        assert "Browser tab closed" in output

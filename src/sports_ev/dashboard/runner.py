"""Launch Streamlit and exit when the browser tab closes."""
from __future__ import annotations
import argparse
import atexit
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from sports_ev.dashboard.reuse import try_reuse_running
from sports_ev.dashboard.heartbeat import (
    DEFAULT_IDLE_SHUTDOWN_SECONDS,
    HEARTBEAT_INTERVAL_SECONDS,
    TAB_CLOSE_SHUTDOWN_SECONDS,
    HeartbeatServer,
    clear_heartbeat,
    count_local_connections,
    heartbeat_listen_port,
    seconds_since_heartbeat,
    tab_close_requested,
)
DEFAULT_PORT = 8501
POLL_SECONDS = 2
DEFAULT_STARTUP_GRACE_SECONDS = 120
def _startup_grace_seconds() -> int:
    raw = os.environ.get("SPORTS_EV_STARTUP_GRACE_SECONDS")
    if raw:
        try:
            return max(5, int(raw))
        except ValueError:
            pass
    return DEFAULT_STARTUP_GRACE_SECONDS
def _kill_process_tree(proc: subprocess.Popen[bytes]) -> None:
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            capture_output=True,
            check=False,
        )
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
        return
    proc.terminate()
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=5)
def _browser_active(port: int, *, fresh_seconds: float) -> bool:
    """True while the dashboard tab appears open (heartbeat and/or TCP)."""
    age = seconds_since_heartbeat(port)
    if age is not None and age <= fresh_seconds:
        return True
    return count_local_connections(port) > 0
def run_dashboard(
    *,
    port: int = DEFAULT_PORT,
    idle_shutdown_seconds: int = DEFAULT_IDLE_SHUTDOWN_SECONDS,
) -> int:
    if try_reuse_running(port=port):
        return 0
    app_path = Path(__file__).resolve().parent / "app.py"
    clear_heartbeat(port)
    heartbeat_port = heartbeat_listen_port(port)
    hb_server = HeartbeatServer(streamlit_port=port, listen_port=heartbeat_port)
    hb_server.start()
    env = os.environ.copy()
    env["SPORTS_EV_DASHBOARD_PORT"] = str(port)
    env["SPORTS_EV_HEARTBEAT_PORT"] = str(heartbeat_port)
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        f"--server.port={port}",
        "--browser.gatherUsageStats=false",
    ]
    headless = os.environ.get("SPORTS_EV_HEADLESS", "").lower() in ("1", "true", "yes")
    cmd.append("--server.headless=true" if headless else "--server.headless=false")
    popen_kwargs: dict = {
        "env": env,
        # Avoid stdout pipe deadlocks if the parent console/reader stalls;
        # tab-close shutdown must not wait on Streamlit log writes.
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    proc = subprocess.Popen(cmd, **popen_kwargs)
    started_at = time.monotonic()
    had_browser = False
    idle_since: float | None = None
    def _shutdown(*_args) -> None:
        _kill_process_tree(proc)
    if sys.platform != "win32":
        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGTERM, _shutdown)
    else:
        signal.signal(signal.SIGINT, _shutdown)
        signal.signal(signal.SIGBREAK, _shutdown)
    atexit.register(lambda: (_shutdown(), hb_server.stop(), clear_heartbeat(port)))
    print(f"Dashboard running on http://localhost:{port}")
    print(
        "Close the browser tab to stop the server within "
        f"~{TAB_CLOSE_SHUTDOWN_SECONDS}-{idle_shutdown_seconds}s "
        "(or press Ctrl+C in this window)."
    )
    # taskkill / Streamlit often exit non-zero; intentional stops should still
    # return 0 so launch-dashboard.bat does not pause and leave the console open.
    intentional_stop = False
    closing = False
    fresh_seconds = HEARTBEAT_INTERVAL_SECONDS + POLL_SECONDS
    try:
        while proc.poll() is None:
            # Latch tab-close so a late /ping cannot clear /bye and cancel shutdown.
            if closing or tab_close_requested(port):
                closing = True
                had_browser = True
                limit = TAB_CLOSE_SHUTDOWN_SECONDS
                reason = "Browser tab closed"
            elif _browser_active(port, fresh_seconds=fresh_seconds):
                had_browser = True
                idle_since = None
                time.sleep(POLL_SECONDS)
                continue
            elif had_browser:
                limit = idle_shutdown_seconds
                reason = "No browser heartbeat or connection"
            elif time.monotonic() - started_at >= _startup_grace_seconds():
                print("No browser connected. Shutting down dashboard...")
                intentional_stop = True
                _shutdown()
                break
            else:
                time.sleep(POLL_SECONDS)
                continue
            if had_browser:
                if idle_since is None:
                    idle_since = time.monotonic()
                elif time.monotonic() - idle_since >= limit:
                    print(f"{reason}. Shutting down dashboard...")
                    intentional_stop = True
                    _shutdown()
                    break
            time.sleep(POLL_SECONDS)
    finally:
        if proc.poll() is None:
            _shutdown()
        hb_server.stop()
        clear_heartbeat(port)
        child_code = proc.wait()
    if intentional_stop:
        return 0
    return child_code if child_code is not None else 0
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run SportsPredictor dashboard with idle shutdown")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--idle-seconds", type=int, default=DEFAULT_IDLE_SHUTDOWN_SECONDS)
    args = parser.parse_args(argv)
    return run_dashboard(port=args.port, idle_shutdown_seconds=args.idle_seconds)
if __name__ == "__main__":
    raise SystemExit(main())

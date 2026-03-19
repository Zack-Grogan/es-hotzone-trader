"""launchd helpers for running es-hotzone-trader as a macOS user agent."""
from __future__ import annotations

import os
import plistlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


LAUNCHD_LABEL = "com.gtrade.es-hotzone-trader"


@dataclass
class LaunchdResult:
    ok: bool
    message: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def launch_agents_dir() -> Path:
    return Path.home() / "Library" / "LaunchAgents"


def plist_path() -> Path:
    return launch_agents_dir() / f"{LAUNCHD_LABEL}.plist"


def service_logs_dir() -> Path:
    return project_root() / "logs"


def stdout_log_path() -> Path:
    return service_logs_dir() / "launchd.stdout.log"


def stderr_log_path() -> Path:
    return service_logs_dir() / "launchd.stderr.log"


def launchctl_target() -> str:
    return f"gui/{os.getuid()}/{LAUNCHD_LABEL}"


def render_launchd_plist(config_path: Optional[str] = None) -> bytes:
    args = [sys.executable, "-m", "src.cli", "start"]
    if config_path:
        args.extend(["--config", str(Path(config_path).resolve())])
    stdout_path = stdout_log_path()
    stderr_path = stderr_log_path()
    stdout_path.parent.mkdir(parents=True, exist_ok=True)
    environment = {
        "PYTHONUNBUFFERED": "1",
    }
    internal_token = os.environ.get("GTRADE_INTERNAL_API_TOKEN", "").strip()
    if internal_token:
        environment["GTRADE_INTERNAL_API_TOKEN"] = internal_token
    plist = {
        "Label": LAUNCHD_LABEL,
        "ProgramArguments": args,
        "WorkingDirectory": str(project_root()),
        "RunAtLoad": False,
        "KeepAlive": False,
        "StandardOutPath": str(stdout_path),
        "StandardErrorPath": str(stderr_path),
        "EnvironmentVariables": environment,
    }
    return plistlib.dumps(plist)


def write_launchd_plist(config_path: Optional[str] = None) -> Path:
    path = plist_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(render_launchd_plist(config_path))
    return path


def read_launchd_environment() -> dict[str, str]:
    path = plist_path()
    if not path.exists():
        return {}
    try:
        payload = plistlib.loads(path.read_bytes())
    except Exception:
        return {}
    environment = payload.get("EnvironmentVariables")
    if not isinstance(environment, dict):
        return {}
    return {str(key): str(value) for key, value in environment.items() if value is not None}


def _run_launchctl(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["launchctl", *args],
        capture_output=True,
        text=True,
        cwd=str(project_root()),
        check=False,
    )


def install_launch_agent(config_path: Optional[str] = None) -> LaunchdResult:
    path = write_launchd_plist(config_path)
    return LaunchdResult(True, f"Installed launchd agent plist at {path}.")


def uninstall_launch_agent() -> LaunchdResult:
    stop_launch_agent()
    path = plist_path()
    if path.exists():
        path.unlink()
        return LaunchdResult(True, f"Removed launchd agent plist at {path}.")
    return LaunchdResult(True, f"Launchd agent plist not present at {path}.")


def start_launch_agent() -> LaunchdResult:
    path = plist_path()
    if not path.exists():
        return LaunchdResult(False, f"Launchd agent plist missing at {path}. Install it first.")
    bootstrap = _run_launchctl("bootstrap", f"gui/{os.getuid()}", str(path))
    if bootstrap.returncode not in {0} and "service already loaded" not in (bootstrap.stderr or "").lower():
        return LaunchdResult(False, bootstrap.stderr.strip() or bootstrap.stdout.strip() or "launchctl bootstrap failed")
    kickstart = _run_launchctl("kickstart", "-k", launchctl_target())
    if kickstart.returncode != 0:
        return LaunchdResult(False, kickstart.stderr.strip() or kickstart.stdout.strip() or "launchctl kickstart failed")
    return LaunchdResult(True, f"Started launchd agent {LAUNCHD_LABEL}.")


def stop_launch_agent() -> LaunchdResult:
    result = _run_launchctl("bootout", launchctl_target())
    if result.returncode != 0 and "could not find service" not in (result.stderr or "").lower():
        return LaunchdResult(False, result.stderr.strip() or result.stdout.strip() or "launchctl bootout failed")
    return LaunchdResult(True, f"Stopped launchd agent {LAUNCHD_LABEL}.")


def restart_launch_agent() -> LaunchdResult:
    stopped = stop_launch_agent()
    if not stopped.ok:
        return stopped
    return start_launch_agent()


def launch_agent_status() -> dict[str, object]:
    plist = plist_path()
    result = _run_launchctl("print", launchctl_target())
    return {
        "label": LAUNCHD_LABEL,
        "plist_path": str(plist),
        "installed": plist.exists(),
        "loaded": result.returncode == 0,
        "details": result.stdout.strip() if result.returncode == 0 else (result.stderr.strip() or result.stdout.strip()),
        "stdout_log_path": str(stdout_log_path()),
        "stderr_log_path": str(stderr_log_path()),
    }

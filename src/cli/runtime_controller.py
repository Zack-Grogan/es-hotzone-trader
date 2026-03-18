"""Runtime controller primitives shared by CLI and TUI surfaces."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from typing import Any, Optional

from src.config import get_config, load_config, set_config
from src.market import get_client


@dataclass
class RuntimeActionResult:
    ok: bool
    message: str
    active_pid: Optional[int] = None


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_config(config_path: Optional[str]) -> tuple[Any, str]:
    if config_path:
        cfg = load_config(config_path)
    else:
        cfg = get_config()
    set_config(cfg)
    resolved = str(Path(config_path).resolve()) if config_path else str(_project_root() / "config" / "default.yaml")
    return cfg, resolved


def _runtime_paths(cfg) -> dict[str, Path]:
    project_root = _project_root()
    log_path = Path(cfg.logging.file)
    if not log_path.is_absolute():
        log_path = project_root / log_path
    runtime_dir = log_path.resolve().parent / "runtime"
    return {
        "runtime_dir": runtime_dir,
        "pid_file": runtime_dir / "trader.pid",
        "request_file": runtime_dir / "lifecycle_request.json",
        "status_file": runtime_dir / "runtime_status.json",
        "operator_request_file": runtime_dir / "operator_request.json",
    }


def _read_json_file(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _read_pid(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def runtime_status(cfg) -> Optional[dict[str, Any]]:
    return _read_json_file(_runtime_paths(cfg)["status_file"])


def active_pid(cfg) -> Optional[int]:
    pid = _read_pid(_runtime_paths(cfg)["pid_file"])
    if pid is None:
        return None
    return pid if _pid_is_running(pid) else None


def _env_path() -> Path:
    return _project_root() / ".env"


def _upsert_env_var(path: Path, key: str, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
    replaced = False
    updated: list[str] = []
    for line in lines:
        if line.startswith(f"{key}="):
            updated.append(f"{key}={value}")
            replaced = True
        else:
            updated.append(line)
    if not replaced:
        updated.append(f"{key}={value}")
    path.write_text("\n".join(updated).rstrip() + "\n", encoding="utf-8")


def persist_preferred_account_id(account_id: str) -> RuntimeActionResult:
    cleaned = str(account_id).strip()
    if not cleaned:
        return RuntimeActionResult(False, "Account ID is required.")
    try:
        os.environ["PREFERRED_ACCOUNT_ID"] = cleaned
        _upsert_env_var(_env_path(), "PREFERRED_ACCOUNT_ID", cleaned)
    except Exception as exc:
        return RuntimeActionResult(False, f"Failed to persist preferred account id: {exc}")
    return RuntimeActionResult(True, f"Preferred account updated to {cleaned}.")


def list_available_accounts(config_path: Optional[str] = None) -> tuple[list[dict[str, Any]], RuntimeActionResult]:
    cfg, _ = resolve_config(config_path)
    _ = cfg
    client = get_client(force_recreate=True)
    if not client.authenticate():
        return [], RuntimeActionResult(False, "Authentication failed while listing accounts.")
    accounts = client.list_accounts(only_active_accounts=True)
    active_id = client.get_active_account_id()
    rows: list[dict[str, Any]] = []
    for account in accounts:
        account_id = str(account.get("id"))
        rows.append(
            {
                "id": account_id,
                "name": str(account.get("name", account_id)),
                "balance": account.get("balance"),
                "can_trade": bool(account.get("canTrade", False)),
                "visible": bool(account.get("isVisible", False)),
                "simulated": bool(account.get("simulated", False)),
                "selected": str(active_id) == account_id if active_id is not None else False,
            }
        )
    return rows, RuntimeActionResult(True, f"Loaded {len(rows)} account(s).")


def launch_start_process(config_path: Optional[str] = None) -> RuntimeActionResult:
    cfg, resolved_config = resolve_config(config_path)
    pid = active_pid(cfg)
    if pid is not None:
        return RuntimeActionResult(False, f"Runtime already active (PID {pid}).", active_pid=pid)
    project_root = _project_root()
    args = [sys.executable, "-m", "src.cli", "start"]
    if config_path:
        args.extend(["--config", resolved_config])
    try:
        proc = subprocess.Popen(
            args,
            cwd=str(project_root),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception as exc:
        return RuntimeActionResult(False, f"Failed to launch runtime: {exc}")
    return RuntimeActionResult(True, f"Runtime launch requested (PID {proc.pid}).", active_pid=proc.pid)


def request_runtime_action(
    action: str,
    *,
    reason: str,
    config_path: Optional[str] = None,
    timeout_seconds: int = 20,
    source: str = "runtime_controller",
) -> RuntimeActionResult:
    cfg, _ = resolve_config(config_path)
    paths = _runtime_paths(cfg)
    pid = active_pid(cfg)
    if pid is None:
        if action == "start":
            return launch_start_process(config_path)
        return RuntimeActionResult(False, "Runtime is not active.")

    payload = {
        "request_id": f"{int(time.time())}-{os.getpid()}-{action}",
        "action": action,
        "reason": reason,
        "source": source,
        "requester_pid": os.getpid(),
        "requested_at": time.time(),
    }
    _write_json_file(paths["request_file"], payload)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return RuntimeActionResult(False, "Runtime process disappeared before signal.")

    if action == "stop":
        deadline = time.time() + max(timeout_seconds, 1)
        while time.time() < deadline:
            if not _pid_is_running(pid):
                return RuntimeActionResult(True, f"Stopped runtime PID {pid}.", active_pid=pid)
            time.sleep(0.2)
        return RuntimeActionResult(False, f"Timed out stopping PID {pid}.", active_pid=pid)

    if action == "restart":
        deadline = time.time() + max(timeout_seconds, 1)
        while time.time() < deadline:
            if not _pid_is_running(pid):
                launched = launch_start_process(config_path)
                if launched.ok:
                    return RuntimeActionResult(True, f"Restarted runtime from PID {pid}.", active_pid=launched.active_pid)
                return launched
            time.sleep(0.2)
        return RuntimeActionResult(False, f"Timed out restarting PID {pid}.", active_pid=pid)

    return RuntimeActionResult(True, f"Requested action '{action}' for PID {pid}.", active_pid=pid)


def switch_account(
    account_id: str,
    *,
    allow_hot_switch: bool = True,
    config_path: Optional[str] = None,
    timeout_seconds: int = 30,
    source: str = "runtime_controller",
) -> RuntimeActionResult:
    cfg, _ = resolve_config(config_path)
    persist_result = persist_preferred_account_id(account_id)
    if not persist_result.ok:
        return persist_result

    pid = active_pid(cfg)
    if pid is None:
        return RuntimeActionResult(
            True,
            f"{persist_result.message} Runtime is not active; next start will use this account.",
        )

    # Hot switch can only be attempted when the runtime process explicitly advertises support.
    # Current runtime does not expose a safe in-process account switch endpoint, so fallback to restart.
    if allow_hot_switch:
        status = runtime_status(cfg) or {}
        if bool(status.get("supports_live_account_switch")):
            return RuntimeActionResult(
                False,
                "Runtime reported live account switching support, but no safe switch endpoint is configured. Falling back to restart is required.",
                active_pid=pid,
            )

    restarted = request_runtime_action(
        "restart",
        reason=f"account_switch:{account_id}",
        config_path=config_path,
        timeout_seconds=timeout_seconds,
        source=source,
    )
    if restarted.ok:
        return RuntimeActionResult(
            True,
            f"{persist_result.message} Restarted runtime to apply account switch.",
            active_pid=restarted.active_pid,
        )
    return RuntimeActionResult(
        False,
        f"{persist_result.message} Failed to restart runtime for account switch: {restarted.message}",
        active_pid=restarted.active_pid,
    )


def write_operator_request(
    action: str,
    *,
    config_path: Optional[str] = None,
    source: str = "cli",
) -> RuntimeActionResult:
    """Ask the running engine to perform force_reconcile or clear_unresolved (safe-gated)."""
    if action not in ("force_reconcile", "clear_unresolved"):
        return RuntimeActionResult(False, f"Unknown operator action: {action}")
    cfg, _ = resolve_config(config_path)
    paths = _runtime_paths(cfg)
    if active_pid(cfg) is None:
        return RuntimeActionResult(False, "Runtime is not active.")
    payload = {
        "action": action,
        "source": source,
        "requested_at": time.time(),
    }
    try:
        _write_json_file(paths["operator_request_file"], payload)
    except Exception as exc:
        return RuntimeActionResult(False, str(exc))
    return RuntimeActionResult(True, f"Requested '{action}'. Engine will process on next loop.")


def run_replay(path: str, config_path: Optional[str] = None) -> RuntimeActionResult:
    cfg, resolved_config = resolve_config(config_path)
    if active_pid(cfg) is not None:
        return RuntimeActionResult(False, "Runtime is active. Stop it before replay.")
    project_root = _project_root()
    args = [sys.executable, "-m", "src.cli", "replay", "--path", path]
    if config_path:
        args.extend(["--config", resolved_config])
    try:
        proc = subprocess.run(args, cwd=str(project_root), capture_output=True, text=True, check=False)
    except Exception as exc:
        return RuntimeActionResult(False, f"Replay failed to launch: {exc}")
    if proc.returncode != 0:
        return RuntimeActionResult(False, proc.stderr.strip() or proc.stdout.strip() or "Replay failed.")
    return RuntimeActionResult(True, proc.stdout.strip() or "Replay completed.")

"""CLI commands module."""
import click
from datetime import UTC, datetime, timedelta
import hashlib
import logging
import json
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path
import requests
import signal
import sys
import threading
import time
from typing import Any, Optional

from src.config import get_config, load_config, set_config
from src.cli.launchd import (
    install_launch_agent,
    launch_agent_status,
    read_launchd_environment,
    restart_launch_agent,
    start_launch_agent,
    stderr_log_path,
    stdout_log_path,
    stop_launch_agent,
    uninstall_launch_agent,
)
from src.server import get_server, get_state, set_state
from src.market import get_client
from src.execution import get_executor
from src.engine import ReplayRunner, get_scheduler, get_risk_manager, get_trading_engine
from src.observability import get_observability_store
from src.observability.provenance import collect_run_provenance
from src.bridge import start_railway_bridge
from src.bridge.outbox import RailwayOutbox
from src.bridge.railway_bridge import rebuild_outbox_from_observability

logger = logging.getLogger(__name__)


class _ConsoleFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",
        logging.INFO: "\033[32m",
        logging.WARNING: "\033[33m",
        logging.ERROR: "\033[31m",
        logging.CRITICAL: "\033[35m",
    }
    RESET = "\033[0m"

    def __init__(self, fmt: str, use_colors: bool):
        super().__init__(fmt)
        self._use_colors = use_colors

    def format(self, record: logging.LogRecord) -> str:
        rendered = super().format(record)
        if not self._use_colors:
            return rendered
        color = self.COLORS.get(record.levelno)
        return f"{color}{rendered}{self.RESET}" if color else rendered


class _ObservabilityLogHandler(logging.Handler):
    _local = threading.local()
    _formatter = logging.Formatter()

    def emit(self, record: logging.LogRecord) -> None:
        if getattr(self._local, "active", False):
            return
        if record.name.startswith("src.observability.store"):
            return
        try:
            self._local.active = True
            exception_text = None
            if record.exc_info:
                exception_text = self._formatter.formatException(record.exc_info)
            payload = {
                "logged_at": datetime.fromtimestamp(record.created, UTC).isoformat(),
                "logger_name": record.name,
                "level": record.levelname,
                "source": "local-runtime",
                "service_name": "es-hotzone-trader",
                "process_id": os.getpid(),
                "line_hash": hashlib.sha1(
                    f"{record.pathname}:{record.lineno}:{record.getMessage()}".encode("utf-8")
                ).hexdigest()[:16],
                "thread_name": record.threadName,
                "message": record.getMessage(),
                "exception_text": exception_text,
                "pathname": record.pathname,
                "lineno": record.lineno,
            }
            get_observability_store().record_runtime_log(payload)
        except Exception:
            pass
        finally:
            self._local.active = False


def _log_uncaught_exception(exc_type, exc_value, exc_traceback) -> None:
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    get_observability_store().record_event(
        category="system",
        event_type="uncaught_exception",
        source=__name__,
        payload={"exception_type": exc_type.__name__, "message": str(exc_value)},
        action="uncaught_exception",
        reason=str(exc_value),
    )
    logging.getLogger(__name__).critical(
        "Uncaught exception",
        exc_info=(exc_type, exc_value, exc_traceback),
    )


def _log_thread_exception(args: threading.ExceptHookArgs) -> None:
    if issubclass(args.exc_type, KeyboardInterrupt):
        return
    get_observability_store().record_event(
        category="system",
        event_type="thread_exception",
        source=__name__,
        payload={
            "thread_name": args.thread.name if args.thread else "unknown",
            "exception_type": args.exc_type.__name__,
            "message": str(args.exc_value),
        },
        action="thread_exception",
        reason=str(args.exc_value),
    )
    logging.getLogger(__name__).critical(
        "Unhandled thread exception in %s",
        args.thread.name if args.thread else "unknown",
        exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
    )


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _resolve_log_path(cfg) -> Path:
    project_root = Path(__file__).resolve().parent.parent.parent
    log_path = Path(cfg.logging.file)
    if not log_path.is_absolute():
        log_path = project_root / log_path
    return log_path


def _configure_logging(cfg) -> Path:
    log_path = _resolve_log_path(cfg)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    log_level = getattr(logging, str(cfg.logging.level).upper(), logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(threadName)s %(message)s"
    )

    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:
            pass

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(log_level)
    stream_handler.setFormatter(
        _ConsoleFormatter(
            "%(asctime)s %(levelname)s [%(name)s] %(threadName)s %(message)s",
            bool(getattr(cfg.logging, "console_colors", True)) and getattr(sys.stderr, "isatty", lambda: False)(),
        )
    )

    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=int(cfg.logging.max_bytes),
        backupCount=int(cfg.logging.backup_count),
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)

    observability_handler = _ObservabilityLogHandler()
    observability_handler.setLevel(log_level)

    root_logger.setLevel(log_level)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(observability_handler)
    logging.captureWarnings(True)
    sys.excepthook = _log_uncaught_exception
    threading.excepthook = _log_thread_exception
    logger.info("Logging configured at %s", log_path)
    return log_path


def _runtime_control_paths(cfg, log_path: Optional[Path] = None) -> dict[str, Path]:
    resolved_log_path = log_path or _resolve_log_path(cfg)
    runtime_dir = resolved_log_path.parent / "runtime"
    return {
        "runtime_dir": runtime_dir,
        "pid_file": runtime_dir / "trader.pid",
        "request_file": runtime_dir / "lifecycle_request.json",
        "status_file": runtime_dir / "runtime_status.json",
    }


def _write_json_file(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_name(f".{path.name}.tmp")
    temp_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temp_path.replace(path)


def _read_json_file(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to read JSON file at %s", path, exc_info=True)
        return None


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


def _read_pid_file(path: Path) -> Optional[int]:
    if not path.exists():
        return None
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        logger.warning("Failed to parse pid file at %s", path, exc_info=True)
        return None


def _write_pid_file(path: Path, pid: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{pid}\n", encoding="utf-8")


def _remove_file_if_exists(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except Exception:
        logger.warning("Failed to remove file at %s", path, exc_info=True)


def _build_lifecycle_request(*, action: str, reason: str, source: str) -> dict[str, Any]:
    requested_at = _utc_now().isoformat()
    return {
        "request_id": f"{int(time.time())}-{os.getpid()}-{action}",
        "action": action,
        "reason": reason,
        "source": source,
        "requester_pid": os.getpid(),
        "requested_at": requested_at,
    }


def _read_lifecycle_request(cfg, log_path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    return _read_json_file(_runtime_control_paths(cfg, log_path)["request_file"])


def _write_lifecycle_request(cfg, payload: dict[str, Any], log_path: Optional[Path] = None) -> Path:
    path = _runtime_control_paths(cfg, log_path)["request_file"]
    _write_json_file(path, payload)
    return path


def _clear_lifecycle_request(cfg, log_path: Optional[Path] = None) -> None:
    _remove_file_if_exists(_runtime_control_paths(cfg, log_path)["request_file"])


def _write_runtime_status(
    cfg,
    *,
    log_path: Path,
    phase: str,
    config_path: str,
    running: bool,
    data_mode: str,
    lifecycle_request: Optional[dict[str, Any]] = None,
    run_id: Optional[str] = None,
    status: Optional[str] = None,
) -> dict[str, Any]:
    payload = {
        "phase": phase,
        "status": status or phase,
        "pid": os.getpid(),
        "running": running,
        "data_mode": data_mode,
        "config_path": config_path,
        "log_path": str(log_path),
        "run_id": run_id,
        "updated_at": _utc_now().isoformat(),
        "lifecycle_request": lifecycle_request,
    }
    _write_json_file(_runtime_control_paths(cfg, log_path)["status_file"], payload)
    return payload


def _read_runtime_status(cfg, log_path: Optional[Path] = None) -> Optional[dict[str, Any]]:
    return _read_json_file(_runtime_control_paths(cfg, log_path)["status_file"])


def _mark_runtime_active(cfg, *, log_path: Path, config_path: str, data_mode: str, lifecycle_request: Optional[dict[str, Any]] = None) -> None:
    paths = _runtime_control_paths(cfg, log_path)
    _write_pid_file(paths["pid_file"], os.getpid())
    _write_runtime_status(
        cfg,
        log_path=log_path,
        phase="starting",
        config_path=config_path,
        running=True,
        data_mode=data_mode,
        lifecycle_request=lifecycle_request,
        status="starting",
    )


def _mark_runtime_phase(
    cfg,
    *,
    log_path: Path,
    config_path: str,
    data_mode: str,
    phase: str,
    running: bool,
    lifecycle_request: Optional[dict[str, Any]] = None,
    run_id: Optional[str] = None,
    status: Optional[str] = None,
) -> None:
    _write_runtime_status(
        cfg,
        log_path=log_path,
        phase=phase,
        config_path=config_path,
        running=running,
        data_mode=data_mode,
        lifecycle_request=lifecycle_request,
        run_id=run_id,
        status=status,
    )


def _mark_runtime_inactive(
    cfg,
    *,
    log_path: Path,
    config_path: str,
    data_mode: str,
    lifecycle_request: Optional[dict[str, Any]] = None,
    run_id: Optional[str] = None,
    status: str = "stopped",
) -> None:
    paths = _runtime_control_paths(cfg, log_path)
    _remove_file_if_exists(paths["pid_file"])
    _write_runtime_status(
        cfg,
        log_path=log_path,
        phase="stopped" if status == "stopped" else "error",
        config_path=config_path,
        running=False,
        data_mode=data_mode,
        lifecycle_request=lifecycle_request,
        run_id=run_id,
        status=status,
    )


def _set_lifecycle_state(**updates: Any) -> dict[str, Any]:
    current = dict(getattr(get_state(), "lifecycle", {}) or {})
    current.update(updates)
    set_state(lifecycle=current)
    return current


def _ensure_runtime_is_not_active(cfg, *, log_path: Path) -> None:
    paths = _runtime_control_paths(cfg, log_path)
    active_pid = _read_pid_file(paths["pid_file"])
    if active_pid is not None and _pid_is_running(active_pid):
        raise click.ClickException(f"ES Hot-Zone Trader is already running with PID {active_pid}.")
    if active_pid is not None:
        _remove_file_if_exists(paths["pid_file"])


def _resolve_shutdown_request(
    cfg,
    *,
    log_path: Path,
    fallback_reason: str,
    fallback_action: str = "stop",
    signal_name: Optional[str] = None,
) -> dict[str, Any]:
    request = _read_lifecycle_request(cfg, log_path) or {}
    return {
        "request_id": request.get("request_id"),
        "requested_action": request.get("action") or fallback_action,
        "operator_reason": request.get("reason") or fallback_reason,
        "request_source": request.get("source"),
        "requester_pid": request.get("requester_pid"),
        "requested_at": request.get("requested_at"),
        "signal_name": signal_name,
    }


def _record_system_event(observability, *, event_type: str, payload: dict[str, Any], symbol: Optional[str], action: str, reason: str) -> None:
    observability.record_event(
        category="system",
        event_type=event_type,
        source="src.cli.commands",
        payload=payload,
        symbol=symbol,
        action=action,
        reason=reason,
    )


def _request_runtime_action(
    cfg,
    *,
    log_path: Path,
    action: str,
    reason: str,
    timeout_seconds: int,
    request_source: str,
) -> tuple[Optional[int], Optional[dict[str, Any]], dict[str, Any]]:
    paths = _runtime_control_paths(cfg, log_path)
    active_pid = _read_pid_file(paths["pid_file"])
    runtime_status = _read_runtime_status(cfg, log_path)
    request = _build_lifecycle_request(action=action, reason=reason, source=request_source)
    _write_lifecycle_request(cfg, request, log_path)
    if active_pid is None or not _pid_is_running(active_pid):
        if active_pid is not None:
            _remove_file_if_exists(paths["pid_file"])
        if runtime_status and runtime_status.get("running"):
            _mark_runtime_inactive(
                cfg,
                log_path=log_path,
                config_path=str(runtime_status.get("config_path") or _resolve_config_path(None)),
                data_mode=str(runtime_status.get("data_mode") or "unknown"),
                lifecycle_request=request,
                run_id=runtime_status.get("run_id"),
                status="stopped",
            )
        return None, runtime_status, request

    os.kill(active_pid, signal.SIGTERM)
    deadline = time.time() + max(timeout_seconds, 1)
    while time.time() < deadline:
        if not _pid_is_running(active_pid):
            return active_pid, runtime_status, request
        time.sleep(0.25)

    raise click.ClickException(f"Timed out waiting for PID {active_pid} to {action}.")


def _shutdown_runtime(
    *,
    cfg,
    config_path: str,
    log_path: Path,
    observability,
    server,
    engine,
    symbol: Optional[str],
    shutdown_request: dict[str, Any],
    started_engine: bool,
    started_server: bool,
    startup_completed: bool,
) -> None:
    shutdown_reason = str(shutdown_request.get("operator_reason") or "shutdown_requested")
    requested_action = str(shutdown_request.get("requested_action") or "stop")
    signal_name = shutdown_request.get("signal_name")
    lifecycle_state = _set_lifecycle_state(
        phase="stopping",
        requested_action=requested_action,
        operator_reason=shutdown_reason,
        signal_name=signal_name,
        request_id=shutdown_request.get("request_id"),
        requested_at=shutdown_request.get("requested_at"),
        request_source=shutdown_request.get("request_source"),
        requester_pid=shutdown_request.get("requester_pid"),
        last_shutdown_started_at=_utc_now().isoformat(),
    )
    set_state(status="stopping", running=False)
    _mark_runtime_phase(
        cfg,
        log_path=log_path,
        config_path=config_path,
        data_mode="live",
        phase="stopping",
        running=False,
        lifecycle_request=shutdown_request,
        run_id=observability.get_run_id() if observability else None,
        status="stopping",
    )
    if observability:
        _record_system_event(
            observability,
            event_type="shutdown_requested",
            payload={
                "requested_action": requested_action,
                "operator_reason": shutdown_reason,
                "signal_name": signal_name,
                "request_id": shutdown_request.get("request_id"),
                "request_source": shutdown_request.get("request_source"),
                "requester_pid": shutdown_request.get("requester_pid"),
                "startup_completed": startup_completed,
            },
            symbol=symbol,
            action="stop" if requested_action != "restart" else "restart",
            reason=shutdown_reason,
        )
    shutdown_error: Optional[str] = None
    if started_engine:
        try:
            engine.stop()
        except Exception as exc:
            shutdown_error = str(exc)
            logger.exception("Engine shutdown failed")
            if observability:
                _record_system_event(
                    observability,
                    event_type="shutdown_engine_failed",
                    payload={"error": shutdown_error},
                    symbol=symbol,
                    action="stop",
                    reason="engine_shutdown_failed",
                )
    if started_server:
        try:
            server.stop()
        except Exception as exc:
            logger.exception("Debug server shutdown failed")
            shutdown_error = shutdown_error or str(exc)
            if observability:
                _record_system_event(
                    observability,
                    event_type="shutdown_server_failed",
                    payload={"error": str(exc)},
                    symbol=symbol,
                    action="stop",
                    reason="server_shutdown_failed",
                )
    final_status = "error" if shutdown_error else "stopped"
    final_reason = shutdown_error or shutdown_reason
    if observability:
        _record_system_event(
            observability,
            event_type="shutdown",
            payload={
                "requested_action": requested_action,
                "operator_reason": shutdown_reason,
                "signal_name": signal_name,
                "request_id": shutdown_request.get("request_id"),
                "request_source": shutdown_request.get("request_source"),
                "requester_pid": shutdown_request.get("requester_pid"),
                "startup_completed": startup_completed,
                "shutdown_error": shutdown_error,
            },
            symbol=symbol,
            action="stop" if requested_action != "restart" else "restart",
            reason=final_reason,
        )
        observability.force_flush()
    lifecycle_state = dict(lifecycle_state)
    lifecycle_state.update(
        {
            "phase": "stopped" if final_status == "stopped" else "error",
            "last_shutdown_completed_at": _utc_now().isoformat(),
            "last_shutdown_reason": final_reason,
            "last_requested_action": requested_action,
        }
    )
    set_state(
        running=False,
        status=final_status,
        lifecycle=lifecycle_state,
    )
    _mark_runtime_inactive(
        cfg,
        log_path=log_path,
        config_path=config_path,
        data_mode="live",
        lifecycle_request=shutdown_request,
        run_id=observability.get_run_id() if observability else None,
        status=final_status,
    )
    if observability:
        observability.stop()


def _print_banner(title: str, color: str = "cyan") -> None:
    click.secho("=" * 50, fg=color)
    click.secho(title, fg=color, bold=True)
    click.secho("=" * 50, fg=color)


def _log_startup_summary(cfg, log_path: Path, current_zone: Optional[str], zone_state: str) -> None:
    mcp_url = getattr(cfg.server, "railway_mcp_url", None) or ""
    mcp_url = mcp_url.strip() if mcp_url else "disabled"
    logger.info(
        "startup_summary capital=%s max_contracts=%s hot_zones=%s matrix_version=%s preferred_account_match=%s trade_outside_hotzones=%s log_file=%s",
        cfg.account.capital,
        cfg.account.max_contracts,
        len(cfg.hot_zones),
        cfg.alpha.matrix_version,
        cfg.safety.preferred_account_match,
        cfg.strategy.trade_outside_hotzones,
        log_path,
    )
    logger.info(
        "startup_endpoints health_url=%s debug_url=%s mcp_url=%s current_zone=%s zone_state=%s",
        f"http://{cfg.server.host}:{cfg.server.health_port}/health",
        f"http://{cfg.server.host}:{cfg.server.debug_port}/debug",
        mcp_url,
        current_zone,
        zone_state,
    )


def _startup_payload(
    cfg,
    log_path: Path,
    current_zone: Optional[str],
    zone_state: str,
    lifecycle_request: Optional[dict[str, Any]] = None,
) -> dict:
    payload = {
        "capital": cfg.account.capital,
        "max_contracts": cfg.account.max_contracts,
        "symbols": list(cfg.symbols),
        "hot_zones": len(cfg.hot_zones),
        "matrix_version": cfg.alpha.matrix_version,
        "preferred_account_match": cfg.safety.preferred_account_match,
        "trade_outside_hotzones": cfg.strategy.trade_outside_hotzones,
        "log_file": str(log_path),
        "health_port": cfg.server.health_port,
        "debug_port": cfg.server.debug_port,
        "mcp_path": cfg.server.mcp_path,
        "current_zone": current_zone,
        "zone_state": zone_state,
    }
    if lifecycle_request:
        payload.update(
            {
                "lifecycle_request_id": lifecycle_request.get("request_id"),
                "requested_action": lifecycle_request.get("action"),
                "operator_reason": lifecycle_request.get("reason"),
                "request_source": lifecycle_request.get("source"),
                "requester_pid": lifecycle_request.get("requester_pid"),
                "requested_at": lifecycle_request.get("requested_at"),
            }
        )
    return payload


def _resolve_config_path(config: Optional[str]) -> str:
    if config:
        return str(Path(config).expanduser().resolve())
    return str((Path(__file__).resolve().parent.parent.parent / "config" / "default.yaml").resolve())


def _runtime_urls(cfg) -> dict:
    mcp_url = getattr(cfg.server, "railway_mcp_url", None)
    return {
        "health_url": f"http://{cfg.server.host}:{cfg.server.health_port}/health",
        "debug_url": f"http://{cfg.server.host}:{cfg.server.debug_port}/debug",
        "mcp_url": mcp_url.strip() if mcp_url else None,
    }


def _fetch_remote_debug_state(cfg, timeout_seconds: float = 1.5) -> Optional[dict[str, Any]]:
    debug_url = f"http://{cfg.server.host}:{cfg.server.debug_port}/debug"
    try:
        response = requests.get(debug_url, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _fetch_remote_health(cfg, timeout_seconds: float = 1.5) -> Optional[dict[str, Any]]:
    health_url = f"http://{cfg.server.host}:{cfg.server.health_port}/health"
    try:
        response = requests.get(health_url, timeout=timeout_seconds)
        response.raise_for_status()
        payload = response.json()
        return payload if isinstance(payload, dict) else None
    except Exception:
        return None


def _record_runtime_provenance(cfg, observability, *, config_path: str, log_path: Path, data_mode: str):
    urls = _runtime_urls(cfg)
    manifest = collect_run_provenance(
        cfg,
        config_path=config_path,
        log_path=log_path,
        sqlite_path=observability.get_db_path(),
        data_mode=data_mode,
        health_url=urls["health_url"],
        debug_url=urls["debug_url"],
        mcp_url=urls["mcp_url"],
    )
    debug_state = _fetch_remote_debug_state(cfg) or {}
    account_payload = debug_state.get("account") if isinstance(debug_state.get("account"), dict) else {}
    if account_payload:
        manifest["account"] = {
            "id": account_payload.get("id"),
            "name": account_payload.get("name"),
            "daily_pnl": account_payload.get("daily_pnl"),
            "is_practice": account_payload.get("is_practice"),
        }
    manifest["run_id"] = observability.get_run_id()
    if getattr(cfg.observability, "capture_run_provenance", True):
        observability.record_run_manifest(manifest)
    backfill_result = {"checked": 0, "backfilled": 0, "run_id": observability.get_run_id()}
    if getattr(cfg.observability, "backfill_missing_trade_records", True):
        backfill_result = observability.backfill_completed_trades_from_events()
        observability.record_event(
            category="system",
            event_type="trade_backfill_checked",
            source=__name__,
            payload=backfill_result,
            symbol=cfg.symbols[0] if cfg.symbols else None,
            action="backfill_completed_trades",
            reason="startup_backfill_check",
        )
    set_state(
        run_id=manifest["run_id"],
        code_version=manifest.get("app_version"),
        git_commit=manifest.get("git_commit"),
        git_branch=manifest.get("git_branch"),
        config_path=manifest.get("config_path"),
        config_hash=manifest.get("config_hash"),
        observability_db_path=manifest.get("sqlite_path"),
        mcp_url=manifest.get("mcp_url"),
        last_backfill=backfill_result,
    )
    return manifest, backfill_result


def _sync_account_trade_history(
    cfg,
    observability,
    *,
    source: str,
    lookback_hours: int,
    run_id: Optional[str] = None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "source": source,
        "run_id": run_id or observability.get_run_id(),
        "requested_lookback_hours": int(lookback_hours),
        "account_id": None,
        "account_name": None,
        "account_mode": None,
        "checked": 0,
        "imported": 0,
    }
    try:
        client = get_client()
        if not client._access_token and not client.authenticate():
            result["error"] = "authentication_failed"
            return result
        account = client.get_account()
        if account is None:
            result["error"] = "account_unavailable"
            return result
        result["account_id"] = account.account_id
        result["account_name"] = account.name
        result["account_mode"] = "practice" if account.is_practice else "live"
        window_hours = max(int(lookback_hours), 1)
        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(hours=window_hours)
        broker_trades = client.search_trades(
            start_timestamp=start_time.isoformat(),
            end_timestamp=end_time.isoformat(),
            account_id=int(account.account_id),
        )
        result["checked"] = len(broker_trades)
        for trade in broker_trades:
            payload = dict(trade)
            payload.setdefault("run_id", run_id or observability.get_run_id())
            payload.setdefault("account_id", account.account_id)
            payload.setdefault("account_name", account.name)
            payload.setdefault("account_is_practice", account.is_practice)
            if observability.record_account_trade(payload, run_id=run_id, source=source):
                result["imported"] += 1
        return result
    except Exception as exc:
        logger.exception("Failed to sync account trade history")
        result["error"] = str(exc)
        return result


def _print_json(payload: Any) -> None:
    click.echo(json.dumps(payload, indent=2, default=str))


def _tail_file_lines(path: Path, *, lines: int = 100) -> list[str]:
    if not path.exists():
        return []
    try:
        content = path.read_text(encoding="utf-8").splitlines()
        return content[-max(int(lines), 1):]
    except Exception:
        logger.exception("Failed to read log file at %s", path)
        return []


def _service_doctor_payload(cfg) -> dict[str, Any]:
    log_path = _resolve_log_path(cfg)
    runtime_status = _read_runtime_status(cfg, log_path)
    observability = get_observability_store()
    outbox = RailwayOutbox(cfg.observability.outbox_path)
    try:
        launchd_env = read_launchd_environment()
        return {
            "launchd": launch_agent_status(),
            "runtime_status": runtime_status,
            "remote_health": _fetch_remote_health(cfg),
            "remote_debug": _fetch_remote_debug_state(cfg),
            "observability_db_path": observability.get_db_path(),
            "outbox_stats": outbox.get_queue_stats(),
            "delivery_state": outbox.get_delivery_state(),
            "log_path": str(log_path),
            "launchd_stdout_log": str(stdout_log_path()),
            "launchd_stderr_log": str(stderr_log_path()),
            "bridge_url": cfg.observability.railway_ingest_url,
            "internal_auth_configured": bool(
                (
                    os.environ.get("GTRADE_INTERNAL_API_TOKEN")
                    or launchd_env.get("GTRADE_INTERNAL_API_TOKEN")
                    or cfg.observability.internal_api_token
                    or ""
                ).strip()
            ),
            "legacy_ingest_key_configured": bool(
                (os.environ.get("RAILWAY_INGEST_API_KEY") or cfg.observability.railway_ingest_api_key or "").strip()
            ),
        }
    finally:
        outbox.close()


@click.group()
def cli():
    """ES Hot-Zone Day Trading CLI."""
    pass


@cli.command()
@click.option('--config', type=click.Path(exists=True), help='Config file path')
def start(config: Optional[str]):
    """Start the trading engine (live: real Topstep API, practice or funded account)."""
    _print_banner("ES Hot-Zone Trader Starting...")
    
    # Load config
    config_path = _resolve_config_path(config)
    if config:
        cfg = load_config(config)
    else:
        cfg = get_config()
    set_config(cfg)
    log_path = _configure_logging(cfg)
    _ensure_runtime_is_not_active(cfg, log_path=log_path)
    startup_request = _read_lifecycle_request(cfg, log_path)
    observability = get_observability_store(force_recreate=True)
    observability.start()
    client = get_client(force_recreate=True)
    executor = get_executor(force_recreate=True)
    scheduler = get_scheduler(force_recreate=True)
    get_risk_manager(force_recreate=True)
    engine = get_trading_engine(force_recreate=True)
    symbol = cfg.symbols[0] if cfg.symbols else None
    started_server = False
    started_engine = False
    startup_completed = False
    shutdown_signal = {"name": None}
    shutdown_requested = threading.Event()
    previous_signal_handlers: dict[int, Any] = {}
    
    click.secho(f"Config loaded: ${cfg.account.capital} account, max {cfg.account.max_contracts} contracts", fg="green")
    click.secho(f"Hot zones: {len(cfg.hot_zones)} configured", fg="green")
    click.secho(f"Alpha matrix: {cfg.alpha.matrix_version}", fg="green")
    click.secho(f"Preferred account match: {cfg.safety.preferred_account_match}", fg="green")
    click.secho(f"Trade outside hot zones: {cfg.strategy.trade_outside_hotzones}", fg="green")
    click.secho(f"Log file: {log_path}", fg="green")
    click.secho("Live: Topstep API (real market data and orders).", fg="green")
    _mark_runtime_active(cfg, log_path=log_path, config_path=config_path, data_mode="live", lifecycle_request=startup_request)
    _set_lifecycle_state(
        phase="starting",
        requested_action=(startup_request or {}).get("action") or "start",
        operator_reason=(startup_request or {}).get("reason") or "live_start",
        request_id=(startup_request or {}).get("request_id"),
        requested_at=(startup_request or {}).get("requested_at"),
        request_source=(startup_request or {}).get("source"),
        requester_pid=(startup_request or {}).get("requester_pid"),
        last_start_requested_at=_utc_now().isoformat(),
    )
    
    # Start debug servers
    server = get_server(force_recreate=True)
    server.start()
    started_server = True
    click.secho(f"Health server: http://127.0.0.1:{cfg.server.health_port}/health", fg="blue")
    click.secho(f"Debug server:  http://127.0.0.1:{cfg.server.debug_port}/debug", fg="blue")
    railway_mcp = getattr(cfg.server, "railway_mcp_url", None)
    if railway_mcp and railway_mcp.strip():
        click.secho(f"MCP (Railway): {railway_mcp.strip()}", fg="blue")
    if start_railway_bridge():
        click.secho("Railway bridge: started (outbox → ingest)", fg="blue")

    # Update state
    set_state(
        running=True,
        status="running",
        start_time=time.time(),
        data_mode="live",
        replay_summary=None,
    )
    _record_runtime_provenance(cfg, observability, config_path=config_path, log_path=log_path, data_mode="live")
    if getattr(cfg.observability, "sync_account_trade_history_on_startup", False):
        account_trade_sync = _sync_account_trade_history(
            cfg,
            observability,
            source="startup_account_history_sync",
            lookback_hours=getattr(cfg.observability, "account_trade_history_lookback_hours", 168),
        )
        observability.record_event(
            category="system",
            event_type="account_trade_history_sync",
            source=__name__,
            payload=account_trade_sync,
            symbol=cfg.symbols[0] if cfg.symbols else None,
            action="sync_account_trade_history",
            reason="startup_account_history_sync",
        )
        observability.update_run_manifest_payload(
            observability.get_run_id(),
            {"last_account_trade_sync": account_trade_sync},
        )
    
    # Show current zone
    zone = scheduler.get_current_zone()
    if zone:
        click.secho(f"Current zone: {zone.name} ({zone.state.value})", fg="cyan")
        current_zone_name = zone.name
        zone_state = zone.state.value
    else:
        click.secho(
            "Current zone: Outside (active)" if cfg.strategy.trade_outside_hotzones else "Currently outside all trading zones",
            fg="cyan",
        )
        current_zone_name = "Outside" if cfg.strategy.trade_outside_hotzones else None
        zone_state = "active" if cfg.strategy.trade_outside_hotzones else "inactive"
    _log_startup_summary(cfg, log_path, current_zone_name, zone_state)
    _record_system_event(
        observability,
        event_type="startup",
        payload=_startup_payload(cfg, log_path, current_zone_name, zone_state, startup_request),
        symbol=symbol,
        action="start",
        reason="cli_start",
    )

    def _request_shutdown(signum, _frame) -> None:
        shutdown_signal["name"] = signal.Signals(signum).name
        shutdown_requested.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        previous_signal_handlers[sig] = signal.getsignal(sig)
        signal.signal(sig, _request_shutdown)
    
    try:
        engine.start()
        started_engine = True
    except Exception as exc:
        logger.exception("Engine startup failed")
        _record_system_event(
            observability,
            event_type="startup_failed",
            payload={
                "error": str(exc),
                "request_id": (startup_request or {}).get("request_id"),
                "requested_action": (startup_request or {}).get("action"),
                "operator_reason": (startup_request or {}).get("reason"),
            },
            symbol=symbol,
            action="start",
            reason="engine_startup_failed",
        )
        observability.force_flush()
        _set_lifecycle_state(
            phase="error",
            last_startup_failed_at=_utc_now().isoformat(),
            last_startup_error=str(exc),
        )
        _mark_runtime_inactive(
            cfg,
            log_path=log_path,
            config_path=config_path,
            data_mode="live",
            lifecycle_request=startup_request,
            run_id=observability.get_run_id(),
            status="error",
        )
        set_state(running=False, status="error")
        _clear_lifecycle_request(cfg, log_path)
        if started_server:
            server.stop()
        observability.stop()
        raise click.ClickException(str(exc))
    startup_completed = True
    _set_lifecycle_state(
        phase="running",
        last_startup_completed_at=_utc_now().isoformat(),
        last_startup_error=None,
        active_run_id=observability.get_run_id(),
    )
    _mark_runtime_phase(
        cfg,
        log_path=log_path,
        config_path=config_path,
        data_mode="live",
        phase="running",
        running=True,
        lifecycle_request=startup_request,
        run_id=observability.get_run_id(),
        status="running",
    )
    _clear_lifecycle_request(cfg, log_path)
    click.secho("\nTrading engine is running...", fg="green", bold=True)
    click.secho("Press Ctrl+C to stop", fg="yellow")
    
    try:
        while not shutdown_requested.wait(timeout=1.0):
            pass
    finally:
        for sig, previous in previous_signal_handlers.items():
            signal.signal(sig, previous)

    click.secho("\nShutting down...", fg="yellow", bold=True)
    shutdown_request = _resolve_shutdown_request(
        cfg,
        log_path=log_path,
        fallback_reason="signal_shutdown" if shutdown_signal["name"] else "shutdown_requested",
        signal_name=shutdown_signal["name"],
    )
    _shutdown_runtime(
        cfg=cfg,
        config_path=config_path,
        log_path=log_path,
        observability=observability,
        server=server,
        engine=engine,
        symbol=symbol,
        shutdown_request=shutdown_request,
        started_engine=started_engine,
        started_server=started_server,
        startup_completed=startup_completed,
    )
    if shutdown_request.get("requested_action") != "restart":
        _clear_lifecycle_request(cfg, log_path)
    if shutdown_request.get("requested_action") == "restart":
        click.secho("Stopped and ready for restart.", fg="green")
    else:
        click.secho("Done.", fg="green")


@cli.command()
@click.option('--path', required=True, type=click.Path(exists=True), help='Replay CSV or JSONL file')
@click.option('--config', type=click.Path(exists=True), help='Config file path')
def replay(path: str, config: Optional[str]):
    """Replay historical events through the live engine path."""
    config_path = _resolve_config_path(config)
    if config:
        cfg = load_config(config)
    else:
        cfg = get_config()
    set_config(cfg)
    log_path = _configure_logging(cfg)
    _ensure_runtime_is_not_active(cfg, log_path=log_path)
    observability = get_observability_store(force_recreate=True)
    observability.start()
    _mark_runtime_active(cfg, log_path=log_path, config_path=config_path, data_mode="replay")
    _set_lifecycle_state(
        phase="replay",
        requested_action="replay",
        operator_reason="cli_replay",
        last_start_requested_at=_utc_now().isoformat(),
    )
    set_state(status="replay", running=False, data_mode="replay", replay_summary=None, start_time=time.time())
    get_client(force_recreate=True)
    get_executor(force_recreate=True)
    get_scheduler(force_recreate=True)
    get_risk_manager(force_recreate=True)
    engine = get_trading_engine(force_recreate=True)
    _record_runtime_provenance(cfg, observability, config_path=config_path, log_path=log_path, data_mode="replay")

    try:
        runner = ReplayRunner(config=cfg, engine=engine)
        result = runner.run(path)
        run_id = observability.get_run_id()
        if getattr(cfg.observability, "backfill_missing_trade_records", True):
            backfill_result = observability.backfill_completed_trades_from_events(run_id=run_id)
            observability.record_event(
                category="system",
                event_type="trade_backfill_checked",
                source=__name__,
                payload=backfill_result,
                symbol=cfg.symbols[0] if cfg.symbols else None,
                action="backfill_completed_trades",
                reason="replay_backfill_check",
            )
        observability.update_run_manifest_payload(
            run_id,
            {"replay_path": path, "replay_events": result.events, "replay_summary": result.summary},
        )
        _record_system_event(
            observability,
            event_type="replay_completed",
            payload={"path": str(path), "events": result.events, "segments": result.segments},
            symbol=cfg.symbols[0] if cfg.symbols else None,
            action="replay",
            reason="cli_replay",
        )
        _set_lifecycle_state(
            phase="stopped",
            last_startup_completed_at=_utc_now().isoformat(),
            last_shutdown_completed_at=_utc_now().isoformat(),
            last_shutdown_reason="replay_completed",
            last_requested_action="replay",
        )
        _mark_runtime_inactive(
            cfg,
            log_path=log_path,
            config_path=config_path,
            data_mode="replay",
            run_id=observability.get_run_id(),
            status="stopped",
        )
        observability.stop()
        click.echo(
            json.dumps(
                {
                    "path": result.path,
                    "events": result.events,
                    "segments": result.segments,
                    "summary": result.summary,
                },
                indent=2,
                default=str,
            )
        )
    except Exception:
        _mark_runtime_inactive(
            cfg,
            log_path=log_path,
            config_path=config_path,
            data_mode="replay",
            run_id=observability.get_run_id(),
            status="error",
        )
        observability.stop()
        raise


@cli.command()
@click.option('--reason', default='cli_stop', show_default=True, type=str, help='Reason recorded for the shutdown request')
@click.option('--timeout-seconds', default=20, show_default=True, type=int, help='Seconds to wait for the process to exit cleanly')
def stop(reason: str, timeout_seconds: int):
    """Stop the trading engine."""
    cfg = get_config()
    log_path = _resolve_log_path(cfg)
    active_pid, _, _ = _request_runtime_action(
        cfg,
        log_path=log_path,
        action="stop",
        reason=reason,
        timeout_seconds=timeout_seconds,
        request_source="src.cli.commands.stop",
    )
    if active_pid is None:
        _clear_lifecycle_request(cfg, log_path)
        click.echo("Trading engine is not running.")
        return
    click.echo(f"Trading engine stopped (PID {active_pid}, reason={reason}).")


@cli.command()
@click.option('--config', type=click.Path(exists=True), help='Config file path for the restarted process')
@click.option('--reason', default='cli_restart', show_default=True, type=str, help='Reason recorded for the restart request')
@click.option('--timeout-seconds', default=20, show_default=True, type=int, help='Seconds to wait for the prior process to exit cleanly')
@click.pass_context
def restart(ctx: click.Context, config: Optional[str], reason: str, timeout_seconds: int):
    """Restart the trading engine (always live)."""
    cfg = load_config(config) if config else get_config()
    set_config(cfg)
    log_path = _resolve_log_path(cfg)
    active_pid, runtime_status, _ = _request_runtime_action(
        cfg,
        log_path=log_path,
        action="restart",
        reason=reason,
        timeout_seconds=timeout_seconds,
        request_source="src.cli.commands.restart",
    )
    restart_config_path = config or ((runtime_status or {}).get("config_path") if runtime_status else None)
    if active_pid is None:
        click.echo("Trading engine is not running. Starting a fresh process with restart intent.")
    else:
        click.echo(f"Trading engine stopped for restart (PID {active_pid}, reason={reason}).")
    ctx.invoke(start, config=restart_config_path)


@cli.command()
def status():
    """Show current trading status."""
    cfg = get_config()
    remote = _fetch_remote_debug_state(cfg)
    if remote:
        status_value = str(remote.get("status", "unknown"))
        running_value = bool(remote.get("running", False))
        data_mode = str(remote.get("data_mode", "unknown"))
        zone_name = str((remote.get("zone") or {}).get("name") or "None")
        zone_state = str((remote.get("zone") or {}).get("state") or "inactive")
        strategy = str(remote.get("strategy", "None"))
        position_contracts = int((remote.get("position") or {}).get("contracts") or 0)
        position_pnl = float((remote.get("position") or {}).get("unrealized_pnl") or 0.0)
        daily_pnl = float((remote.get("account") or {}).get("daily_pnl") or 0.0)
        risk_state = str((remote.get("risk") or {}).get("state") or "normal")
    else:
        state = get_state()
        status_value = str(state.effective_status())
        running_value = bool(state.running)
        data_mode = str(state.data_mode)
        zone_name = str(state.current_zone or "None")
        zone_state = str(state.zone_state)
        strategy = str(state.current_strategy or "None")
        position_contracts = int(state.position)
        position_pnl = float(state.position_pnl)
        daily_pnl = float(state.daily_pnl)
        risk_state = str(state.risk_state)

    _print_banner("Trading Status", color="blue")
    click.secho(f"Status:    {status_value}", fg="green" if status_value in {"healthy", "running"} else "yellow")
    click.secho(f"Running:   {running_value}", fg="green" if running_value else "yellow")
    click.secho(f"Data Mode: {data_mode}", fg="blue")
    click.secho(f"Zone:      {zone_name} ({zone_state})", fg="cyan")
    click.secho(f"Strategy:  {strategy}", fg="cyan")
    click.secho(f"Position:  {position_contracts} contracts", fg="magenta")
    click.secho(f"Position PnL: ${position_pnl:.2f}", fg="magenta")
    click.secho(f"Daily PnL: ${daily_pnl:.2f}", fg="magenta")
    click.secho(f"Risk State: {risk_state}", fg="red" if risk_state.lower() != "normal" else "green")
    click.secho("=" * 50, fg="blue")
    if not remote:
        click.secho("Note: runtime endpoints unavailable; showing local process state only.", fg="yellow")


@cli.command()
def health():
    """Show health check results."""
    cfg = get_config()
    remote_health = _fetch_remote_health(cfg)
    health = remote_health or get_state().to_health_dict()

    click.echo("Health Check:")
    click.echo(f"  Status:      {health['status']}")
    click.echo(f"  Data Mode:   {health['data_mode']}")
    click.echo(f"  Zone:        {health['zone']}")
    click.echo(f"  Position:    {health['position']}")
    click.echo(f"  Daily PnL:   ${health['daily_pnl']:.2f}")
    click.echo(f"  Risk State:  {health['risk_state']}")
    if not remote_health:
        click.echo("  Note: runtime health endpoint unavailable; showing local process state.")


@cli.command()
def debug():
    """Show full debug information."""
    cfg = get_config()
    data = _fetch_remote_debug_state(cfg) or get_state().to_dict()
    click.echo(json.dumps(data, indent=2))


@cli.command()
@click.option('--limit', default=50, show_default=True, type=int, help='Maximum number of events to return')
@click.option('--category', type=str, help='Filter by category')
@click.option('--event-type', type=str, help='Filter by event type')
@click.option('--since-minutes', type=int, help='Only include events from the last N minutes')
@click.option('--search', type=str, help='Search across reasons, payloads, symbols, and sources')
def events(limit: int, category: Optional[str], event_type: Optional[str], since_minutes: Optional[int], search: Optional[str]):
    """Query recent observability events."""
    rows = get_observability_store().query_events(
        limit=limit,
        category=category,
        event_type=event_type,
        since_minutes=since_minutes,
        search=search,
    )
    click.echo(json.dumps(rows, indent=2, default=str))


@cli.group()
def service():
    """Manage the macOS launchd service wrapper."""


@service.command("install")
@click.option("--config", type=click.Path(exists=True), help="Config file path embedded into the launchd plist")
def service_install(config: Optional[str]) -> None:
    result = install_launch_agent(_resolve_config_path(config) if config else None)
    if not result.ok:
        raise click.ClickException(result.message)
    click.echo(result.message)


@service.command("uninstall")
def service_uninstall() -> None:
    result = uninstall_launch_agent()
    if not result.ok:
        raise click.ClickException(result.message)
    click.echo(result.message)


@service.command("start")
def service_start() -> None:
    result = start_launch_agent()
    if not result.ok:
        raise click.ClickException(result.message)
    click.echo(result.message)


@service.command("stop")
def service_stop() -> None:
    result = stop_launch_agent()
    if not result.ok:
        raise click.ClickException(result.message)
    click.echo(result.message)


@service.command("restart")
def service_restart() -> None:
    result = restart_launch_agent()
    if not result.ok:
        raise click.ClickException(result.message)
    click.echo(result.message)


@service.command("status")
def service_status() -> None:
    cfg = get_config()
    _print_json(_service_doctor_payload(cfg))


@service.command("logs")
@click.option("--lines", default=100, show_default=True, type=int, help="Number of lines to tail")
@click.option("--source", "source_name", default="app", show_default=True, type=click.Choice(["app", "launchd-stdout", "launchd-stderr"]), help="Log source to display")
def service_logs(lines: int, source_name: str) -> None:
    cfg = get_config()
    source_map = {
        "app": _resolve_log_path(cfg),
        "launchd-stdout": stdout_log_path(),
        "launchd-stderr": stderr_log_path(),
    }
    path = source_map[source_name]
    rows = _tail_file_lines(path, lines=lines)
    if not rows:
        click.echo(f"No log lines available at {path}")
        return
    click.echo("\n".join(rows))


@service.command("doctor")
def service_doctor() -> None:
    cfg = get_config()
    _print_json(_service_doctor_payload(cfg))


@cli.group()
def db():
    """Inspect local durability and recovery state."""


@db.command("runs")
@click.option("--limit", default=25, show_default=True, type=int)
@click.option("--search", type=str)
def db_runs(limit: int, search: Optional[str]) -> None:
    rows = get_observability_store().query_run_manifests(limit=limit, search=search)
    _print_json(rows)


@db.command("events")
@click.option("--limit", default=100, show_default=True, type=int)
@click.option("--category", type=str)
@click.option("--event-type", type=str)
@click.option("--search", type=str)
@click.option("--run-id", type=str)
def db_events(limit: int, category: Optional[str], event_type: Optional[str], search: Optional[str], run_id: Optional[str]) -> None:
    rows = get_observability_store().query_events(
        limit=limit,
        category=category,
        event_type=event_type,
        search=search,
        run_id=run_id,
    )
    _print_json(rows)


@db.command("snapshots")
@click.option("--limit", default=100, show_default=True, type=int)
@click.option("--kind", default="state", show_default=True, type=click.Choice(["state", "decision", "market", "order"]))
@click.option("--run-id", type=str)
@click.option("--search", type=str)
def db_snapshots(limit: int, kind: str, run_id: Optional[str], search: Optional[str]) -> None:
    store = get_observability_store()
    if kind == "state":
        rows = store.query_state_snapshots(limit=limit, run_id=run_id, search=search)
    elif kind == "decision":
        rows = store.query_decision_snapshots(limit=limit, run_id=run_id, search=search)
    elif kind == "market":
        rows = store.query_market_tape(limit=limit, run_id=run_id, search=search)
    else:
        rows = store.query_order_lifecycle(limit=limit, run_id=run_id, search=search)
    _print_json(rows)


@db.command("bridge-health")
@click.option("--limit", default=100, show_default=True, type=int)
@click.option("--run-id", type=str)
@click.option("--search", type=str)
def db_bridge_health(limit: int, run_id: Optional[str], search: Optional[str]) -> None:
    rows = get_observability_store().query_bridge_health(limit=limit, run_id=run_id, search=search)
    _print_json(rows)


@db.command("logs")
@click.option("--limit", default=100, show_default=True, type=int)
@click.option("--run-id", type=str)
@click.option("--level", "level_name", type=str)
@click.option("--search", type=str)
def db_logs(limit: int, run_id: Optional[str], level_name: Optional[str], search: Optional[str]) -> None:
    rows = get_observability_store().query_runtime_logs(limit=limit, run_id=run_id, level=level_name, search=search)
    _print_json(rows)


@db.command("account-trades")
@click.option("--limit", default=100, show_default=True, type=int)
@click.option("--run-id", type=str)
@click.option("--account-id", type=str)
@click.option("--search", type=str)
def db_account_trades(limit: int, run_id: Optional[str], account_id: Optional[str], search: Optional[str]) -> None:
    rows = get_observability_store().query_account_trades(limit=limit, run_id=run_id, account_id=account_id, search=search)
    _print_json(rows)


@db.command("sync-account-trades")
@click.option("--hours", default=168, show_default=True, type=int, help="Look back this many hours in broker account history.")
def db_sync_account_trades(hours: int) -> None:
    cfg = get_config()
    observability = get_observability_store()
    payload = _sync_account_trade_history(
        cfg,
        observability,
        source="cli_account_history_sync",
        lookback_hours=hours,
    )
    observability.record_event(
        category="system",
        event_type="account_trade_history_sync",
        source=__name__,
        payload=payload,
        symbol=cfg.symbols[0] if cfg.symbols else None,
        action="sync_account_trade_history",
        reason="cli_account_history_sync",
    )
    _print_json(payload)


@db.command("replay-missing")
@click.option("--run-id", type=str, help="Replay a single run. Default: all local runs newer than the delivery cursor.")
@click.option("--include-sent", is_flag=True, help="Ignore delivery cursors and replay all matching local records.")
@click.option("--limit-per-kind", default=1000, show_default=True, type=int)
def db_replay_missing(run_id: Optional[str], include_sent: bool, limit_per_kind: int) -> None:
    cfg = get_config()
    outbox = RailwayOutbox(cfg.observability.outbox_path)
    try:
        counts = rebuild_outbox_from_observability(
            outbox,
            run_id=run_id,
            include_sent=include_sent,
            limit_per_kind=limit_per_kind,
        )
        payload = {
            "replayed": counts,
            "outbox_stats": outbox.get_queue_stats(),
            "delivery_state": outbox.get_delivery_state(),
        }
    finally:
        outbox.close()
    _print_json(payload)


@cli.command()
def config():
    """Show current configuration."""
    cfg = get_config()
    
    click.echo("Configuration:")
    click.echo(f"  Account Capital:    ${cfg.account.capital}")
    click.echo(f"  Max Contracts:      {cfg.account.max_contracts}")
    click.echo(f"  Default Contracts:  {cfg.account.default_contracts}")
    click.echo(f"  Risk per Contract:  ${cfg.account.risk_per_contract}")
    click.echo("")
    click.echo("Hot Zones:")
    for hz in cfg.hot_zones:
        click.echo(f"  {hz.name}: {hz.start}-{hz.end} ({hz.timezone}) [{'enabled' if hz.enabled else 'disabled'}]")
    click.echo("")
    click.echo(f"Alpha Matrix Version: {cfg.alpha.matrix_version}")
    click.echo(f"Min Entry Score:      {cfg.alpha.min_entry_score}")
    click.echo(f"Min Score Gap:        {cfg.alpha.min_score_gap}")
    click.echo(f"PRAC Only:            {cfg.safety.prac_only}")
    click.echo(f"Preferred Match:      {cfg.safety.preferred_account_match}")
    click.echo("")
    click.echo("Risk Limits:")
    click.echo(f"  Max Daily Loss:      ${cfg.risk.max_daily_loss}")
    click.echo(f"  Max Position Loss:   ${cfg.risk.max_position_loss}")
    click.echo(f"  Max Consecutive:     {cfg.risk.max_consecutive_losses}")
    click.echo(f"  Max Trades/Hour:     {cfg.risk.max_trades_per_hour}")
    click.echo("")
    click.echo("Servers:")
    click.echo(f"  Health: {cfg.server.host}:{cfg.server.health_port}")
    click.echo(f"  Debug:  {cfg.server.host}:{cfg.server.debug_port}")


@cli.command()
def balance():
    """Show account balance."""
    client = get_client()
    
    # Auto-authenticate if not already
    if not client._access_token:
        click.echo("Authenticating with TopstepX...")
        if not client.authenticate():
            click.echo("Authentication failed. Check your .env credentials.")
            return
    
    account = client.get_account()
    
    if account:
        click.echo("Account:")
        click.echo(f"  Account ID:    {account.account_id}")
        click.echo(f"  Balance:       ${account.balance:.2f}")
        click.echo(f"  Available:     ${account.available:.2f}")
        click.echo(f"  Margin Used:   ${account.margin_used:.2f}")
        click.echo(f"  Open PnL:      ${account.open_pnl:.2f}")
        click.echo(f"  Realized PnL:  ${account.realized_pnl:.2f}")
    else:
        click.echo("Unable to fetch account. Make sure you're authenticated.")


def main(argv: Optional[list[str]] = None):
    """Main entry point."""
    args = list(argv if argv is not None else sys.argv[1:])
    if not args:
        cli(["--help"], prog_name="es-trade")
        return
    cli(args=args, prog_name="es-trade")


if __name__ == '__main__':
    main()

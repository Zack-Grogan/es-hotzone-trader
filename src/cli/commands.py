"""CLI commands module."""
import click
import logging
import json
from logging.handlers import RotatingFileHandler
from pathlib import Path
import sys
import threading
from typing import Optional

from src.config import get_config, load_config, set_config
from src.server import get_server, get_state, set_state
from src.market import get_client
from src.execution import get_executor
from src.engine import ReplayRunner, get_scheduler, get_risk_manager, get_trading_engine
from src.observability import get_observability_store
from src.observability.provenance import collect_run_provenance

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


def _configure_logging(cfg) -> Path:
    project_root = Path(__file__).resolve().parent.parent.parent
    log_path = Path(cfg.logging.file)
    if not log_path.is_absolute():
        log_path = project_root / log_path
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

    root_logger.setLevel(log_level)
    root_logger.addHandler(stream_handler)
    root_logger.addHandler(file_handler)
    logging.captureWarnings(True)
    sys.excepthook = _log_uncaught_exception
    threading.excepthook = _log_thread_exception
    logger.info("Logging configured at %s", log_path)
    return log_path


def _print_banner(title: str, color: str = "cyan") -> None:
    click.secho("=" * 50, fg=color)
    click.secho(title, fg=color, bold=True)
    click.secho("=" * 50, fg=color)


def _log_startup_summary(cfg, log_path: Path, mock: bool, current_zone: Optional[str], zone_state: str) -> None:
    mcp_url = f"http://{cfg.server.host}:{cfg.server.debug_port}{cfg.server.mcp_path}" if cfg.server.mcp_enabled else "disabled"
    logger.info(
        "startup_summary capital=%s max_contracts=%s hot_zones=%s matrix_version=%s preferred_account_match=%s trade_outside_hotzones=%s mock_mode=%s log_file=%s",
        cfg.account.capital,
        cfg.account.max_contracts,
        len(cfg.hot_zones),
        cfg.alpha.matrix_version,
        cfg.safety.preferred_account_match,
        cfg.strategy.trade_outside_hotzones,
        mock,
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


def _startup_payload(cfg, log_path: Path, mock: bool, current_zone: Optional[str], zone_state: str) -> dict:
    return {
        "capital": cfg.account.capital,
        "max_contracts": cfg.account.max_contracts,
        "symbols": list(cfg.symbols),
        "hot_zones": len(cfg.hot_zones),
        "matrix_version": cfg.alpha.matrix_version,
        "preferred_account_match": cfg.safety.preferred_account_match,
        "trade_outside_hotzones": cfg.strategy.trade_outside_hotzones,
        "mock_mode": mock,
        "log_file": str(log_path),
        "health_port": cfg.server.health_port,
        "debug_port": cfg.server.debug_port,
        "mcp_path": cfg.server.mcp_path,
        "current_zone": current_zone,
        "zone_state": zone_state,
    }


def _resolve_config_path(config: Optional[str]) -> str:
    if config:
        return str(Path(config).expanduser().resolve())
    return str((Path(__file__).resolve().parent.parent.parent / "config" / "default.yaml").resolve())


def _runtime_urls(cfg) -> dict:
    return {
        "health_url": f"http://{cfg.server.host}:{cfg.server.health_port}/health",
        "debug_url": f"http://{cfg.server.host}:{cfg.server.debug_port}/debug",
        "mcp_url": f"http://{cfg.server.host}:{cfg.server.debug_port}{cfg.server.mcp_path}" if cfg.server.mcp_enabled else None,
    }


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


@click.group()
def cli():
    """ES Hot-Zone Day Trading CLI."""
    pass


@cli.command()
@click.option('--mock', is_flag=True, help='Run in mock mode (no API)')
@click.option('--config', type=click.Path(exists=True), help='Config file path')
def start(mock: bool, config: Optional[str]):
    """Start the trading engine."""
    _print_banner("ES Hot-Zone Trader Starting...")
    
    # Load config
    config_path = _resolve_config_path(config)
    if config:
        cfg = load_config(config)
    else:
        cfg = get_config()
    set_config(cfg)
    log_path = _configure_logging(cfg)
    observability = get_observability_store(force_recreate=True)
    observability.start()
    client = get_client(force_recreate=True)
    executor = get_executor(force_recreate=True)
    scheduler = get_scheduler(force_recreate=True)
    get_risk_manager(force_recreate=True)
    engine = get_trading_engine(force_recreate=True)
    
    click.secho(f"Config loaded: ${cfg.account.capital} account, max {cfg.account.max_contracts} contracts", fg="green")
    click.secho(f"Hot zones: {len(cfg.hot_zones)} configured", fg="green")
    click.secho(f"Alpha matrix: {cfg.alpha.matrix_version}", fg="green")
    click.secho(f"Preferred account match: {cfg.safety.preferred_account_match}", fg="green")
    click.secho(f"Trade outside hot zones: {cfg.strategy.trade_outside_hotzones}", fg="green")
    click.secho(f"Log file: {log_path}", fg="green")
    
    # Initialize mock mode if requested
    if mock:
        click.secho("Running in MOCK mode (no real trading)", fg="yellow", bold=True)
        client.enable_mock_mode()
        executor.enable_mock_mode()
    
    # Start debug servers
    server = get_server(force_recreate=True)
    server.start()
    click.secho(f"Health server: http://127.0.0.1:{cfg.server.health_port}/health", fg="blue")
    click.secho(f"Debug server:  http://127.0.0.1:{cfg.server.debug_port}/debug", fg="blue")
    click.secho(f"MCP server:    http://127.0.0.1:{cfg.server.debug_port}{cfg.server.mcp_path}", fg="blue")
    
    # Update state
    set_state(
        running=True,
        status="running",
        start_time=__import__('time').time(),
        data_mode="mock" if mock else "live",
        replay_summary=None,
    )
    _record_runtime_provenance(cfg, observability, config_path=config_path, log_path=log_path, data_mode="mock" if mock else "live")
    
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
    _log_startup_summary(cfg, log_path, mock, current_zone_name, zone_state)
    observability.record_event(
        category="system",
        event_type="startup",
        source="src.cli.commands",
        payload=_startup_payload(cfg, log_path, mock, current_zone_name, zone_state),
        symbol=cfg.symbols[0] if cfg.symbols else None,
        zone=current_zone_name,
        action="start",
        reason="cli_start",
    )
    
    try:
        engine.start(mock=mock)
    except Exception as exc:
        logger.exception("Engine startup failed")
        observability.record_event(
            category="system",
            event_type="startup_failed",
            source="src.cli.commands",
            payload={"error": str(exc), "mock_mode": mock},
            symbol=cfg.symbols[0] if cfg.symbols else None,
            zone=current_zone_name,
            action="start",
            reason="engine_startup_failed",
        )
        observability.stop()
        set_state(running=False, status="error")
        server.stop()
        raise click.ClickException(str(exc))
    click.secho("\nTrading engine is running...", fg="green", bold=True)
    click.secho("Press Ctrl+C to stop", fg="yellow")
    
    try:
        import time
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        click.secho("\nShutting down...", fg="yellow", bold=True)
        engine.stop()
        observability.record_event(
            category="system",
            event_type="shutdown",
            source="src.cli.commands",
            payload={"mock_mode": mock},
            symbol=cfg.symbols[0] if cfg.symbols else None,
            action="stop",
            reason="keyboard_interrupt",
        )
        observability.stop()
        set_state(running=False, status="stopped")
        server.stop()
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
    observability = get_observability_store(force_recreate=True)
    observability.start()
    set_state(status="replay", running=False, data_mode="replay", replay_summary=None, start_time=__import__('time').time())
    get_client(force_recreate=True)
    get_executor(force_recreate=True)
    get_scheduler(force_recreate=True)
    get_risk_manager(force_recreate=True)
    engine = get_trading_engine(force_recreate=True)
    _record_runtime_provenance(cfg, observability, config_path=config_path, log_path=log_path, data_mode="replay")

    runner = ReplayRunner(config=cfg, engine=engine)
    result = runner.run(path)
    observability.record_event(
        category="system",
        event_type="replay_completed",
        source="src.cli.commands",
        payload={"path": str(path), "events": result.events, "segments": result.segments},
        symbol=cfg.symbols[0] if cfg.symbols else None,
        action="replay",
        reason="cli_replay",
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


@cli.command()
def stop():
    """Stop the trading engine."""
    get_trading_engine().stop()
    observability = get_observability_store()
    observability.record_event(
        category="system",
        event_type="shutdown",
        source="src.cli.commands",
        payload={"invoked_by": "cli_stop"},
        action="stop",
        reason="cli_stop",
    )
    observability.stop()
    server = get_server()
    server.stop()
    set_state(running=False, status="stopped")
    click.echo("Trading engine stopped.")


@cli.command()
def status():
    """Show current trading status."""
    state = get_state()
    
    _print_banner("Trading Status", color="blue")
    effective_status = state.effective_status()
    click.secho(f"Status:    {effective_status}", fg="green" if effective_status == "healthy" else "yellow")
    click.secho(f"Running:   {state.running}", fg="green" if state.running else "yellow")
    click.secho(f"Data Mode: {state.data_mode}", fg="blue")
    click.secho(f"Zone:      {state.current_zone or 'None'} ({state.zone_state})", fg="cyan")
    click.secho(f"Strategy:  {state.current_strategy or 'None'}", fg="cyan")
    click.secho(f"Position:  {state.position} contracts", fg="magenta")
    click.secho(f"Position PnL: ${state.position_pnl:.2f}", fg="magenta")
    click.secho(f"Daily PnL: ${state.daily_pnl:.2f}", fg="magenta")
    click.secho(f"Risk State: {state.risk_state}", fg="red" if str(state.risk_state).lower() != "normal" else "green")
    click.secho("=" * 50, fg="blue")


@cli.command()
def health():
    """Show health check results."""
    state = get_state()
    health = state.to_health_dict()
    
    click.echo("Health Check:")
    click.echo(f"  Status:      {health['status']}")
    click.echo(f"  Data Mode:   {health['data_mode']}")
    click.echo(f"  Zone:        {health['zone']}")
    click.echo(f"  Position:    {health['position']}")
    click.echo(f"  Daily PnL:   ${health['daily_pnl']:.2f}")
    click.echo(f"  Risk State:  {health['risk_state']}")


@cli.command()
def debug():
    """Show full debug information."""
    state = get_state()
    data = state.to_dict()
    
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


def main():
    """Main entry point."""
    cli()


if __name__ == '__main__':
    main()

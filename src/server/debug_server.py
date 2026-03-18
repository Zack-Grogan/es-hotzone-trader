"""Debug and Health HTTP Server."""
from datetime import UTC, datetime
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from src.config import get_config, ServerConfig
from src.observability import get_observability_store

logger = logging.getLogger(__name__)


class TradingState:
    """Global trading state container."""
    
    def __init__(self):
        """Initialize state."""
        self.status: str = "stopped"
        self.running: bool = False
        self.current_zone: Optional[str] = None
        self.zone_state: str = "inactive"
        self.current_strategy: Optional[str] = None
        self.position: int = 0
        self.position_pnl: float = 0
        self.daily_pnl: float = 0
        self.account_balance: float = 50000
        self.account_equity: float = 0
        self.account_available: float = 0
        self.account_margin_used: float = 0
        self.account_open_pnl: float = 0
        self.account_realized_pnl: float = 0
        self.account_id: Optional[str] = None
        self.account_name: Optional[str] = None
        self.account_is_practice: Optional[bool] = None
        self.data_mode: str = "unknown"
        self.long_score: float = 0
        self.short_score: float = 0
        self.flat_bias: float = 0
        self.active_vetoes: list = []
        self.matrix_version: Optional[str] = None
        self.last_entry_reason: Optional[str] = None
        self.last_exit_reason: Optional[str] = None
        self.active_session: Optional[str] = None
        self.anchored_vwaps: Dict[str, Any] = {}
        self.vwap_bands: Dict[str, Any] = {}
        self.volume_profile: Dict[str, Any] = {}
        self.order_flow: Dict[str, Any] = {}
        self.regime: Dict[str, Any] = {"state": None, "reason": None}
        self.execution: Dict[str, Any] = {}
        self.heartbeat: Dict[str, Any] = {}
        self.event_context: Dict[str, Any] = {}
        self.replay_summary: Optional[Dict[str, Any]] = None
        self.last_signal: Optional[Dict] = None
        self.last_price: Optional[float] = None
        self.uptime_seconds: float = 0
        self.start_time: float = 0
        self.risk_state: str = "normal"
        self.trades_today: int = 0
        self.trades_this_hour: int = 0
        self.trades_this_zone: int = 0
        self.max_daily_loss: float = 0
        self.consecutive_losses: int = 0
        self.errors: list = []
        self.run_id: Optional[str] = None
        self.code_version: Optional[str] = None
        self.git_commit: Optional[str] = None
        self.git_branch: Optional[str] = None
        self.config_path: Optional[str] = None
        self.config_hash: Optional[str] = None
        self.observability_db_path: Optional[str] = None
        self.mcp_url: Optional[str] = None
        self.last_backfill: Optional[Dict[str, Any]] = None
        self.lifecycle: Dict[str, Any] = {}

    def effective_status(self) -> str:
        """Derive an externally meaningful status from runtime state."""
        heartbeat = self.heartbeat or {}
        mode = str(self.data_mode or "unknown").lower()
        raw_status = str(self.status or "stopped").lower()

        if not self.running:
            return raw_status
        if mode == "replay":
            return mode
        if raw_status not in {"running", "healthy"}:
            return raw_status

        degraded = any(
            [
                bool(heartbeat.get("market_stream_error")),
                bool(heartbeat.get("feed_stale")),
                bool(heartbeat.get("broker_ack_stale")),
                bool(heartbeat.get("protection_timeout")),
                bool(heartbeat.get("fail_safe_lockout")),
                mode == "live" and heartbeat.get("market_stream_connected") is False,
            ]
        )
        return "degraded" if degraded else "healthy"
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'status': self.effective_status(),
            'process_status': self.status,
            'running': self.running,
            'data_mode': self.data_mode,
            'zone': {
                'name': self.current_zone,
                'state': self.zone_state
            },
            'strategy': self.current_strategy,
            'alpha': {
                'long_score': self.long_score,
                'short_score': self.short_score,
                'flat_bias': self.flat_bias,
                'active_vetoes': self.active_vetoes,
                'matrix_version': self.matrix_version,
                'last_entry_reason': self.last_entry_reason,
                'last_exit_reason': self.last_exit_reason,
                'active_session': self.active_session,
            },
            'session_context': {
                'anchored_vwaps': self.anchored_vwaps,
                'vwap_bands': self.vwap_bands,
                'volume_profile': self.volume_profile,
            },
            'order_flow': self.order_flow,
            'regime': self.regime,
            'position': {
                'contracts': self.position,
                'pnl': self.position_pnl
            },
            'account': {
                'id': self.account_id,
                'name': self.account_name,
                'balance': self.account_balance,
                'equity': self.account_equity,
                'available': self.account_available,
                'margin_used': self.account_margin_used,
                'open_pnl': self.account_open_pnl,
                'realized_pnl': self.account_realized_pnl,
                'daily_pnl': self.daily_pnl,
                'is_practice': self.account_is_practice,
            },
            'risk': {
                'state': self.risk_state,
                'trades_today': self.trades_today,
                'trades_this_hour': self.trades_this_hour,
                'trades_this_zone': self.trades_this_zone,
                'max_daily_loss': self.max_daily_loss,
                'consecutive_losses': self.consecutive_losses
            },
            'execution': self.execution,
            'heartbeat': self.heartbeat,
            'event_context': self.event_context,
            'replay_summary': self.replay_summary,
            'last_signal': self.last_signal,
            'last_price': self.last_price,
            'uptime_seconds': self.uptime_seconds,
            'errors': self.errors[-10:],
            'lifecycle': self.lifecycle,
            'observability': {
                'run_id': self.run_id,
                'code_version': self.code_version,
                'git_commit': self.git_commit,
                'git_branch': self.git_branch,
                'config_path': self.config_path,
                'config_hash': self.config_hash,
                'sqlite_path': self.observability_db_path,
                'mcp_url': self.mcp_url,
                'last_backfill': self.last_backfill,
            },
        }
    
    def to_health_dict(self) -> Dict[str, Any]:
        """Convert to health response."""
        return {
            'status': self.effective_status(),
            'data_mode': self.data_mode,
            'zone': self.current_zone or 'inactive',
            'position': self.position,
            'daily_pnl': self.daily_pnl,
            'risk_state': self.risk_state,
            'long_score': self.long_score,
            'short_score': self.short_score,
            'practice_account': self.account_is_practice,
            'market_stream_connected': (self.heartbeat or {}).get('market_stream_connected'),
        }


# Global state
_state = TradingState()


def get_state() -> TradingState:
    """Get global state."""
    return _state


def set_state(**kwargs):
    """Update global state."""
    for key, value in kwargs.items():
        if hasattr(_state, key):
            setattr(_state, key, value)


def record_error(message: str) -> None:
    entry = {"timestamp": datetime.now(UTC).isoformat(), "message": str(message)}
    _state.errors.append(entry)
    if len(_state.errors) > 100:
        _state.errors = _state.errors[-100:]
    get_observability_store().record_event(
        category="system",
        event_type="recorded_error",
        source=__name__,
        payload={"message": str(message)},
        event_time=datetime.now(UTC),
        action="record_error",
        reason=str(message),
    )


class HealthHandler(BaseHTTPRequestHandler):
    """Health endpoint handler."""
    
    def do_GET(self):
        """Handle GET request."""
        if self.path == '/health' or self.path == '/':
            response = _state.to_health_dict()
            self.send_response(200 if response.get("status") == "healthy" else 503)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(response, indent=2).encode())
        else:
            self.send_response(404)
            self.end_headers()
    
    def log_message(self, format, *args):
        """Suppress logging."""
        pass


class DebugHandler(BaseHTTPRequestHandler):
    """Debug endpoint handler."""

    def _write_json_response(self, status: int, payload: Optional[Dict[str, Any]], headers: Optional[Dict[str, str]] = None) -> None:
        self.send_response(status)
        if headers:
            for name, value in headers.items():
                self.send_header(name, value)
        if payload is not None:
            self.send_header('Content-Type', 'application/json')
        self.end_headers()
        if payload is not None:
            self.wfile.write(json.dumps(payload, indent=2, default=str).encode())

    def do_GET(self):
        """Handle GET request."""
        path = urlparse(self.path).path
        if path == '/debug' or path == '/':
            self._write_json_response(200, _state.to_dict())
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self):
        """Debug server has no POST endpoints; MCP runs on Railway."""
        self.send_response(404)
        self.end_headers()

    def log_message(self, format, *args):
        pass


class DebugServer:
    """
    HTTP server for health and debug endpoints.
    
    Runs two servers:
    - Health server on configured port
    - Debug server on configured port + 1
    """
    
    def __init__(self, config: Optional[ServerConfig] = None):
        """Initialize server."""
        self.config = config or get_config().server
        self.host = self.config.host
        self.health_port = self.config.health_port
        self.debug_port = self.config.debug_port
        
        self._health_server: Optional[ThreadingHTTPServer] = None
        self._debug_server: Optional[ThreadingHTTPServer] = None
        self._health_thread: Optional[threading.Thread] = None
        self._debug_thread: Optional[threading.Thread] = None
        self._running = False
    
    def start(self):
        """Start servers."""
        if self._running:
            return
        
        # Start health server
        self._health_server = ThreadingHTTPServer((self.host, self.health_port), HealthHandler)
        logger.info(f"Health server started on {self.host}:{self.health_port}")
        
        # Start debug server
        self._debug_server = ThreadingHTTPServer((self.host, self.debug_port), DebugHandler)
        logger.info(f"Debug server started on {self.host}:{self.debug_port}")

        self._running = True
        self._health_thread = threading.Thread(target=self._health_server.serve_forever, kwargs={"poll_interval": 0.25}, daemon=True)
        self._debug_thread = threading.Thread(target=self._debug_server.serve_forever, kwargs={"poll_interval": 0.25}, daemon=True)
        self._health_thread.start()
        self._debug_thread.start()
    
    def stop(self):
        """Stop servers."""
        self._running = False

        if self._health_server:
            self._health_server.shutdown()
            self._health_server.server_close()
            self._health_server = None

        if self._debug_server:
            self._debug_server.shutdown()
            self._debug_server.server_close()
            self._debug_server = None

        if self._health_thread and self._health_thread.is_alive():
            self._health_thread.join(timeout=2)
        if self._debug_thread and self._debug_thread.is_alive():
            self._debug_thread.join(timeout=2)
        self._health_thread = None
        self._debug_thread = None

        logger.info("Debug servers stopped")


# Global server
_server: Optional[DebugServer] = None


def get_server(force_recreate: bool = False) -> DebugServer:
    """Get global server instance."""
    global _server
    if force_recreate:
        _server = None
    if _server is None:
        _server = DebugServer()
    return _server

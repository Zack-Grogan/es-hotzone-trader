from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
import json
import threading
from typing import Any, Callable
from uuid import uuid4


JsonDict = dict[str, Any]
StateGetter = Callable[[], Any]
MCP_SESSION_HEADER = "Mcp-Session-Id"

_session_lock = threading.RLock()
_http_sessions: dict[str, dict[str, Any]] = {}


def get_mcp_http_metadata(endpoint: str = "/mcp") -> JsonDict:
    return {
        "name": "es-hotzone-trader-mcp",
        "transport": "streamable-http",
        "endpoint": endpoint,
        "session_header": MCP_SESSION_HEADER,
        "methods": ["initialize", "notifications/initialized", "ping", "tools/list", "tools/call", "resources/list", "resources/read"],
    }


def handle_mcp_request(request: JsonDict, state_getter: StateGetter) -> tuple[int, JsonDict]:
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}
    try:
        if method == "initialize":
            return 200, _jsonrpc_result(
                request_id,
                {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "es-hotzone-trader-mcp", "version": _app_version()},
                    "capabilities": {"tools": {}, "resources": {}},
                },
            )
        if method == "ping":
            return 200, _jsonrpc_result(request_id, {})
        if method == "tools/list":
            return 200, _jsonrpc_result(request_id, {"tools": _tool_definitions()})
        if method == "tools/call":
            tool_name = params.get("name")
            tool_args = params.get("arguments") or {}
            result = _call_tool(tool_name, tool_args, state_getter)
            return 200, _jsonrpc_result(
                request_id,
                {
                    "content": [{"type": "text", "text": json.dumps(result, indent=2, default=str)}],
                    "structuredContent": result,
                    "isError": False,
                },
            )
        if method == "resources/list":
            return 200, _jsonrpc_result(request_id, {"resources": _resource_definitions()})
        if method == "resources/read":
            uri = params.get("uri")
            content = _read_resource(uri, state_getter)
            return 200, _jsonrpc_result(
                request_id,
                {
                    "contents": [
                        {
                            "uri": uri,
                            "mimeType": "application/json",
                            "text": json.dumps(content, indent=2, default=str),
                        }
                    ]
                },
            )
        return 404, _jsonrpc_error(request_id, -32601, f"Unsupported MCP method: {method}")
    except Exception as exc:
        return 500, _jsonrpc_error(request_id, -32000, str(exc))


def handle_mcp_http_request(
    request: JsonDict,
    state_getter: StateGetter,
    *,
    session_id: str | None = None,
) -> tuple[int, JsonDict | None, dict[str, str]]:
    method = str(request.get("method") or "")

    try:
        if method == "initialize":
            status, response = handle_mcp_request(request, state_getter)
            created_session_id = _create_http_session()
            return status, response, {MCP_SESSION_HEADER: created_session_id}

        created_session_id: str | None = None
        if not session_id and _allows_implicit_http_session(method):
            created_session_id = _create_http_session(initialized=True)
            session_id = created_session_id

        session_error = _require_http_session(session_id)
        if session_error is not None:
            return session_error

        _touch_http_session(session_id)

        if method.startswith("notifications/"):
            if method == "notifications/initialized":
                _mark_http_session_initialized(session_id)
            return 202, None, {MCP_SESSION_HEADER: session_id or ""}

        if request.get("id") is None:
            return 202, None, {MCP_SESSION_HEADER: session_id or ""}

        status, response = handle_mcp_request(request, state_getter)
        return status, response, {MCP_SESSION_HEADER: session_id or ""}
    except Exception as exc:
        headers = {MCP_SESSION_HEADER: session_id} if session_id else {}
        return 500, {"error": str(exc)}, headers


def reset_mcp_sessions() -> None:
    with _session_lock:
        _http_sessions.clear()


def _create_http_session(*, initialized: bool = False) -> str:
    session_id = uuid4().hex
    with _session_lock:
        _http_sessions[session_id] = {
            "created_at": datetime.now(UTC),
            "last_seen_at": datetime.now(UTC),
            "initialized": initialized,
        }
    return session_id


def _allows_implicit_http_session(method: str) -> bool:
    return method in {
        "ping",
        "tools/list",
        "tools/call",
        "resources/list",
        "resources/read",
    }


def _require_http_session(session_id: str | None) -> tuple[int, JsonDict, dict[str, str]] | None:
    if not session_id:
        return 400, {"error": f"Missing {MCP_SESSION_HEADER} header"}, {}
    with _session_lock:
        if session_id not in _http_sessions:
            return 404, {"error": "Session not found"}, {}
    return None


def _touch_http_session(session_id: str | None) -> None:
    if not session_id:
        return
    with _session_lock:
        session = _http_sessions.get(session_id)
        if session is not None:
            session["last_seen_at"] = datetime.now(UTC)


def _mark_http_session_initialized(session_id: str | None) -> None:
    if not session_id:
        return
    with _session_lock:
        session = _http_sessions.get(session_id)
        if session is not None:
            session["initialized"] = True


def _tool_definitions() -> list[JsonDict]:
    return [
        {
            "name": "get_health",
            "description": "Return the current health snapshot.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "get_debug_state",
            "description": "Return the full in-memory debug state.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "get_runtime_summary",
            "description": "Return current runtime summary, observability context, and recent trade stats.",
            "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False},
        },
        {
            "name": "query_events",
            "description": "Query recent observability events from SQLite.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer"},
                    "category": {"type": "string"},
                    "event_type": {"type": "string"},
                    "since_minutes": {"type": "integer"},
                    "search": {"type": "string"},
                    "run_id": {"type": "string"},
                    "order_id": {"type": "string"},
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "list_runs",
            "description": "List recent run manifests and code/config provenance. Optionally filter by data_mode: 'replay' = backtest from file (offline), 'live' = real API (practice or funded account).",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer"},
                    "search": {"type": "string"},
                    "data_mode": {"type": "string", "description": "Filter runs by data_mode: 'replay' (backtest) or 'live' (real API)."},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "get_run_context",
            "description": "Return manifest, trade summary, and event counts for a run.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string"},
                    "event_limit": {"type": "integer"},
                    "trade_limit": {"type": "integer"},
                },
                "required": ["run_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "list_completed_trades",
            "description": "Query persisted completed trades from SQLite.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer"},
                    "run_id": {"type": "string"},
                    "zone": {"type": "string"},
                    "strategy": {"type": "string"},
                    "search": {"type": "string"},
                    "min_pnl": {"type": "number"},
                    "max_pnl": {"type": "number"},
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "get_performance_summary",
            "description": "Summarize completed trades for the current or specified run.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "get_event_timeline",
            "description": "Return a time-ordered event slice for a run.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["run_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "get_order_event_story",
            "description": "Return all matching events for an order id.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
        {
            "name": "summarize_execution_reconstruction",
            "description": "Summarize entry attempts, fills, and position transitions for a run or time window using observability events.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "run_id": {"type": "string"},
                    "limit": {"type": "integer"},
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        {
            "name": "explain_pnl_window",
            "description": "Return structured evidence for a PnL window using persisted trades and events.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "start_time": {"type": "string"},
                    "end_time": {"type": "string"},
                    "run_id": {"type": "string"},
                    "limit": {"type": "integer"},
                },
                "required": ["start_time", "end_time"],
                "additionalProperties": False,
            },
        },
    ]


def _resource_definitions() -> list[JsonDict]:
    return [
        {"uri": "state://health", "name": "Health", "mimeType": "application/json"},
        {"uri": "state://debug", "name": "Debug State", "mimeType": "application/json"},
        {"uri": "observability://current-run", "name": "Current Run Context", "mimeType": "application/json"},
        {"uri": "observability://runs/recent", "name": "Recent Runs", "mimeType": "application/json"},
        {"uri": "observability://performance/current", "name": "Current Performance", "mimeType": "application/json"},
    ]


def _call_tool(tool_name: str, args: JsonDict, state_getter: StateGetter) -> JsonDict:
    store = _store()
    if tool_name == "get_health":
        return state_getter().to_health_dict()
    if tool_name == "get_debug_state":
        return state_getter().to_dict()
    if tool_name == "get_runtime_summary":
        return _runtime_summary(state_getter)
    if tool_name == "query_events":
        return {"events": store.query_events(**args)}
    if tool_name == "list_runs":
        limit = int(args.get("limit", 25))
        search = args.get("search")
        data_mode = args.get("data_mode")
        runs = store.query_run_manifests(limit=max(limit, 50) if data_mode else limit, search=search)
        if data_mode:
            runs = [r for r in runs if r.get("data_mode") == data_mode][:limit]
        return {"runs": runs}
    if tool_name == "get_run_context":
        run_id = str(args["run_id"])
        event_limit = int(args.get("event_limit", 200))
        trade_limit = int(args.get("trade_limit", 200))
        trades = store.query_completed_trades(limit=trade_limit, run_id=run_id)
        events = store.query_events(limit=event_limit, run_id=run_id)
        return {
            "run": store.get_run_manifest(run_id),
            "performance": _build_performance_summary(trades),
            "trade_count": len(trades),
            "event_count": len(events),
            "event_counts": dict(Counter(event.get("category") for event in events)),
            "trades": trades,
            "recent_events": list(reversed(events)),
        }
    if tool_name == "list_completed_trades":
        return {"trades": store.query_completed_trades(**args)}
    if tool_name == "get_performance_summary":
        run_id = args.get("run_id")
        limit = int(args.get("limit", 1000))
        trades = store.query_completed_trades(limit=limit, run_id=run_id or store.get_run_id())
        if not trades and not run_id:
            trades = _runtime_trades()
        return _build_performance_summary(trades)
    if tool_name == "get_event_timeline":
        run_id = str(args["run_id"])
        limit = int(args.get("limit", 200))
        events = store.query_events(limit=limit, run_id=run_id)
        return {"events": list(reversed(events))}
    if tool_name == "get_order_event_story":
        order_id = str(args["order_id"])
        limit = int(args.get("limit", 200))
        events = store.query_events(limit=limit, order_id=order_id)
        return {"order_id": order_id, "events": list(reversed(events))}
    if tool_name == "summarize_execution_reconstruction":
        limit = int(args.get("limit", 1000))
        run_id = args.get("run_id")
        start_time = args.get("start_time")
        end_time = args.get("end_time")
        events = store.query_events(limit=limit, run_id=run_id, start_time=start_time, end_time=end_time)
        return _summarize_execution_reconstruction(events, run_id=run_id, start_time=start_time, end_time=end_time)
    if tool_name == "explain_pnl_window":
        limit = int(args.get("limit", 1000))
        run_id = args.get("run_id")
        start_time = str(args["start_time"])
        end_time = str(args["end_time"])
        trades = store.query_completed_trades(limit=limit, run_id=run_id, start_time=start_time, end_time=end_time)
        events = store.query_events(limit=limit, run_id=run_id, start_time=start_time, end_time=end_time)
        return {
            "window": {"start_time": start_time, "end_time": end_time, "run_id": run_id},
            "performance": _build_performance_summary(trades),
            "trades": trades,
            "event_counts": dict(Counter(event.get("category") for event in events)),
            "decision_counts": dict(Counter(event.get("payload", {}).get("outcome") for event in events if event.get("category") == "decision")),
            "largest_winners": sorted(trades, key=lambda item: float(item.get("pnl", 0.0)), reverse=True)[:5],
            "largest_losers": sorted(trades, key=lambda item: float(item.get("pnl", 0.0)))[:5],
            "events": list(reversed(events)),
        }
    raise ValueError(f"Unknown MCP tool: {tool_name}")


def _read_resource(uri: str, state_getter: StateGetter) -> JsonDict:
    store = _store()
    if uri == "state://health":
        return state_getter().to_health_dict()
    if uri == "state://debug":
        return state_getter().to_dict()
    if uri == "observability://current-run":
        run_id = store.get_run_id()
        trades = store.query_completed_trades(limit=500, run_id=run_id)
        return {
            "run": store.get_run_manifest(run_id),
            "performance": _build_performance_summary(trades),
            "trade_count": len(trades),
        }
    if uri == "observability://runs/recent":
        return {"runs": store.query_run_manifests(limit=25)}
    if uri == "observability://performance/current":
        return _call_tool("get_performance_summary", {}, state_getter)
    raise ValueError(f"Unknown MCP resource: {uri}")


def _runtime_summary(state_getter: StateGetter) -> JsonDict:
    state = state_getter()
    store = _store()
    trades = store.query_completed_trades(limit=200, run_id=store.get_run_id())
    if not trades:
        trades = _runtime_trades()
    return {
        "state": state.to_dict(),
        "run_id": getattr(state, "run_id", None) or store.get_run_id(),
        "observability_db_path": store.get_db_path(),
        "performance": _build_performance_summary(trades),
        "recent_trades": trades[:10],
    }


def _summarize_execution_reconstruction(
    events: list[JsonDict],
    *,
    run_id: str | None = None,
    start_time: str | None = None,
    end_time: str | None = None,
) -> JsonDict:
    chronological_events = list(reversed(events))
    event_counts = Counter(str(event.get("event_type") or "") for event in chronological_events)
    decision_counts = Counter(
        str((event.get("payload") or {}).get("outcome") or "")
        for event in chronological_events
        if str(event.get("category") or "") == "decision"
    )
    attempts_by_order: dict[str, JsonDict] = {}
    decision_only_attempts: list[JsonDict] = []
    position_transitions: list[JsonDict] = []

    def _attempt(order_id: str) -> JsonDict:
        if order_id not in attempts_by_order:
            attempts_by_order[order_id] = {
                "order_id": order_id,
                "run_id": None,
                "symbol": None,
                "zone": None,
                "side": None,
                "reason": None,
                "decision_timestamp": None,
                "submitted_at": None,
                "order_type": None,
                "limit_price": None,
                "contracts": None,
                "fill_count": 0,
                "filled_quantity": 0.0,
                "last_fill_price": None,
                "position_opened": False,
                "position_closed": False,
                "terminal_event": None,
                "terminal_reason": None,
                "events": [],
            }
        return attempts_by_order[order_id]

    def _record_attempt_event(attempt: JsonDict, event: JsonDict) -> None:
        items = attempt.setdefault("events", [])
        if len(items) >= 8:
            return
        items.append(
            {
                "event_timestamp": event.get("event_timestamp"),
                "event_type": event.get("event_type"),
                "action": event.get("action"),
                "reason": event.get("reason"),
            }
        )

    for event in chronological_events:
        payload = event.get("payload") or {}
        event_type = str(event.get("event_type") or "")
        order_id = event.get("order_id")
        category = str(event.get("category") or "")

        if category == "decision" and str(payload.get("outcome") or "") == "entry_submitted":
            attempt_payload = {
                "run_id": event.get("run_id"),
                "symbol": event.get("symbol"),
                "zone": event.get("zone") or payload.get("zone") or payload.get("zone_name"),
                "side": payload.get("side"),
                "reason": payload.get("outcome_reason") or payload.get("decision_reason") or event.get("reason"),
                "decision_timestamp": event.get("event_timestamp"),
                "submitted_at": event.get("event_timestamp"),
                "order_type": payload.get("order_type"),
                "limit_price": payload.get("limit_price"),
                "contracts": payload.get("contracts"),
            }
            if order_id:
                attempt = _attempt(str(order_id))
                attempt.update({key: value for key, value in attempt_payload.items() if value is not None})
                _record_attempt_event(attempt, event)
            else:
                decision_only_attempts.append(attempt_payload)

        if order_id:
            attempt = _attempt(str(order_id))
            attempt["run_id"] = attempt.get("run_id") or event.get("run_id")
            attempt["symbol"] = attempt.get("symbol") or event.get("symbol")
            attempt["zone"] = attempt.get("zone") or event.get("zone")
            _record_attempt_event(attempt, event)

            if event_type == "order_submitted":
                attempt["submitted_at"] = attempt.get("submitted_at") or event.get("event_timestamp")
                attempt["side"] = attempt.get("side") or payload.get("side")
                attempt["reason"] = attempt.get("reason") or event.get("reason")
                attempt["order_type"] = attempt.get("order_type") or payload.get("order_type")
                attempt["limit_price"] = attempt.get("limit_price") or payload.get("limit_price")
                attempt["contracts"] = attempt.get("contracts") or payload.get("quantity")
            elif event_type == "order_fill":
                attempt["fill_count"] = int(attempt.get("fill_count") or 0) + 1
                attempt["filled_quantity"] = float(attempt.get("filled_quantity") or 0.0) + float(payload.get("filled_quantity", 0.0) or 0.0)
                attempt["last_fill_price"] = payload.get("filled_price") or attempt.get("last_fill_price")
            elif event_type in {"order_cancelled", "order_submission_failed", "order_cancel_failed"}:
                attempt["terminal_event"] = event_type
                attempt["terminal_reason"] = event.get("reason")

        if event_type in {"position_opened", "position_closed", "position_adjusted"}:
            position_transitions.append(
                {
                    "event_timestamp": event.get("event_timestamp"),
                    "run_id": event.get("run_id"),
                    "event_type": event_type,
                    "action": event.get("action"),
                    "reason": event.get("reason"),
                    "zone": event.get("zone"),
                    "signed_position": payload.get("signed_position"),
                    "prior_position": payload.get("prior_position"),
                    "entry_price": payload.get("entry_price"),
                    "exit_price": payload.get("exit_price"),
                    "transition_price": payload.get("transition_price"),
                    "event_tags": payload.get("event_tags"),
                }
            )

    sorted_attempts = sorted(
        attempts_by_order.values(),
        key=lambda item: str(item.get("decision_timestamp") or item.get("submitted_at") or ""),
    )

    for transition in position_transitions:
        if transition.get("event_type") == "position_opened":
            for attempt in reversed(sorted_attempts):
                if attempt.get("position_opened"):
                    continue
                attempt["position_opened"] = True
                break
        elif transition.get("event_type") == "position_closed":
            for attempt in reversed(sorted_attempts):
                if attempt.get("position_opened") and not attempt.get("position_closed"):
                    attempt["position_closed"] = True
                    if not attempt.get("terminal_event"):
                        attempt["terminal_event"] = "position_closed"
                        attempt["terminal_reason"] = transition.get("reason")
                    break

    return {
        "window": {
            "run_id": run_id,
            "start_time": start_time,
            "end_time": end_time,
            "event_count": len(chronological_events),
        },
        "summary": {
            "entry_attempt_count": len(sorted_attempts) + len(decision_only_attempts),
            "filled_attempt_count": sum(1 for attempt in sorted_attempts if float(attempt.get("filled_quantity") or 0.0) > 0.0),
            "open_position_attempt_count": sum(1 for attempt in sorted_attempts if attempt.get("position_opened")),
            "closed_position_attempt_count": sum(1 for attempt in sorted_attempts if attempt.get("position_closed")),
            "decision_only_attempt_count": len(decision_only_attempts),
            "active_entry_order_blocks": int(decision_counts.get("active_entry_order", 0)),
            "position_open_blocks": int(decision_counts.get("position_open", 0)),
            "event_counts": dict(event_counts),
            "decision_counts": {key: value for key, value in dict(decision_counts).items() if key},
        },
        "entry_attempts": sorted_attempts,
        "decision_only_attempts": decision_only_attempts,
        "position_transitions": position_transitions,
    }


def _runtime_trades() -> list[JsonDict]:
    try:
        from src.engine import get_risk_manager

        trades = get_risk_manager().get_trade_history()
    except Exception:
        return []
    return [_normalize_value(asdict(trade) if is_dataclass(trade) else trade) for trade in trades]


def _build_performance_summary(trades: list[JsonDict]) -> JsonDict:
    if not trades:
        return {
            "trade_count": 0,
            "win_rate": 0.0,
            "total_pnl": 0.0,
            "avg_winner": 0.0,
            "avg_loser": 0.0,
            "scratch_trades": 0,
            "zone_stats": {},
            "regime_stats": {},
            "event_tag_stats": {},
        }
    wins = [float(trade.get("pnl", 0.0)) for trade in trades if float(trade.get("pnl", 0.0)) > 0]
    losses = [float(trade.get("pnl", 0.0)) for trade in trades if float(trade.get("pnl", 0.0)) < 0]
    scratch_trades = sum(1 for trade in trades if float(trade.get("pnl", 0.0)) == 0)
    zone_stats: dict[str, JsonDict] = {}
    regime_stats: dict[str, JsonDict] = {}
    event_tag_stats: dict[str, JsonDict] = {}
    for trade in trades:
        pnl = float(trade.get("pnl", 0.0))
        zone = trade.get("zone") or "Unknown"
        regime = trade.get("regime") or "Unknown"
        for bucket, key in ((zone_stats, zone), (regime_stats, regime)):
            stats = bucket.setdefault(key, {"trades": 0, "wins": 0, "pnl": 0.0})
            stats["trades"] += 1
            stats["pnl"] += pnl
            if pnl > 0:
                stats["wins"] += 1
        for tag in trade.get("event_tags") or ["none"]:
            stats = event_tag_stats.setdefault(tag, {"trades": 0, "wins": 0, "pnl": 0.0})
            stats["trades"] += 1
            stats["pnl"] += pnl
            if pnl > 0:
                stats["wins"] += 1
    for bucket in (zone_stats, regime_stats, event_tag_stats):
        for stats in bucket.values():
            stats["win_rate"] = stats["wins"] / max(stats["trades"], 1)
    return {
        "trade_count": len(trades),
        "win_rate": len(wins) / len(trades),
        "total_pnl": sum(float(trade.get("pnl", 0.0)) for trade in trades),
        "avg_winner": sum(wins) / len(wins) if wins else 0.0,
        "avg_loser": sum(losses) / len(losses) if losses else 0.0,
        "scratch_trades": scratch_trades,
        "zone_stats": zone_stats,
        "regime_stats": regime_stats,
        "event_tag_stats": event_tag_stats,
    }


def _store():
    from src.observability import get_observability_store

    return get_observability_store()


def _app_version() -> str:
    try:
        from src.observability.provenance import _read_package_version

        return _read_package_version()
    except Exception:
        return "0.1.0"


def _normalize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_value(item) for item in value]
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat() if value.tzinfo is not None else value.replace(tzinfo=UTC).isoformat()
    return value


def _jsonrpc_result(request_id: Any, result: JsonDict) -> JsonDict:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _jsonrpc_error(request_id: Any, code: int, message: str) -> JsonDict:
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}

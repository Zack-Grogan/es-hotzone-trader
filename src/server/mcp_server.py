from __future__ import annotations

from collections import Counter
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
import json
from typing import Any, Callable


JsonDict = dict[str, Any]
StateGetter = Callable[[], Any]


def get_mcp_http_metadata(endpoint: str = "/mcp") -> JsonDict:
    return {
        "name": "es-hotzone-trader-mcp",
        "transport": "http",
        "endpoint": endpoint,
        "methods": ["initialize", "ping", "tools/list", "tools/call", "resources/list", "resources/read"],
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
            "description": "List recent run manifests and code/config provenance.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "limit": {"type": "integer"},
                    "search": {"type": "string"},
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
        return {"runs": store.query_run_manifests(limit=int(args.get("limit", 25)), search=args.get("search"))}
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

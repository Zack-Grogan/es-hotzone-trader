"""In-process data bridge: collect state/events/trades, write to outbox, drain to Railway ingest."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from datetime import UTC, datetime
from typing import Any, Optional

from src.bridge.outbox import RailwayOutbox
from src.config import get_config
from src.observability import get_observability_store
from src.server import get_state

logger = logging.getLogger(__name__)

_bridge_thread: Optional[threading.Thread] = None
_bridge_stop = threading.Event()

_KIND_TO_CURSOR_KEY = {
    "state_snapshots": "state_snapshots",
    "market_tape": "market_tape",
    "decision_snapshots": "decision_snapshots",
    "order_lifecycle": "order_lifecycle",
    "events": "events",
    "trades": "trades",
    "account_trades": "account_trades",
    "bridge_health": "bridge_health",
    "runtime_logs": "runtime_logs",
}


def _collect_state_snapshot() -> dict[str, Any]:
    """Current debug state for ingest."""
    try:
        state = get_state().to_dict()
        run_id = state.get("run_id") or state.get("observability", {}).get("run_id")
        if run_id:
            state["run_id"] = run_id
        return state
    except Exception:
        logger.exception("Bridge: get_state failed")
        return {}


def _collect_events(limit: int = 200, after_id: int | None = None) -> list[dict[str, Any]]:
    """Recent events from observability store."""
    try:
        store = get_observability_store()
        if not store.enabled():
            return []
        rows = store.query_events(limit=limit, after_id=after_id, ascending=True)
        return [dict(r) for r in rows]
    except Exception:
        logger.exception("Bridge: query_events failed")
        return []


def _collect_trades(limit: int = 100, after_id: int | None = None) -> list[dict[str, Any]]:
    """Recent completed trades."""
    try:
        store = get_observability_store()
        if not store.enabled():
            return []
        rows = store.query_completed_trades(limit=limit, after_id=after_id, ascending=True)
        return [dict(r) for r in rows]
    except Exception:
        logger.exception("Bridge: query_completed_trades failed")
        return []


def _collect_account_trades(limit: int = 100, after_id: int | None = None) -> list[dict[str, Any]]:
    """Recent account-wide broker trade history."""
    try:
        store = get_observability_store()
        if not store.enabled():
            return []
        rows = store.query_account_trades(limit=limit, after_id=after_id, ascending=True)
        return [dict(r) for r in rows]
    except Exception:
        logger.exception("Bridge: query_account_trades failed")
        return []


def _collect_state_snapshots(limit: int = 100, after_id: int | None = None) -> list[dict[str, Any]]:
    try:
        store = get_observability_store()
        if not store.enabled():
            return []
        rows = store.query_state_snapshots(limit=limit, after_id=after_id, ascending=True)
        return [dict(r) for r in rows]
    except Exception:
        logger.exception("Bridge: query_state_snapshots failed")
        return []


def _collect_market_tape(limit: int = 500, after_id: int | None = None) -> list[dict[str, Any]]:
    try:
        store = get_observability_store()
        if not store.enabled():
            return []
        rows = store.query_market_tape(limit=limit, after_id=after_id, ascending=True)
        return [dict(r) for r in rows]
    except Exception:
        logger.exception("Bridge: query_market_tape failed")
        return []


def _collect_decision_snapshots(limit: int = 100, after_id: int | None = None) -> list[dict[str, Any]]:
    try:
        store = get_observability_store()
        if not store.enabled():
            return []
        rows = store.query_decision_snapshots(limit=limit, after_id=after_id, ascending=True)
        return [dict(r) for r in rows]
    except Exception:
        logger.exception("Bridge: query_decision_snapshots failed")
        return []


def _collect_order_lifecycle(limit: int = 100, after_id: int | None = None) -> list[dict[str, Any]]:
    try:
        store = get_observability_store()
        if not store.enabled():
            return []
        rows = store.query_order_lifecycle(limit=limit, after_id=after_id, ascending=True)
        return [dict(r) for r in rows]
    except Exception:
        logger.exception("Bridge: query_order_lifecycle failed")
        return []


def _collect_bridge_health(limit: int = 50, after_id: int | None = None) -> list[dict[str, Any]]:
    try:
        store = get_observability_store()
        if not store.enabled():
            return []
        rows = store.query_bridge_health(limit=limit, after_id=after_id, ascending=True)
        return [dict(r) for r in rows]
    except Exception:
        logger.exception("Bridge: query_bridge_health failed")
        return []


def _collect_runtime_logs(limit: int = 200, after_id: int | None = None) -> list[dict[str, Any]]:
    try:
        store = get_observability_store()
        if not store.enabled():
            return []
        rows = store.query_runtime_logs(limit=limit, after_id=after_id, ascending=True)
        return [dict(r) for r in rows]
    except Exception:
        logger.exception("Bridge: query_runtime_logs failed")
        return []


def _bridge_api_token() -> str:
    cfg = get_config()
    obs = cfg.observability
    return (
        os.environ.get("GTRADE_INTERNAL_API_TOKEN")
        or getattr(obs, "internal_api_token", None)
        or ""
    ).strip()


def _payload_max_local_id(kind: str, payload: dict[str, Any]) -> int:
    cursor_key = _KIND_TO_CURSOR_KEY.get(kind)
    if not cursor_key:
        return 0
    rows = payload.get(cursor_key)
    if not isinstance(rows, list) or not rows:
        return 0
    try:
        return max(int(row.get("id", 0)) for row in rows if isinstance(row, dict))
    except Exception:
        return 0


def _record_delivery_success(
    outbox: RailwayOutbox,
    kind: str,
    payload: dict[str, Any],
    *,
    row_batch_id: str,
) -> None:
    max_local_id = int(((payload.get("_cursor") or {}).get("max_local_id")) or _payload_max_local_id(kind, payload))
    if max_local_id > 0 or kind == "run_manifest":
        outbox.update_delivery_cursor(
            kind,
            max_local_id,
            last_batch_id=row_batch_id,
            last_success_at=datetime.now(UTC).isoformat(),
            last_error=None,
        )


def _is_permanent_http_error(status_code: int) -> bool:
    return status_code in {400, 401, 403, 404, 409, 422}


def rebuild_outbox_from_observability(
    outbox: RailwayOutbox,
    *,
    run_id: Optional[str] = None,
    include_sent: bool = False,
    limit_per_kind: int = 1000,
) -> dict[str, int]:
    store = get_observability_store()
    if not store.enabled():
        return {}
    sent_state = outbox.get_delivery_state()
    counts: dict[str, int] = {}
    collectors: list[tuple[str, Any, str]] = [
        ("events", store.query_events, "events"),
        ("trades", store.query_completed_trades, "trades"),
        ("account_trades", store.query_account_trades, "account_trades"),
        ("state_snapshots", store.query_state_snapshots, "state_snapshots"),
        ("market_tape", store.query_market_tape, "market_tape"),
        ("decision_snapshots", store.query_decision_snapshots, "decision_snapshots"),
        ("order_lifecycle", store.query_order_lifecycle, "order_lifecycle"),
        ("bridge_health", store.query_bridge_health, "bridge_health"),
        ("runtime_logs", store.query_runtime_logs, "runtime_logs"),
    ]
    for kind, query_fn, payload_key in collectors:
        after_id = None
        if not include_sent:
            after_id = int(sent_state.get(kind, {}).get("cursor_value") or 0)
        kwargs = {"limit": limit_per_kind, "after_id": after_id, "ascending": True}
        if run_id and kind != "bridge_health":
            kwargs["run_id"] = run_id
        rows = query_fn(**kwargs)
        if not rows:
            continue
        payload = {payload_key: [dict(row) for row in rows]}
        max_local_id = _payload_max_local_id(kind, payload)
        payload["_cursor"] = {"kind": kind, "max_local_id": max_local_id}
        batch_id = f"replay_{kind}_{run_id or 'all'}_{max_local_id}_{int(time.time() * 1000)}"
        if outbox.enqueue(kind, payload, batch_id=batch_id):
            counts[kind] = len(rows)
    manifest = store.get_run_manifest(run_id) if run_id else None
    if manifest:
        outbox.enqueue("run_manifest", manifest, batch_id=f"replay_run_manifest_{run_id}")
        counts["run_manifest"] = 1
    return counts


def _run_bridge_loop(outbox: RailwayOutbox, ingest_url: str, api_key: str, interval: float) -> None:
    """Collect and enqueue periodically; drain outbox with retry."""
    import requests

    sent_run_manifest = False
    last_state_id_sent = 0
    last_market_id_sent = 0
    last_decision_id_sent = 0
    last_order_id_sent = 0
    last_state_health_id_sent = 0
    last_runtime_log_id_sent = 0
    last_event_id_sent = 0
    last_trade_id_sent = 0
    last_account_trade_id_sent = 0

    while not _bridge_stop.is_set():
        try:
            cfg = get_config()
            obs = cfg.observability
            now = time.time()

            store = get_observability_store()
            run_id = store.get_run_id() if store.enabled() else "unknown"

            if not sent_run_manifest and run_id and run_id != "unknown":
                manifest = store.get_run_manifest(run_id)
                if manifest:
                    outbox.enqueue("run_manifest", manifest, batch_id=f"run_manifest_{run_id}")
                    sent_run_manifest = True

            # Drain locally recorded raw streams in order.
            while True:
                rows = _collect_state_snapshots(limit=250, after_id=last_state_id_sent)
                if not rows:
                    break
                rows.sort(key=lambda item: int(item.get("id", 0)))
                last_state_id_sent = max(last_state_id_sent, max(int(row.get("id", 0)) for row in rows))
                outbox.enqueue("state_snapshots", {"state_snapshots": rows}, batch_id=f"state_snapshots_{run_id}_{last_state_id_sent}")
                if len(rows) < 250:
                    break

            while True:
                rows = _collect_market_tape(limit=500, after_id=last_market_id_sent)
                if not rows:
                    break
                rows.sort(key=lambda item: int(item.get("id", 0)))
                last_market_id_sent = max(last_market_id_sent, max(int(row.get("id", 0)) for row in rows))
                outbox.enqueue("market_tape", {"market_tape": rows}, batch_id=f"market_tape_{run_id}_{last_market_id_sent}")
                if len(rows) < 500:
                    break

            while True:
                rows = _collect_decision_snapshots(limit=250, after_id=last_decision_id_sent)
                if not rows:
                    break
                rows.sort(key=lambda item: int(item.get("id", 0)))
                last_decision_id_sent = max(last_decision_id_sent, max(int(row.get("id", 0)) for row in rows))
                outbox.enqueue("decision_snapshots", {"decision_snapshots": rows}, batch_id=f"decision_snapshots_{run_id}_{last_decision_id_sent}")
                if len(rows) < 250:
                    break

            while True:
                rows = _collect_order_lifecycle(limit=250, after_id=last_order_id_sent)
                if not rows:
                    break
                rows.sort(key=lambda item: int(item.get("id", 0)))
                last_order_id_sent = max(last_order_id_sent, max(int(row.get("id", 0)) for row in rows))
                outbox.enqueue("order_lifecycle", {"order_lifecycle": rows}, batch_id=f"order_lifecycle_{run_id}_{last_order_id_sent}")
                if len(rows) < 250:
                    break

            # Incrementally enqueue events/trades so older blockers are not skipped under bursty load.
            while True:
                events = _collect_events(limit=250, after_id=last_event_id_sent)
                if not events:
                    break
                events.sort(key=lambda item: int(item.get("id", 0)))
                last_event_id_sent = max(last_event_id_sent, max(int(event.get("id", 0)) for event in events))
                batch_id = f"events_{run_id}_{last_event_id_sent}"
                outbox.enqueue("events", {"events": events}, batch_id=batch_id)
                if len(events) < 250:
                    break

            while True:
                trades = _collect_trades(limit=250, after_id=last_trade_id_sent)
                if not trades:
                    break
                trades.sort(key=lambda item: int(item.get("id", 0)))
                last_trade_id_sent = max(last_trade_id_sent, max(int(trade.get("id", 0)) for trade in trades))
                batch_id = f"trades_{run_id}_{last_trade_id_sent}"
                outbox.enqueue("trades", {"trades": trades}, batch_id=batch_id)
                if len(trades) < 250:
                    break

            while True:
                account_trades = _collect_account_trades(limit=250, after_id=last_account_trade_id_sent)
                if not account_trades:
                    break
                account_trades.sort(key=lambda item: int(item.get("id", 0)))
                last_account_trade_id_sent = max(
                    last_account_trade_id_sent,
                    max(int(trade.get("id", 0)) for trade in account_trades),
                )
                outbox.enqueue(
                    "account_trades",
                    {
                        "account_trades": account_trades,
                        "_cursor": {"kind": "account_trades", "max_local_id": last_account_trade_id_sent},
                    },
                    batch_id=f"account_trades_{run_id}_{last_account_trade_id_sent}",
                )
                if len(account_trades) < 250:
                    break

            bridge_health = {
                "run_id": run_id,
                "bridge_status": "running",
                "queue_depth": None,
                "last_flush_at": None,
                "last_success_at": None,
                "last_error": None,
                "observed_at": datetime.now(UTC).isoformat(),
                "bridge_interval_seconds": interval,
                "state_stream_last_id": last_state_id_sent,
                "market_tape_last_id": last_market_id_sent,
                "decision_snapshot_last_id": last_decision_id_sent,
                "order_lifecycle_last_id": last_order_id_sent,
                "event_last_id": last_event_id_sent,
                "trade_last_id": last_trade_id_sent,
                "sent_run_manifest": sent_run_manifest,
            }
            store.record_bridge_health(bridge_health)
            store.force_flush()
            health_rows = _collect_bridge_health(limit=50, after_id=last_state_health_id_sent)
            if health_rows:
                health_rows.sort(key=lambda item: int(item.get("id", 0)))
                last_state_health_id_sent = max(last_state_health_id_sent, max(int(row.get("id", 0)) for row in health_rows))
                outbox.enqueue(
                    "bridge_health",
                    {"bridge_health": health_rows, "_cursor": {"kind": "bridge_health", "max_local_id": last_state_health_id_sent}},
                    batch_id=f"bridge_health_{run_id}_{last_state_health_id_sent}",
                )

            while True:
                rows = _collect_runtime_logs(limit=250, after_id=last_runtime_log_id_sent)
                if not rows:
                    break
                rows.sort(key=lambda item: int(item.get("id", 0)))
                last_runtime_log_id_sent = max(last_runtime_log_id_sent, max(int(row.get("id", 0)) for row in rows))
                outbox.enqueue(
                    "runtime_logs",
                    {"runtime_logs": rows, "_cursor": {"kind": "runtime_logs", "max_local_id": last_runtime_log_id_sent}},
                    batch_id=f"runtime_logs_{run_id}_{last_runtime_log_id_sent}",
                )
                if len(rows) < 250:
                    break

            # Drain outbox
            batches = outbox.dequeue_batch(limit=10)
            for row in batches:
                row_id = row["id"]
                kind = row["kind"]
                try:
                    payload = json.loads(row["payload_json"])
                except Exception as e:
                    outbox.mark_failed(row_id, str(e))
                    continue
                path = {
                    "state_snapshots": "/ingest/state-snapshots",
                    "market_tape": "/ingest/market-tape",
                    "decision_snapshots": "/ingest/decision-snapshots",
                    "order_lifecycle": "/ingest/order-lifecycle",
                    "events": "/ingest/events",
                    "trades": "/ingest/trades",
                    "account_trades": "/ingest/account-trades",
                    "run_manifest": "/ingest/run-manifest",
                    "bridge_health": "/ingest/bridge-health",
                    "runtime_logs": "/ingest/runtime-logs",
                }.get(kind, "")
                if not path:
                    outbox.mark_sent(row_id)
                    continue
                url = ingest_url.rstrip("/") + path
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
                attempts = getattr(obs, "bridge_retry_attempts", 5)
                base_delay = getattr(obs, "bridge_retry_base_seconds", 2.0)
                sent = False
                permanent_error = False
                for attempt in range(attempts):
                    try:
                        r = requests.post(url, json=payload, headers=headers, timeout=15)
                        if 200 <= r.status_code < 300:
                            outbox.mark_sent(row_id)
                            _record_delivery_success(
                                outbox,
                                kind,
                                payload,
                                row_batch_id=row.get("batch_id"),
                            )
                            sent = True
                            break
                        err = f"HTTP {r.status_code}"
                        permanent_error = _is_permanent_http_error(r.status_code)
                        if permanent_error:
                            break
                    except requests.RequestException as e:
                        err = str(e)
                    if attempt < attempts - 1:
                        time.sleep(base_delay * (2**attempt))
                if not sent:
                    outbox.mark_failed(row_id, err, permanent=permanent_error)
                    outbox.update_delivery_cursor(
                        kind,
                        int(((payload.get("_cursor") or {}).get("max_local_id")) or 0),
                        last_batch_id=row.get("batch_id"),
                        last_success_at=None,
                        last_error=err,
                    )
                    store.record_bridge_health(
                        {
                            "run_id": run_id,
                            "bridge_status": "error" if permanent_error else "degraded",
                            "queue_depth": outbox.get_queue_stats().get("total"),
                            "last_error": err,
                            "observed_at": datetime.now(UTC).isoformat(),
                            "kind": kind,
                            "permanent_failure": permanent_error,
                        }
                    )
                    store.force_flush()

        except Exception:
            logger.exception("Bridge loop error")
        _bridge_stop.wait(timeout=interval)


def start_railway_bridge() -> bool:
    """Start the bridge thread if config has a non-empty ingest URL. Returns True if started."""
    global _bridge_thread
    cfg = get_config()
    obs = cfg.observability
    url = (getattr(obs, "railway_ingest_url", None) or "").strip()
    api_key = _bridge_api_token()
    if not url or not api_key:
        logger.debug("Railway bridge disabled: no railway_ingest_url or internal auth token")
        return False
    if _bridge_thread is not None and _bridge_thread.is_alive():
        return True
    _bridge_stop.clear()
    outbox_path = getattr(obs, "outbox_path", "logs/railway_outbox.db") or "logs/railway_outbox.db"
    interval = max(5.0, getattr(obs, "bridge_interval_seconds", 30.0))
    outbox = RailwayOutbox(outbox_path)

    def _run() -> None:
        _run_bridge_loop(outbox, url, api_key, interval)
        outbox.close()

    _bridge_thread = threading.Thread(target=_run, name="railway-bridge", daemon=True)
    _bridge_thread.start()
    logger.info("Railway bridge started (ingest_url=%s)", url.split("/")[2] if "/" in url else url)
    return True


def stop_railway_bridge() -> None:
    """Signal the bridge thread to stop."""
    _bridge_stop.set()

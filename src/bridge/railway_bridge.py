"""In-process data bridge: collect state/events/trades, write to outbox, drain to Railway ingest."""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from typing import Any, Optional

from src.bridge.outbox import RailwayOutbox
from src.config import get_config
from src.observability import get_observability_store
from src.server import get_state

logger = logging.getLogger(__name__)

_bridge_thread: Optional[threading.Thread] = None
_bridge_stop = threading.Event()


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


def _run_bridge_loop(outbox: RailwayOutbox, ingest_url: str, api_key: str, interval: float) -> None:
    """Collect and enqueue periodically; drain outbox with retry."""
    import requests

    last_state_ts = 0.0
    state_interval = max(10.0, interval * 0.5)
    sent_run_manifest = False
    last_event_id_sent = 0
    last_trade_id_sent = 0

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

            # Collect and enqueue state periodically
            if now - last_state_ts >= state_interval:
                state = _collect_state_snapshot()
                if state:
                    outbox.enqueue("state", state, batch_id=f"state_{run_id}_{int(now)}")
                last_state_ts = now

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

            outbox.enqueue(
                "bridge_health",
                {
                    "run_id": run_id,
                    "bridge_status": "running",
                    "bridge_interval_seconds": interval,
                    "state_interval_seconds": state_interval,
                    "sent_run_manifest": sent_run_manifest,
                    "observed_at": time.time(),
                },
                batch_id=f"bridge_health_{run_id}_{int(now)}",
            )

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
                    "state": "/ingest/state",
                    "events": "/ingest/events",
                    "trades": "/ingest/trades",
                    "run_manifest": "/ingest/run-manifest",
                    "bridge_health": "/ingest/bridge-health",
                }.get(kind, "")
                if not path:
                    outbox.mark_sent(row_id)
                    continue
                url = ingest_url.rstrip("/") + path
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
                attempts = getattr(obs, "bridge_retry_attempts", 5)
                base_delay = getattr(obs, "bridge_retry_base_seconds", 2.0)
                sent = False
                for attempt in range(attempts):
                    try:
                        r = requests.post(url, json=payload, headers=headers, timeout=15)
                        if 200 <= r.status_code < 300:
                            outbox.mark_sent(row_id)
                            sent = True
                            break
                        err = f"HTTP {r.status_code}"
                    except requests.RequestException as e:
                        err = str(e)
                    if attempt < attempts - 1:
                        time.sleep(base_delay * (2**attempt))
                if not sent:
                    outbox.mark_failed(row_id, err)

        except Exception:
            logger.exception("Bridge loop error")
        _bridge_stop.wait(timeout=interval)


def start_railway_bridge() -> bool:
    """Start the bridge thread if config has a non-empty ingest URL. Returns True if started."""
    global _bridge_thread
    cfg = get_config()
    obs = cfg.observability
    url = (getattr(obs, "railway_ingest_url", None) or "").strip()
    api_key = (os.environ.get("RAILWAY_INGEST_API_KEY") or getattr(obs, "railway_ingest_api_key", None) or "").strip()
    if not url or not api_key:
        logger.debug("Railway bridge disabled: no railway_ingest_url or railway_ingest_api_key")
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

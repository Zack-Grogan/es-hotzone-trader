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
        return get_state().to_dict()
    except Exception:
        logger.exception("Bridge: get_state failed")
        return {}


def _collect_events(limit: int = 200) -> list[dict[str, Any]]:
    """Recent events from observability store."""
    try:
        store = get_observability_store()
        if not store.enabled():
            return []
        rows = store.query_events(limit=limit)
        return [dict(r) for r in rows]
    except Exception:
        logger.exception("Bridge: query_events failed")
        return []


def _collect_trades(limit: int = 100) -> list[dict[str, Any]]:
    """Recent completed trades."""
    try:
        store = get_observability_store()
        if not store.enabled():
            return []
        rows = store.query_completed_trades(limit=limit)
        return [dict(r) for r in rows]
    except Exception:
        logger.exception("Bridge: query_completed_trades failed")
        return []


def _run_bridge_loop(outbox: RailwayOutbox, ingest_url: str, api_key: str, interval: float) -> None:
    """Collect and enqueue periodically; drain outbox with retry."""
    import requests

    last_state_ts = 0.0
    state_interval = max(10.0, interval * 0.5)

    while not _bridge_stop.is_set():
        try:
            cfg = get_config()
            obs = cfg.observability
            now = time.time()

            store = get_observability_store()
            run_id = store.get_run_id() if store.enabled() else "unknown"

            # Collect and enqueue state periodically
            if now - last_state_ts >= state_interval:
                state = _collect_state_snapshot()
                if state:
                    outbox.enqueue("state", state, batch_id=f"state_{run_id}_{int(now)}")
                last_state_ts = now

            # Enqueue events/trades with deterministic batch_id so we don't duplicate (INSERT OR IGNORE)
            interval_ts = int(now // interval) * int(interval)
            events = _collect_events(limit=100)
            if events:
                outbox.enqueue("events", {"events": events}, batch_id=f"events_{run_id}_{interval_ts}")
            trades = _collect_trades(limit=50)
            if trades:
                outbox.enqueue("trades", {"trades": trades}, batch_id=f"trades_{run_id}_{interval_ts}")

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
                path = {"state": "/ingest/state", "events": "/ingest/events", "trades": "/ingest/trades"}.get(kind, "")
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

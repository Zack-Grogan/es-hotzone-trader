from __future__ import annotations

import json
import logging
import os
import queue
import sqlite3
import threading
import time
from collections import defaultdict
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from src.config import Config, get_config

logger = logging.getLogger(__name__)


class ObservabilityStore:
    def __init__(self, config: Optional[Config] = None):
        root_config = config or get_config()
        self.config = root_config
        self.settings = root_config.observability
        self._db_path = self._resolve_db_path(self.settings.sqlite_path)
        self._queue: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=max(int(self.settings.queue_max_size), 1))
        self._lock = threading.RLock()
        self._worker: Optional[threading.Thread] = None
        self._conn: Optional[sqlite3.Connection] = None
        self._running = False
        self._failed = False
        self._dropped_events = 0
        self._run_id = f"{int(time.time())}-{os.getpid()}"

    def _resolve_db_path(self, sqlite_path: str) -> Path:
        path = Path(sqlite_path)
        if path.is_absolute():
            return path
        project_root = Path(__file__).resolve().parent.parent.parent
        return project_root / path

    def _account_mode(self, is_practice: Any) -> Optional[str]:
        practice_value = self._coerce_bool(is_practice)
        if practice_value is None:
            return None
        return "practice" if practice_value else "live"

    def _extract_account_context(self, payload: dict[str, Any]) -> dict[str, Any]:
        account = payload.get("account") if isinstance(payload.get("account"), dict) else {}
        account_id = payload.get("account_id") or account.get("id") or account.get("account_id")
        account_name = payload.get("account_name") or account.get("name") or account.get("account_name")
        account_is_practice = payload.get("account_is_practice")
        if account_is_practice is None:
            account_is_practice = account.get("is_practice")
        if account_is_practice is None:
            account_is_practice = account.get("practice_account")
        if account_is_practice is None:
            account_is_practice = account.get("practice")
        if account_is_practice is None:
            account_is_practice = account.get("simulated")
        return {
            "account_id": str(account_id) if account_id not in {None, ""} else None,
            "account_name": str(account_name) if account_name not in {None, ""} else None,
            "account_is_practice": self._coerce_bool(account_is_practice),
            "account_mode": self._account_mode(account_is_practice),
        }

    def enabled(self) -> bool:
        return bool(getattr(self.settings, "enabled", False)) and not self._failed

    def start(self) -> None:
        if not self.enabled():
            return
        try:
            with self._lock:
                if self._running:
                    return
                self._db_path.parent.mkdir(parents=True, exist_ok=True)
                self._conn = sqlite3.connect(self._db_path, check_same_thread=False, timeout=5.0)
                self._conn.row_factory = sqlite3.Row
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
                self._conn.execute("PRAGMA temp_store=MEMORY")
                self._ensure_schema()
                self._prune_old_records_locked()
                self._running = True
                self._worker = threading.Thread(target=self._worker_loop, name="observability-store", daemon=True)
                self._worker.start()
                logger.info("Observability store started at %s", self._db_path)
        except Exception:
            self._failed = True
            logger.exception("Failed to start observability store")
            with self._lock:
                if self._conn is not None:
                    try:
                        self._conn.close()
                    except Exception:
                        pass
                    self._conn = None
                self._running = False

    def stop(self) -> None:
        worker: Optional[threading.Thread] = None
        with self._lock:
            if not self._running and self._conn is None:
                return
            self._running = False
            worker = self._worker
            self._worker = None
        if worker is not None:
            worker.join(timeout=5)
        with self._lock:
            self._flush_locked()
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def force_flush(self) -> None:
        if not self.enabled():
            return
        self.start()
        with self._lock:
            self._flush_locked()

    def record_event(
        self,
        *,
        category: str,
        event_type: str,
        source: str,
        payload: Optional[dict[str, Any]] = None,
        event_time: Optional[datetime] = None,
        symbol: Optional[str] = None,
        zone: Optional[str] = None,
        action: Optional[str] = None,
        reason: Optional[str] = None,
        order_id: Optional[str] = None,
        risk_state: Optional[str] = None,
    ) -> None:
        if not self.enabled():
            return
        try:
            self.start()
            if not self.enabled():
                return
            event = {
                "record_type": "event",
                "event_timestamp": self._serialize_datetime(event_time or datetime.now(UTC)),
                "inserted_at": self._serialize_datetime(datetime.now(UTC)),
                "run_id": self._run_id,
                "process_id": os.getpid(),
                "category": str(category),
                "event_type": str(event_type),
                "source": str(source),
                "symbol": symbol,
                "zone": zone,
                "action": action,
                "reason": reason,
                "order_id": order_id,
                "risk_state": risk_state,
                "payload_json": json.dumps(self._normalize_value(payload or {}), separators=(",", ":"), sort_keys=True),
            }
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                self._dropped_events += 1
                if self._dropped_events in {1, 10, 100} or self._dropped_events % 1000 == 0:
                    logger.warning("Observability queue full; dropped %s events", self._dropped_events)
        except Exception:
            self._failed = True
            logger.exception("Failed to record observability event category=%s event_type=%s", category, event_type)

    def record_state_snapshot(self, snapshot: dict[str, Any], event_time: Optional[datetime] = None) -> None:
        if not self.enabled():
            return
        try:
            self.start()
            payload = self._normalize_value(snapshot or {})
            observability = payload.get("observability") if isinstance(payload.get("observability"), dict) else {}
            account_context = self._extract_account_context(payload)
            record = {
                "record_type": "state_snapshot",
                "captured_at": self._coerce_datetime_value(event_time or datetime.now(UTC)),
                "inserted_at": self._serialize_datetime(datetime.now(UTC)),
                "run_id": str(payload.get("run_id") or self._run_id),
                "process_id": int(payload.get("process_id") or os.getpid()),
                "status": payload.get("status"),
                "data_mode": payload.get("data_mode"),
                "symbol": payload.get("symbol") or (observability.get("symbols") or [None])[0],
                "zone": (payload.get("zone") or {}).get("name") if isinstance(payload.get("zone"), dict) else payload.get("zone"),
                "zone_state": (payload.get("zone") or {}).get("state") if isinstance(payload.get("zone"), dict) else payload.get("zone_state"),
                "position": (payload.get("position") or {}).get("contracts") if isinstance(payload.get("position"), dict) else payload.get("position"),
                "position_pnl": (payload.get("position") or {}).get("pnl") if isinstance(payload.get("position"), dict) else payload.get("position_pnl"),
                "daily_pnl": (payload.get("account") or {}).get("daily_pnl") if isinstance(payload.get("account"), dict) else payload.get("daily_pnl"),
                "risk_state": (payload.get("risk") or {}).get("state") if isinstance(payload.get("risk"), dict) else payload.get("risk_state"),
                "account_id": account_context["account_id"],
                "account_name": account_context["account_name"],
                "account_mode": account_context["account_mode"],
                "account_is_practice": account_context["account_is_practice"],
                "decision_id": ((payload.get("execution") or {}).get("decision_id") if isinstance(payload.get("execution"), dict) else None) or payload.get("decision_id"),
                "attempt_id": ((payload.get("execution") or {}).get("attempt_id") if isinstance(payload.get("execution"), dict) else None) or payload.get("attempt_id"),
                "position_id": ((payload.get("execution") or {}).get("position_id") if isinstance(payload.get("execution"), dict) else None) or payload.get("position_id"),
                "trade_id": ((payload.get("execution") or {}).get("trade_id") if isinstance(payload.get("execution"), dict) else None) or payload.get("trade_id"),
                "payload_json": self._json_dumps(payload),
            }
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped_events += 1
            if self._dropped_events in {1, 10, 100} or self._dropped_events % 1000 == 0:
                logger.warning("Observability queue full; dropped %s records", self._dropped_events)
        except Exception:
            self._failed = True
            logger.exception("Failed to record state snapshot")

    def record_market_tick(self, tick: dict[str, Any]) -> None:
        if not self.enabled():
            return
        try:
            self.start()
            payload = self._normalize_value(tick or {})
            record = {
                "record_type": "market_tape",
                "captured_at": self._coerce_datetime_value(payload.get("captured_at") or payload.get("event_time") or payload.get("timestamp") or datetime.now(UTC)),
                "inserted_at": self._serialize_datetime(datetime.now(UTC)),
                "run_id": str(payload.get("run_id") or self._run_id),
                "process_id": int(payload.get("process_id") or os.getpid()),
                "symbol": payload.get("symbol"),
                "contract_id": payload.get("contract_id"),
                "bid": payload.get("bid"),
                "ask": payload.get("ask"),
                "last": payload.get("last"),
                "volume": payload.get("volume"),
                "bid_size": payload.get("bid_size"),
                "ask_size": payload.get("ask_size"),
                "last_size": payload.get("last_size"),
                "volume_is_cumulative": self._coerce_bool(payload.get("volume_is_cumulative")),
                "quote_is_synthetic": self._coerce_bool(payload.get("quote_is_synthetic")),
                "trade_side": payload.get("trade_side"),
                "latency_ms": payload.get("latency_ms"),
                "source": payload.get("source"),
                "sequence": payload.get("sequence"),
                "payload_json": self._json_dumps(payload),
            }
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped_events += 1
            if self._dropped_events in {1, 10, 100} or self._dropped_events % 1000 == 0:
                logger.warning("Observability queue full; dropped %s records", self._dropped_events)
        except Exception:
            self._failed = True
            logger.exception("Failed to record market tick")

    def record_decision_snapshot(self, snapshot: dict[str, Any]) -> None:
        if not self.enabled():
            return
        try:
            self.start()
            payload = self._normalize_value(snapshot or {})
            record = {
                "record_type": "decision_snapshot",
                "decided_at": self._coerce_datetime_value(payload.get("decided_at") or payload.get("event_time") or datetime.now(UTC)),
                "inserted_at": self._serialize_datetime(datetime.now(UTC)),
                "run_id": str(payload.get("run_id") or self._run_id),
                "process_id": int(payload.get("process_id") or os.getpid()),
                "decision_id": str(payload.get("decision_id") or f"{self._run_id}:decision:{int(time.time() * 1000)}"),
                "attempt_id": payload.get("attempt_id"),
                "symbol": payload.get("symbol"),
                "zone": payload.get("zone"),
                "action": payload.get("action"),
                "reason": payload.get("reason"),
                "outcome": payload.get("outcome"),
                "outcome_reason": payload.get("outcome_reason"),
                "long_score": payload.get("long_score"),
                "short_score": payload.get("short_score"),
                "flat_bias": payload.get("flat_bias"),
                "score_gap": payload.get("score_gap"),
                "dominant_side": payload.get("dominant_side"),
                "current_price": payload.get("current_price"),
                "allow_entries": self._coerce_bool(payload.get("allow_entries")),
                "execution_tradeable": self._coerce_bool(payload.get("execution_tradeable")),
                "contracts": payload.get("contracts"),
                "order_type": payload.get("order_type"),
                "limit_price": payload.get("limit_price"),
                "decision_price": payload.get("decision_price"),
                "side": payload.get("side"),
                "stop_loss": payload.get("stop_loss"),
                "take_profit": payload.get("take_profit"),
                "max_hold_minutes": payload.get("max_hold_minutes"),
                "regime_state": payload.get("regime_state"),
                "regime_reason": payload.get("regime_reason"),
                "active_session": payload.get("active_session"),
                "active_vetoes_json": self._json_dumps(list(payload.get("active_vetoes") or [])),
                "feature_snapshot_json": self._json_dumps(payload.get("feature_snapshot") or {}),
                "entry_guard_json": self._json_dumps(payload.get("entry_guard") or {}),
                "unresolved_entry_json": self._json_dumps(payload.get("unresolved_entry") or {}),
                "event_context_json": self._json_dumps(payload.get("event_context") or {}),
                "order_flow_json": self._json_dumps(payload.get("order_flow") or {}),
                "payload_json": self._json_dumps(payload),
            }
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped_events += 1
            if self._dropped_events in {1, 10, 100} or self._dropped_events % 1000 == 0:
                logger.warning("Observability queue full; dropped %s records", self._dropped_events)
        except Exception:
            self._failed = True
            logger.exception("Failed to record decision snapshot")

    def record_order_lifecycle(self, lifecycle: dict[str, Any]) -> None:
        if not self.enabled():
            return
        try:
            self.start()
            payload = self._normalize_value(lifecycle or {})
            record = {
                "record_type": "order_lifecycle",
                "observed_at": self._coerce_datetime_value(payload.get("observed_at") or payload.get("event_time") or datetime.now(UTC)),
                "inserted_at": self._serialize_datetime(datetime.now(UTC)),
                "run_id": str(payload.get("run_id") or self._run_id),
                "process_id": int(payload.get("process_id") or os.getpid()),
                "decision_id": payload.get("decision_id"),
                "attempt_id": payload.get("attempt_id"),
                "order_id": payload.get("order_id"),
                "position_id": payload.get("position_id"),
                "trade_id": payload.get("trade_id"),
                "symbol": payload.get("symbol"),
                "event_type": payload.get("event_type"),
                "status": payload.get("status"),
                "side": payload.get("side"),
                "role": payload.get("role"),
                "is_protective": self._coerce_bool(payload.get("is_protective")),
                "order_type": payload.get("order_type"),
                "quantity": payload.get("quantity"),
                "contracts": payload.get("contracts"),
                "limit_price": payload.get("limit_price"),
                "stop_price": payload.get("stop_price"),
                "expected_fill_price": payload.get("expected_fill_price"),
                "filled_price": payload.get("filled_price"),
                "filled_quantity": payload.get("filled_quantity"),
                "remaining_quantity": payload.get("remaining_quantity"),
                "zone": payload.get("zone"),
                "reason": payload.get("reason"),
                "lifecycle_state": payload.get("lifecycle_state"),
                "payload_json": self._json_dumps(payload),
            }
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped_events += 1
            if self._dropped_events in {1, 10, 100} or self._dropped_events % 1000 == 0:
                logger.warning("Observability queue full; dropped %s records", self._dropped_events)
        except Exception:
            self._failed = True
            logger.exception("Failed to record order lifecycle")

    def record_bridge_health(self, health: dict[str, Any]) -> None:
        if not self.enabled():
            return
        try:
            self.start()
            payload = self._normalize_value(health or {})
            record = {
                "record_type": "bridge_health",
                "observed_at": self._coerce_datetime_value(payload.get("observed_at") or datetime.now(UTC)),
                "inserted_at": self._serialize_datetime(datetime.now(UTC)),
                "run_id": str(payload.get("run_id") or self._run_id),
                "bridge_status": payload.get("bridge_status"),
                "queue_depth": payload.get("queue_depth"),
                "last_flush_at": self._coerce_optional_datetime_value(payload.get("last_flush_at")),
                "last_success_at": self._coerce_optional_datetime_value(payload.get("last_success_at")),
                "last_error": payload.get("last_error"),
                "payload_json": self._json_dumps(payload),
            }
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped_events += 1
            if self._dropped_events in {1, 10, 100} or self._dropped_events % 1000 == 0:
                logger.warning("Observability queue full; dropped %s records", self._dropped_events)
        except Exception:
            self._failed = True
            logger.exception("Failed to record bridge health")

    def record_runtime_log(self, entry: dict[str, Any]) -> None:
        if not self.enabled():
            return
        try:
            self.start()
            payload = self._normalize_value(entry or {})
            record = {
                "record_type": "runtime_log",
                "logged_at": self._coerce_datetime_value(payload.get("logged_at") or payload.get("observed_at") or datetime.now(UTC)),
                "inserted_at": self._serialize_datetime(datetime.now(UTC)),
                "run_id": str(payload.get("run_id") or self._run_id),
                "logger_name": payload.get("logger_name"),
                "level": payload.get("level") or payload.get("level_name"),
                "source": payload.get("source"),
                "service_name": payload.get("service_name"),
                "process_id": payload.get("process_id"),
                "line_hash": payload.get("line_hash"),
                "thread_name": payload.get("thread_name"),
                "message": payload.get("message"),
                "exception_text": payload.get("exception_text"),
                "payload_json": self._json_dumps(payload),
            }
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped_events += 1
            if self._dropped_events in {1, 10, 100} or self._dropped_events % 1000 == 0:
                logger.warning("Observability queue full; dropped %s records", self._dropped_events)
        except Exception:
            self._failed = True
            logger.exception("Failed to record runtime log")

    def query_events(
        self,
        *,
        limit: int = 100,
        after_id: Optional[int] = None,
        ascending: bool = False,
        category: Optional[str] = None,
        event_type: Optional[str] = None,
        since_minutes: Optional[int] = None,
        search: Optional[str] = None,
        run_id: Optional[str] = None,
        order_id: Optional[str] = None,
        start_time: Optional[datetime | str] = None,
        end_time: Optional[datetime | str] = None,
    ) -> list[dict[str, Any]]:
        if not self.enabled():
            return []
        try:
            self.start()
            if not self.enabled():
                return []
            clauses: list[str] = []
            params: list[Any] = []
            if category:
                clauses.append("category = ?")
                params.append(category)
            if event_type:
                clauses.append("event_type = ?")
                params.append(event_type)
            if after_id is not None:
                clauses.append("id > ?")
                params.append(int(after_id))
            if run_id:
                clauses.append("run_id = ?")
                params.append(run_id)
            if order_id:
                clauses.append("order_id = ?")
                params.append(order_id)
            if since_minutes is not None:
                since_time = datetime.now(UTC) - timedelta(minutes=max(int(since_minutes), 0))
                clauses.append("event_timestamp >= ?")
                params.append(self._serialize_datetime(since_time))
            if start_time is not None:
                clauses.append("event_timestamp >= ?")
                params.append(self._coerce_datetime_value(start_time))
            if end_time is not None:
                clauses.append("event_timestamp <= ?")
                params.append(self._coerce_datetime_value(end_time))
            if search:
                clauses.append("(reason LIKE ? OR action LIKE ? OR symbol LIKE ? OR zone LIKE ? OR payload_json LIKE ? OR event_type LIKE ? OR source LIKE ?)")
                pattern = f"%{search}%"
                params.extend([pattern, pattern, pattern, pattern, pattern, pattern, pattern])
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            order_clause = "ORDER BY id ASC" if ascending else "ORDER BY id DESC"
            query = (
                "SELECT id, event_timestamp, inserted_at, run_id, process_id, category, event_type, source, symbol, zone, action, reason, order_id, risk_state, payload_json "
                f"FROM events {where} {order_clause} LIMIT ?"
            )
            params.append(max(int(limit), 1))
            with self._lock:
                self._flush_locked()
                assert self._conn is not None
                rows = self._conn.execute(query, params).fetchall()
            results: list[dict[str, Any]] = []
            for row in rows:
                item = dict(row)
                item["payload"] = json.loads(item.pop("payload_json") or "{}")
                results.append(item)
            return results
        except Exception:
            self._failed = True
            logger.exception("Failed to query observability events")
            return []

    def query_state_snapshots(
        self,
        *,
        limit: int = 100,
        after_id: Optional[int] = None,
        ascending: bool = False,
        run_id: Optional[str] = None,
        symbol: Optional[str] = None,
        search: Optional[str] = None,
        start_time: Optional[datetime | str] = None,
        end_time: Optional[datetime | str] = None,
    ) -> list[dict[str, Any]]:
        if not self.enabled():
            return []
        try:
            self.start()
            clauses: list[str] = []
            params: list[Any] = []
            if after_id is not None:
                clauses.append("id > ?")
                params.append(int(after_id))
            if run_id:
                clauses.append("run_id = ?")
                params.append(run_id)
            if symbol:
                clauses.append("symbol = ?")
                params.append(symbol)
            if start_time is not None:
                clauses.append("captured_at >= ?")
                params.append(self._coerce_datetime_value(start_time))
            if end_time is not None:
                clauses.append("captured_at <= ?")
                params.append(self._coerce_datetime_value(end_time))
            if search:
                pattern = f"%{search}%"
                clauses.append("(status LIKE ? OR data_mode LIKE ? OR symbol LIKE ? OR zone LIKE ? OR zone_state LIKE ? OR account_name LIKE ? OR payload_json LIKE ?)")
                params.extend([pattern, pattern, pattern, pattern, pattern, pattern, pattern])
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            order_clause = "ORDER BY id ASC" if ascending else "ORDER BY id DESC"
            with self._lock:
                assert self._conn is not None
                rows = self._conn.execute(
                    f"""
                    SELECT id, captured_at, inserted_at, run_id, process_id, status, data_mode, symbol, zone, zone_state,
                           position, position_pnl, daily_pnl, risk_state, account_id, account_name, account_mode,
                           account_is_practice, decision_id, attempt_id, position_id, trade_id,
                           payload_json
                    FROM state_snapshots
                    {where}
                    {order_clause}
                    LIMIT ?
                    """,
                    [*params, max(int(limit), 1)],
                ).fetchall()
            return [self._decode_state_snapshot_row(row) for row in rows]
        except Exception:
            self._failed = True
            logger.exception("Failed to query state snapshots")
            return []

    def query_market_tape(
        self,
        *,
        limit: int = 100,
        after_id: Optional[int] = None,
        ascending: bool = False,
        run_id: Optional[str] = None,
        symbol: Optional[str] = None,
        search: Optional[str] = None,
        start_time: Optional[datetime | str] = None,
        end_time: Optional[datetime | str] = None,
    ) -> list[dict[str, Any]]:
        if not self.enabled():
            return []
        try:
            self.start()
            clauses: list[str] = []
            params: list[Any] = []
            if after_id is not None:
                clauses.append("id > ?")
                params.append(int(after_id))
            if run_id:
                clauses.append("run_id = ?")
                params.append(run_id)
            if symbol:
                clauses.append("symbol = ?")
                params.append(symbol)
            if start_time is not None:
                clauses.append("captured_at >= ?")
                params.append(self._coerce_datetime_value(start_time))
            if end_time is not None:
                clauses.append("captured_at <= ?")
                params.append(self._coerce_datetime_value(end_time))
            if search:
                pattern = f"%{search}%"
                clauses.append("(symbol LIKE ? OR contract_id LIKE ? OR trade_side LIKE ? OR source LIKE ? OR payload_json LIKE ?)")
                params.extend([pattern, pattern, pattern, pattern, pattern])
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            order_clause = "ORDER BY id ASC" if ascending else "ORDER BY id DESC"
            with self._lock:
                assert self._conn is not None
                rows = self._conn.execute(
                    f"""
                    SELECT id, captured_at, inserted_at, run_id, process_id, symbol, contract_id, bid, ask, last,
                           volume, bid_size, ask_size, last_size, volume_is_cumulative, quote_is_synthetic,
                           trade_side, latency_ms, source, sequence, payload_json
                    FROM market_tape
                    {where}
                    {order_clause}
                    LIMIT ?
                    """,
                    [*params, max(int(limit), 1)],
                ).fetchall()
            return [self._decode_market_tape_row(row) for row in rows]
        except Exception:
            self._failed = True
            logger.exception("Failed to query market tape")
            return []

    def query_decision_snapshots(
        self,
        *,
        limit: int = 100,
        after_id: Optional[int] = None,
        ascending: bool = False,
        run_id: Optional[str] = None,
        symbol: Optional[str] = None,
        search: Optional[str] = None,
        start_time: Optional[datetime | str] = None,
        end_time: Optional[datetime | str] = None,
    ) -> list[dict[str, Any]]:
        if not self.enabled():
            return []
        try:
            self.start()
            clauses: list[str] = []
            params: list[Any] = []
            if after_id is not None:
                clauses.append("id > ?")
                params.append(int(after_id))
            if run_id:
                clauses.append("run_id = ?")
                params.append(run_id)
            if symbol:
                clauses.append("symbol = ?")
                params.append(symbol)
            if start_time is not None:
                clauses.append("decided_at >= ?")
                params.append(self._coerce_datetime_value(start_time))
            if end_time is not None:
                clauses.append("decided_at <= ?")
                params.append(self._coerce_datetime_value(end_time))
            if search:
                pattern = f"%{search}%"
                clauses.append("(decision_id LIKE ? OR action LIKE ? OR reason LIKE ? OR outcome LIKE ? OR zone LIKE ? OR payload_json LIKE ?)")
                params.extend([pattern, pattern, pattern, pattern, pattern, pattern])
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            order_clause = "ORDER BY id ASC" if ascending else "ORDER BY id DESC"
            with self._lock:
                assert self._conn is not None
                rows = self._conn.execute(
                    f"""
                    SELECT id, decided_at, inserted_at, run_id, process_id, decision_id, attempt_id, symbol, zone,
                           action, reason, outcome, outcome_reason, long_score, short_score, flat_bias, score_gap,
                           dominant_side, current_price, allow_entries, execution_tradeable, contracts, order_type,
                           limit_price, decision_price, side, stop_loss, take_profit, max_hold_minutes, regime_state,
                           regime_reason, active_session, active_vetoes_json, feature_snapshot_json, entry_guard_json,
                           unresolved_entry_json, event_context_json, order_flow_json, payload_json
                    FROM decision_snapshots
                    {where}
                    {order_clause}
                    LIMIT ?
                    """,
                    [*params, max(int(limit), 1)],
                ).fetchall()
            return [self._decode_decision_snapshot_row(row) for row in rows]
        except Exception:
            self._failed = True
            logger.exception("Failed to query decision snapshots")
            return []

    def query_order_lifecycle(
        self,
        *,
        limit: int = 100,
        after_id: Optional[int] = None,
        ascending: bool = False,
        run_id: Optional[str] = None,
        order_id: Optional[str] = None,
        symbol: Optional[str] = None,
        search: Optional[str] = None,
        start_time: Optional[datetime | str] = None,
        end_time: Optional[datetime | str] = None,
    ) -> list[dict[str, Any]]:
        if not self.enabled():
            return []
        try:
            self.start()
            clauses: list[str] = []
            params: list[Any] = []
            if after_id is not None:
                clauses.append("id > ?")
                params.append(int(after_id))
            if run_id:
                clauses.append("run_id = ?")
                params.append(run_id)
            if order_id:
                clauses.append("order_id = ?")
                params.append(order_id)
            if symbol:
                clauses.append("symbol = ?")
                params.append(symbol)
            if start_time is not None:
                clauses.append("observed_at >= ?")
                params.append(self._coerce_datetime_value(start_time))
            if end_time is not None:
                clauses.append("observed_at <= ?")
                params.append(self._coerce_datetime_value(end_time))
            if search:
                pattern = f"%{search}%"
                clauses.append("(decision_id LIKE ? OR attempt_id LIKE ? OR order_id LIKE ? OR status LIKE ? OR event_type LIKE ? OR reason LIKE ? OR payload_json LIKE ?)")
                params.extend([pattern, pattern, pattern, pattern, pattern, pattern, pattern])
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            order_clause = "ORDER BY id ASC" if ascending else "ORDER BY id DESC"
            with self._lock:
                assert self._conn is not None
                rows = self._conn.execute(
                    f"""
                    SELECT id, observed_at, inserted_at, run_id, process_id, decision_id, attempt_id, order_id,
                           position_id, trade_id, symbol, event_type, status, side, role, is_protective, order_type,
                           quantity, contracts, limit_price, stop_price, expected_fill_price, filled_price,
                           filled_quantity, remaining_quantity, zone, reason, lifecycle_state, payload_json
                    FROM order_lifecycle
                    {where}
                    {order_clause}
                    LIMIT ?
                    """,
                    [*params, max(int(limit), 1)],
                ).fetchall()
            return [self._decode_order_lifecycle_row(row) for row in rows]
        except Exception:
            self._failed = True
            logger.exception("Failed to query order lifecycle")
            return []

    def query_bridge_health(
        self,
        *,
        limit: int = 100,
        after_id: Optional[int] = None,
        ascending: bool = False,
        run_id: Optional[str] = None,
        search: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        if not self.enabled():
            return []
        try:
            self.start()
            clauses: list[str] = []
            params: list[Any] = []
            if after_id is not None:
                clauses.append("id > ?")
                params.append(int(after_id))
            if run_id:
                clauses.append("run_id = ?")
                params.append(run_id)
            if search:
                pattern = f"%{search}%"
                clauses.append("(bridge_status LIKE ? OR last_error LIKE ? OR payload_json LIKE ?)")
                params.extend([pattern, pattern, pattern])
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            order_clause = "ORDER BY id ASC" if ascending else "ORDER BY id DESC"
            with self._lock:
                assert self._conn is not None
                rows = self._conn.execute(
                    f"""
                    SELECT id, observed_at, inserted_at, run_id, bridge_status, queue_depth, last_flush_at,
                           last_success_at, last_error, payload_json
                    FROM bridge_health
                    {where}
                    {order_clause}
                    LIMIT ?
                    """,
                    [*params, max(int(limit), 1)],
                ).fetchall()
            return [self._decode_bridge_health_row(row) for row in rows]
        except Exception:
            self._failed = True
            logger.exception("Failed to query bridge health")
            return []

    def query_runtime_logs(
        self,
        *,
        limit: int = 100,
        after_id: Optional[int] = None,
        ascending: bool = False,
        run_id: Optional[str] = None,
        level: Optional[str] = None,
        search: Optional[str] = None,
        start_time: Optional[datetime | str] = None,
        end_time: Optional[datetime | str] = None,
    ) -> list[dict[str, Any]]:
        if not self.enabled():
            return []
        try:
            self.start()
            clauses: list[str] = []
            params: list[Any] = []
            if after_id is not None:
                operator = ">" if ascending else "<"
                clauses.append(f"id {operator} ?")
                params.append(int(after_id))
            if run_id:
                clauses.append("run_id = ?")
                params.append(str(run_id))
            if level:
                clauses.append("level = ?")
                params.append(str(level).upper())
            start_value = self._coerce_optional_datetime_value(start_time)
            if start_value:
                clauses.append("logged_at >= ?")
                params.append(start_value)
            end_value = self._coerce_optional_datetime_value(end_time)
            if end_value:
                clauses.append("logged_at <= ?")
                params.append(end_value)
            if search:
                pattern = f"%{search}%"
                clauses.append("(logger_name LIKE ? OR message LIKE ? OR exception_text LIKE ? OR source LIKE ? OR service_name LIKE ? OR payload_json LIKE ?)")
                params.extend([pattern, pattern, pattern, pattern, pattern, pattern])
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            order_clause = "ORDER BY id ASC" if ascending else "ORDER BY id DESC"
            with self._lock:
                assert self._conn is not None
                rows = self._conn.execute(
                    f"""
                    SELECT id, logged_at, inserted_at, run_id, logger_name, level, source, service_name,
                           process_id, line_hash, thread_name, message, exception_text, payload_json
                    FROM runtime_logs
                    {where}
                    {order_clause}
                    LIMIT ?
                    """,
                    [*params, max(int(limit), 1)],
                ).fetchall()
            return [self._decode_runtime_log_row(row) for row in rows]
        except Exception:
            self._failed = True
            logger.exception("Failed to query runtime logs")
            return []

    def query_account_trades(
        self,
        *,
        limit: int = 100,
        after_id: Optional[int] = None,
        ascending: bool = False,
        run_id: Optional[str] = None,
        account_id: Optional[str] = None,
        search: Optional[str] = None,
        start_time: Optional[datetime | str] = None,
        end_time: Optional[datetime | str] = None,
    ) -> list[dict[str, Any]]:
        if not self.enabled():
            return []
        try:
            self.start()
            clauses: list[str] = []
            params: list[Any] = []
            if after_id is not None:
                clauses.append("id > ?")
                params.append(int(after_id))
            if run_id:
                clauses.append("run_id = ?")
                params.append(run_id)
            if account_id:
                clauses.append("account_id = ?")
                params.append(account_id)
            start_value = self._coerce_optional_datetime_value(start_time)
            if start_value:
                clauses.append("occurred_at >= ?")
                params.append(start_value)
            end_value = self._coerce_optional_datetime_value(end_time)
            if end_value:
                clauses.append("occurred_at <= ?")
                params.append(end_value)
            if search:
                pattern = f"%{search}%"
                clauses.append(
                    "(account_name LIKE ? OR contract_id LIKE ? OR broker_trade_id LIKE ? OR broker_order_id LIKE ? OR payload_json LIKE ?)"
                )
                params.extend([pattern, pattern, pattern, pattern, pattern])
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            order_clause = "ORDER BY id ASC" if ascending else "ORDER BY id DESC"
            with self._lock:
                assert self._conn is not None
                rows = self._conn.execute(
                    f"""
                    SELECT id, run_id, inserted_at, occurred_at, account_id, account_name, account_mode,
                           account_is_practice, broker_trade_id, broker_order_id, contract_id, side, size,
                           price, profit_and_loss, fees, voided, source, payload_json
                    FROM account_trades
                    {where}
                    {order_clause}
                    LIMIT ?
                    """,
                    [*params, max(int(limit), 1)],
                ).fetchall()
            return [self._decode_account_trade_row(row) for row in rows]
        except Exception:
            self._failed = True
            logger.exception("Failed to query account trades")
            return []

    def get_db_path(self) -> str:
        return str(self._db_path)

    def get_run_id(self) -> str:
        return self._run_id

    def record_account_trade(self, trade: dict[str, Any], *, run_id: Optional[str] = None, source: str = "broker_history") -> bool:
        if not self.enabled():
            return False
        try:
            self.start()
            payload = self._normalize_value(trade or {})
            account_context = self._extract_account_context(payload)
            broker_trade_id = payload.get("broker_trade_id") or payload.get("trade_id") or payload.get("id")
            occurred_at = (
                self._coerce_optional_datetime_value(payload.get("occurred_at"))
                or self._coerce_optional_datetime_value(payload.get("creationTimestamp"))
                or self._coerce_optional_datetime_value(payload.get("creation_timestamp"))
                or self._coerce_optional_datetime_value(payload.get("timestamp"))
                or self._serialize_datetime(datetime.now(UTC))
            )
            record = {
                "record_type": "account_trade",
                "run_id": str(run_id or payload.get("run_id") or self._run_id),
                "inserted_at": self._serialize_datetime(datetime.now(UTC)),
                "occurred_at": occurred_at,
                "account_id": account_context["account_id"] or str(payload.get("accountId") or payload.get("account_id") or ""),
                "account_name": account_context["account_name"],
                "account_mode": account_context["account_mode"],
                "account_is_practice": account_context["account_is_practice"],
                "broker_trade_id": str(broker_trade_id or ""),
                "broker_order_id": payload.get("broker_order_id") or payload.get("orderId") or payload.get("order_id"),
                "contract_id": payload.get("contract_id") or payload.get("contractId"),
                "side": payload.get("side"),
                "size": payload.get("size"),
                "price": payload.get("price"),
                "profit_and_loss": payload.get("profit_and_loss") if "profit_and_loss" in payload else payload.get("profitAndLoss"),
                "fees": payload.get("fees"),
                "voided": self._coerce_bool(payload.get("voided")),
                "source": source,
                "payload_json": self._json_dumps(payload),
            }
            if not record["account_id"] or not record["broker_trade_id"]:
                return False
            with self._lock:
                assert self._conn is not None
                before = self._conn.total_changes
                self._write_account_trades_locked([record])
                return self._conn.total_changes > before
        except Exception:
            self._failed = True
            logger.exception("Failed to record account trade")
            return False

    def record_run_manifest(self, manifest: dict[str, Any]) -> None:
        if not self.enabled():
            return
        try:
            self.start()
            payload = self._normalize_value(manifest or {})
            account_context = self._extract_account_context(payload)
            run_id = str(payload.get("run_id") or self._run_id)
            created_at = str(payload.get("started_at") or payload.get("created_at") or self._serialize_datetime(datetime.now(UTC)))
            process_id = int(payload.get("process_id") or os.getpid())
            with self._lock:
                assert self._conn is not None
                self._conn.execute(
                    """
                    INSERT INTO run_manifests (
                        run_id,
                        created_at,
                        process_id,
                        data_mode,
                        symbol,
                        account_id,
                        account_name,
                        account_mode,
                        account_is_practice,
                        config_path,
                        config_hash,
                        log_path,
                        sqlite_path,
                        git_commit,
                        git_branch,
                        git_dirty,
                        git_available,
                        app_version,
                        payload_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        created_at=excluded.created_at,
                        process_id=excluded.process_id,
                        data_mode=excluded.data_mode,
                        symbol=excluded.symbol,
                        account_id=excluded.account_id,
                        account_name=excluded.account_name,
                        account_mode=excluded.account_mode,
                        account_is_practice=excluded.account_is_practice,
                        config_path=excluded.config_path,
                        config_hash=excluded.config_hash,
                        log_path=excluded.log_path,
                        sqlite_path=excluded.sqlite_path,
                        git_commit=excluded.git_commit,
                        git_branch=excluded.git_branch,
                        git_dirty=excluded.git_dirty,
                        git_available=excluded.git_available,
                        app_version=excluded.app_version,
                        payload_json=excluded.payload_json
                    """,
                    (
                        run_id,
                        created_at,
                        process_id,
                        payload.get("data_mode"),
                        (payload.get("symbols") or [None])[0],
                        account_context["account_id"],
                        account_context["account_name"],
                        account_context["account_mode"],
                        account_context["account_is_practice"],
                        payload.get("config_path"),
                        payload.get("config_hash"),
                        payload.get("log_path"),
                        payload.get("sqlite_path"),
                        payload.get("git_commit"),
                        payload.get("git_branch"),
                        self._coerce_bool(payload.get("git_dirty")),
                        self._coerce_bool(payload.get("git_available")),
                        payload.get("app_version"),
                        self._json_dumps(payload),
                    ),
                )
                self._conn.commit()
        except Exception:
            self._failed = True
            logger.exception("Failed to record run manifest")

    def query_run_manifests(self, *, limit: int = 25, search: Optional[str] = None) -> list[dict[str, Any]]:
        if not self.enabled():
            return []
        try:
            self.start()
            clauses: list[str] = []
            params: list[Any] = []
            if search:
                pattern = f"%{search}%"
                clauses.append("(run_id LIKE ? OR data_mode LIKE ? OR symbol LIKE ? OR account_name LIKE ? OR git_branch LIKE ? OR git_commit LIKE ? OR payload_json LIKE ?)")
                params.extend([pattern, pattern, pattern, pattern, pattern, pattern, pattern])
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            with self._lock:
                assert self._conn is not None
                rows = self._conn.execute(
                    f"""
                    SELECT run_id, created_at, process_id, data_mode, symbol, account_id, account_name, account_mode,
                           account_is_practice, config_path, config_hash,
                           log_path, sqlite_path, git_commit, git_branch, git_dirty, git_available,
                           app_version, payload_json
                    FROM run_manifests
                    {where}
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    [*params, max(int(limit), 1)],
                ).fetchall()
            return [self._decode_run_manifest_row(row) for row in rows]
        except Exception:
            self._failed = True
            logger.exception("Failed to query run manifests")
            return []

    def get_run_manifest(self, run_id: str) -> Optional[dict[str, Any]]:
        if not self.enabled():
            return None
        try:
            self.start()
            with self._lock:
                assert self._conn is not None
                row = self._conn.execute(
                    """
                    SELECT run_id, created_at, process_id, data_mode, symbol, account_id, account_name, account_mode,
                           account_is_practice, config_path, config_hash,
                           log_path, sqlite_path, git_commit, git_branch, git_dirty, git_available,
                           app_version, payload_json
                    FROM run_manifests
                    WHERE run_id = ?
                    """,
                    (run_id,),
                ).fetchone()
            return self._decode_run_manifest_row(row) if row is not None else None
        except Exception:
            self._failed = True
            logger.exception("Failed to load run manifest for run_id=%s", run_id)
            return None

    def update_run_manifest_payload(self, run_id: str, extra: dict[str, Any]) -> None:
        """Merge extra keys into the payload_json of an existing run manifest."""
        if not self.enabled() or not extra:
            return
        try:
            self.start()
            with self._lock:
                assert self._conn is not None
                row = self._conn.execute(
                    "SELECT payload_json FROM run_manifests WHERE run_id = ?",
                    (run_id,),
                ).fetchone()
                if row is None:
                    return
                payload = json.loads(row["payload_json"] or "{}")
                payload.update(self._normalize_value(extra))
                self._conn.execute(
                    "UPDATE run_manifests SET payload_json = ? WHERE run_id = ?",
                    (self._json_dumps(payload), run_id),
                )
                self._conn.commit()
        except Exception:
            self._failed = True
            logger.exception("Failed to update run manifest payload for run_id=%s", run_id)

    def record_completed_trade(
        self,
        trade: Any,
        *,
        run_id: Optional[str] = None,
        source: str = "runtime",
        backfilled: bool = False,
    ) -> bool:
        if not self.enabled():
            return False
        try:
            self.start()
            payload = self._normalize_value(asdict(trade) if is_dataclass(trade) else dict(trade))
            account_context = self._extract_account_context(payload)
            record = {
                "run_id": str(run_id or payload.get("run_id") or self._run_id),
                "inserted_at": self._serialize_datetime(datetime.now(UTC)),
                "entry_time": self._coerce_optional_datetime_value(payload.get("entry_time")),
                "exit_time": self._coerce_optional_datetime_value(payload.get("exit_time")) or self._serialize_datetime(datetime.now(UTC)),
                "direction": int(payload.get("direction") or 0),
                "contracts": int(payload.get("contracts") or 0),
                "entry_price": float(payload.get("entry_price") or 0.0),
                "exit_price": float(payload.get("exit_price") or 0.0),
                "pnl": float(payload.get("pnl") or 0.0),
                "zone": payload.get("zone"),
                "strategy": payload.get("strategy"),
                "regime": payload.get("regime"),
                "event_tags_json": self._json_dumps(list(payload.get("event_tags") or [])),
                "source": source,
                "backfilled": self._coerce_bool(backfilled),
                "trade_id": payload.get("trade_id"),
                "position_id": payload.get("position_id"),
                "decision_id": payload.get("decision_id"),
                "attempt_id": payload.get("attempt_id"),
                "account_id": account_context["account_id"],
                "account_name": account_context["account_name"],
                "account_mode": account_context["account_mode"],
                "account_is_practice": account_context["account_is_practice"],
                "payload_json": self._json_dumps(payload),
            }
            with self._lock:
                assert self._conn is not None
                before = self._conn.total_changes
                self._insert_completed_trade_locked(record)
                return self._conn.total_changes > before
        except Exception:
            self._failed = True
            logger.exception("Failed to record completed trade")
            return False

    def query_completed_trades(
        self,
        *,
        limit: int = 100,
        after_id: Optional[int] = None,
        ascending: bool = False,
        run_id: Optional[str] = None,
        account_id: Optional[str] = None,
        zone: Optional[str] = None,
        strategy: Optional[str] = None,
        min_pnl: Optional[float] = None,
        max_pnl: Optional[float] = None,
        search: Optional[str] = None,
        start_time: Optional[datetime | str] = None,
        end_time: Optional[datetime | str] = None,
    ) -> list[dict[str, Any]]:
        if not self.enabled():
            return []
        try:
            self.start()
            clauses: list[str] = []
            params: list[Any] = []
            if after_id is not None:
                clauses.append("id > ?")
                params.append(int(after_id))
            if run_id:
                clauses.append("run_id = ?")
                params.append(run_id)
            if account_id:
                clauses.append("account_id = ?")
                params.append(account_id)
            if zone:
                clauses.append("zone = ?")
                params.append(zone)
            if strategy:
                clauses.append("strategy = ?")
                params.append(strategy)
            if min_pnl is not None:
                clauses.append("pnl >= ?")
                params.append(float(min_pnl))
            if max_pnl is not None:
                clauses.append("pnl <= ?")
                params.append(float(max_pnl))
            if start_time is not None:
                clauses.append("exit_time >= ?")
                params.append(self._coerce_datetime_value(start_time))
            if end_time is not None:
                clauses.append("exit_time <= ?")
                params.append(self._coerce_datetime_value(end_time))
            if search:
                pattern = f"%{search}%"
                clauses.append("(zone LIKE ? OR strategy LIKE ? OR regime LIKE ? OR payload_json LIKE ? OR event_tags_json LIKE ?)")
                params.extend([pattern, pattern, pattern, pattern, pattern])
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            order_clause = "ORDER BY id ASC" if ascending else "ORDER BY id DESC"
            with self._lock:
                assert self._conn is not None
                rows = self._conn.execute(
                    f"""
                    SELECT id, run_id, inserted_at, entry_time, exit_time, direction, contracts, entry_price,
                           exit_price, pnl, zone, strategy, regime, event_tags_json, source, backfilled,
                           trade_id, position_id, decision_id, attempt_id, account_id, account_name, account_mode,
                           account_is_practice, payload_json
                    FROM completed_trades
                    {where}
                    {order_clause}, exit_time DESC
                    LIMIT ?
                    """,
                    [*params, max(int(limit), 1)],
                ).fetchall()
            return [self._decode_completed_trade_row(row) for row in rows]
        except Exception:
            self._failed = True
            logger.exception("Failed to query completed trades")
            return []

    def backfill_completed_trades_from_events(self, *, run_id: Optional[str] = None) -> dict[str, Any]:
        if not self.enabled():
            return {"checked": 0, "backfilled": 0, "run_id": run_id}
        try:
            self.start()
            checked = 0
            backfilled = 0
            with self._lock:
                self._flush_locked()
                assert self._conn is not None
                query = """
                    SELECT run_id, event_timestamp, zone, payload_json
                    FROM events
                    WHERE event_type = 'trade_recorded'
                """
                params: list[Any] = []
                if run_id:
                    query += " AND run_id = ?"
                    params.append(run_id)
                query += " ORDER BY id ASC"
                rows = self._conn.execute(query, params).fetchall()
                for row in rows:
                    checked += 1
                    payload = json.loads(row["payload_json"] or "{}")
                    account_context = self._extract_account_context(payload)
                    inserted = self._insert_completed_trade_locked(
                        {
                            "run_id": row["run_id"],
                            "inserted_at": self._serialize_datetime(datetime.now(UTC)),
                            "entry_time": self._coerce_optional_datetime_value(payload.get("entry_time")),
                            "exit_time": self._coerce_optional_datetime_value(payload.get("exit_time")) or row["event_timestamp"],
                            "direction": int(payload.get("direction") or 0),
                            "contracts": int(payload.get("contracts") or 0),
                            "entry_price": float(payload.get("entry_price") or 0.0),
                            "exit_price": float(payload.get("exit_price") or 0.0),
                            "pnl": float(payload.get("pnl") or 0.0),
                            "zone": row["zone"] or payload.get("zone"),
                            "strategy": payload.get("strategy"),
                            "regime": payload.get("regime"),
                            "event_tags_json": self._json_dumps(list(payload.get("event_tags") or [])),
                            "source": "event_backfill",
                            "backfilled": 1,
                            "trade_id": payload.get("trade_id"),
                            "position_id": payload.get("position_id"),
                            "decision_id": payload.get("decision_id"),
                            "attempt_id": payload.get("attempt_id"),
                            "account_id": account_context["account_id"],
                            "account_name": account_context["account_name"],
                            "account_mode": account_context["account_mode"],
                            "account_is_practice": account_context["account_is_practice"],
                            "payload_json": self._json_dumps(payload),
                        }
                    )
                    if inserted:
                        backfilled += 1
            return {"checked": checked, "backfilled": backfilled, "run_id": run_id or self._run_id}
        except Exception:
            self._failed = True
            logger.exception("Failed to backfill completed trades from events")
            return {"checked": 0, "backfilled": 0, "run_id": run_id, "error": "backfill_failed"}

    def _worker_loop(self) -> None:
        flush_interval = max(float(self.settings.flush_interval_ms), 1.0) / 1000.0
        batch_size = max(int(self.settings.batch_size), 1)
        pending: list[dict[str, Any]] = []
        last_flush = time.time()
        while self._running or not self._queue.empty() or pending:
            timeout = max(flush_interval - (time.time() - last_flush), 0.0)
            try:
                event = self._queue.get(timeout=timeout if timeout > 0 else flush_interval)
                pending.append(event)
                while len(pending) < batch_size:
                    pending.append(self._queue.get_nowait())
            except queue.Empty:
                pass
            if pending and (len(pending) >= batch_size or (time.time() - last_flush) >= flush_interval or not self._running):
                try:
                    with self._lock:
                        self._write_records_locked(pending)
                except Exception:
                    self._failed = True
                    logger.exception("Failed to flush observability events")
                    return
                pending = []
                last_flush = time.time()

    def _flush_locked(self) -> None:
        drained: list[dict[str, Any]] = []
        while True:
            try:
                drained.append(self._queue.get_nowait())
            except queue.Empty:
                break
        if drained:
            self._write_records_locked(drained)

    def _ensure_schema(self) -> None:
        assert self._conn is not None
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_timestamp TEXT NOT NULL,
                inserted_at TEXT NOT NULL,
                run_id TEXT NOT NULL,
                process_id INTEGER NOT NULL,
                category TEXT NOT NULL,
                event_type TEXT NOT NULL,
                source TEXT NOT NULL,
                symbol TEXT,
                zone TEXT,
                action TEXT,
                reason TEXT,
                order_id TEXT,
                risk_state TEXT,
                payload_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(event_timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_events_category ON events(category, event_type, event_timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_events_symbol ON events(symbol, event_timestamp DESC);
            CREATE INDEX IF NOT EXISTS idx_events_order_id ON events(order_id, event_timestamp DESC);

            CREATE TABLE IF NOT EXISTS state_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                captured_at TEXT NOT NULL,
                inserted_at TEXT NOT NULL,
                run_id TEXT NOT NULL,
                process_id INTEGER NOT NULL,
                status TEXT,
                data_mode TEXT,
                symbol TEXT,
                zone TEXT,
                zone_state TEXT,
                position INTEGER,
                position_pnl REAL,
                daily_pnl REAL,
                risk_state TEXT,
                account_id TEXT,
                account_name TEXT,
                account_mode TEXT,
                account_is_practice INTEGER,
                decision_id TEXT,
                attempt_id TEXT,
                position_id TEXT,
                trade_id TEXT,
                payload_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_state_snapshots_run_id ON state_snapshots(run_id, captured_at DESC);
            CREATE INDEX IF NOT EXISTS idx_state_snapshots_symbol ON state_snapshots(symbol, captured_at DESC);

            CREATE TABLE IF NOT EXISTS market_tape (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                captured_at TEXT NOT NULL,
                inserted_at TEXT NOT NULL,
                run_id TEXT NOT NULL,
                process_id INTEGER NOT NULL,
                symbol TEXT NOT NULL,
                contract_id TEXT,
                bid REAL,
                ask REAL,
                last REAL,
                volume INTEGER,
                bid_size REAL,
                ask_size REAL,
                last_size REAL,
                volume_is_cumulative INTEGER,
                quote_is_synthetic INTEGER,
                trade_side TEXT,
                latency_ms INTEGER,
                source TEXT,
                sequence INTEGER,
                payload_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_market_tape_run_id ON market_tape(run_id, captured_at DESC);
            CREATE INDEX IF NOT EXISTS idx_market_tape_symbol ON market_tape(symbol, captured_at DESC);

            CREATE TABLE IF NOT EXISTS decision_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                decided_at TEXT NOT NULL,
                inserted_at TEXT NOT NULL,
                run_id TEXT NOT NULL,
                process_id INTEGER NOT NULL,
                decision_id TEXT NOT NULL,
                attempt_id TEXT,
                symbol TEXT,
                zone TEXT,
                action TEXT,
                reason TEXT,
                outcome TEXT,
                outcome_reason TEXT,
                long_score REAL,
                short_score REAL,
                flat_bias REAL,
                score_gap REAL,
                dominant_side TEXT,
                current_price REAL,
                allow_entries INTEGER,
                execution_tradeable INTEGER,
                contracts INTEGER,
                order_type TEXT,
                limit_price REAL,
                decision_price REAL,
                side TEXT,
                stop_loss REAL,
                take_profit REAL,
                max_hold_minutes INTEGER,
                regime_state TEXT,
                regime_reason TEXT,
                active_session TEXT,
                active_vetoes_json TEXT,
                feature_snapshot_json TEXT,
                entry_guard_json TEXT,
                unresolved_entry_json TEXT,
                event_context_json TEXT,
                order_flow_json TEXT,
                payload_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_decision_snapshots_run_id ON decision_snapshots(run_id, decided_at DESC);
            CREATE INDEX IF NOT EXISTS idx_decision_snapshots_decision_id ON decision_snapshots(decision_id);

            CREATE TABLE IF NOT EXISTS order_lifecycle (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observed_at TEXT NOT NULL,
                inserted_at TEXT NOT NULL,
                run_id TEXT NOT NULL,
                process_id INTEGER NOT NULL,
                decision_id TEXT,
                attempt_id TEXT,
                order_id TEXT,
                position_id TEXT,
                trade_id TEXT,
                symbol TEXT,
                event_type TEXT,
                status TEXT,
                side TEXT,
                role TEXT,
                is_protective INTEGER,
                order_type TEXT,
                quantity INTEGER,
                contracts INTEGER,
                limit_price REAL,
                stop_price REAL,
                expected_fill_price REAL,
                filled_price REAL,
                filled_quantity INTEGER,
                remaining_quantity INTEGER,
                zone TEXT,
                reason TEXT,
                lifecycle_state TEXT,
                payload_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_order_lifecycle_run_id ON order_lifecycle(run_id, observed_at DESC);
            CREATE INDEX IF NOT EXISTS idx_order_lifecycle_order_id ON order_lifecycle(order_id, observed_at DESC);

            CREATE TABLE IF NOT EXISTS bridge_health (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                observed_at TEXT NOT NULL,
                inserted_at TEXT NOT NULL,
                run_id TEXT,
                bridge_status TEXT,
                queue_depth INTEGER,
                last_flush_at TEXT,
                last_success_at TEXT,
                last_error TEXT,
                payload_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_bridge_health_run_id ON bridge_health(run_id, observed_at DESC);
            CREATE INDEX IF NOT EXISTS idx_bridge_health_observed_at ON bridge_health(observed_at DESC);

            CREATE TABLE IF NOT EXISTS runtime_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                logged_at TEXT NOT NULL,
                inserted_at TEXT NOT NULL,
                run_id TEXT,
                logger_name TEXT,
                level TEXT,
                source TEXT,
                service_name TEXT,
                process_id INTEGER,
                line_hash TEXT,
                thread_name TEXT,
                message TEXT,
                exception_text TEXT,
                payload_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_runtime_logs_run_id ON runtime_logs(run_id, logged_at DESC);
            CREATE INDEX IF NOT EXISTS idx_runtime_logs_level ON runtime_logs(level, logged_at DESC);
            CREATE INDEX IF NOT EXISTS idx_runtime_logs_logged_at ON runtime_logs(logged_at DESC);

            CREATE TABLE IF NOT EXISTS run_manifests (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                process_id INTEGER NOT NULL,
                data_mode TEXT,
                symbol TEXT,
                account_id TEXT,
                account_name TEXT,
                account_mode TEXT,
                account_is_practice INTEGER,
                config_path TEXT,
                config_hash TEXT,
                log_path TEXT,
                sqlite_path TEXT,
                git_commit TEXT,
                git_branch TEXT,
                git_dirty INTEGER,
                git_available INTEGER,
                app_version TEXT,
                payload_json TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_run_manifests_created_at ON run_manifests(created_at DESC);
            CREATE INDEX IF NOT EXISTS idx_run_manifests_symbol ON run_manifests(symbol, created_at DESC);

            CREATE TABLE IF NOT EXISTS completed_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                inserted_at TEXT NOT NULL,
                entry_time TEXT,
                exit_time TEXT NOT NULL,
                direction INTEGER NOT NULL,
                contracts INTEGER NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL NOT NULL,
                pnl REAL NOT NULL,
                zone TEXT,
                strategy TEXT,
                regime TEXT,
                event_tags_json TEXT NOT NULL,
                source TEXT NOT NULL,
                backfilled INTEGER NOT NULL DEFAULT 0,
                trade_id TEXT,
                position_id TEXT,
                decision_id TEXT,
                attempt_id TEXT,
                account_id TEXT,
                account_name TEXT,
                account_mode TEXT,
                account_is_practice INTEGER,
                payload_json TEXT NOT NULL,
                UNIQUE(run_id, exit_time, direction, contracts, entry_price, exit_price, pnl, zone, strategy)
            );

            CREATE INDEX IF NOT EXISTS idx_completed_trades_run_id ON completed_trades(run_id, exit_time DESC);
            CREATE INDEX IF NOT EXISTS idx_completed_trades_pnl ON completed_trades(pnl DESC, exit_time DESC);
            CREATE INDEX IF NOT EXISTS idx_completed_trades_zone ON completed_trades(zone, exit_time DESC);

            CREATE TABLE IF NOT EXISTS account_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT,
                inserted_at TEXT NOT NULL,
                occurred_at TEXT NOT NULL,
                account_id TEXT NOT NULL,
                account_name TEXT,
                account_mode TEXT,
                account_is_practice INTEGER,
                broker_trade_id TEXT NOT NULL,
                broker_order_id TEXT,
                contract_id TEXT,
                side INTEGER,
                size INTEGER,
                price REAL,
                profit_and_loss REAL,
                fees REAL,
                voided INTEGER,
                source TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                UNIQUE(account_id, broker_trade_id)
            );

            CREATE INDEX IF NOT EXISTS idx_account_trades_account_id ON account_trades(account_id, occurred_at DESC);
            CREATE INDEX IF NOT EXISTS idx_account_trades_run_id ON account_trades(run_id, occurred_at DESC);
            CREATE INDEX IF NOT EXISTS idx_account_trades_broker_trade_id ON account_trades(broker_trade_id);

            """
        )
        self._ensure_column_locked("state_snapshots", "account_id", "TEXT")
        self._ensure_column_locked("state_snapshots", "account_name", "TEXT")
        self._ensure_column_locked("state_snapshots", "account_mode", "TEXT")
        self._ensure_column_locked("state_snapshots", "account_is_practice", "INTEGER")
        self._ensure_column_locked("run_manifests", "account_id", "TEXT")
        self._ensure_column_locked("run_manifests", "account_name", "TEXT")
        self._ensure_column_locked("run_manifests", "account_mode", "TEXT")
        self._ensure_column_locked("run_manifests", "account_is_practice", "INTEGER")
        self._ensure_column_locked("completed_trades", "trade_id", "TEXT")
        self._ensure_column_locked("completed_trades", "position_id", "TEXT")
        self._ensure_column_locked("completed_trades", "decision_id", "TEXT")
        self._ensure_column_locked("completed_trades", "attempt_id", "TEXT")
        self._ensure_column_locked("completed_trades", "account_id", "TEXT")
        self._ensure_column_locked("completed_trades", "account_name", "TEXT")
        self._ensure_column_locked("completed_trades", "account_mode", "TEXT")
        self._ensure_column_locked("completed_trades", "account_is_practice", "INTEGER")
        # `completed_trades` existed before account-aware durability landed, so create
        # the new account index only after the ALTER TABLE migration has run.
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_completed_trades_account_id ON completed_trades(account_id, exit_time DESC)"
        )
        self._conn.execute(
            "INSERT INTO metadata(key, value) VALUES('schema_version', '5') ON CONFLICT(key) DO UPDATE SET value=excluded.value"
        )
        self._conn.commit()

    def _ensure_column_locked(self, table_name: str, column_name: str, column_type: str) -> None:
        assert self._conn is not None
        columns = {
            row["name"]
            for row in self._conn.execute(f"PRAGMA table_info({table_name})").fetchall()
        }
        if column_name in columns:
            return
        self._conn.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")

    def _write_records_locked(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in records:
            grouped[str(record.get("record_type") or "event")].append(record)
        if grouped.get("event"):
            self._write_events_locked(grouped["event"])
        if grouped.get("state_snapshot"):
            self._write_state_snapshots_locked(grouped["state_snapshot"])
        if grouped.get("market_tape"):
            self._write_market_tape_locked(grouped["market_tape"])
        if grouped.get("decision_snapshot"):
            self._write_decision_snapshots_locked(grouped["decision_snapshot"])
        if grouped.get("order_lifecycle"):
            self._write_order_lifecycle_locked(grouped["order_lifecycle"])
        if grouped.get("bridge_health"):
            self._write_bridge_health_locked(grouped["bridge_health"])
        if grouped.get("runtime_log"):
            self._write_runtime_logs_locked(grouped["runtime_log"])
        if grouped.get("account_trade"):
            self._write_account_trades_locked(grouped["account_trade"])

    def _write_events_locked(self, events: list[dict[str, Any]]) -> None:
        if not events:
            return
        assert self._conn is not None
        self._conn.executemany(
            """
            INSERT INTO events (
                event_timestamp,
                inserted_at,
                run_id,
                process_id,
                category,
                event_type,
                source,
                symbol,
                zone,
                action,
                reason,
                order_id,
                risk_state,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    event["event_timestamp"],
                    event["inserted_at"],
                    event["run_id"],
                    event["process_id"],
                    event["category"],
                    event["event_type"],
                    event["source"],
                    event["symbol"],
                    event["zone"],
                    event["action"],
                    event["reason"],
                    event["order_id"],
                    event["risk_state"],
                    event["payload_json"],
                )
                for event in events
            ],
        )
        self._conn.commit()

    def _write_state_snapshots_locked(self, snapshots: list[dict[str, Any]]) -> None:
        if not snapshots:
            return
        assert self._conn is not None
        self._conn.executemany(
            """
            INSERT INTO state_snapshots (
                captured_at,
                inserted_at,
                run_id,
                process_id,
                status,
                data_mode,
                symbol,
                zone,
                zone_state,
                position,
                position_pnl,
                daily_pnl,
                risk_state,
                account_id,
                account_name,
                account_mode,
                account_is_practice,
                decision_id,
                attempt_id,
                position_id,
                trade_id,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item["captured_at"],
                    item["inserted_at"],
                    item["run_id"],
                    item["process_id"],
                    item["status"],
                    item["data_mode"],
                    item["symbol"],
                    item["zone"],
                    item["zone_state"],
                    item["position"],
                    item["position_pnl"],
                    item["daily_pnl"],
                    item["risk_state"],
                    item["account_id"],
                    item["account_name"],
                    item["account_mode"],
                    item["account_is_practice"],
                    item["decision_id"],
                    item["attempt_id"],
                    item["position_id"],
                    item["trade_id"],
                    item["payload_json"],
                )
                for item in snapshots
            ],
        )
        self._conn.commit()

    def _write_market_tape_locked(self, tape_rows: list[dict[str, Any]]) -> None:
        if not tape_rows:
            return
        assert self._conn is not None
        self._conn.executemany(
            """
            INSERT INTO market_tape (
                captured_at,
                inserted_at,
                run_id,
                process_id,
                symbol,
                contract_id,
                bid,
                ask,
                last,
                volume,
                bid_size,
                ask_size,
                last_size,
                volume_is_cumulative,
                quote_is_synthetic,
                trade_side,
                latency_ms,
                source,
                sequence,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item["captured_at"],
                    item["inserted_at"],
                    item["run_id"],
                    item["process_id"],
                    item["symbol"],
                    item["contract_id"],
                    item["bid"],
                    item["ask"],
                    item["last"],
                    item["volume"],
                    item["bid_size"],
                    item["ask_size"],
                    item["last_size"],
                    item["volume_is_cumulative"],
                    item["quote_is_synthetic"],
                    item["trade_side"],
                    item["latency_ms"],
                    item["source"],
                    item["sequence"],
                    item["payload_json"],
                )
                for item in tape_rows
            ],
        )
        self._conn.commit()

    def _write_decision_snapshots_locked(self, snapshots: list[dict[str, Any]]) -> None:
        if not snapshots:
            return
        assert self._conn is not None
        self._conn.executemany(
            """
            INSERT INTO decision_snapshots (
                decided_at,
                inserted_at,
                run_id,
                process_id,
                decision_id,
                attempt_id,
                symbol,
                zone,
                action,
                reason,
                outcome,
                outcome_reason,
                long_score,
                short_score,
                flat_bias,
                score_gap,
                dominant_side,
                current_price,
                allow_entries,
                execution_tradeable,
                contracts,
                order_type,
                limit_price,
                decision_price,
                side,
                stop_loss,
                take_profit,
                max_hold_minutes,
                regime_state,
                regime_reason,
                active_session,
                active_vetoes_json,
                feature_snapshot_json,
                entry_guard_json,
                unresolved_entry_json,
                event_context_json,
                order_flow_json,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item["decided_at"],
                    item["inserted_at"],
                    item["run_id"],
                    item["process_id"],
                    item["decision_id"],
                    item["attempt_id"],
                    item["symbol"],
                    item["zone"],
                    item["action"],
                    item["reason"],
                    item["outcome"],
                    item["outcome_reason"],
                    item["long_score"],
                    item["short_score"],
                    item["flat_bias"],
                    item["score_gap"],
                    item["dominant_side"],
                    item["current_price"],
                    item["allow_entries"],
                    item["execution_tradeable"],
                    item["contracts"],
                    item["order_type"],
                    item["limit_price"],
                    item["decision_price"],
                    item["side"],
                    item["stop_loss"],
                    item["take_profit"],
                    item["max_hold_minutes"],
                    item["regime_state"],
                    item["regime_reason"],
                    item["active_session"],
                    item["active_vetoes_json"],
                    item["feature_snapshot_json"],
                    item["entry_guard_json"],
                    item["unresolved_entry_json"],
                    item["event_context_json"],
                    item["order_flow_json"],
                    item["payload_json"],
                )
                for item in snapshots
            ],
        )
        self._conn.commit()

    def _write_order_lifecycle_locked(self, lifecycles: list[dict[str, Any]]) -> None:
        if not lifecycles:
            return
        assert self._conn is not None
        self._conn.executemany(
            """
            INSERT INTO order_lifecycle (
                observed_at,
                inserted_at,
                run_id,
                process_id,
                decision_id,
                attempt_id,
                order_id,
                position_id,
                trade_id,
                symbol,
                event_type,
                status,
                side,
                role,
                is_protective,
                order_type,
                quantity,
                contracts,
                limit_price,
                stop_price,
                expected_fill_price,
                filled_price,
                filled_quantity,
                remaining_quantity,
                zone,
                reason,
                lifecycle_state,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item["observed_at"],
                    item["inserted_at"],
                    item["run_id"],
                    item["process_id"],
                    item["decision_id"],
                    item["attempt_id"],
                    item["order_id"],
                    item["position_id"],
                    item["trade_id"],
                    item["symbol"],
                    item["event_type"],
                    item["status"],
                    item["side"],
                    item["role"],
                    item["is_protective"],
                    item["order_type"],
                    item["quantity"],
                    item["contracts"],
                    item["limit_price"],
                    item["stop_price"],
                    item["expected_fill_price"],
                    item["filled_price"],
                    item["filled_quantity"],
                    item["remaining_quantity"],
                    item["zone"],
                    item["reason"],
                    item["lifecycle_state"],
                    item["payload_json"],
                )
                for item in lifecycles
            ],
        )
        self._conn.commit()

    def _write_bridge_health_locked(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        assert self._conn is not None
        self._conn.executemany(
            """
            INSERT INTO bridge_health (
                observed_at,
                inserted_at,
                run_id,
                bridge_status,
                queue_depth,
                last_flush_at,
                last_success_at,
                last_error,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item["observed_at"],
                    item["inserted_at"],
                    item["run_id"],
                    item["bridge_status"],
                    item["queue_depth"],
                    item["last_flush_at"],
                    item["last_success_at"],
                    item["last_error"],
                    item["payload_json"],
                )
                for item in records
            ],
        )
        self._conn.commit()

    def _write_runtime_logs_locked(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        assert self._conn is not None
        self._conn.executemany(
            """
            INSERT INTO runtime_logs (
                logged_at,
                inserted_at,
                run_id,
                logger_name,
                level,
                source,
                service_name,
                process_id,
                line_hash,
                thread_name,
                message,
                exception_text,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item["logged_at"],
                    item["inserted_at"],
                    item["run_id"],
                    item["logger_name"],
                    item["level"],
                    item["source"],
                    item["service_name"],
                    item["process_id"],
                    item["line_hash"],
                    item["thread_name"],
                    item["message"],
                    item["exception_text"],
                    item["payload_json"],
                )
                for item in records
            ],
        )
        self._conn.commit()

    def _prune_old_records_locked(self) -> None:
        retention_days = int(getattr(self.settings, "retention_days", 0) or 0)
        if retention_days <= 0:
            return
        assert self._conn is not None
        cutoff = datetime.now(UTC) - timedelta(days=retention_days)
        self._conn.execute("DELETE FROM events WHERE event_timestamp < ?", (self._serialize_datetime(cutoff),))
        self._conn.execute("DELETE FROM state_snapshots WHERE captured_at < ?", (self._serialize_datetime(cutoff),))
        self._conn.execute("DELETE FROM market_tape WHERE captured_at < ?", (self._serialize_datetime(cutoff),))
        self._conn.execute("DELETE FROM decision_snapshots WHERE decided_at < ?", (self._serialize_datetime(cutoff),))
        self._conn.execute("DELETE FROM order_lifecycle WHERE observed_at < ?", (self._serialize_datetime(cutoff),))
        self._conn.execute("DELETE FROM bridge_health WHERE observed_at < ?", (self._serialize_datetime(cutoff),))
        self._conn.execute("DELETE FROM runtime_logs WHERE logged_at < ?", (self._serialize_datetime(cutoff),))
        self._conn.execute("DELETE FROM account_trades WHERE occurred_at < ?", (self._serialize_datetime(cutoff),))
        self._conn.commit()

    def _normalize_value(self, value: Any) -> Any:
        if is_dataclass(value):
            return self._normalize_value(asdict(value))
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, datetime):
            return self._serialize_datetime(value)
        if isinstance(value, dict):
            return {str(key): self._normalize_value(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._normalize_value(item) for item in value]
        if isinstance(value, Path):
            return str(value)
        return value

    def _serialize_datetime(self, value: datetime) -> str:
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC).isoformat()
        return value.astimezone(UTC).isoformat()

    def _coerce_datetime_value(self, value: datetime | str) -> str:
        if isinstance(value, datetime):
            return self._serialize_datetime(value)
        return str(value)

    def _coerce_optional_datetime_value(self, value: Optional[datetime | str]) -> Optional[str]:
        if value in {None, ""}:
            return None
        return self._coerce_datetime_value(value)

    def _coerce_bool(self, value: Any) -> int:
        if value is None:
            return None
        return 1 if bool(value) else 0

    def _json_dumps(self, payload: Any) -> str:
        return json.dumps(self._normalize_value(payload), separators=(",", ":"), sort_keys=True)

    def _decode_run_manifest_row(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["git_dirty"] = bool(item["git_dirty"]) if item["git_dirty"] is not None else None
        item["git_available"] = bool(item["git_available"]) if item["git_available"] is not None else None
        item["account_is_practice"] = bool(item["account_is_practice"]) if item.get("account_is_practice") is not None else None
        item["payload"] = json.loads(item.pop("payload_json") or "{}")
        return item

    def _decode_state_snapshot_row(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["account_is_practice"] = bool(item["account_is_practice"]) if item.get("account_is_practice") is not None else None
        item["payload"] = json.loads(item.pop("payload_json") or "{}")
        return item

    def _decode_market_tape_row(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["volume_is_cumulative"] = bool(item["volume_is_cumulative"]) if item["volume_is_cumulative"] is not None else None
        item["quote_is_synthetic"] = bool(item["quote_is_synthetic"]) if item["quote_is_synthetic"] is not None else None
        item["payload"] = json.loads(item.pop("payload_json") or "{}")
        return item

    def _decode_decision_snapshot_row(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["allow_entries"] = bool(item["allow_entries"]) if item["allow_entries"] is not None else None
        item["execution_tradeable"] = bool(item["execution_tradeable"]) if item["execution_tradeable"] is not None else None
        item["active_vetoes"] = json.loads(item.pop("active_vetoes_json") or "[]")
        item["feature_snapshot"] = json.loads(item.pop("feature_snapshot_json") or "{}")
        item["entry_guard"] = json.loads(item.pop("entry_guard_json") or "{}")
        item["unresolved_entry"] = json.loads(item.pop("unresolved_entry_json") or "{}")
        item["event_context"] = json.loads(item.pop("event_context_json") or "{}")
        item["order_flow"] = json.loads(item.pop("order_flow_json") or "{}")
        item["payload"] = json.loads(item.pop("payload_json") or "{}")
        return item

    def _decode_order_lifecycle_row(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["is_protective"] = bool(item["is_protective"]) if item["is_protective"] is not None else None
        item["payload"] = json.loads(item.pop("payload_json") or "{}")
        return item

    def _decode_bridge_health_row(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json") or "{}")
        return item

    def _decode_runtime_log_row(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["payload"] = json.loads(item.pop("payload_json") or "{}")
        return item

    def _decode_completed_trade_row(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["backfilled"] = bool(item["backfilled"])
        item["account_is_practice"] = bool(item["account_is_practice"]) if item.get("account_is_practice") is not None else None
        item["event_tags"] = json.loads(item.pop("event_tags_json") or "[]")
        item["payload"] = json.loads(item.pop("payload_json") or "{}")
        return item

    def _decode_account_trade_row(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["account_is_practice"] = bool(item["account_is_practice"]) if item.get("account_is_practice") is not None else None
        item["voided"] = bool(item["voided"]) if item.get("voided") is not None else None
        item["payload"] = json.loads(item.pop("payload_json") or "{}")
        return item

    def _insert_completed_trade_locked(self, record: dict[str, Any]) -> bool:
        assert self._conn is not None
        before = self._conn.total_changes
        self._conn.execute(
            """
            INSERT OR IGNORE INTO completed_trades (
                run_id,
                inserted_at,
                entry_time,
                exit_time,
                direction,
                contracts,
                entry_price,
                exit_price,
                pnl,
                zone,
                strategy,
                regime,
                event_tags_json,
                source,
                backfilled,
                trade_id,
                position_id,
                decision_id,
                attempt_id,
                account_id,
                account_name,
                account_mode,
                account_is_practice,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["run_id"],
                record["inserted_at"],
                record["entry_time"],
                record["exit_time"],
                record["direction"],
                record["contracts"],
                record["entry_price"],
                record["exit_price"],
                record["pnl"],
                record["zone"],
                record["strategy"],
                record["regime"],
                record["event_tags_json"],
                record["source"],
                record["backfilled"],
                record.get("trade_id"),
                record.get("position_id"),
                record.get("decision_id"),
                record.get("attempt_id"),
                record.get("account_id"),
                record.get("account_name"),
                record.get("account_mode"),
                record.get("account_is_practice"),
                record["payload_json"],
            ),
        )
        self._conn.commit()
        return self._conn.total_changes > before

    def _write_account_trades_locked(self, records: list[dict[str, Any]]) -> None:
        if not records:
            return
        assert self._conn is not None
        self._conn.executemany(
            """
            INSERT OR IGNORE INTO account_trades (
                run_id,
                inserted_at,
                occurred_at,
                account_id,
                account_name,
                account_mode,
                account_is_practice,
                broker_trade_id,
                broker_order_id,
                contract_id,
                side,
                size,
                price,
                profit_and_loss,
                fees,
                voided,
                source,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    item["run_id"],
                    item["inserted_at"],
                    item["occurred_at"],
                    item["account_id"],
                    item["account_name"],
                    item["account_mode"],
                    item["account_is_practice"],
                    item["broker_trade_id"],
                    item["broker_order_id"],
                    item["contract_id"],
                    item["side"],
                    item["size"],
                    item["price"],
                    item["profit_and_loss"],
                    item["fees"],
                    item["voided"],
                    item["source"],
                    item["payload_json"],
                )
                for item in records
            ],
        )
        self._conn.commit()


_store: Optional[ObservabilityStore] = None


def get_observability_store(force_recreate: bool = False, config: Optional[Config] = None) -> ObservabilityStore:
    global _store
    if force_recreate and _store is not None:
        _store.stop()
        _store = None
    if _store is None:
        _store = ObservabilityStore(config=config)
    return _store

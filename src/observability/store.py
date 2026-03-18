from __future__ import annotations

import json
import logging
import os
import queue
import sqlite3
import threading
import time
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
                self._prune_old_events_locked()
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

    def query_events(
        self,
        *,
        limit: int = 100,
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
            query = (
                "SELECT id, event_timestamp, inserted_at, run_id, process_id, category, event_type, source, symbol, zone, action, reason, order_id, risk_state, payload_json "
                f"FROM events {where} ORDER BY id DESC LIMIT ?"
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

    def get_db_path(self) -> str:
        return str(self._db_path)

    def get_run_id(self) -> str:
        return self._run_id

    def record_run_manifest(self, manifest: dict[str, Any]) -> None:
        if not self.enabled():
            return
        try:
            self.start()
            payload = self._normalize_value(manifest or {})
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
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(run_id) DO UPDATE SET
                        created_at=excluded.created_at,
                        process_id=excluded.process_id,
                        data_mode=excluded.data_mode,
                        symbol=excluded.symbol,
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
                clauses.append("(run_id LIKE ? OR data_mode LIKE ? OR symbol LIKE ? OR git_branch LIKE ? OR git_commit LIKE ? OR payload_json LIKE ?)")
                params.extend([pattern, pattern, pattern, pattern, pattern, pattern])
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            with self._lock:
                assert self._conn is not None
                rows = self._conn.execute(
                    f"""
                    SELECT run_id, created_at, process_id, data_mode, symbol, config_path, config_hash,
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
                    SELECT run_id, created_at, process_id, data_mode, symbol, config_path, config_hash,
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
        run_id: Optional[str] = None,
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
            if run_id:
                clauses.append("run_id = ?")
                params.append(run_id)
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
            with self._lock:
                assert self._conn is not None
                rows = self._conn.execute(
                    f"""
                    SELECT id, run_id, inserted_at, entry_time, exit_time, direction, contracts, entry_price,
                           exit_price, pnl, zone, strategy, regime, event_tags_json, source, backfilled, payload_json
                    FROM completed_trades
                    {where}
                    ORDER BY exit_time DESC, id DESC
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
                        self._write_events_locked(pending)
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
            self._write_events_locked(drained)

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

            CREATE TABLE IF NOT EXISTS run_manifests (
                run_id TEXT PRIMARY KEY,
                created_at TEXT NOT NULL,
                process_id INTEGER NOT NULL,
                data_mode TEXT,
                symbol TEXT,
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
                payload_json TEXT NOT NULL,
                UNIQUE(run_id, exit_time, direction, contracts, entry_price, exit_price, pnl, zone, strategy)
            );

            CREATE INDEX IF NOT EXISTS idx_completed_trades_run_id ON completed_trades(run_id, exit_time DESC);
            CREATE INDEX IF NOT EXISTS idx_completed_trades_pnl ON completed_trades(pnl DESC, exit_time DESC);
            CREATE INDEX IF NOT EXISTS idx_completed_trades_zone ON completed_trades(zone, exit_time DESC);
            """
        )
        self._conn.execute(
            "INSERT INTO metadata(key, value) VALUES('schema_version', '2') ON CONFLICT(key) DO UPDATE SET value=excluded.value"
        )
        self._conn.commit()

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

    def _prune_old_events_locked(self) -> None:
        retention_days = int(getattr(self.settings, "retention_days", 0) or 0)
        if retention_days <= 0:
            return
        assert self._conn is not None
        cutoff = datetime.now(UTC) - timedelta(days=retention_days)
        self._conn.execute("DELETE FROM events WHERE event_timestamp < ?", (self._serialize_datetime(cutoff),))
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
        item["payload"] = json.loads(item.pop("payload_json") or "{}")
        return item

    def _decode_completed_trade_row(self, row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["backfilled"] = bool(item["backfilled"])
        item["event_tags"] = json.loads(item.pop("event_tags_json") or "[]")
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
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
                record["payload_json"],
            ),
        )
        self._conn.commit()
        return self._conn.total_changes > before


_store: Optional[ObservabilityStore] = None


def get_observability_store(force_recreate: bool = False, config: Optional[Config] = None) -> ObservabilityStore:
    global _store
    if force_recreate and _store is not None:
        _store.stop()
        _store = None
    if _store is None:
        _store = ObservabilityStore(config=config)
    return _store

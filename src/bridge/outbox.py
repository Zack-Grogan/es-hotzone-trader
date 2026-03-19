"""Durable outbox for Railway ingest payloads (SQLite-backed)."""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RailwayOutbox:
    """Persistent queue for state/event/trade batches. At-least-once delivery."""

    def __init__(self, outbox_path: str, project_root: Optional[Path] = None):
        path = Path(outbox_path)
        if not path.is_absolute():
            root = project_root or Path(__file__).resolve().parent.parent.parent
            path = root / path
        self._path = path
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None

    def _connect(self) -> sqlite3.Connection:
        with self._lock:
            if self._conn is not None:
                return self._conn
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self._path), timeout=10.0)
            self._conn.row_factory = sqlite3.Row
            self._ensure_schema()
            return self._conn

    def _ensure_schema(self) -> None:
        assert self._conn is not None
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS outbox (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_id TEXT NOT NULL UNIQUE,
                kind TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                permanent_failure INTEGER NOT NULL DEFAULT 0
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_outbox_kind_created ON outbox(kind, created_at)"
        )
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS delivery_state (
                kind TEXT PRIMARY KEY,
                cursor_value INTEGER NOT NULL DEFAULT 0,
                last_batch_id TEXT,
                last_success_at TEXT,
                last_error TEXT
            )
            """
        )
        columns = {
            row["name"]
            for row in self._conn.execute("PRAGMA table_info(outbox)").fetchall()
        }
        if "permanent_failure" not in columns:
            self._conn.execute(
                "ALTER TABLE outbox ADD COLUMN permanent_failure INTEGER NOT NULL DEFAULT 0"
            )
        self._conn.commit()

    def enqueue(self, kind: str, payload: dict[str, Any], batch_id: Optional[str] = None) -> bool:
        """Append a batch to the outbox. Returns True if enqueued (False if duplicate batch_id)."""
        try:
            conn = self._connect()
            bid = batch_id or f"{kind}_{int(time.time() * 1000)}_{os.getpid()}"
            with self._lock:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO outbox (batch_id, kind, payload_json, created_at, attempts) VALUES (?, ?, ?, ?, 0)",
                    (bid, kind, json.dumps(payload, default=str, separators=(",", ":")), datetime.now(UTC).isoformat()),
                )
                conn.commit()
                return cur.rowcount > 0
        except Exception:
            logger.exception("Outbox enqueue failed kind=%s", kind)
            return False

    def dequeue_batch(self, limit: int = 50, *, include_permanent: bool = False) -> list[dict[str, Any]]:
        """Return up to `limit` rows (oldest first) for draining."""
        try:
            conn = self._connect()
            where = "" if include_permanent else "WHERE permanent_failure = 0"
            with self._lock:
                rows = conn.execute(
                    f"SELECT id, batch_id, kind, payload_json, created_at, attempts, last_error, permanent_failure FROM outbox {where} ORDER BY id ASC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(row) for row in rows]
        except Exception:
            logger.exception("Outbox dequeue_batch failed")
            return []

    def mark_sent(self, row_id: int) -> None:
        """Remove a row after successful send."""
        try:
            conn = self._connect()
            with self._lock:
                conn.execute("DELETE FROM outbox WHERE id = ?", (row_id,))
                conn.commit()
        except Exception:
            logger.exception("Outbox mark_sent failed id=%s", row_id)

    def mark_failed(self, row_id: int, error: str, *, permanent: bool = False) -> None:
        """Increment attempts and store error for retry later."""
        try:
            conn = self._connect()
            with self._lock:
                conn.execute(
                    "UPDATE outbox SET attempts = attempts + 1, last_error = ?, permanent_failure = ? WHERE id = ?",
                    (error[:500], 1 if permanent else 0, row_id),
                )
                conn.commit()
        except Exception:
            logger.exception("Outbox mark_failed failed id=%s", row_id)

    def query_pending(
        self,
        *,
        limit: int = 50,
        kind: Optional[str] = None,
        include_permanent: bool = True,
    ) -> list[dict[str, Any]]:
        try:
            conn = self._connect()
            clauses: list[str] = []
            params: list[Any] = []
            if kind:
                clauses.append("kind = ?")
                params.append(kind)
            if not include_permanent:
                clauses.append("permanent_failure = 0")
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            with self._lock:
                rows = conn.execute(
                    f"""SELECT id, batch_id, kind, payload_json, created_at, attempts, last_error, permanent_failure
                        FROM outbox
                        {where}
                        ORDER BY id ASC
                        LIMIT ?""",
                    [*params, max(int(limit), 1)],
                ).fetchall()
            return [dict(row) for row in rows]
        except Exception:
            logger.exception("Outbox query_pending failed")
            return []

    def get_queue_stats(self) -> dict[str, Any]:
        try:
            conn = self._connect()
            with self._lock:
                total_row = conn.execute(
                    "SELECT COUNT(*) AS total, COALESCE(SUM(CASE WHEN permanent_failure = 1 THEN 1 ELSE 0 END), 0) AS permanent_failures, MAX(attempts) AS max_attempts FROM outbox"
                ).fetchone()
                oldest_row = conn.execute(
                    "SELECT created_at FROM outbox ORDER BY id ASC LIMIT 1"
                ).fetchone()
            return {
                "total": int(total_row["total"] or 0),
                "permanent_failures": int(total_row["permanent_failures"] or 0),
                "max_attempts": int(total_row["max_attempts"] or 0),
                "oldest_created_at": oldest_row["created_at"] if oldest_row else None,
            }
        except Exception:
            logger.exception("Outbox get_queue_stats failed")
            return {"total": 0, "permanent_failures": 0, "max_attempts": 0, "oldest_created_at": None}

    def update_delivery_cursor(
        self,
        kind: str,
        cursor_value: int,
        *,
        last_batch_id: Optional[str] = None,
        last_success_at: Optional[str] = None,
        last_error: Optional[str] = None,
    ) -> None:
        try:
            conn = self._connect()
            observed_at = last_success_at or datetime.now(UTC).isoformat()
            with self._lock:
                conn.execute(
                    """
                    INSERT INTO delivery_state (kind, cursor_value, last_batch_id, last_success_at, last_error)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(kind) DO UPDATE SET
                        cursor_value = excluded.cursor_value,
                        last_batch_id = COALESCE(excluded.last_batch_id, delivery_state.last_batch_id),
                        last_success_at = excluded.last_success_at,
                        last_error = excluded.last_error
                    """,
                    (kind, max(int(cursor_value), 0), last_batch_id, observed_at, last_error),
                )
                conn.commit()
        except Exception:
            logger.exception("Outbox update_delivery_cursor failed kind=%s", kind)

    def get_delivery_state(self, kind: Optional[str] = None) -> dict[str, dict[str, Any]]:
        try:
            conn = self._connect()
            with self._lock:
                if kind:
                    rows = conn.execute(
                        "SELECT kind, cursor_value, last_batch_id, last_success_at, last_error FROM delivery_state WHERE kind = ?",
                        (kind,),
                    ).fetchall()
                else:
                    rows = conn.execute(
                        "SELECT kind, cursor_value, last_batch_id, last_success_at, last_error FROM delivery_state ORDER BY kind ASC"
                    ).fetchall()
            return {
                str(row["kind"]): {
                    "cursor_value": int(row["cursor_value"] or 0),
                    "last_batch_id": row["last_batch_id"],
                    "last_success_at": row["last_success_at"],
                    "last_error": row["last_error"],
                }
                for row in rows
            }
        except Exception:
            logger.exception("Outbox get_delivery_state failed")
            return {}

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

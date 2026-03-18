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
                last_error TEXT
            )
            """
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_outbox_kind_created ON outbox(kind, created_at)"
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

    def dequeue_batch(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return up to `limit` rows (oldest first) for draining."""
        try:
            conn = self._connect()
            with self._lock:
                rows = conn.execute(
                    "SELECT id, batch_id, kind, payload_json, created_at, attempts FROM outbox ORDER BY id ASC LIMIT ?",
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

    def mark_failed(self, row_id: int, error: str) -> None:
        """Increment attempts and store error for retry later."""
        try:
            conn = self._connect()
            with self._lock:
                conn.execute(
                    "UPDATE outbox SET attempts = attempts + 1, last_error = ? WHERE id = ?",
                    (error[:500], row_id),
                )
                conn.commit()
        except Exception:
            logger.exception("Outbox mark_failed failed id=%s", row_id)

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None

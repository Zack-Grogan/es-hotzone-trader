from __future__ import annotations

import os
import plistlib
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.bridge.outbox import RailwayOutbox
from src.bridge.railway_bridge import rebuild_outbox_from_observability
from src.cli.commands import _sync_account_trade_history
from src.cli.launchd import LAUNCHD_LABEL, render_launchd_plist
from src.config import get_config, load_config, set_config
from src.observability import get_observability_store


class RuntimeOpsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.previous_config = get_config()

    def tearDown(self) -> None:
        set_config(self.previous_config)

    def _build_config(self, temp_dir: str):
        config = load_config()
        config.observability.enabled = True
        config.observability.sqlite_path = str(Path(temp_dir) / "observability.db")
        config.observability.outbox_path = str(Path(temp_dir) / "railway_outbox.db")
        config.logging.file = str(Path(temp_dir) / "trading.log")
        set_config(config)
        return config

    def test_observability_store_persists_runtime_logs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._build_config(temp_dir)
            store = get_observability_store(force_recreate=True, config=config)
            store.start()
            store.record_runtime_log(
                {
                    "logger_name": "tests.runtime",
                    "level": "ERROR",
                    "process_id": 42,
                    "service_name": "es-hotzone-trader",
                    "source": "local-runtime",
                    "line_hash": "abc123",
                    "thread_name": "MainThread",
                    "message": "bridge auth failed",
                    "exception_text": "HTTP 401",
                }
            )
            store.force_flush()

            rows = store.query_runtime_logs(limit=10, level="ERROR", search="auth")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["message"], "bridge auth failed")
            self.assertEqual(rows[0]["payload"]["exception_text"], "HTTP 401")
            store.stop()

    def test_observability_store_upgrades_existing_completed_trades_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._build_config(temp_dir)
            db_path = Path(config.observability.sqlite_path)
            with sqlite3.connect(db_path) as conn:
                conn.execute(
                    """
                    CREATE TABLE completed_trades (
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
                        payload_json TEXT NOT NULL,
                        UNIQUE(run_id, exit_time, direction, contracts, entry_price, exit_price, pnl, zone, strategy)
                    )
                    """
                )
                conn.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
                conn.execute(
                    "INSERT INTO metadata(key, value) VALUES (?, ?)",
                    ("schema_version", "4"),
                )
                conn.commit()

            store = get_observability_store(force_recreate=True, config=config)
            store.start()
            store.stop()

            with sqlite3.connect(db_path) as conn:
                columns = [row[1] for row in conn.execute("PRAGMA table_info(completed_trades)").fetchall()]
            self.assertIn("account_id", columns)
            self.assertIn("account_name", columns)
            self.assertIn("account_mode", columns)
            self.assertIn("account_is_practice", columns)

    def test_startup_account_trade_sync_authenticates_before_reading_account(self) -> None:
        class FakeAccount:
            account_id = "20139389"
            name = "PRAC-V2-546557-70802903"
            is_practice = True

        class FakeClient:
            def __init__(self) -> None:
                self._access_token = None
                self.authenticate_calls = 0
                self.search_trades_calls = 0

            def authenticate(self) -> bool:
                self.authenticate_calls += 1
                self._access_token = "token"
                return True

            def get_account(self):
                if not self._access_token:
                    return None
                return FakeAccount()

            def search_trades(self, *, start_timestamp: str, end_timestamp: str, account_id: int):
                self.search_trades_calls += 1
                self.last_account_id = account_id
                return [
                    {
                        "id": "trade-1",
                        "accountId": account_id,
                        "side": 1,
                        "size": 1,
                        "price": 6789.25,
                        "profitAndLoss": 42.5,
                        "creationTimestamp": "2026-03-18T14:30:00Z",
                    }
                ]

        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._build_config(temp_dir)
            store = get_observability_store(force_recreate=True, config=config)
            store.start()
            fake_client = FakeClient()
            with patch("src.cli.commands.get_client", return_value=fake_client):
                result = _sync_account_trade_history(
                    config,
                    store,
                    source="startup_account_history_sync",
                    lookback_hours=1,
                )
            self.assertEqual(result["error"] if "error" in result else None, None)
            self.assertEqual(result["checked"], 1)
            self.assertEqual(result["imported"], 1)
            self.assertEqual(fake_client.authenticate_calls, 1)
            self.assertEqual(fake_client.search_trades_calls, 1)
            self.assertEqual(fake_client.last_account_id, 20139389)

            rows = store.query_account_trades(limit=10)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["broker_trade_id"], "trade-1")
            self.assertEqual(rows[0]["account_id"], "20139389")
            store.stop()

    def test_outbox_tracks_permanent_failures_and_delivery_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            outbox = RailwayOutbox(str(Path(temp_dir) / "outbox.db"))
            try:
                outbox.enqueue("events", {"events": [{"id": 1}]}, batch_id="events_1")
                batch = outbox.dequeue_batch(limit=10)[0]
                outbox.mark_failed(batch["id"], "HTTP 401", permanent=True)

                self.assertEqual(outbox.dequeue_batch(limit=10), [])
                pending = outbox.query_pending(limit=10, include_permanent=True)
                self.assertEqual(len(pending), 1)
                self.assertEqual(pending[0]["permanent_failure"], 1)

                outbox.update_delivery_cursor("events", 99, last_batch_id="events_99")
                state = outbox.get_delivery_state("events")
                self.assertEqual(state["events"]["cursor_value"], 99)
                self.assertEqual(state["events"]["last_batch_id"], "events_99")
            finally:
                outbox.close()

    def test_rebuild_outbox_from_observability_enqueues_unsent_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            config = self._build_config(temp_dir)
            store = get_observability_store(force_recreate=True, config=config)
            store.start()
            store.record_event(
                category="system",
                event_type="startup",
                source="tests.runtime",
                payload={"value": 1},
                action="start",
                reason="test",
            )
            store.record_runtime_log(
                {
                    "logger_name": "tests.runtime",
                    "level": "INFO",
                    "process_id": 42,
                    "service_name": "es-hotzone-trader",
                    "source": "local-runtime",
                    "line_hash": "def456",
                    "thread_name": "MainThread",
                    "message": "runtime online",
                }
            )
            store.force_flush()
            outbox = RailwayOutbox(config.observability.outbox_path)
            try:
                counts = rebuild_outbox_from_observability(outbox, include_sent=False)
                self.assertEqual(counts["events"], 1)
                self.assertEqual(counts["runtime_logs"], 1)
                pending = outbox.query_pending(limit=10)
                self.assertTrue(any(row["kind"] == "events" for row in pending))
                self.assertTrue(any(row["kind"] == "runtime_logs" for row in pending))
            finally:
                outbox.close()
                store.stop()

    def test_render_launchd_plist_embeds_cli_start(self) -> None:
        payload = render_launchd_plist("/tmp/gtrade-config.yaml")
        parsed = plistlib.loads(payload)
        self.assertEqual(parsed["Label"], LAUNCHD_LABEL)
        self.assertIn("src.cli", parsed["ProgramArguments"])
        self.assertIn("--config", parsed["ProgramArguments"])

    def test_render_launchd_plist_embeds_internal_auth_token(self) -> None:
        with patch.dict(os.environ, {"GTRADE_INTERNAL_API_TOKEN": "test-internal-token"}, clear=False):
            payload = render_launchd_plist()
        parsed = plistlib.loads(payload)
        self.assertEqual(parsed["EnvironmentVariables"]["GTRADE_INTERNAL_API_TOKEN"], "test-internal-token")

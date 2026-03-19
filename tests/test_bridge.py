"""Tests for Railway data bridge and outbox."""
from __future__ import annotations

import json
import os
import tempfile
import unittest

from src.config import get_config, load_config, set_config
from src.bridge.outbox import RailwayOutbox
from src.bridge.railway_bridge import _record_delivery_success, start_railway_bridge, stop_railway_bridge


class TestRailwayOutbox(unittest.TestCase):
    def setUp(self) -> None:
        self._dir = tempfile.mkdtemp()
        self.outbox_path = os.path.join(self._dir, "outbox.db")

    def tearDown(self) -> None:
        try:
            import shutil
            shutil.rmtree(self._dir, ignore_errors=True)
        except Exception:
            pass

    def test_enqueue_dequeue_mark_sent(self) -> None:
        outbox = RailwayOutbox(self.outbox_path)
        ok = outbox.enqueue("state", {"run_id": "r1", "x": 1}, batch_id="b1")
        self.assertTrue(ok)
        outbox.enqueue("state", {"run_id": "r1", "x": 2}, batch_id="b1")  # duplicate batch_id may be ignored
        batches = outbox.dequeue_batch(limit=10)
        self.assertEqual(len(batches), 1)
        self.assertEqual(batches[0]["kind"], "state")
        self.assertEqual(json.loads(batches[0]["payload_json"])["run_id"], "r1")
        outbox.mark_sent(batches[0]["id"])
        batches2 = outbox.dequeue_batch(limit=10)
        self.assertEqual(len(batches2), 0)
        outbox.close()

    def test_mark_failed_increments_attempts(self) -> None:
        outbox = RailwayOutbox(self.outbox_path)
        outbox.enqueue("events", {"events": []}, batch_id="e1")
        batches = outbox.dequeue_batch(limit=10)
        self.assertEqual(len(batches), 1)
        outbox.mark_failed(batches[0]["id"], "HTTP 500")
        batches2 = outbox.dequeue_batch(limit=10)
        self.assertEqual(len(batches2), 1)
        self.assertEqual(batches2[0]["attempts"], 1)
        outbox.close()


class TestRailwayBridgeStart(unittest.TestCase):
    def test_start_returns_false_when_no_url(self) -> None:
        stop_railway_bridge()
        previous = get_config()
        config = load_config()
        config.observability.railway_ingest_url = ""
        config.observability.railway_ingest_api_key = ""
        set_config(config)
        try:
            started = start_railway_bridge()
            self.assertFalse(started)
        finally:
            stop_railway_bridge()
            set_config(previous)


class TestRailwayBridgeDeliveryState(unittest.TestCase):
    def test_run_manifest_success_clears_stale_error(self) -> None:
        class _FakeOutbox:
            def __init__(self) -> None:
                self.calls: list[tuple[str, int, str | None, str | None, str | None]] = []

            def update_delivery_cursor(
                self,
                kind: str,
                cursor_value: int,
                *,
                last_batch_id: str | None = None,
                last_success_at: str | None = None,
                last_error: str | None = None,
            ) -> None:
                self.calls.append((kind, cursor_value, last_batch_id, last_success_at, last_error))

        outbox = _FakeOutbox()
        _record_delivery_success(outbox, "run_manifest", {"run_manifest": {"run_id": "r1"}}, row_batch_id="batch-1")

        self.assertEqual(len(outbox.calls), 1)
        kind, cursor_value, last_batch_id, last_success_at, last_error = outbox.calls[0]
        self.assertEqual(kind, "run_manifest")
        self.assertEqual(cursor_value, 0)
        self.assertEqual(last_batch_id, "batch-1")
        self.assertIsNotNone(last_success_at)
        self.assertIsNone(last_error)

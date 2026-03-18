"""Data bridge to Railway ingest: outbox and drain loop."""
from __future__ import annotations

from src.bridge.railway_bridge import start_railway_bridge, stop_railway_bridge

__all__ = ["start_railway_bridge", "stop_railway_bridge"]

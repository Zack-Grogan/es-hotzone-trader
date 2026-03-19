"""Trading engine orchestration."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
import logging
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, Optional

import pandas as pd
import pytz

from src.config import Config, get_config
from src.engine.decision_matrix import DecisionMatrixEvaluator, MatrixDecision
from src.engine.event_provider import EventContext, LocalEventProvider
from src.engine.market_context import MicrostructureTracker, OrderFlowSnapshot
from src.engine.risk_manager import RiskManager, get_risk_manager
from src.engine.scheduler import HotZoneScheduler, ZoneInfo, get_scheduler
from src.execution import OrderExecutor, get_executor
from src.indicators import atr
from src.market import MarketData, TopstepClient, get_client
from src.observability import get_observability_store
from src.server import get_state, set_state

logger = logging.getLogger(__name__)


class BarAggregator:
    """Aggregate quote updates into one-minute OHLCV bars."""

    def __init__(self, timezone_name: str):
        self.tz = pytz.timezone(timezone_name)
        self._bucket_start: Optional[datetime] = None
        self._bar: Optional[dict] = None
        self._last_total_volume: Optional[int] = None

    def update(self, market_data: MarketData) -> Optional[dict]:
        timestamp = market_data.timestamp
        if timestamp.tzinfo is None:
            timestamp = pytz.utc.localize(timestamp)
        timestamp = timestamp.astimezone(self.tz)
        bucket_start = timestamp.replace(second=0, microsecond=0)
        price = market_data.last or market_data.mid
        if price <= 0:
            return None

        total_volume = max(int(market_data.volume), 0)
        if not market_data.volume_is_cumulative:
            volume_delta = max(total_volume, 1)
        elif self._last_total_volume is None:
            volume_delta = max(total_volume, 1)
        elif total_volume >= self._last_total_volume:
            volume_delta = max(total_volume - self._last_total_volume, 1)
        else:
            volume_delta = max(total_volume, 1)
        self._last_total_volume = total_volume if market_data.volume_is_cumulative else None

        if self._bucket_start is None:
            self._bucket_start = bucket_start
            self._bar = {"timestamp": bucket_start, "open": price, "high": price, "low": price, "close": price, "volume": volume_delta}
            return None

        if bucket_start < self._bucket_start:
            logger.warning("Out-of-order tick ignored: %s < %s", bucket_start.isoformat(), self._bucket_start.isoformat())
            get_observability_store().record_event(
                category="market",
                event_type="out_of_order_tick",
                source="src.engine.trading_engine",
                payload={"bucket_start": bucket_start.isoformat(), "current_bucket_start": self._bucket_start.isoformat()},
                symbol=market_data.symbol,
                action="ignore_tick",
                reason="out_of_order_tick",
                event_time=market_data.timestamp,
            )
            return None

        if bucket_start != self._bucket_start:
            completed_bar = self._bar
            self._bucket_start = bucket_start
            self._bar = {"timestamp": bucket_start, "open": price, "high": price, "low": price, "close": price, "volume": volume_delta}
            return completed_bar

        assert self._bar is not None
        self._bar["high"] = max(self._bar["high"], price)
        self._bar["low"] = min(self._bar["low"], price)
        self._bar["close"] = price
        self._bar["volume"] += volume_delta
        return None

    def flush(self) -> Optional[dict]:
        completed_bar = self._bar
        self._bar = None
        self._bucket_start = None
        return completed_bar


@dataclass
class WatchdogState:
    feed_stale: bool = False
    broker_ack_stale: bool = False
    protection_timeout: bool = False
    fail_safe_lockout: bool = False
    last_feed_time: Optional[datetime] = None


class TradingEngine:
    """Wire together market data, matrix alpha, risk, and execution."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        self.symbol = self.config.symbols[0]
        self.scheduler: HotZoneScheduler = get_scheduler()
        self.risk_manager: RiskManager = get_risk_manager()
        self.executor: OrderExecutor = get_executor()
        self.client: TopstepClient = get_client()
        self.matrix = DecisionMatrixEvaluator(self.config)
        self.default_tz = self.config.hot_zones[0].timezone if self.config.hot_zones else self.config.sessions.timezone
        self.event_provider = LocalEventProvider(self.config.event_provider, self.config.blackout, Path(__file__).parent.parent.parent)
        self.bar_aggregator = BarAggregator(self.default_tz)
        self.microstructure = MicrostructureTracker(self.config.order_flow)

        self._bars = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        self._bars.index = pd.DatetimeIndex([], tz=self.default_tz)
        self._lock = threading.RLock()
        self._running = False
        self._worker: Optional[threading.Thread] = None
        self._mock_mode = False

        self._active_zone_name: Optional[str] = None
        self._last_price: float = 0.0
        self._last_position: int = 0
        self._latest_market_data: Optional[MarketData] = None
        self._latest_flow_snapshot: OrderFlowSnapshot = OrderFlowSnapshot()
        self._latest_event_context: EventContext = EventContext()
        self._stop_loss: Optional[float] = None
        self._take_profit: Optional[float] = None
        self._pending_entry_zone: str = ""
        self._pending_entry_reason: str = ""
        self._position_entry_time: Optional[pd.Timestamp] = None
        self._last_decision: Optional[MatrixDecision] = None
        self._last_entry_reason: Optional[str] = None
        self._last_exit_reason: Optional[str] = None
        self._pending_expected_fill_price: Optional[float] = None
        self._pending_entry_submitted_at: Optional[datetime] = None
        self._last_fill_drift_ticks: Optional[float] = None
        self._last_entry_fill_price: Optional[float] = None
        self._last_protection_attach_latency_seconds: Optional[float] = None
        self._protection_failure_count: int = 0
        self._fail_safe_count: int = 0
        self._position_sync_requested: bool = False
        self._last_entry_guard_snapshot: Dict[str, Any] = {}
        self._last_entry_block_reason: Optional[str] = None
        self._last_decision_price: Optional[float] = None
        self._decision_events_since_heartbeat: int = 0
        self._last_runtime_heartbeat_log_at: float = time.time()
        self._unresolved_entry_submission_count: int = 0
        self._unresolved_entry_side: Optional[str] = None
        self._unresolved_entry_order_ids: list[str] = []
        self._unresolved_entry_first_submitted_at: Optional[datetime] = None
        self._unresolved_entry_last_submitted_at: Optional[datetime] = None
        self._entry_contamination_detected: bool = False
        self._watchdog_state = WatchdogState()
        self._last_unresolved_reconcile_at: float = 0.0
        self._unresolved_reconcile_interval_seconds: float = 30.0
        self._adopted_broker_position_at_startup: bool = False
        self._adopted_broker_orders_at_startup: bool = False
        self._adoption_source: Optional[str] = None
        self._last_reconciliation_at: Optional[datetime] = None
        self._last_reconciliation_reason: Optional[str] = None
        self._position_high_water: Optional[float] = None
        self._position_low_water: Optional[float] = None
        self._protection_mode: str = "static"
        self._last_dynamic_exit_update_at: float = 0.0
        self._last_sizing_telemetry: Dict[str, Any] = {}
        self._decision_sequence: int = 0
        self._attempt_sequence: int = 0
        self._position_sequence: int = 0
        self._trade_sequence: int = 0
        self._pending_decision_id: Optional[str] = None
        self._pending_attempt_id: Optional[str] = None
        self._pending_position_id: Optional[str] = None
        self._pending_trade_id: Optional[str] = None
        self._current_position_id: Optional[str] = None
        self._current_trade_id: Optional[str] = None
        self._last_decision_id: Optional[str] = None
        self._last_attempt_id: Optional[str] = None
        self.observability = get_observability_store()

    @property
    def running(self) -> bool:
        return self._running

    def _next_stable_id(self, kind: str) -> str:
        sequence_name = f"_{kind}_sequence"
        sequence = getattr(self, sequence_name, 0) + 1
        setattr(self, sequence_name, sequence)
        return f"{self.observability.get_run_id()}:{kind}:{sequence}"

    def _record_event(
        self,
        *,
        category: str,
        event_type: str,
        payload: Optional[dict] = None,
        event_time: Optional[datetime] = None,
        action: Optional[str] = None,
        reason: Optional[str] = None,
        zone: Optional[str] = None,
        order_id: Optional[str] = None,
        risk_state: Optional[str] = None,
    ) -> None:
        self.observability.record_event(
            category=category,
            event_type=event_type,
            source=__name__,
            payload=payload or {},
            event_time=event_time,
            symbol=self.symbol,
            zone=zone if zone is not None else self._active_zone_name,
            action=action,
            reason=reason,
            order_id=order_id,
            risk_state=risk_state or self.risk_manager.get_state().value,
        )

    def _record_decision_event(
        self,
        decision: MatrixDecision,
        *,
        zone: Optional[ZoneInfo],
        current_time,
        current_price: float,
        allow_entries: bool,
        outcome: str,
        outcome_reason: Optional[str] = None,
        contracts: Optional[int] = None,
        order_type: Optional[str] = None,
        limit_price: Optional[float] = None,
        order_id: Optional[str] = None,
        decision_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
        position_id: Optional[str] = None,
        trade_id: Optional[str] = None,
    ) -> None:
        decision_id = decision_id or self._next_stable_id("decision")
        self._last_decision_id = decision_id
        if attempt_id:
            self._last_attempt_id = attempt_id
        dominant_side = "long" if decision.long_score >= decision.short_score else "short"
        dominant_score = decision.long_score if dominant_side == "long" else decision.short_score
        opposing_score = decision.short_score if dominant_side == "long" else decision.long_score
        event_time = current_time.to_pydatetime() if hasattr(current_time, "to_pydatetime") else current_time
        payload = {
            "decision_id": decision_id,
            "attempt_id": attempt_id,
            "position_id": position_id or self._current_position_id,
            "trade_id": trade_id or self._current_trade_id,
            "symbol": self.symbol,
            "zone_state": zone.state.value if zone else ("active" if self.config.strategy.trade_outside_hotzones else "inactive"),
            "decision_reason": decision.reason,
            "long_score": decision.long_score,
            "short_score": decision.short_score,
            "flat_bias": decision.flat_bias,
            "dominant_side": dominant_side,
            "dominant_score": dominant_score,
            "opposing_score": opposing_score,
            "score_gap": round(dominant_score - opposing_score, 4),
            "active_vetoes": list(decision.active_vetoes),
            "execution_tradeable": decision.execution_tradeable,
            "allow_entries": allow_entries,
            "current_position": self._last_position,
            "fail_safe_lockout": self._watchdog_state.fail_safe_lockout,
            "current_price": current_price,
            "current_time": event_time,
            "size_fraction": decision.size_fraction,
            "side": decision.side,
            "stop_loss": decision.stop_loss,
            "take_profit": decision.take_profit,
            "max_hold_minutes": decision.max_hold_minutes,
            "regime_state": decision.feature_snapshot.regime_state,
            "regime_reason": decision.feature_snapshot.regime_reason,
            "active_session": decision.feature_snapshot.active_session,
            "event_tags": list(decision.feature_snapshot.event_tags),
            "event_context_reason": self._latest_event_context.reason,
            "post_event_cooling": self._latest_event_context.post_event_cooling,
            "order_flow": {
                "ofi_zscore": decision.feature_snapshot.long_features.get("ofi_zscore", 0.0),
                "quote_rate_state": decision.feature_snapshot.long_features.get("quote_rate_state", 0.0),
                "spread_regime": decision.feature_snapshot.long_features.get("spread_regime", 0.0),
                "volume_pace": decision.feature_snapshot.long_features.get("volume_pace", 0.0),
            },
            "feature_snapshot": {
                "zone_name": decision.feature_snapshot.zone_name,
                "current_price": decision.feature_snapshot.current_price,
                "atr_value": decision.feature_snapshot.atr_value,
                "long_features": decision.feature_snapshot.long_features,
                "short_features": decision.feature_snapshot.short_features,
                "flat_features": decision.feature_snapshot.flat_features,
                "signed_features": decision.feature_snapshot.signed_features,
                "diagnostics": decision.feature_snapshot.diagnostics,
                "mean_reversion_ready_long": decision.feature_snapshot.mean_reversion_ready_long,
                "mean_reversion_ready_short": decision.feature_snapshot.mean_reversion_ready_short,
                "execution_tradeable": decision.feature_snapshot.execution_tradeable,
                "active_session": decision.feature_snapshot.active_session,
                "regime_state": decision.feature_snapshot.regime_state,
                "regime_reason": decision.feature_snapshot.regime_reason,
                "event_tags": list(decision.feature_snapshot.event_tags),
                "capabilities": decision.feature_snapshot.capabilities,
            },
            "outcome": outcome,
            "outcome_reason": outcome_reason,
            "contracts": contracts,
            "order_type": order_type,
            "limit_price": limit_price,
            "decision_price": self._last_decision_price,
            "entry_guard": dict(self._last_entry_guard_snapshot),
            "unresolved_entry": self._unresolved_entry_snapshot(),
        }
        self.observability.record_decision_snapshot(
            {
                "decided_at": event_time,
                "run_id": self.observability.get_run_id(),
                "process_id": os.getpid(),
                **payload,
            }
        )
        self._record_event(
            category="decision",
            event_type="decision_evaluated",
            payload=payload,
            event_time=event_time,
            action=decision.action,
            reason=decision.reason,
            zone=decision.zone_name,
            order_id=order_id,
        )
        self._decision_events_since_heartbeat += 1

    def reset_runtime_state(self, clear_history: bool = True) -> None:
        """Reset engine runtime state for a fresh live or replay session."""
        with self._lock:
            self._bars = pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
            self._bars.index = pd.DatetimeIndex([], tz=self.default_tz)
            self.bar_aggregator = BarAggregator(self.default_tz)
            self.microstructure.reset()
            self._active_zone_name = None
            self._last_price = 0.0
            self._last_position = 0
            self._latest_market_data = None
            self._latest_flow_snapshot = OrderFlowSnapshot()
            self._latest_event_context = EventContext()
            self._stop_loss = None
            self._take_profit = None
            self._pending_entry_zone = ""
            self._pending_entry_reason = ""
            self._position_entry_time = None
            self._last_decision = None
            self._last_entry_reason = None
            self._last_exit_reason = None
            self._pending_expected_fill_price = None
            self._pending_entry_submitted_at = None
            self._last_fill_drift_ticks = None
            self._last_entry_fill_price = None
            self._last_protection_attach_latency_seconds = None
            self._protection_failure_count = 0
            self._fail_safe_count = 0
            self._position_sync_requested = False
            self._last_entry_guard_snapshot = {}
            self._last_entry_block_reason = None
            self._last_decision_price = None
            self._unresolved_entry_submission_count = 0
            self._unresolved_entry_side = None
            self._unresolved_entry_order_ids = []
            self._unresolved_entry_first_submitted_at = None
            self._unresolved_entry_last_submitted_at = None
            self._entry_contamination_detected = False
            self._adopted_broker_position_at_startup = False
            self._adopted_broker_orders_at_startup = False
            self._adoption_source = None
            self._last_reconciliation_at = None
            self._last_reconciliation_reason = None
            self._position_high_water = None
            self._position_low_water = None
            self._protection_mode = "static"
            self._last_dynamic_exit_update_at = 0.0
            self._last_sizing_telemetry = {}
            self._decision_sequence = 0
            self._attempt_sequence = 0
            self._position_sequence = 0
            self._trade_sequence = 0
            self._pending_decision_id = None
            self._pending_attempt_id = None
            self._pending_position_id = None
            self._pending_trade_id = None
            self._current_position_id = None
            self._current_trade_id = None
            self._last_decision_id = None
            self._last_attempt_id = None
            self._watchdog_state = WatchdogState()
            self._last_unresolved_reconcile_at = 0.0
            self.risk_manager.reset_state(clear_history=clear_history)
            self.executor.reset_state(mock_mode=self._mock_mode)
            self.client.reset_mock_state()
            set_state(
                current_zone=None,
                zone_state="inactive",
                data_mode="unknown",
                current_strategy="WEIGHTED_SCORE_MATRIX",
                position=0,
                position_pnl=0,
                daily_pnl=0,
                account_id=None,
                account_name=None,
                account_is_practice=None,
                account_balance=0,
                account_equity=0,
                account_available=0,
                account_margin_used=0,
                account_open_pnl=0,
                account_realized_pnl=0,
                last_signal=None,
                last_price=None,
                long_score=0,
                short_score=0,
                flat_bias=0,
                active_vetoes=[],
                last_entry_reason=None,
                last_exit_reason=None,
                active_session=None,
                anchored_vwaps={},
                vwap_bands={},
                volume_profile={},
                order_flow={},
                regime={"state": None, "reason": None},
                execution={
                    "state": self.executor.get_lifecycle_state(),
                    "protected": False,
                    "expected_fill_price": None,
                    "entry_fill_price": None,
                    "fill_drift_ticks": None,
                    "protection_attach_latency_seconds": None,
                    "protection_failures": 0,
                    "fail_safe_count": 0,
                    "entry_guard": {},
                    "last_entry_block_reason": None,
                    "decision_price": None,
                    "unresolved_entry": self._unresolved_entry_snapshot(),
                },
                heartbeat={"feed": None, "broker_ack": None, "fail_safe_lockout": False},
                event_context={"active_tags": [], "reason": "", "post_event_cooling": False},
                risk_state="normal",
                trades_today=0,
                trades_this_hour=0,
                trades_this_zone=0,
                max_daily_loss=float(self.config.risk.max_daily_loss),
                consecutive_losses=0,
                replay_summary=None,
            )

    def enable_mock_mode(self) -> None:
        """Put the engine into offline execution (no broker). Used only by replay and tests. Practice account runs use start() (live)."""
        with self._lock:
            self._mock_mode = True
            self.client.enable_mock_mode()
            self.executor.enable_mock_mode()

    def start(self) -> None:
        """Start the engine (live: real Topstep API, practice or funded account)."""
        if self._running:
            return
        self._mock_mode = False
        self._running = True
        self._record_event(
            category="system",
            event_type="engine_starting",
            payload={"symbol": self.symbol},
            event_time=self._current_event_time(),
            action="start",
            reason="engine_start",
        )
        set_state(
            status="running",
            running=True,
            data_mode="live",
            replay_summary=None,
            current_strategy="WEIGHTED_SCORE_MATRIX",
            matrix_version=self.config.alpha.matrix_version,
        )

        if not self.client._access_token and not self.client.authenticate():
            self._running = False
            self._record_event(
                category="system",
                event_type="engine_start_failed",
                payload={"error": "TopstepX authentication failed"},
                event_time=self._current_event_time(),
                action="start",
                reason="authentication_failed",
            )
            raise RuntimeError("TopstepX authentication failed")
        account = self.client.get_account()
        if account is None:
            self._running = False
            self._record_event(
                category="system",
                event_type="engine_start_failed",
                payload={"error": "No tradable Topstep account available"},
                event_time=self._current_event_time(),
                action="start",
                reason="account_unavailable",
            )
            raise RuntimeError("No tradable Topstep account available")
        set_state(
            account_balance=account.balance,
            account_equity=account.equity,
            account_available=account.available,
            account_margin_used=account.margin_used,
            account_open_pnl=account.open_pnl,
            account_realized_pnl=account.realized_pnl,
            account_id=account.account_id,
            account_name=account.name,
            account_is_practice=account.is_practice,
        )
        self.client.start_market_stream(
            symbol=self.symbol,
            on_market_data=self.on_market_data,
            on_order_update=self.on_order_update,
            on_position_update=self.on_position_update,
        )
        if not self.client.wait_for_market_stream(timeout=max(float(self.config.watchdog.feed_stale_seconds), 10.0)):
            self._running = False
            self.client.stop_market_stream()
            self._record_event(
                category="market",
                event_type="stream_not_ready",
                payload={"error": self.client.get_last_stream_error(), "timeout_seconds": max(float(self.config.watchdog.feed_stale_seconds), 10.0)},
                event_time=self._current_event_time(),
                action="connect_stream",
                reason="stream_not_ready",
            )
            raise RuntimeError(self.client.get_last_stream_error() or "Live market data stream did not become ready")

        self._adopt_broker_state_at_startup()
        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        self._position_sync_requested = True
        self._record_event(
            category="system",
            event_type="engine_started",
            payload={"symbol": self.symbol},
            event_time=self._current_event_time(),
            action="start",
            reason="engine_started",
        )

    def stop(self) -> None:
        """Stop the engine."""
        self._running = False
        self._record_event(
            category="system",
            event_type="engine_stopping",
            payload={"position": self._last_position},
            event_time=self._current_event_time(),
            action="stop",
            reason="engine_stop",
        )
        if not self._mock_mode:
            try:
                self.client.stop_market_stream()
            except Exception:
                logger.exception("Failed to stop market stream cleanly")
        if self._worker:
            self._worker.join(timeout=5)
            self._worker = None
        self._record_event(
            category="system",
            event_type="engine_stopped",
            payload={"position": self._last_position},
            event_time=self._current_event_time(),
            action="stop",
            reason="engine_stopped",
        )

    def on_order_update(self, payload: dict) -> None:
        """Handle external order updates when available from the broker stream."""
        order_id = payload.get("orderId") or payload.get("id")
        if not order_id:
            return
        status_map = {
            "filled": "FILLED",
            "open": "OPEN",
            "working": "OPEN",
            "cancelled": "CANCELLED",
            "rejected": "REJECTED",
        }
        status = status_map.get(str(payload.get("status", "")).lower())
        if status is None:
            return
        from src.execution.executor import OrderStatus  # Local import avoids cycle.

        fill_info = {
            "filled_quantity": payload.get("filledQuantity", payload.get("filled_quantity", 0)),
            "filled_price": payload.get("filledPrice", payload.get("filled_price", 0.0)),
        }
        self.executor.update_order_status(str(order_id), OrderStatus[status], fill_info)
        logger.info(
            "broker_order_update order_id=%s status=%s filled_quantity=%s filled_price=%s",
            order_id,
            status.lower(),
            fill_info["filled_quantity"],
            fill_info["filled_price"],
        )
        self._record_event(
            category="execution",
            event_type="broker_order_update",
            payload={"payload": payload, "filled_quantity": fill_info["filled_quantity"], "filled_price": fill_info["filled_price"]},
            event_time=self._current_event_time(),
            action=status.lower(),
            reason="broker_order_update",
            order_id=str(order_id),
        )
        if status in {"CANCELLED", "REJECTED"} and self._last_position == 0 and not self.executor.has_active_entry_order(self.symbol):
            self._clear_unresolved_entry_state(status.lower())
        self._position_sync_requested = True

    def on_position_update(self, payload: dict) -> None:
        """Request a position sync after any broker-side position event."""
        self._record_event(
            category="execution",
            event_type="broker_position_update",
            payload={"payload": payload},
            event_time=self._current_event_time(),
            action="sync_position",
            reason="broker_position_update",
        )
        self._position_sync_requested = True

    def on_market_data(self, market_data: MarketData) -> None:
        """Handle incoming market data."""
        with self._lock:
            self._latest_market_data = market_data
            self._watchdog_state.last_feed_time = market_data.timestamp
            self.risk_manager.observe_time(market_data.timestamp)
            self._last_price = market_data.last or market_data.mid or self._last_price
            self.risk_manager.observe_market_price(market_data.mid or market_data.last, market_data.timestamp)
            self._latest_flow_snapshot = self.microstructure.update(market_data)
            set_state(last_price=self._last_price)

            if self._last_position != 0 and self._last_price:
                self._position_high_water = max(self._position_high_water or self._last_price, self._last_price)
                self._position_low_water = min(self._position_low_water or self._last_price, self._last_price)
            if self.executor.process_market_data(market_data):
                self._position_sync_requested = True
                self._sync_position_state()

            flattened = self._enforce_tick_level_risk(self._last_price)
            completed_bar = self.bar_aggregator.update(market_data)
            if completed_bar is not None:
                self._append_bar(completed_bar)
            if flattened:
                return
            if completed_bar is not None:
                self._evaluate_current_state(allow_entries=not self._watchdog_state.fail_safe_lockout)
            elif self.config.alpha.decision_interval == "tick_and_bar" and self._last_position != 0 and not self._bars.empty:
                self._evaluate_current_state(allow_entries=False)

    def flush_pending_bar(self) -> None:
        """Flush the in-progress bar and evaluate it."""
        with self._lock:
            pending_bar = self.bar_aggregator.flush()
            if pending_bar is not None:
                self._append_bar(pending_bar)
                self._evaluate_current_state(allow_entries=not self._watchdog_state.fail_safe_lockout)

    def build_performance_summary(self) -> dict:
        """Summarize current trade history."""
        trades = self.risk_manager.get_trade_history()
        if not trades:
            return {
                "trade_count": 0,
                "win_rate": 0.0,
                "total_pnl": 0.0,
                "avg_winner": 0.0,
                "avg_loser": 0.0,
                "scratch_trades": 0,
                "zone_stats": {},
                "regime_stats": {},
                "event_tag_stats": {},
            }

        wins = [trade.pnl for trade in trades if trade.pnl > 0]
        losses = [trade.pnl for trade in trades if trade.pnl < 0]
        scratch_trades = sum(1 for trade in trades if trade.pnl == 0)
        zone_stats: dict[str, dict[str, float]] = {}
        regime_stats: dict[str, dict[str, float]] = {}
        event_tag_stats: dict[str, dict[str, float]] = {}

        for trade in trades:
            for bucket, key in ((zone_stats, trade.zone or "Unknown"), (regime_stats, trade.regime or "Unknown")):
                stats = bucket.setdefault(key, {"trades": 0, "wins": 0, "pnl": 0.0})
                stats["trades"] += 1
                stats["pnl"] += trade.pnl
                if trade.pnl > 0:
                    stats["wins"] += 1
            for tag in trade.event_tags or ["none"]:
                stats = event_tag_stats.setdefault(tag, {"trades": 0, "wins": 0, "pnl": 0.0})
                stats["trades"] += 1
                stats["pnl"] += trade.pnl
                if trade.pnl > 0:
                    stats["wins"] += 1

        for bucket in (zone_stats, regime_stats, event_tag_stats):
            for stats in bucket.values():
                stats["win_rate"] = stats["wins"] / max(stats["trades"], 1)

        return {
            "trade_count": len(trades),
            "win_rate": len(wins) / len(trades),
            "total_pnl": sum(trade.pnl for trade in trades),
            "avg_winner": sum(wins) / len(wins) if wins else 0.0,
            "avg_loser": sum(losses) / len(losses) if losses else 0.0,
            "scratch_trades": scratch_trades,
            "zone_stats": zone_stats,
            "regime_stats": regime_stats,
            "event_tag_stats": event_tag_stats,
        }

    def _operator_request_path(self) -> Path:
        log_path = Path(self.config.logging.file)
        if not log_path.is_absolute():
            log_path = Path.cwd() / log_path
        return log_path.resolve().parent / "runtime" / "operator_request.json"

    def _process_operator_request(self) -> None:
        path = self._operator_request_path()
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            action = str(data.get("action", "")).strip().lower()
        except Exception:
            path.unlink(missing_ok=True)
            return
        if action == "force_reconcile":
            self._position_sync_requested = True
            self._record_event(
                category="system",
                event_type="operator_force_reconcile",
                payload={"source": data.get("source", "cli")},
                event_time=self._current_event_time(),
                action="operator_request",
                reason="force_reconcile",
            )
        elif action == "clear_unresolved":
            if self._last_position == 0 and self._unresolved_entry_submission_count > 0:
                self._clear_unresolved_entry_state("operator_clear")
                self._record_event(
                    category="system",
                    event_type="operator_clear_unresolved",
                    payload={"source": data.get("source", "cli")},
                    event_time=self._current_event_time(),
                    action="operator_request",
                    reason="clear_unresolved",
                )
        path.unlink(missing_ok=True)

    def _run(self) -> None:
        """Background state reconciliation loop."""
        while self._running:
            try:
                self._process_operator_request()
                self.executor.reconcile_pending_orders()
                if self._unresolved_entry_submission_count > 0 and self._last_position == 0:
                    now_ts = time.time()
                    if now_ts - self._last_unresolved_reconcile_at >= self._unresolved_reconcile_interval_seconds:
                        self._position_sync_requested = True
                        self._last_unresolved_reconcile_at = now_ts
                if self._position_sync_requested or self._last_position != 0:
                    self._sync_position_state()
                if self._last_position != 0:
                    now_ts = time.time()
                    cadence = getattr(self.config.strategy, "dynamic_exit_update_cadence_seconds", 10.0) or 10.0
                    if now_ts - self._last_dynamic_exit_update_at >= cadence:
                        with self._lock:
                            self._refresh_dynamic_exit()
                        self._last_dynamic_exit_update_at = now_ts
                self._handle_watchdogs()
                with self._lock:
                    self._update_server_state()
                    now_ts = time.time()
                    if now_ts - self._last_runtime_heartbeat_log_at >= 60:
                        logger.info(
                            "runtime_heartbeat mode=%s zone=%s position=%s last_price=%s decisions_last_min=%s fail_safe=%s",
                            "replay" if self._mock_mode else "live",
                            self._active_zone_name or "Outside",
                            self._last_position,
                            self._last_price,
                            self._decision_events_since_heartbeat,
                            self._watchdog_state.fail_safe_lockout,
                        )
                        self._last_runtime_heartbeat_log_at = now_ts
                        self._decision_events_since_heartbeat = 0
            except Exception:
                logger.exception("Trading engine loop error")
                self._record_event(
                    category="system",
                    event_type="engine_loop_error",
                    payload={},
                    event_time=self._current_event_time(),
                    action="loop",
                    reason="engine_loop_error",
                )
            time.sleep(1)

    def _append_bar(self, bar: dict) -> None:
        bar_index = pd.DatetimeIndex([bar["timestamp"]], tz=self.default_tz)
        row = pd.DataFrame(
            {"open": [bar["open"]], "high": [bar["high"]], "low": [bar["low"]], "close": [bar["close"]], "volume": [bar["volume"]]},
            index=bar_index,
        )
        self._bars = pd.concat([self._bars, row]).tail(1200)

    def _unresolved_entry_snapshot(self) -> Dict[str, Any]:
        return {
            "submission_count": self._unresolved_entry_submission_count,
            "side": self._unresolved_entry_side,
            "order_ids": list(self._unresolved_entry_order_ids),
            "first_submitted_at": self._unresolved_entry_first_submitted_at.isoformat() if self._unresolved_entry_first_submitted_at else None,
            "last_submitted_at": self._unresolved_entry_last_submitted_at.isoformat() if self._unresolved_entry_last_submitted_at else None,
            "contamination_detected": self._entry_contamination_detected,
        }

    def _clear_unresolved_entry_state(self, reason: str) -> None:
        if (
            self._unresolved_entry_submission_count == 0
            and not self._unresolved_entry_order_ids
            and self._unresolved_entry_side is None
        ):
            return
        self._record_event(
            category="execution",
            event_type="unresolved_entry_cleared",
            payload={"reason": reason, "prior_state": self._unresolved_entry_snapshot()},
            event_time=self._current_event_time(),
            action="clear_unresolved_entry",
            reason=reason,
        )
        self._unresolved_entry_submission_count = 0
        self._unresolved_entry_side = None
        self._unresolved_entry_order_ids = []
        self._unresolved_entry_first_submitted_at = None
        self._unresolved_entry_last_submitted_at = None
        self._entry_contamination_detected = False
        self._adopted_broker_position_at_startup = False
        self._adopted_broker_orders_at_startup = False
        self._adoption_source = None
        self._last_reconciliation_at = None
        self._last_reconciliation_reason = None

    def _track_unresolved_entry_submission(
        self,
        *,
        side: str,
        order_id: str,
        decision_price: float,
        expected_fill_price: float,
    ) -> None:
        event_time = self._current_event_time()
        if self._unresolved_entry_submission_count == 0:
            self._unresolved_entry_first_submitted_at = event_time
        self._unresolved_entry_last_submitted_at = event_time
        self._unresolved_entry_submission_count += 1
        self._unresolved_entry_side = side
        if order_id not in self._unresolved_entry_order_ids:
            self._unresolved_entry_order_ids.append(order_id)
        self._entry_contamination_detected = self._unresolved_entry_submission_count > 1
        self._record_event(
            category="execution",
            event_type="unresolved_entry_tracked",
            payload={
                "side": side,
                "order_id": order_id,
                "decision_price": decision_price,
                "expected_fill_price": expected_fill_price,
                "submission_count": self._unresolved_entry_submission_count,
                "contamination_detected": self._entry_contamination_detected,
            },
            event_time=event_time,
            action="track_unresolved_entry",
            reason=side,
            order_id=order_id,
        )

    def _broker_position_from_snapshot(self, positions: Optional[Dict[str, Any]]) -> int:
        if not positions:
            return 0
        if self.symbol in positions:
            return int(positions[self.symbol].quantity)
        requested = self.symbol.upper()
        for contract_id, position in positions.items():
            normalized = str(contract_id).upper()
            if requested in normalized or normalized in requested:
                return int(position.quantity)
        return 0

    def _broker_entry_price_from_snapshot(self, positions: Optional[Dict[str, Any]]) -> Optional[float]:
        if not positions:
            return None
        if self.symbol in positions:
            return float(positions[self.symbol].entry_price or 0) or None
        requested = self.symbol.upper()
        for contract_id, position in positions.items():
            normalized = str(contract_id).upper()
            if requested in normalized or normalized in requested:
                return float(position.entry_price or 0) or None
        return None

    def _adopt_broker_state_at_startup(self) -> None:
        """Pull broker positions/open orders and hydrate local state; attach protection to adopted positions."""
        if self._mock_mode:
            return
        positions, position_error = self.client.get_positions_snapshot()
        broker_orders, order_error = self.client.get_open_orders_snapshot(self.symbol)
        broker_position = self._broker_position_from_snapshot(positions)
        broker_entry_price = self._broker_entry_price_from_snapshot(positions)
        broker_order_ids = [
            str(o.get("id", o.get("orderId", "")))
            for o in (broker_orders or [])
            if str(o.get("id", o.get("orderId", "")))
        ]
        event_time = self._current_event_time()
        with self._lock:
            if position_error:
                return
            if broker_position != 0:
                entry_price = broker_entry_price or self._last_price or 0.0
                atr_value = self._calculate_current_atr()
                tick_size = 0.25
                if atr_value is not None and atr_value > 0:
                    stop_atr = getattr(self.config.alpha, "stop_loss_atr", 1.2) or 1.2
                    tp_atr = 2.0
                    if broker_position > 0:
                        stop_price = entry_price - stop_atr * atr_value
                        take_profit = entry_price + tp_atr * atr_value
                    else:
                        stop_price = entry_price + stop_atr * atr_value
                        take_profit = entry_price - tp_atr * atr_value
                else:
                    default_stop_pts = 10
                    default_tp_pts = 20
                    if broker_position > 0:
                        stop_price = entry_price - default_stop_pts * tick_size
                        take_profit = entry_price + default_tp_pts * tick_size
                    else:
                        stop_price = entry_price + default_stop_pts * tick_size
                        take_profit = entry_price - default_tp_pts * tick_size
                self._stop_loss = stop_price
                self._take_profit = take_profit
                self._last_position = broker_position
                self._last_entry_fill_price = entry_price
                self._position_high_water = entry_price
                self._position_low_water = entry_price
                self._position_entry_time = pd.Timestamp(event_time)
                zone = self.scheduler.get_current_zone(current_time=self._bars.index[-1] if not self._bars.empty else None)
                zone_name = zone.name if zone else "Outside"
                regime_state = self._last_decision.feature_snapshot.regime_state if self._last_decision else ""
                event_tags = self._latest_event_context.active_tags if self._latest_event_context else []
                self.risk_manager.sync_position(
                    signed_position=broker_position,
                    entry_price=entry_price,
                    transition_price=entry_price,
                    zone=zone_name,
                    regime=regime_state,
                    event_tags=event_tags,
                    strategy="WEIGHTED_SCORE_MATRIX",
                    current_time=event_time,
                )
                self.executor.mark_position_open()
                self.executor.ensure_protection(
                    symbol=self.symbol,
                    quantity=abs(broker_position),
                    direction=1 if broker_position > 0 else -1,
                    stop_price=stop_price,
                    take_profit=take_profit,
                )
                self._adopted_broker_position_at_startup = True
                self._adoption_source = "broker_startup"
                self._last_reconciliation_at = event_time
                self._last_reconciliation_reason = "startup_adopt_position"
                self._record_event(
                    category="execution",
                    event_type="broker_position_adopted",
                    payload={
                        "broker_position": broker_position,
                        "entry_price": entry_price,
                        "stop_price": stop_price,
                        "take_profit": take_profit,
                        "adoption_source": "broker_startup",
                    },
                    event_time=event_time,
                    action="startup_adopt",
                    reason="broker_startup",
                )
                logger.info(
                    "Adopted broker position at startup: position=%s entry=%s stop=%s tp=%s",
                    broker_position, entry_price, stop_price, take_profit,
                )
                return
            if broker_position == 0 and broker_order_ids:
                first = (broker_orders or [])[0] if broker_orders else {}
                raw_side = str(first.get("side", first.get("orderSide", first.get("orderType", "")))).lower()
                side = "buy" if raw_side in ("buy", "long", "bid", "b") else "sell"
                self._unresolved_entry_order_ids = list(broker_order_ids)
                self._unresolved_entry_side = side
                self._unresolved_entry_submission_count = 1
                self._unresolved_entry_first_submitted_at = event_time
                self._unresolved_entry_last_submitted_at = event_time
                self._adopted_broker_orders_at_startup = True
                self._adoption_source = "broker_startup_orders"
                self._last_reconciliation_at = event_time
                self._last_reconciliation_reason = "startup_adopt_orders"
                self._record_event(
                    category="execution",
                    event_type="broker_orders_adopted",
                    payload={
                        "order_ids": broker_order_ids,
                        "side": side,
                        "adoption_source": "broker_startup_orders",
                    },
                    event_time=event_time,
                    action="startup_adopt",
                    reason="broker_startup_orders",
                )
                logger.info("Adopted broker open orders at startup: order_ids=%s side=%s", broker_order_ids, side)

    def _evaluate_live_entry_guard(self, *, side: str, decision_price: float, expected_fill_price: float) -> tuple[bool, Optional[str], Dict[str, Any]]:
        local_active_entry_orders = self.executor.get_active_orders(symbol=self.symbol, is_protective=False)
        snapshot: Dict[str, Any] = {
            "checked": True,
            "mode": "replay" if self._mock_mode else "live",
            "side": side,
            "decision_price": decision_price,
            "expected_fill_price": expected_fill_price,
            "local_position": self._last_position,
            "local_active_entry_order_ids": list(local_active_entry_orders.keys()),
            "local_active_entry_orders": len(local_active_entry_orders),
            "broker_position": None,
            "broker_entry_price": None,
            "broker_open_order_ids": [],
            "broker_open_order_count": 0,
            "broker_position_error": None,
            "broker_open_orders_error": None,
            "position_mismatch": False,
            "open_order_mismatch": False,
            "unresolved_entry": self._unresolved_entry_snapshot(),
            "reason": None,
        }
        if self._mock_mode:
            return True, None, snapshot

        positions, position_error = self.client.get_positions_snapshot()
        broker_orders, order_error = self.client.get_open_orders_snapshot(self.symbol)
        broker_position = self._broker_position_from_snapshot(positions)
        broker_entry_price = self._broker_entry_price_from_snapshot(positions)
        broker_order_ids = [
            str(order.get("id", order.get("orderId", "")))
            for order in (broker_orders or [])
            if str(order.get("id", order.get("orderId", "")))
        ]
        snapshot.update(
            {
                "broker_position": broker_position,
                "broker_entry_price": broker_entry_price,
                "broker_open_order_ids": broker_order_ids,
                "broker_open_order_count": len(broker_order_ids),
                "broker_position_error": position_error,
                "broker_open_orders_error": order_error,
                "position_mismatch": broker_position != self._last_position,
                "open_order_mismatch": bool(local_active_entry_orders) != bool(broker_order_ids),
            }
        )

        if position_error:
            snapshot["reason"] = "broker_position_unavailable"
            return False, "broker_position_unavailable", snapshot
        if order_error:
            snapshot["reason"] = "broker_open_orders_unavailable"
            return False, "broker_open_orders_unavailable", snapshot
        if broker_position != self._last_position:
            snapshot["reason"] = "broker_position_mismatch"
            return False, "broker_position_mismatch", snapshot
        if broker_position != 0:
            snapshot["reason"] = "broker_position_not_flat"
            return False, "broker_position_not_flat", snapshot
        if broker_order_ids:
            snapshot["reason"] = "broker_open_orders_present"
            return False, "broker_open_orders_present", snapshot
        if self._unresolved_entry_submission_count > 0:
            if self._unresolved_entry_side == side and self._unresolved_entry_submission_count + 1 >= 2:
                snapshot["reason"] = "duplicate_unresolved_entry"
                return False, "duplicate_unresolved_entry", snapshot
            snapshot["reason"] = "unresolved_entry_pending"
            return False, "unresolved_entry_pending", snapshot
        return True, None, snapshot

    def _evaluate_current_state(self, allow_entries: bool) -> None:
        if self._bars.empty:
            return

        current_time = self._bars.index[-1]
        current_price = float(self._last_price or self._bars["close"].iloc[-1])
        self._last_decision_price = current_price
        self._last_entry_block_reason = None
        self._last_entry_guard_snapshot = {
            "checked": False,
            "mode": "replay" if self._mock_mode else "live",
            "decision_price": current_price,
            "reason": None,
            "unresolved_entry": self._unresolved_entry_snapshot(),
        }
        self.risk_manager.observe_time(current_time.to_pydatetime() if hasattr(current_time, "to_pydatetime") else current_time)
        zone = self.scheduler.get_current_zone(current_time=current_time)
        self._latest_event_context = self.event_provider.get_context(current_time.to_pydatetime() if hasattr(current_time, "to_pydatetime") else current_time)
        self.risk_manager.set_blackout(self._latest_event_context.blackout_active, self._latest_event_context.reason)

        atr_value = self._calculate_current_atr()
        if atr_value is not None:
            self.risk_manager.update_volatility(atr_value)

        self._handle_zone_transition(zone)
        display_zone_name = zone.name if zone else ("Outside" if self.config.strategy.trade_outside_hotzones else None)
        display_zone_state = zone.state.value if zone else ("active" if self.config.strategy.trade_outside_hotzones else "inactive")
        set_state(current_zone=display_zone_name, zone_state=display_zone_state)

        should_flatten, flatten_reason = self.risk_manager.should_flatten_position(
            current_price,
            current_time.to_pydatetime() if hasattr(current_time, "to_pydatetime") else current_time,
        )
        if should_flatten:
            self._flatten_position(flatten_reason)
            return

        decision = self.matrix.evaluate(
            bars=self._bars.copy(),
            zone=zone,
            market_data=self._latest_market_data,
            risk_state=self.risk_manager.get_state(),
            blackout_active=self._latest_event_context.blackout_active,
            current_position=self._last_position,
            allow_entries=allow_entries and not self._watchdog_state.fail_safe_lockout,
            current_entry_time=self._position_entry_time,
            event_context=self._latest_event_context,
            flow_snapshot=self._latest_flow_snapshot,
        )
        self._last_decision = decision
        self._record_matrix_state(decision)

        if decision.action == "FLAT":
            self._record_decision_event(
                decision,
                zone=zone,
                current_time=current_time,
                current_price=current_price,
                allow_entries=allow_entries,
                outcome="flatten_request" if self._last_position != 0 else "already_flat",
                outcome_reason=decision.reason,
            )
            if self._last_position != 0:
                self._flatten_position(decision.reason)
            return
        if decision.action in {"NO_TRADE", "HOLD"}:
            self._record_decision_event(
                decision,
                zone=zone,
                current_time=current_time,
                current_price=current_price,
                allow_entries=allow_entries,
                outcome=decision.action.lower(),
                outcome_reason=decision.reason,
            )
            return
        if not allow_entries:
            self._record_decision_event(
                decision,
                zone=zone,
                current_time=current_time,
                current_price=current_price,
                allow_entries=allow_entries,
                outcome="entries_disabled",
                outcome_reason="allow_entries_false",
            )
            return
        if self._last_position != 0:
            self._record_decision_event(
                decision,
                zone=zone,
                current_time=current_time,
                current_price=current_price,
                allow_entries=allow_entries,
                outcome="position_open",
                outcome_reason="position_already_open",
            )
            return
        if self._watchdog_state.fail_safe_lockout:
            self._record_decision_event(
                decision,
                zone=zone,
                current_time=current_time,
                current_price=current_price,
                allow_entries=allow_entries,
                outcome="fail_safe_lockout",
                outcome_reason="watchdog_lockout",
            )
            return
        if self.executor.has_active_entry_order(self.symbol):
            logger.info("Entry skipped because an active entry order is already working for %s", self.symbol)
            self._record_decision_event(
                decision,
                zone=zone,
                current_time=current_time,
                current_price=current_price,
                allow_entries=allow_entries,
                outcome="active_entry_order",
                outcome_reason="active_entry_order",
            )
            return

        allowed, reason = self.risk_manager.can_trade(decision.zone_name, current_time=current_time.to_pydatetime() if hasattr(current_time, "to_pydatetime") else current_time)
        if not allowed:
            set_state(active_vetoes=decision.active_vetoes + [reason])
            logger.info("Trade blocked by risk manager: %s", reason)
            self._record_decision_event(
                decision,
                zone=zone,
                current_time=current_time,
                current_price=current_price,
                allow_entries=allow_entries,
                outcome="risk_blocked",
                outcome_reason=reason,
            )
            return

        contracts = self._determine_contracts(decision)
        if contracts <= 0 or decision.side is None:
            self._record_decision_event(
                decision,
                zone=zone,
                current_time=current_time,
                current_price=current_price,
                allow_entries=allow_entries,
                outcome="size_zero" if contracts <= 0 else "missing_side",
                outcome_reason="contracts_non_positive" if contracts <= 0 else "missing_order_side",
                contracts=contracts,
            )
            return

        order_type, limit_price = self._choose_entry_order(decision.side)
        expected_fill_price = limit_price if limit_price is not None else current_price
        decision_id = self._next_stable_id("decision")
        attempt_id = self._next_stable_id("attempt")
        position_id = self._next_stable_id("position")
        trade_id = self._next_stable_id("trade")
        self._pending_decision_id = decision_id
        self._pending_attempt_id = attempt_id
        self._pending_position_id = position_id
        self._pending_trade_id = trade_id
        guard_allows_entry, guard_reason, guard_snapshot = self._evaluate_live_entry_guard(
            side=decision.side,
            decision_price=current_price,
            expected_fill_price=expected_fill_price,
        )
        self._last_entry_guard_snapshot = guard_snapshot
        if not guard_allows_entry:
            self._last_entry_block_reason = guard_reason
            if guard_reason == "duplicate_unresolved_entry":
                self._entry_contamination_detected = True
                self._record_event(
                    category="risk",
                    event_type="duplicate_unresolved_entry_detected",
                    payload=guard_snapshot,
                    event_time=self._current_event_time(),
                    action="fail_safe",
                    reason=guard_reason,
                )
                self._trigger_fail_safe("duplicate_entry_submission_unresolved")
                self._record_decision_event(
                    decision,
                    zone=zone,
                    current_time=current_time,
                    current_price=current_price,
                    allow_entries=allow_entries,
                    outcome="fail_safe_lockout",
                    outcome_reason=guard_reason,
                    contracts=contracts,
                    order_type=order_type,
                    limit_price=limit_price,
                )
                return
            self._record_decision_event(
                decision,
                zone=zone,
                current_time=current_time,
                current_price=current_price,
                allow_entries=allow_entries,
                outcome="broker_entry_guard_blocked",
                outcome_reason=guard_reason,
                contracts=contracts,
                order_type=order_type,
                limit_price=limit_price,
            )
            return
        decision_id = self._next_stable_id("decision")
        attempt_id = self._next_stable_id("attempt")
        position_id = self._next_stable_id("position")
        trade_id = self._next_stable_id("trade")
        self._pending_decision_id = decision_id
        self._pending_attempt_id = attempt_id
        self._pending_position_id = position_id
        self._pending_trade_id = trade_id
        order = self.executor.place_order(
            symbol=self.symbol,
            quantity=contracts,
            side=decision.side,
            order_type=order_type,
            limit_price=limit_price,
            decision_id=decision_id,
            attempt_id=attempt_id,
            position_id=position_id,
            trade_id=trade_id,
        )
        if order is None:
            self._record_decision_event(
                decision,
                zone=zone,
                current_time=current_time,
                current_price=current_price,
                allow_entries=allow_entries,
                outcome="order_submit_failed",
                outcome_reason="executor_place_order_returned_none",
                contracts=contracts,
                order_type=order_type,
                limit_price=limit_price,
                decision_id=decision_id,
                attempt_id=attempt_id,
                position_id=position_id,
                trade_id=trade_id,
            )
            self._pending_decision_id = None
            self._pending_attempt_id = None
            self._pending_position_id = None
            self._pending_trade_id = None
            return

        self._stop_loss = decision.stop_loss
        self._take_profit = decision.take_profit
        self._pending_entry_zone = decision.zone_name
        self._pending_entry_reason = decision.reason
        self._pending_expected_fill_price = expected_fill_price
        self._pending_entry_submitted_at = self._current_event_time()
        self._last_entry_reason = decision.reason
        self._last_entry_block_reason = None
        self._track_unresolved_entry_submission(
            side=decision.side,
            order_id=order.order_id,
            decision_price=current_price,
            expected_fill_price=expected_fill_price,
        )
        self._record_decision_event(
            decision,
            zone=zone,
            current_time=current_time,
            current_price=current_price,
            allow_entries=allow_entries,
            outcome="order_submitted",
            outcome_reason="entry_order_placed",
            contracts=contracts,
            order_type=order_type,
            limit_price=limit_price,
            order_id=order.order_id,
            decision_id=decision_id,
            attempt_id=attempt_id,
            position_id=position_id,
            trade_id=trade_id,
        )
        set_state(
            last_signal={
                "direction": decision.action,
                "price": current_price,
                "decision_price": current_price,
                "reason": decision.reason,
                "strategy": "WEIGHTED_SCORE_MATRIX",
                "zone": decision.zone_name,
                "regime": decision.feature_snapshot.regime_state,
                "expected_fill_price": self._pending_expected_fill_price,
                "decision_id": decision_id,
                "attempt_id": attempt_id,
                "position_id": position_id,
                "trade_id": trade_id,
            },
            last_entry_reason=decision.reason,
        )
        logger.info(
            "decision_trace action=%s zone=%s regime=%s reason=%s expected_fill=%s order_type=%s",
            decision.action,
            decision.zone_name,
            decision.feature_snapshot.regime_state,
            decision.reason,
            self._pending_expected_fill_price,
            order_type,
        )
        if self._mock_mode:
            self._sync_position_state()
        else:
            self._position_sync_requested = True
        logger.info(
            "engine_entry_submitted side=%s contracts=%s order_type=%s price=%s zone=%s reason=%s",
            decision.side,
            contracts,
            order_type,
            limit_price or "market",
            decision.zone_name,
            decision.reason,
        )
        self._record_decision_event(
            decision,
            zone=zone,
            current_time=current_time,
            current_price=current_price,
            allow_entries=allow_entries,
            outcome="entry_submitted",
            outcome_reason=decision.reason,
            contracts=contracts,
            order_type=order_type,
            limit_price=limit_price,
            order_id=order.order_id,
        )

    def _record_matrix_state(self, decision: MatrixDecision) -> None:
        snapshot = decision.feature_snapshot
        set_state(
            current_strategy="WEIGHTED_SCORE_MATRIX",
            long_score=decision.long_score,
            short_score=decision.short_score,
            flat_bias=decision.flat_bias,
            active_vetoes=decision.active_vetoes,
            matrix_version=self.config.alpha.matrix_version,
            active_session=snapshot.active_session,
            anchored_vwaps={"rth": snapshot.diagnostics.get("rth_vwap"), "eth": snapshot.diagnostics.get("eth_vwap")},
            vwap_bands={
                "rth_sigma": snapshot.diagnostics.get("rth_sigma"),
                "eth_sigma": snapshot.diagnostics.get("eth_sigma"),
            },
            volume_profile={
                "poc": snapshot.diagnostics.get("poc"),
                "vah": snapshot.diagnostics.get("vah"),
                "val": snapshot.diagnostics.get("val"),
            },
            order_flow={
                "ofi_zscore": snapshot.long_features.get("ofi_zscore", 0.0),
                "quote_rate_state": snapshot.long_features.get("quote_rate_state", 0.0),
                "spread_regime": snapshot.long_features.get("spread_regime", 0.0),
                "volume_pace": snapshot.long_features.get("volume_pace", 0.0),
            },
            regime={"state": snapshot.regime_state, "reason": snapshot.regime_reason},
            event_context={
                "active_tags": snapshot.event_tags,
                "reason": self._latest_event_context.reason,
                "post_event_cooling": self._latest_event_context.post_event_cooling,
            },
        )

    def _handle_zone_transition(self, zone: Optional[ZoneInfo]) -> None:
        zone_name = zone.name if zone else ("Outside" if self.config.strategy.trade_outside_hotzones else None)
        if zone_name == self._active_zone_name:
            return
        previous_zone = self._active_zone_name
        self._active_zone_name = zone_name
        self._record_event(
            category="market",
            event_type="zone_transition",
            payload={"previous_zone": previous_zone, "new_zone": zone_name, "zone_state": zone.state.value if zone else "inactive"},
            event_time=self._current_event_time(),
            action="zone_transition",
            reason="zone_changed",
            zone=zone_name,
        )

    def _determine_contracts(self, decision: MatrixDecision) -> int:
        direction = 1 if decision.action == "LONG" else -1
        atr_val = decision.feature_snapshot.atr_value or 0.0
        base_contracts, risk_telemetry = self.risk_manager.calculate_position_size_with_telemetry(atr_val, direction)
        if base_contracts <= 0:
            self._last_sizing_telemetry = {**risk_telemetry, "size_fraction": decision.size_fraction, "final_contracts": 0}
            return 0
        contracts = base_contracts if decision.size_fraction >= 1.0 else max(1, int(round(base_contracts * decision.size_fraction)))
        if self.risk_manager.is_reduced_risk():
            contracts = min(contracts, 1)
            if "reduced_risk_cap" not in risk_telemetry.get("guardrail_reasons", []):
                risk_telemetry.setdefault("guardrail_reasons", []).append("reduced_risk_cap")
        contracts = max(1, min(contracts, self.config.account.max_contracts))
        if contracts >= self.config.account.max_contracts and base_contracts >= self.config.account.max_contracts:
            risk_telemetry.setdefault("guardrail_reasons", []).append("max_contracts_cap")
        self._last_sizing_telemetry = {
            **risk_telemetry,
            "size_fraction": decision.size_fraction,
            "final_contracts": contracts,
        }
        return contracts

    def _choose_entry_order(self, side: str) -> tuple[str, Optional[float]]:
        market_data = self._latest_market_data or self.client.get_market_data(self.symbol)
        if not self.config.order_execution.use_limit_orders or market_data is None:
            return "market", None

        tick_size = 0.25
        offset = self.config.order_execution.limit_offset_ticks * tick_size
        max_slippage = self.config.order_execution.max_slippage_ticks * tick_size
        anchor_price = market_data.last or market_data.mid or self._last_price
        if side == "buy":
            anchor = market_data.ask or anchor_price
            limit_price = min(anchor + offset, anchor_price + max_slippage)
        else:
            anchor = market_data.bid or anchor_price
            limit_price = max(anchor - offset, anchor_price - max_slippage)
        return "limit", round(limit_price / tick_size) * tick_size

    def _flatten_position(self, reason: str) -> None:
        self._last_exit_reason = reason
        set_state(last_exit_reason=reason)
        self.executor.clear_protection(self.symbol)
        self.executor.cancel_all_orders()
        self._record_event(
            category="execution",
            event_type="flatten_requested",
            payload={"reason": reason, "position": self._last_position},
            event_time=self._current_event_time(),
            action="flatten",
            reason=reason,
        )
        if self._last_position == 0:
            return
        if self.executor.flatten(self.symbol):
            logger.warning("Flatten requested: %s", reason)
            if self._mock_mode:
                self._sync_position_state()
            else:
                self._position_sync_requested = True

    def _enforce_tick_level_risk(self, current_price: float) -> bool:
        if self._last_position == 0 or self.executor.is_protected(self.symbol):
            return False
        if self._stop_loss is not None:
            if self._last_position > 0 and current_price <= self._stop_loss:
                self._flatten_position("stop_loss")
                return True
            if self._last_position < 0 and current_price >= self._stop_loss:
                self._flatten_position("stop_loss")
                return True
        if self._take_profit is not None:
            if self._last_position > 0 and current_price >= self._take_profit:
                self._flatten_position("take_profit")
                return True
            if self._last_position < 0 and current_price <= self._take_profit:
                self._flatten_position("take_profit")
                return True
        return False

    def _sync_position_state(self) -> None:
        if self._mock_mode:
            signed_position = self.executor.get_position(self.symbol)
            entry_price = self.executor.get_average_price()
            position_authoritative = True
        else:
            position = self.client.get_position(self.symbol)
            signed_position = position.quantity
            entry_price = position.entry_price
            position_authoritative = getattr(position, "authoritative", True)

        event_time = self._current_event_time()
        with self._lock:
            if not self._mock_mode and not position_authoritative:
                if signed_position != self._last_position:
                    self._record_event(
                        category="execution",
                        event_type="position_sync_skipped_unavailable",
                        payload={
                            "signed_position": signed_position,
                            "last_position": self._last_position,
                        },
                        event_time=event_time,
                        action="sync_position",
                        reason="position_lookup_unavailable",
                    )
                self._position_sync_requested = False
                self._last_reconciliation_at = event_time
                self._last_reconciliation_reason = "position_lookup_unavailable"
                return

            if not self._mock_mode and signed_position == 0 and self._unresolved_entry_submission_count > 0:
                broker_orders, order_error = self.client.get_open_orders_snapshot(self.symbol)
                if order_error is None and (broker_orders is None or len(broker_orders) == 0):
                    self._clear_unresolved_entry_state("broker_flat_no_orders")
                    self._last_position = 0
                    self._position_sync_requested = False
                    self._last_reconciliation_at = event_time
                    self._last_reconciliation_reason = "broker_flat_no_orders"
                    self._record_event(
                        category="execution",
                        event_type="reconciliation_broker_truth",
                        payload={"reason": "broker_flat_no_orders", "signed_position": 0},
                        event_time=event_time,
                        action="sync_position",
                        reason="broker_flat_no_orders",
                    )
                    return

            if signed_position != self._last_position:
                fills = self.executor.consume_fills(self.symbol)
                delta = signed_position - self._last_position
                transition_side = "buy" if delta > 0 else "sell"
                fill_qty = 0.0
                fill_notional = 0.0
                for fill in fills:
                    if str(fill.get("side", "")).lower() != transition_side:
                        continue
                    quantity = float(fill.get("quantity", 0) or 0)
                    price = float(fill.get("filled_price", 0) or 0)
                    if quantity <= 0 or price <= 0:
                        continue
                    fill_qty += quantity
                    fill_notional += quantity * price

                transition_price = (fill_notional / fill_qty) if fill_qty > 0 else (entry_price or self._last_price)
                actual_entry_price = entry_price or transition_price or self._last_price
                zone = self.scheduler.get_current_zone(current_time=self._bars.index[-1] if not self._bars.empty else None)
                regime_state = self._last_decision.feature_snapshot.regime_state if self._last_decision else ""
                event_tags = self._latest_event_context.active_tags if self._latest_event_context else []
                zone_name = self._pending_entry_zone or self._active_zone_name or (zone.name if zone else ("Outside" if self.config.strategy.trade_outside_hotzones else "sync_recovery"))
                prior_position = self._last_position
                position_id = self._pending_position_id or self._current_position_id or self._next_stable_id("position")
                trade_id = self._pending_trade_id or self._current_trade_id or self._next_stable_id("trade")
                decision_id = self._pending_decision_id or self._last_decision_id
                attempt_id = self._pending_attempt_id or self._last_attempt_id

                self.risk_manager.sync_position(
                    signed_position=signed_position,
                    entry_price=actual_entry_price,
                    transition_price=transition_price or actual_entry_price,
                    zone=zone_name,
                    regime=regime_state,
                    event_tags=event_tags,
                    strategy="WEIGHTED_SCORE_MATRIX",
                    current_time=event_time,
                    decision_id=decision_id,
                    attempt_id=attempt_id,
                    position_id=position_id,
                    trade_id=trade_id,
                )
                self._clear_unresolved_entry_state("position_transition_confirmed")

                if prior_position == 0 and signed_position != 0:
                    self._current_position_id = position_id
                    self._current_trade_id = trade_id
                    self._position_entry_time = pd.Timestamp(event_time)
                    self._position_high_water = actual_entry_price
                    self._position_low_water = actual_entry_price
                    self.executor.mark_position_open()
                    self.executor.ensure_protection(
                        symbol=self.symbol,
                        quantity=abs(signed_position),
                        direction=1 if signed_position > 0 else -1,
                        stop_price=self._stop_loss,
                        take_profit=self._take_profit,
                        decision_id=decision_id,
                        attempt_id=attempt_id,
                        position_id=position_id,
                        trade_id=trade_id,
                    )
                    self._last_entry_fill_price = actual_entry_price
                    if self._pending_expected_fill_price is not None and actual_entry_price:
                        self._last_fill_drift_ticks = round((actual_entry_price - self._pending_expected_fill_price) / 0.25, 4)
                    else:
                        self._last_fill_drift_ticks = None
                    if self._pending_entry_submitted_at is not None:
                        self._last_protection_attach_latency_seconds = max(
                            (event_time - self._pending_entry_submitted_at).total_seconds(),
                            0.0,
                        )
                    else:
                        self._last_protection_attach_latency_seconds = None
                    logger.info(
                        "execution_trace event=entry zone=%s regime=%s fill=%s drift_ticks=%s protection_attach_s=%s",
                        zone_name,
                        regime_state,
                        actual_entry_price,
                        self._last_fill_drift_ticks,
                        self._last_protection_attach_latency_seconds,
                    )
                    self._record_event(
                        category="execution",
                        event_type="position_opened",
                        payload={
                            "prior_position": prior_position,
                            "signed_position": signed_position,
                            "entry_price": actual_entry_price,
                            "transition_price": transition_price,
                            "zone": zone_name,
                            "regime_state": regime_state,
                            "event_tags": event_tags,
                            "fill_drift_ticks": self._last_fill_drift_ticks,
                            "protection_attach_latency_seconds": self._last_protection_attach_latency_seconds,
                            "decision_id": decision_id,
                            "attempt_id": attempt_id,
                            "position_id": position_id,
                            "trade_id": trade_id,
                        },
                        event_time=event_time,
                        action="entry_fill",
                        reason=self._pending_entry_reason or self._last_entry_reason,
                        zone=zone_name,
                    )
                    self._pending_expected_fill_price = None
                    self._pending_entry_submitted_at = None
                    self._pending_decision_id = None
                    self._pending_attempt_id = None
                    self._pending_position_id = None
                    self._pending_trade_id = None
                elif prior_position != 0 and signed_position == 0:
                    protective_reason = self.executor.pop_last_protective_fill_reason()
                    if protective_reason == "stop_loss":
                        self._last_exit_reason = "stop_loss"
                    elif protective_reason == "take_profit":
                        self._last_exit_reason = "take_profit"
                    if self._last_exit_reason:
                        set_state(last_exit_reason=self._last_exit_reason)
                    self.executor.clear_protection(self.symbol)
                    self.executor.mark_position_flat()
                    close_position_id = self._current_position_id or position_id
                    close_trade_id = self._current_trade_id or trade_id
                    logger.info(
                        "execution_trace event=exit reason=%s fill=%s zone=%s regime=%s",
                        self._last_exit_reason or "flat_sync",
                        transition_price or self._last_price,
                        zone_name,
                        regime_state,
                    )
                    self._record_event(
                        category="execution",
                        event_type="position_closed",
                        payload={
                            "prior_position": prior_position,
                            "signed_position": signed_position,
                            "exit_price": transition_price or self._last_price,
                            "zone": zone_name,
                            "regime_state": regime_state,
                            "event_tags": event_tags,
                            "decision_id": decision_id,
                            "attempt_id": attempt_id,
                            "position_id": close_position_id,
                            "trade_id": close_trade_id,
                        },
                        event_time=event_time,
                        action="exit_fill",
                        reason=self._last_exit_reason or "flat_sync",
                        zone=zone_name,
                    )
                    self._stop_loss = None
                    self._take_profit = None
                    self._pending_entry_zone = ""
                    self._pending_entry_reason = ""
                    self._position_entry_time = None
                    self._position_high_water = None
                    self._position_low_water = None
                    self._protection_mode = "static"
                    self._current_position_id = None
                    self._current_trade_id = None
                    self._pending_decision_id = None
                    self._pending_attempt_id = None
                    self._pending_position_id = None
                    self._pending_trade_id = None
                else:
                    self.executor.mark_position_open()
                    self.executor.ensure_protection(
                        symbol=self.symbol,
                        quantity=abs(signed_position),
                        direction=1 if signed_position > 0 else -1,
                        stop_price=self._stop_loss,
                        take_profit=self._take_profit,
                        position_id=self._current_position_id,
                        trade_id=self._current_trade_id,
                    )
                    if prior_position * signed_position < 0:
                        self._position_entry_time = pd.Timestamp(event_time)
                        self._last_entry_fill_price = actual_entry_price
                        self._position_high_water = actual_entry_price
                        self._position_low_water = actual_entry_price
                    self._record_event(
                        category="execution",
                        event_type="position_adjusted",
                        payload={
                            "prior_position": prior_position,
                            "signed_position": signed_position,
                            "entry_price": actual_entry_price,
                            "transition_price": transition_price,
                            "zone": zone_name,
                            "regime_state": regime_state,
                            "event_tags": event_tags,
                            "decision_id": decision_id,
                            "attempt_id": attempt_id,
                            "position_id": self._current_position_id or position_id,
                            "trade_id": self._current_trade_id or trade_id,
                        },
                        event_time=event_time,
                        action="position_adjustment",
                        reason="position_adjusted",
                        zone=zone_name,
                    )
                self._last_position = signed_position
                self._position_sync_requested = False
                self._last_reconciliation_at = event_time
                self._last_reconciliation_reason = "position_sync"
            self._position_sync_requested = False
            self._last_reconciliation_at = event_time
            self._last_reconciliation_reason = "position_sync"

    def _refresh_dynamic_exit(self) -> None:
        """Update stop to breakeven/profit-lock/trailing when price moved in our favor; idempotent protection refresh."""
        if self._mock_mode or self._last_position == 0 or self._stop_loss is None or self._last_entry_fill_price is None:
            return
        atr_value = self._calculate_current_atr()
        if atr_value is None or atr_value <= 0:
            return
        cfg = self.config.strategy
        be_atr = getattr(cfg, "breakeven_trigger_atr", 0.5) or 0.5
        pl_atr = getattr(cfg, "profit_lock_atr", 0.5) or 0.5
        trail_atr = getattr(cfg, "trailing_stop_atr", 1.0) or 1.0
        entry = self._last_entry_fill_price
        price = self._last_price or entry
        if self._last_position > 0:
            favorable = price - entry
            candidate = self._stop_loss
            if favorable >= be_atr * atr_value:
                candidate = max(candidate, entry)
            if favorable >= pl_atr * atr_value:
                candidate = max(candidate, entry + pl_atr * atr_value)
            hw = self._position_high_water or price
            candidate = max(candidate, hw - trail_atr * atr_value)
            if candidate > self._stop_loss and candidate <= price:
                self._stop_loss = candidate
                self._protection_mode = "breakeven" if candidate <= entry else "trailing"
                self.executor.ensure_protection(
                    symbol=self.symbol,
                    quantity=abs(self._last_position),
                    direction=1,
                    stop_price=self._stop_loss,
                    take_profit=self._take_profit,
                )
                self._record_event(
                    category="execution",
                    event_type="dynamic_exit_updated",
                    payload={"stop_price": candidate, "protection_mode": self._protection_mode, "high_water": hw},
                    event_time=self._current_event_time(),
                    action="refresh_protection",
                    reason="dynamic_exit",
                )
        else:
            favorable = entry - price
            candidate = self._stop_loss
            if favorable >= be_atr * atr_value:
                candidate = min(candidate, entry)
            if favorable >= pl_atr * atr_value:
                candidate = min(candidate, entry - pl_atr * atr_value)
            lw = self._position_low_water or price
            candidate = min(candidate, lw + trail_atr * atr_value)
            if candidate < self._stop_loss and candidate >= price:
                self._stop_loss = candidate
                self._protection_mode = "breakeven" if candidate >= entry else "trailing"
                self.executor.ensure_protection(
                    symbol=self.symbol,
                    quantity=abs(self._last_position),
                    direction=-1,
                    stop_price=self._stop_loss,
                    take_profit=self._take_profit,
                )
                self._record_event(
                    category="execution",
                    event_type="dynamic_exit_updated",
                    payload={"stop_price": candidate, "protection_mode": self._protection_mode, "low_water": lw},
                    event_time=self._current_event_time(),
                    action="refresh_protection",
                    reason="dynamic_exit",
                )

    def _handle_watchdogs(self) -> None:
        """Enforce feed/protection watchdogs and fail-safe lockout."""
        if not self.config.watchdog.enabled:
            return
        now = self._current_event_time()
        feed_stale = False
        if self._watchdog_state.last_feed_time is not None:
            feed_stale = (now - self._watchdog_state.last_feed_time).total_seconds() >= self.config.watchdog.feed_stale_seconds

        last_ack = self.executor.get_last_ack_time()
        broker_ack_stale = False
        if last_ack is not None and self._last_position != 0 and not self.executor.is_protected(self.symbol):
            broker_ack_stale = (now - last_ack).total_seconds() >= self.config.watchdog.broker_ack_stale_seconds

        protection_timeout = self.executor.protection_pending_too_long(self.symbol, now, self.config.watchdog.protection_ack_seconds)

        self._watchdog_state.feed_stale = feed_stale
        self._watchdog_state.broker_ack_stale = broker_ack_stale
        self._watchdog_state.protection_timeout = protection_timeout

        if feed_stale:
            self._record_event(
                category="risk",
                event_type="watchdog_triggered",
                payload={"feed_stale": feed_stale, "broker_ack_stale": broker_ack_stale, "protection_timeout": protection_timeout},
                event_time=now,
                action="watchdog",
                reason="feed_stale",
            )
            self._trigger_fail_safe("feed_stale")
        elif protection_timeout:
            self._record_event(
                category="risk",
                event_type="watchdog_triggered",
                payload={"feed_stale": feed_stale, "broker_ack_stale": broker_ack_stale, "protection_timeout": protection_timeout},
                event_time=now,
                action="watchdog",
                reason="protection_ack_timeout",
            )
            self._trigger_fail_safe("protection_ack_timeout")
        elif broker_ack_stale:
            self._record_event(
                category="risk",
                event_type="watchdog_triggered",
                payload={"feed_stale": feed_stale, "broker_ack_stale": broker_ack_stale, "protection_timeout": protection_timeout},
                event_time=now,
                action="watchdog",
                reason="broker_ack_stale",
            )
            self._trigger_fail_safe("broker_ack_stale")

    def _trigger_fail_safe(self, reason: str) -> None:
        if self._watchdog_state.fail_safe_lockout:
            return
        self._watchdog_state.fail_safe_lockout = True
        self._fail_safe_count += 1
        if reason in {"protection_ack_timeout", "broker_ack_stale"}:
            self._protection_failure_count += 1
        logger.error("Fail-safe lockout activated: %s", reason)
        self._record_event(
            category="risk",
            event_type="fail_safe_activated",
            payload={"reason": reason, "position": self._last_position, "protection_failures": self._protection_failure_count},
            event_time=self._current_event_time(),
            action="fail_safe",
            reason=reason,
        )
        if self._last_position != 0:
            self._flatten_position(reason)

    def _calculate_current_atr(self) -> Optional[float]:
        if len(self._bars) < self.config.strategy.atr_length:
            return None
        atr_series = atr(self._bars["high"], self._bars["low"], self._bars["close"], self.config.strategy.atr_length)
        atr_value = atr_series.iloc[-1]
        if pd.isna(atr_value):
            return None
        return float(atr_value)

    def _update_server_state(self) -> None:
        metrics = self.risk_manager.get_metrics()
        state = get_state()
        uptime = time.time() - state.start_time if state.start_time else 0
        execution_snapshot = self.executor.get_watchdog_snapshot(self.symbol)
        execution_snapshot.update(
            {
                "expected_fill_price": self._pending_expected_fill_price,
                "entry_fill_price": self._last_entry_fill_price,
                "fill_drift_ticks": self._last_fill_drift_ticks,
                "protection_attach_latency_seconds": self._last_protection_attach_latency_seconds,
                "protection_failures": self._protection_failure_count,
                "fail_safe_count": self._fail_safe_count,
                "entry_guard": dict(self._last_entry_guard_snapshot),
                "last_entry_block_reason": self._last_entry_block_reason,
                "decision_price": self._last_decision_price,
                "unresolved_entry": self._unresolved_entry_snapshot(),
                "active_zone": self._active_zone_name,
                "active_regime": self._last_decision.feature_snapshot.regime_state if self._last_decision else None,
                "adoption_source": self._adoption_source,
                "adopted_broker_position_at_startup": self._adopted_broker_position_at_startup,
                "adopted_broker_orders_at_startup": self._adopted_broker_orders_at_startup,
                "last_reconciliation_at": self._last_reconciliation_at.isoformat() if self._last_reconciliation_at else None,
                "last_reconciliation_reason": self._last_reconciliation_reason,
                "desync_flags": {
                    "unresolved_entry_pending": self._unresolved_entry_submission_count > 0,
                    "position_sync_requested": self._position_sync_requested,
                },
                "protection_mode": self._protection_mode,
                "current_stop": self._stop_loss,
                "current_target": self._take_profit,
                "high_water": self._position_high_water,
                "low_water": self._position_low_water,
            }
        )
        if self._last_sizing_telemetry:
            execution_snapshot["sizing_telemetry"] = dict(self._last_sizing_telemetry)
        account = self.client._account
        set_state(
            position=self._last_position,
            position_pnl=metrics.current_position_pnl,
            daily_pnl=metrics.daily_pnl,
            data_mode="replay" if (self.client.is_mock_mode() or self._mock_mode) else "live",
            account_balance=account.balance if account else 0,
            account_equity=account.equity if account else 0,
            account_available=account.available if account else 0,
            account_margin_used=account.margin_used if account else 0,
            account_open_pnl=account.open_pnl if account else 0,
            account_realized_pnl=account.realized_pnl if account else 0,
            account_id=account.account_id if account else None,
            account_name=account.name if account else None,
            account_is_practice=account.is_practice if account else None,
            risk_state=metrics.risk_state.value,
            trades_today=metrics.trades_today,
            trades_this_hour=metrics.trades_this_hour,
            trades_this_zone=metrics.trades_this_zone,
            max_daily_loss=metrics.max_daily_loss,
            consecutive_losses=metrics.consecutive_losses,
            uptime_seconds=uptime,
            last_entry_reason=self._last_entry_reason,
            last_exit_reason=self._last_exit_reason,
            execution=execution_snapshot,
            heartbeat={
                "feed": self._watchdog_state.last_feed_time.isoformat() if self._watchdog_state.last_feed_time else None,
                "broker_ack": self.executor.get_last_ack_time().isoformat() if self.executor.get_last_ack_time() else None,
                "fail_safe_lockout": self._watchdog_state.fail_safe_lockout,
                "feed_stale": self._watchdog_state.feed_stale,
                "broker_ack_stale": self._watchdog_state.broker_ack_stale,
                "protection_timeout": self._watchdog_state.protection_timeout,
                "market_stream_connected": self.client._connected,
                "market_stream_error": self.client.get_last_stream_error(),
            },
        )
        self.observability.record_state_snapshot(get_state().to_dict(), event_time=self._current_event_time())

    def _current_event_time(self) -> datetime:
        if self._latest_market_data is not None and self._latest_market_data.timestamp is not None:
            timestamp = self._latest_market_data.timestamp
            return timestamp if timestamp.tzinfo else timestamp.replace(tzinfo=UTC)
        if not self._bars.empty:
            current_time = self._bars.index[-1]
            return current_time.to_pydatetime() if hasattr(current_time, "to_pydatetime") else current_time
        return datetime.now(UTC)


_trading_engine: Optional[TradingEngine] = None


def get_trading_engine(force_recreate: bool = False) -> TradingEngine:
    """Return the global trading engine instance."""
    global _trading_engine
    if force_recreate:
        _trading_engine = None
    if _trading_engine is None:
        _trading_engine = TradingEngine()
    return _trading_engine

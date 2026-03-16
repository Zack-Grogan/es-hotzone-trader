"""Trading engine orchestration."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
import logging
from pathlib import Path
import threading
import time
from typing import Optional

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
        self._watchdog_state = WatchdogState()
        self.observability = get_observability_store()

    @property
    def running(self) -> bool:
        return self._running

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
    ) -> None:
        dominant_side = "long" if decision.long_score >= decision.short_score else "short"
        dominant_score = decision.long_score if dominant_side == "long" else decision.short_score
        opposing_score = decision.short_score if dominant_side == "long" else decision.long_score
        event_time = current_time.to_pydatetime() if hasattr(current_time, "to_pydatetime") else current_time
        payload = {
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
            "outcome": outcome,
            "outcome_reason": outcome_reason,
            "contracts": contracts,
            "order_type": order_type,
            "limit_price": limit_price,
        }
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
            self._watchdog_state = WatchdogState()
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
                },
                heartbeat={"feed": None, "broker_ack": None, "fail_safe_lockout": False},
                event_context={"active_tags": [], "reason": "", "post_event_cooling": False},
                replay_summary=None,
            )

    def enable_mock_mode(self) -> None:
        """Put the engine and its dependencies into replay/mock mode."""
        with self._lock:
            self._mock_mode = True
            self.client.enable_mock_mode()
            self.executor.enable_mock_mode()

    def start(self, mock: bool = False) -> None:
        """Start the engine."""
        if self._running:
            return
        self._mock_mode = mock
        self._running = True
        self._record_event(
            category="system",
            event_type="engine_starting",
            payload={"mock_mode": mock, "symbol": self.symbol},
            event_time=self._current_event_time(),
            action="start",
            reason="engine_start",
        )
        set_state(
            status="running",
            running=True,
            data_mode="mock" if mock else "live",
            replay_summary=None,
            current_strategy="WEIGHTED_SCORE_MATRIX",
            matrix_version=self.config.alpha.matrix_version,
        )

        if mock:
            self.client.enable_mock_mode()
            self.executor.enable_mock_mode()
            account = self.client._account
            if account is not None:
                set_state(
                    account_balance=account.balance,
                    account_id=account.account_id,
                    account_name=account.name,
                    account_is_practice=account.is_practice,
                )
        else:
            if not self.client._access_token and not self.client.authenticate():
                self._running = False
                self._record_event(
                    category="system",
                    event_type="engine_start_failed",
                    payload={"mock_mode": mock, "error": "TopstepX authentication failed"},
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
                    payload={"mock_mode": mock, "error": "No tradable Topstep account available"},
                    event_time=self._current_event_time(),
                    action="start",
                    reason="account_unavailable",
                )
                raise RuntimeError("No tradable Topstep account available")
            set_state(
                account_balance=account.balance,
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

        self._worker = threading.Thread(target=self._run, daemon=True)
        self._worker.start()
        self._record_event(
            category="system",
            event_type="engine_started",
            payload={"mock_mode": mock, "symbol": self.symbol},
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
            payload={"mock_mode": self._mock_mode, "position": self._last_position},
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
            payload={"mock_mode": self._mock_mode, "position": self._last_position},
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

    def _run(self) -> None:
        """Background state reconciliation loop."""
        while self._running:
            try:
                self.executor.reconcile_pending_orders()
                if self._position_sync_requested or self._last_position != 0:
                    self._sync_position_state()
                self._handle_watchdogs()
                with self._lock:
                    self._update_server_state()
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

    def _evaluate_current_state(self, allow_entries: bool) -> None:
        if self._bars.empty:
            return

        current_time = self._bars.index[-1]
        current_price = float(self._last_price or self._bars["close"].iloc[-1])
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
        order = self.executor.place_order(
            symbol=self.symbol,
            quantity=contracts,
            side=decision.side,
            order_type=order_type,
            limit_price=limit_price,
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
            )
            return

        self._stop_loss = decision.stop_loss
        self._take_profit = decision.take_profit
        self._pending_entry_zone = decision.zone_name
        self._pending_entry_reason = decision.reason
        self._pending_expected_fill_price = limit_price if limit_price is not None else current_price
        self._pending_entry_submitted_at = self._current_event_time()
        self._last_entry_reason = decision.reason
        set_state(
            last_signal={
                "direction": decision.action,
                "price": current_price,
                "reason": decision.reason,
                "strategy": "WEIGHTED_SCORE_MATRIX",
                "zone": decision.zone_name,
                "regime": decision.feature_snapshot.regime_state,
                "expected_fill_price": self._pending_expected_fill_price,
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
        base_contracts = self.risk_manager.calculate_position_size(decision.feature_snapshot.atr_value, direction)
        if base_contracts <= 0:
            return 0
        contracts = base_contracts if decision.size_fraction >= 1.0 else max(1, int(round(base_contracts * decision.size_fraction)))
        if self.risk_manager.is_reduced_risk():
            contracts = min(contracts, 1)
        return max(1, min(contracts, self.config.account.max_contracts))

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
        else:
            position = self.client.get_position(self.symbol)
            signed_position = position.quantity
            entry_price = position.entry_price

        event_time = self._current_event_time()
        with self._lock:
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

                self.risk_manager.sync_position(
                    signed_position=signed_position,
                    entry_price=actual_entry_price,
                    transition_price=transition_price or actual_entry_price,
                    zone=zone_name,
                    regime=regime_state,
                    event_tags=event_tags,
                    strategy="WEIGHTED_SCORE_MATRIX",
                    current_time=event_time,
                )

                if prior_position == 0 and signed_position != 0:
                    self._position_entry_time = pd.Timestamp(event_time)
                    self.executor.mark_position_open()
                    self.executor.ensure_protection(
                        symbol=self.symbol,
                        quantity=abs(signed_position),
                        direction=1 if signed_position > 0 else -1,
                        stop_price=self._stop_loss,
                        take_profit=self._take_profit,
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
                        },
                        event_time=event_time,
                        action="entry_fill",
                        reason=self._pending_entry_reason or self._last_entry_reason,
                        zone=zone_name,
                    )
                    self._pending_expected_fill_price = None
                    self._pending_entry_submitted_at = None
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
                else:
                    self.executor.mark_position_open()
                    self.executor.ensure_protection(
                        symbol=self.symbol,
                        quantity=abs(signed_position),
                        direction=1 if signed_position > 0 else -1,
                        stop_price=self._stop_loss,
                        take_profit=self._take_profit,
                    )
                    if prior_position * signed_position < 0:
                        self._position_entry_time = pd.Timestamp(event_time)
                        self._last_entry_fill_price = actual_entry_price
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
                        },
                        event_time=event_time,
                        action="position_adjustment",
                        reason="position_adjusted",
                        zone=zone_name,
                    )
                self._last_position = signed_position
                self._position_sync_requested = False

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
                "active_zone": self._active_zone_name,
                "active_regime": self._last_decision.feature_snapshot.regime_state if self._last_decision else None,
            }
        )
        set_state(
            position=self._last_position,
            position_pnl=metrics.current_position_pnl,
            daily_pnl=metrics.daily_pnl,
            data_mode="mock" if self.client.is_mock_mode() or self._mock_mode else "live",
            account_id=self.client._account.account_id if self.client._account else None,
            account_name=self.client._account.name if self.client._account else None,
            account_is_practice=self.client._account.is_practice if self.client._account else None,
            risk_state=metrics.risk_state.value,
            trades_today=metrics.trades_today,
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

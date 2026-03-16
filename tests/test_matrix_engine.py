"""Matrix engine and replay tests."""
from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from src.cli.commands import _configure_logging, _log_startup_summary
from src.config import BlackoutConfig, Config, EventProviderConfig, HotZoneConfig, set_config
from src.engine import DecisionMatrixEvaluator, FeatureSnapshot, HotZoneScheduler, MatrixDecision, ReplayRunner, TradingEngine, ZoneInfo, ZoneState, get_risk_manager, get_scheduler
from src.engine.event_provider import EventContext, LocalEventProvider
from src.engine.market_context import OrderFlowSnapshot
from src.engine.risk_manager import RiskManager, RiskState, TradeRecord
from src.execution.executor import OrderExecutor, get_executor
from src.indicators.rsi import rsi
from src.market import Account, MarketData, TopstepClient, get_client
from src.observability import get_observability_store
from src.server.mcp_server import handle_mcp_request
from src.server.debug_server import DebugServer, TradingState
from src.server import get_state


def build_config() -> Config:
    """Create a deterministic config for matrix tests."""
    config = Config(
        hot_zones=[
            HotZoneConfig(name="Pre-Open", start="06:30", end="08:30", timezone="America/Chicago"),
            HotZoneConfig(name="Post-Open", start="09:00", end="11:00", timezone="America/Chicago"),
            HotZoneConfig(name="Midday", start="12:00", end="13:00", timezone="America/Chicago"),
            HotZoneConfig(name="Close-Scalp", start="12:45", end="13:00", timezone="America/Chicago", mode="flatten_only"),
        ]
    )
    config.alpha.min_entry_score = 1.25
    config.alpha.full_size_score = 3.5
    config.alpha.min_score_gap = 0.25
    config.alpha.flat_bias_buffer = 0.0
    config.alpha.zone_vetoes["Midday"]["max_atr_percentile"] = 1.0
    config.regime.stress_quote_rate = 0.0
    config.regime.trend_slope_threshold = 0.05
    config.regime.trend_ofi_threshold = 0.2
    config.replay.segment_size = 2
    config.validation.walk_forward_train_bars = 2
    config.validation.walk_forward_test_bars = 2
    config.validation.synthetic_quote_policy = "reject"
    config.event_provider.calendar_path = "config/does-not-exist.yaml"
    config.event_provider.emergency_halt_path = "config/does-not-exist.flag"
    config.observability.enabled = False
    return config


def zone(
    name: str,
    ts: pd.Timestamp,
    minutes_remaining: float = 30.0,
    state: ZoneState = ZoneState.ACTIVE,
    start_time: pd.Timestamp | None = None,
) -> ZoneInfo:
    """Build a zone info object for tests."""
    return ZoneInfo(
        name=name,
        state=state,
        start_time=(start_time or ts).floor("min"),
        end_time=ts.floor("min") + pd.Timedelta(minutes=minutes_remaining),
        minutes_remaining=minutes_remaining,
        is_first_bar=False,
        is_last_bar=False,
    )


def bars_from_prices(start: str, prices: list[float], timezone: str = "America/Chicago") -> pd.DataFrame:
    """Build simple OHLCV bars from close prices."""
    index = pd.date_range(start, periods=len(prices), freq="min", tz=timezone)
    rows = []
    prior = prices[0]
    for price in prices:
        rows.append(
            {
                "open": prior,
                "high": max(prior, price) + 0.4,
                "low": min(prior, price) - 0.4,
                "close": price,
                "volume": 100,
            }
        )
        prior = price
    return pd.DataFrame(rows, index=index)


class DecisionMatrixTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = build_config()
        set_config(self.config)
        self.evaluator = DecisionMatrixEvaluator(self.config)
        self.healthy_flow = OrderFlowSnapshot(
            ofi=12.0,
            ofi_zscore=1.2,
            quote_rate_per_minute=30.0,
            quote_rate_state=0.8,
            spread_regime=1.4,
            volume_pace=1.0,
        )

    def test_session_profile_features_are_populated(self) -> None:
        prices = [100.0 + (i * 0.2) for i in range(30)]
        bars = bars_from_prices("2026-03-13 09:00", prices)
        current_zone = zone("Post-Open", bars.index[-1], minutes_remaining=50, start_time=bars.index[0])

        snapshot = self.evaluator.extract_features(
            bars,
            current_zone,
            None,
            EventContext(),
            self.healthy_flow,
            current_position=0,
        )

        self.assertEqual(snapshot.active_session, "RTH")
        self.assertIn("rth_vwap", snapshot.diagnostics)
        self.assertIn("eth_vwap", snapshot.diagnostics)
        self.assertGreater(snapshot.diagnostics["vah"], 0)
        self.assertGreater(snapshot.diagnostics["poc"], 0)
        self.assertIn("poc_distance", snapshot.long_features)

    def test_held_minutes_handles_mixed_timezone_inputs(self) -> None:
        minutes = self.evaluator._held_minutes(
            pd.Timestamp("2026-03-13 09:30", tz="America/Chicago"),
            pd.Timestamp("2026-03-13 10:00"),
        )

        self.assertEqual(minutes, 30.0)

    def test_pre_open_breakout_scores_long_with_flow_support(self) -> None:
        prices = [100.0 + (i * 0.18) for i in range(15)] + [103.0, 103.5, 104.0, 104.5, 104.9]
        bars = bars_from_prices("2026-03-13 06:30", prices)
        current_zone = zone("Pre-Open", bars.index[-1], minutes_remaining=40, start_time=bars.index[0])
        market_data = MarketData(symbol="ES", bid=104.75, ask=105.0, last=104.9, volume=2000, timestamp=bars.index[-1].tz_convert("UTC").to_pydatetime())

        decision = self.evaluator.evaluate(
            bars,
            current_zone,
            market_data,
            RiskState.NORMAL,
            False,
            0,
            True,
            event_context=EventContext(),
            flow_snapshot=self.healthy_flow,
        )

        self.assertEqual(decision.action, "LONG")
        self.assertGreater(decision.long_score, decision.short_score)

    def test_midday_range_setup_can_score_long(self) -> None:
        prices = [100.0] * 12 + [99.8, 99.7, 99.6, 99.2, 98.8, 98.5, 98.6, 98.9]
        bars = bars_from_prices("2026-03-13 12:00", prices)
        bars.iloc[-1, bars.columns.get_loc("low")] = 97.6
        bars.iloc[-1, bars.columns.get_loc("close")] = 98.4
        bars.iloc[-1, bars.columns.get_loc("high")] = 98.9
        current_zone = zone("Midday", bars.index[-1], minutes_remaining=20, start_time=bars.index[0])
        market_data = MarketData(symbol="ES", bid=98.2, ask=98.45, last=98.4, volume=2500, timestamp=bars.index[-1].tz_convert("UTC").to_pydatetime())
        range_flow = OrderFlowSnapshot(
            ofi=0.0,
            ofi_zscore=0.1,
            quote_rate_per_minute=20.0,
            quote_rate_state=0.1,
            spread_regime=1.2,
            volume_pace=0.2,
        )

        decision = self.evaluator.evaluate(
            bars,
            current_zone,
            market_data,
            RiskState.NORMAL,
            False,
            0,
            True,
            event_context=EventContext(),
            flow_snapshot=range_flow,
        )

        self.assertEqual(decision.action, "LONG")
        self.assertTrue(decision.feature_snapshot.mean_reversion_ready_long)
        self.assertEqual(decision.feature_snapshot.regime_state, "RANGE")

    def test_regime_classifier_marks_trend(self) -> None:
        prices = [100.0 + (i * 0.5) for i in range(20)]
        bars = bars_from_prices("2026-03-13 12:00", prices)
        current_zone = zone("Midday", bars.index[-1], minutes_remaining=20, start_time=bars.index[0])
        trend_flow = OrderFlowSnapshot(
            ofi=12.0,
            ofi_zscore=1.3,
            quote_rate_per_minute=35.0,
            quote_rate_state=1.1,
            spread_regime=1.6,
            volume_pace=1.0,
        )

        snapshot = self.evaluator.extract_features(
            bars,
            current_zone,
            None,
            EventContext(),
            trend_flow,
            current_position=0,
        )

        self.assertEqual(snapshot.regime_state, "TREND")

    def test_close_scalp_is_flatten_only(self) -> None:
        prices = [100.0 + (i * 0.1) for i in range(20)]
        bars = bars_from_prices("2026-03-13 12:45", prices)
        current_zone = zone("Close-Scalp", bars.index[-1], minutes_remaining=8, state=ZoneState.FLATTEN_ONLY, start_time=bars.index[0])
        market_data = MarketData(symbol="ES", bid=101.85, ask=102.1, last=102.0, volume=1500, timestamp=bars.index[-1].tz_convert("UTC").to_pydatetime())

        flat_decision = self.evaluator.evaluate(bars, current_zone, market_data, RiskState.NORMAL, False, 0, True, event_context=EventContext(), flow_snapshot=self.healthy_flow)
        held_decision = self.evaluator.evaluate(bars, current_zone, market_data, RiskState.NORMAL, False, 1, True, event_context=EventContext(), flow_snapshot=self.healthy_flow)

        self.assertEqual(flat_decision.action, "NO_TRADE")
        self.assertEqual(held_decision.action, "FLAT")

    def test_outside_hotzones_can_be_enabled_with_dedicated_profile(self) -> None:
        config = build_config()
        config.strategy.trade_outside_hotzones = True
        set_config(config)
        evaluator = DecisionMatrixEvaluator(config)
        prices = [100.0 + (i * 0.2) for i in range(20)]
        bars = bars_from_prices("2026-03-13 14:10", prices)
        market_data = MarketData(symbol="ES", bid=103.75, ask=104.0, last=103.9, volume=2000, timestamp=bars.index[-1].tz_convert("UTC").to_pydatetime())

        decision = evaluator.evaluate(
            bars,
            None,
            market_data,
            RiskState.NORMAL,
            False,
            0,
            True,
            event_context=EventContext(),
            flow_snapshot=self.healthy_flow,
        )

        self.assertEqual(decision.zone_name, "Outside")
        self.assertNotEqual(decision.reason, "outside_zone")
        self.assertIn(decision.action, {"LONG", "SHORT", "NO_TRADE"})

    def test_outside_hotzones_disabled_keeps_outside_zone_veto(self) -> None:
        prices = [100.0 + (i * 0.2) for i in range(20)]
        bars = bars_from_prices("2026-03-13 14:10", prices)

        decision = self.evaluator.evaluate(
            bars,
            None,
            None,
            RiskState.NORMAL,
            False,
            0,
            True,
            event_context=EventContext(),
            flow_snapshot=self.healthy_flow,
        )

        self.assertEqual(decision.zone_name, "Outside")
        self.assertEqual(decision.reason, "outside_zone")
        self.assertIn("outside_zone", decision.active_vetoes)

    def test_zone_spread_veto_is_enforced(self) -> None:
        config = build_config()
        config.order_execution.max_slippage_ticks = 10
        set_config(config)
        evaluator = DecisionMatrixEvaluator(config)
        prices = [100.0 + (i * 0.2) for i in range(20)]
        bars = bars_from_prices("2026-03-13 09:00", prices)
        current_zone = zone("Post-Open", bars.index[-1], minutes_remaining=35, start_time=bars.index[0])
        wide_market = MarketData(symbol="ES", bid=103.0, ask=104.5, last=104.0, volume=2000, timestamp=bars.index[-1].tz_convert("UTC").to_pydatetime())

        decision = evaluator.evaluate(
            bars,
            current_zone,
            wide_market,
            RiskState.NORMAL,
            False,
            0,
            True,
            event_context=EventContext(),
            flow_snapshot=self.healthy_flow,
        )

        self.assertIn("spread_too_wide", decision.active_vetoes)

    def test_time_stop_exit_path(self) -> None:
        prices = [100.0 + (i * 0.15) for i in range(20)]
        bars = bars_from_prices("2026-03-13 09:00", prices)
        current_zone = zone("Post-Open", bars.index[-1], minutes_remaining=40, start_time=bars.index[0])
        decision = self.evaluator.evaluate(
            bars,
            current_zone,
            None,
            RiskState.NORMAL,
            False,
            1,
            True,
            current_entry_time=bars.index[-1] - pd.Timedelta(minutes=120),
            event_context=EventContext(),
            flow_snapshot=self.healthy_flow,
        )

        self.assertEqual(decision.action, "FLAT")
        self.assertEqual(decision.reason, "time_stop")

    def test_feature_pruning_validation_rejects_overweight_zone(self) -> None:
        config = build_config()
        config.alpha.zone_weights["Pre-Open"]["long"]["extra_feature_1"] = 0.1
        config.alpha.zone_weights["Pre-Open"]["long"]["extra_feature_2"] = 0.1
        set_config(config)

        with self.assertRaises(ValueError):
            DecisionMatrixEvaluator(config)


class IndicatorRegressionTests(unittest.TestCase):
    def test_rsi_exact_period_length_does_not_raise(self) -> None:
        close = pd.Series([float(100 + i) for i in range(14)])

        values = rsi(close, period=14)

        self.assertEqual(len(values), len(close))
        self.assertTrue(values.isna().all())


class StateReportingTests(unittest.TestCase):
    def test_health_reports_degraded_when_live_feed_is_disconnected(self) -> None:
        state = TradingState()
        state.status = "running"
        state.running = True
        state.data_mode = "live"
        state.heartbeat = {"market_stream_connected": False, "market_stream_error": "socket dropped"}

        health = state.to_health_dict()

        self.assertEqual(health["status"], "degraded")
        self.assertFalse(health["market_stream_connected"])

    def test_health_reports_mock_mode_as_mock_not_healthy(self) -> None:
        state = TradingState()
        state.status = "running"
        state.running = True
        state.data_mode = "mock"
        state.account_is_practice = True

        health = state.to_health_dict()

        self.assertEqual(health["status"], "mock")
        self.assertEqual(health["data_mode"], "mock")
        self.assertTrue(health["practice_account"])


class ExecutionLifecycleTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = build_config()
        set_config(self.config)
        get_client(force_recreate=True)
        self.engine = TradingEngine(self.config)
        self.engine.reset_runtime_state()

    def test_executor_counts_only_active_orders_in_watchdog_snapshot(self) -> None:
        executor = self.engine.executor
        executor.enable_mock_mode()
        active = executor._place_mock_order("ES", 1, "buy", "limit", 100.0, None, False, "entry")
        inactive = executor._place_mock_order("ES", 1, "buy", "limit", 99.0, None, False, "entry")
        inactive.status = type(inactive.status).CANCELLED

        snapshot = executor.get_watchdog_snapshot("ES")

        self.assertEqual(snapshot["pending_orders"], 1)
        self.assertEqual(snapshot["tracked_orders"], 2)
        self.assertEqual(snapshot["active_entry_orders"], 1)
        self.assertIn(active.order_id, executor.get_active_orders(symbol="ES", is_protective=False))

    def test_engine_skips_new_entry_when_active_entry_order_exists(self) -> None:
        executor = self.engine.executor
        executor.enable_mock_mode()
        self.engine._bars = bars_from_prices("2026-03-13 09:00", [100.0 + (i * 0.2) for i in range(30)])
        self.engine._last_price = float(self.engine._bars["close"].iloc[-1])
        self.engine._latest_market_data = MarketData(
            symbol="ES",
            bid=self.engine._last_price - 0.25,
            ask=self.engine._last_price + 0.25,
            last=self.engine._last_price,
            volume=1000,
            timestamp=self.engine._bars.index[-1].tz_convert("UTC").to_pydatetime(),
        )
        self.engine._latest_flow_snapshot = OrderFlowSnapshot(
            ofi=12.0,
            ofi_zscore=1.2,
            quote_rate_per_minute=30.0,
            quote_rate_state=0.8,
            spread_regime=1.4,
            volume_pace=1.0,
        )
        executor._place_mock_order("ES", 1, "buy", "limit", self.engine._last_price, None, False, "entry")

        with patch.object(self.engine.scheduler, "get_current_zone", return_value=zone("Post-Open", self.engine._bars.index[-1], start_time=self.engine._bars.index[0])):
            with patch.object(self.engine.risk_manager, "can_trade", return_value=(True, "")):
                with patch.object(executor, "place_order", wraps=executor.place_order) as place_order:
                    self.engine._evaluate_current_state(allow_entries=True)

        place_order.assert_not_called()

    def test_cancelled_order_update_no_longer_counts_as_pending(self) -> None:
        executor = self.engine.executor
        executor.enable_mock_mode()
        order = executor._place_mock_order("ES", 1, "buy", "limit", 100.0, None, False, "entry")

        executor.update_order_status(order.order_id, type(order.status).CANCELLED)

        snapshot = executor.get_watchdog_snapshot("ES")
        self.assertEqual(snapshot["pending_orders"], 0)
        self.assertEqual(snapshot["active_entry_orders"], 0)

    def test_live_limit_order_can_fall_back_to_market(self) -> None:
        executor = self.engine.executor
        executor.reset_state(mock_mode=False)

        with patch.object(executor.client, "place_order", side_effect=[None, "fallback-market-id"]) as place_order:
            order = executor.place_order("ES", 1, "buy", "limit", 100.0)

        assert order is not None
        self.assertEqual(order.order_type, "market")
        self.assertIsNone(order.limit_price)
        self.assertEqual(place_order.call_count, 2)
        self.assertEqual(place_order.call_args_list[0].kwargs["order_type"], "limit")
        self.assertEqual(place_order.call_args_list[1].kwargs["order_type"], "market")

    def test_reconcile_pending_orders_cancels_stale_entry(self) -> None:
        executor = self.engine.executor
        executor.enable_mock_mode()
        order = executor._place_mock_order("ES", 1, "buy", "limit", 100.0, None, False, "entry")
        order.created_time = order.created_time - timedelta(seconds=self.config.watchdog.stale_order_seconds + 5)

        cancelled = executor.reconcile_pending_orders()

        self.assertEqual(cancelled, 1)
        self.assertFalse(executor.has_active_entry_order("ES"))
        self.assertNotIn(order.order_id, executor.get_orders())

    def test_engine_on_order_update_clears_cancelled_entry(self) -> None:
        executor = self.engine.executor
        executor.enable_mock_mode()
        order = executor._place_mock_order("ES", 1, "buy", "limit", 100.0, None, False, "entry")

        self.engine.on_order_update({"orderId": order.order_id, "status": "cancelled"})

        self.assertFalse(executor.has_active_entry_order("ES"))
        self.assertTrue(self.engine._position_sync_requested)

    def test_executor_filled_order_callback_records_fill_for_sync_consumers(self) -> None:
        executor = self.engine.executor
        executor.enable_mock_mode()
        order = executor._place_mock_order("ES", 2, "buy", "limit", 100.0, None, False, "entry")

        executor.update_order_status(
            order.order_id,
            type(order.status).FILLED,
            {"filled_quantity": 2, "filled_price": 100.25},
        )

        fills = executor.consume_fills("ES")

        self.assertEqual(len(fills), 1)
        self.assertEqual(fills[0]["quantity"], 2)
        self.assertEqual(fills[0]["filled_price"], 100.25)
        self.assertEqual(executor.get_lifecycle_state(), "FILLED")


class ReplayRunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = build_config()
        set_config(self.config)
        get_client(force_recreate=True)
        self.engine = TradingEngine(self.config)

    def test_live_and_replay_share_engine_handler(self) -> None:
        with patch.object(self.engine, "_evaluate_current_state", wraps=self.engine._evaluate_current_state) as wrapped:
            self.engine.reset_runtime_state()
            self.engine._mock_mode = True
            self.engine.client.enable_mock_mode()
            self.engine.executor.enable_mock_mode()
            first = MarketData(symbol="ES", bid=100.0, ask=100.25, last=100.1, volume=100, timestamp=pd.Timestamp("2026-03-13 14:30:01Z").to_pydatetime())
            second = MarketData(symbol="ES", bid=100.6, ask=100.85, last=100.7, volume=180, timestamp=pd.Timestamp("2026-03-13 14:31:01Z").to_pydatetime())
            self.engine.on_market_data(first)
            self.engine.on_market_data(second)
            self.assertGreaterEqual(wrapped.call_count, 1)

        with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as handle:
            handle.write("timestamp,bid,ask,last,volume,bid_size,ask_size,last_size,trade_side,latency_ms\n")
            handle.write("2026-03-13T14:30:01Z,100.0,100.25,100.1,100,5,5,2,buy,25\n")
            handle.write("2026-03-13T14:31:01Z,100.6,100.85,100.7,180,5,5,2,buy,25\n")
            handle.flush()
            replay_path = handle.name

        runner = ReplayRunner(config=self.config, engine=self.engine)
        with patch.object(self.engine, "_evaluate_current_state", wraps=self.engine._evaluate_current_state) as wrapped:
            result = runner.run(replay_path)
            self.assertGreaterEqual(wrapped.call_count, 1)
            self.assertEqual(result.events, 2)

    def test_replay_runner_uses_engine_mock_mode_api(self) -> None:
        with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as handle:
            handle.write("timestamp,bid,ask,last,volume\n")
            handle.write("2026-03-13T14:30:01Z,100.0,100.25,100.1,100\n")
            handle.flush()
            replay_path = handle.name

        runner = ReplayRunner(config=self.config, engine=self.engine)
        with patch.object(self.engine, "enable_mock_mode", wraps=self.engine.enable_mock_mode) as enable_mock:
            runner.run(replay_path)

        enable_mock.assert_called_once()

    def test_replay_produces_segment_summary_and_dsr(self) -> None:
        with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as handle:
            handle.write("timestamp,bid,ask,last,volume,bid_size,ask_size,last_size,trade_side\n")
            handle.write("2026-03-13T14:30:01Z,100.0,100.25,100.1,100,5,5,2,buy\n")
            handle.write("2026-03-13T14:31:01Z,100.6,100.85,100.7,180,5,5,2,buy\n")
            handle.write("2026-03-13T14:32:01Z,101.0,101.25,101.1,260,5,5,2,buy\n")
            handle.flush()
            replay_path = handle.name

        result = ReplayRunner(config=self.config, engine=self.engine).run(replay_path)

        self.assertEqual(result.events, 3)
        self.assertGreaterEqual(len(result.segments), 1)
        self.assertIn("trade_count", result.summary)
        self.assertIn("deflated_sharpe_ratio_approx", result.summary)
        self.assertIn("walk_forward", result.summary)
        self.assertIn("matrix", result.summary)
        self.assertIn("benchmarks", result.summary)
        self.assertIn("acceptance", result.summary)

    def test_replay_size_field_is_treated_as_per_event_volume(self) -> None:
        event = ReplayRunner(config=self.config, engine=self.engine)._event_to_market_data(
            {
                "timestamp": "2026-03-13T14:30:01Z",
                "bid": 100.0,
                "ask": 100.25,
                "last": 100.1,
                "size": 7,
            }
        )

        self.assertEqual(event.volume, 7)
        self.assertFalse(event.volume_is_cumulative)

    def test_replay_event_parser_treats_nan_quotes_as_synthetic(self) -> None:
        event = ReplayRunner(config=self.config, engine=self.engine)._event_to_market_data(
            {
                "timestamp": "2026-03-13T14:30:01Z",
                "bid": float("nan"),
                "ask": float("nan"),
                "last": 100.1,
                "volume": float("nan"),
                "size": float("nan"),
            }
        )

        self.assertTrue(event.quote_is_synthetic)
        self.assertEqual(event.bid, 99.85)
        self.assertEqual(event.ask, 100.35)
        self.assertEqual(event.volume, 0)
        self.assertFalse(event.volume_is_cumulative)

    def test_replay_marks_synthetic_quotes_as_not_decision_ready(self) -> None:
        with tempfile.NamedTemporaryFile("w+", suffix=".csv", delete=False) as handle:
            handle.write("timestamp,last,volume\n")
            handle.write("2026-03-13T14:30:01Z,100.1,100\n")
            handle.write("2026-03-13T14:31:01Z,100.7,180\n")
            handle.write("2026-03-13T14:32:01Z,101.1,260\n")
            handle.flush()
            replay_path = handle.name

        result = ReplayRunner(config=self.config, engine=self.engine).run(replay_path)

        self.assertTrue(result.summary["synthetic_quotes_detected"])
        self.assertFalse(result.summary["decision_ready"])
        self.assertEqual(result.summary["decision_ready_reason"], "synthetic_quotes_rejected")

    def test_costed_trade_summary_reduces_net_pnl(self) -> None:
        runner = ReplayRunner(config=self.config, engine=self.engine)
        trade = TradeRecord(
            entry_time=pd.Timestamp("2026-03-13T14:30:01Z").to_pydatetime(),
            exit_time=pd.Timestamp("2026-03-13T14:31:01Z").to_pydatetime(),
            direction=1,
            contracts=1,
            entry_price=100.0,
            exit_price=101.0,
            pnl=50.0,
            zone="Post-Open",
            strategy="WEIGHTED_SCORE_MATRIX",
            regime="TREND",
            event_tags=["none"],
        )

        summary = runner._costed_trade_summary([trade], strategy_name="WEIGHTED_SCORE_MATRIX")

        self.assertEqual(summary["gross_pnl"], 50.0)
        self.assertLess(summary["net_pnl"], summary["gross_pnl"])
        self.assertLess(summary["stressed_net_pnl"], summary["net_pnl"])

    def test_walk_forward_segments_are_deterministic(self) -> None:
        runner = ReplayRunner(config=self.config, engine=self.engine)
        bars = bars_from_prices("2026-03-13 09:00", [100.0, 100.5, 101.0, 101.5, 102.0, 102.5])
        trades = [
            TradeRecord(
                entry_time=bars.index[2].to_pydatetime(),
                exit_time=bars.index[3].to_pydatetime(),
                direction=1,
                contracts=1,
                entry_price=101.0,
                exit_price=101.5,
                pnl=25.0,
                zone="Post-Open",
                strategy="WEIGHTED_SCORE_MATRIX",
                regime="TREND",
                event_tags=["none"],
            )
        ]

        first = runner._walk_forward_segments(bars, trades, [])
        second = runner._walk_forward_segments(bars, trades, [])

        self.assertEqual(first, second)
        self.assertGreaterEqual(len(first), 1)

    def test_acceptance_gates_require_benchmark_and_stress_pass(self) -> None:
        runner = ReplayRunner(config=self.config, engine=self.engine)
        state = get_state()
        state.execution = {
            "fill_drift_ticks": 0.5,
            "protection_failures": 0,
            "fail_safe_count": 0,
        }

        gates = runner._acceptance_gates(
            matrix_cost_summary={
                "net_pnl": 100.0,
                "stressed_net_pnl": 20.0,
                "zone_stats": {
                    "Pre-Open": {"net_pnl": 60.0},
                    "Post-Open": {"net_pnl": 40.0},
                },
            },
            benchmark_cost_summary={"net_pnl": 50.0, "stressed_net_pnl": 0.0, "trade_count": 1},
            walk_forward=[{"matrix_positive": True}, {"matrix_positive": False}, {"matrix_positive": True}],
            synthetic_quotes_detected=False,
        )

        self.assertTrue(gates["decision_ready"])
        self.assertTrue(gates["benchmark_outperformance"])
        self.assertTrue(gates["stress_survival_ok"])
        self.assertTrue(gates["walk_forward_ok"])
        self.assertTrue(gates["promotable"])


class EngineRiskExecutionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = build_config()
        set_config(self.config)
        get_client(force_recreate=True)
        self.engine = TradingEngine(self.config)
        self.engine.reset_runtime_state()
        self.engine._mock_mode = True
        self.engine.client.enable_mock_mode()
        self.engine.executor.enable_mock_mode()

    def test_stop_loss_flattens_position(self) -> None:
        self.engine.executor.place_order("ES", 1, "buy", "market")
        self.engine._sync_position_state()
        self.engine._stop_loss = 99.0
        self.engine._take_profit = 102.0

        self.engine.on_market_data(
            MarketData(
                symbol="ES",
                bid=98.5,
                ask=98.75,
                last=98.5,
                volume=100,
                bid_size=5,
                ask_size=5,
                last_size=1,
                timestamp=pd.Timestamp("2026-03-13T15:00:01Z").to_pydatetime(),
            )
        )

        self.assertEqual(self.engine.executor.get_position(), 0)
        self.assertEqual(self.engine._last_exit_reason, "stop_loss")

    def test_take_profit_flattens_position(self) -> None:
        self.engine.executor.place_order("ES", 1, "buy", "market")
        self.engine._sync_position_state()
        self.engine._stop_loss = 99.0
        self.engine._take_profit = 101.0

        self.engine.on_market_data(
            MarketData(
                symbol="ES",
                bid=101.0,
                ask=101.25,
                last=101.1,
                volume=100,
                bid_size=5,
                ask_size=5,
                last_size=1,
                timestamp=pd.Timestamp("2026-03-13T15:01:01Z").to_pydatetime(),
            )
        )

        self.assertEqual(self.engine.executor.get_position(), 0)
        self.assertEqual(self.engine._last_exit_reason, "take_profit")

    def test_watchdog_locks_out_after_stale_feed(self) -> None:
        self.engine._watchdog_state.last_feed_time = pd.Timestamp("2026-03-13T15:00:00Z").to_pydatetime()
        self.engine._latest_market_data = MarketData(
            symbol="ES",
            bid=100.0,
            ask=100.25,
            last=100.1,
            volume=100,
            timestamp=pd.Timestamp("2026-03-13T15:00:30Z").to_pydatetime(),
        )

        self.engine._handle_watchdogs()

        self.assertTrue(self.engine._watchdog_state.fail_safe_lockout)

    def test_bar_aggregator_ignores_out_of_order_ticks(self) -> None:
        first = MarketData(symbol="ES", bid=100.0, ask=100.25, last=100.1, volume=100, timestamp=pd.Timestamp("2026-03-13T15:00:30Z").to_pydatetime())
        second = MarketData(symbol="ES", bid=100.2, ask=100.45, last=100.3, volume=120, timestamp=pd.Timestamp("2026-03-13T14:59:59Z").to_pydatetime())

        self.engine.bar_aggregator.update(first)
        completed = self.engine.bar_aggregator.update(second)

        self.assertIsNone(completed)

    def test_bar_aggregator_supports_per_event_volume(self) -> None:
        first = MarketData(
            symbol="ES",
            bid=100.0,
            ask=100.25,
            last=100.1,
            volume=7,
            volume_is_cumulative=False,
            timestamp=pd.Timestamp("2026-03-13T15:00:10Z").to_pydatetime(),
        )
        second = MarketData(
            symbol="ES",
            bid=100.1,
            ask=100.35,
            last=100.2,
            volume=9,
            volume_is_cumulative=False,
            timestamp=pd.Timestamp("2026-03-13T15:00:40Z").to_pydatetime(),
        )
        third = MarketData(
            symbol="ES",
            bid=100.2,
            ask=100.45,
            last=100.3,
            volume=5,
            volume_is_cumulative=False,
            timestamp=pd.Timestamp("2026-03-13T15:01:01Z").to_pydatetime(),
        )

        self.engine.bar_aggregator.update(first)
        self.engine.bar_aggregator.update(second)
        completed = self.engine.bar_aggregator.update(third)

        self.assertIsNotNone(completed)
        assert completed is not None
        self.assertEqual(completed["volume"], 16)

    def test_performance_summary_tracks_scratches_separately(self) -> None:
        self.engine.risk_manager._trade_history = [
            TradeRecord(
                entry_time=pd.Timestamp("2026-03-13T15:00:00Z").to_pydatetime(),
                exit_time=pd.Timestamp("2026-03-13T15:01:00Z").to_pydatetime(),
                direction=1,
                contracts=1,
                entry_price=100.0,
                exit_price=101.0,
                pnl=50.0,
                zone="Midday",
                strategy="WEIGHTED_SCORE_MATRIX",
            ),
            TradeRecord(
                entry_time=pd.Timestamp("2026-03-13T15:02:00Z").to_pydatetime(),
                exit_time=pd.Timestamp("2026-03-13T15:03:00Z").to_pydatetime(),
                direction=1,
                contracts=1,
                entry_price=100.0,
                exit_price=100.0,
                pnl=0.0,
                zone="Midday",
                strategy="WEIGHTED_SCORE_MATRIX",
            ),
            TradeRecord(
                entry_time=pd.Timestamp("2026-03-13T15:04:00Z").to_pydatetime(),
                exit_time=pd.Timestamp("2026-03-13T15:05:00Z").to_pydatetime(),
                direction=-1,
                contracts=1,
                entry_price=100.0,
                exit_price=101.0,
                pnl=-50.0,
                zone="Midday",
                strategy="WEIGHTED_SCORE_MATRIX",
            ),
        ]

        summary = self.engine.build_performance_summary()

        self.assertEqual(summary["scratch_trades"], 1)
        self.assertEqual(summary["avg_loser"], -50.0)

    def test_sync_position_records_partial_exit_trade(self) -> None:
        self.engine.executor.place_order("ES", 2, "buy", "market")
        self.engine._sync_position_state()

        self.engine.executor.place_order("ES", 1, "sell", "market")
        self.engine._sync_position_state()

        history = self.engine.risk_manager.get_trade_history()

        self.assertEqual(self.engine.executor.get_position(), 1)
        self.assertEqual(self.engine.risk_manager.get_metrics().current_position, 1)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].contracts, 1)

    def test_sync_position_records_reversal_close_before_new_open(self) -> None:
        self.engine.executor.place_order("ES", 1, "buy", "market")
        self.engine._sync_position_state()

        self.engine.executor.place_order("ES", 2, "sell", "market")
        self.engine._sync_position_state()

        history = self.engine.risk_manager.get_trade_history()

        self.assertEqual(self.engine.executor.get_position(), -1)
        self.assertEqual(self.engine.risk_manager.get_metrics().current_position, -1)
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0].contracts, 1)


class SchedulerTests(unittest.TestCase):
    def test_cross_midnight_zone_uses_prior_session_start_after_midnight(self) -> None:
        config = build_config()
        config.hot_zones = [
            HotZoneConfig(name="Night", start="23:00", end="01:00", timezone="America/Chicago"),
        ]
        scheduler = HotZoneScheduler(config)

        zone = scheduler.get_current_zone(pd.Timestamp("2026-04-01 00:30", tz="America/Chicago").to_pydatetime())

        assert zone is not None
        self.assertEqual(zone.start_time, pd.Timestamp("2026-03-31 23:00", tz="America/Chicago").to_pydatetime())
        self.assertEqual(zone.end_time, pd.Timestamp("2026-04-01 01:00", tz="America/Chicago").to_pydatetime())
        self.assertEqual(zone.minutes_remaining, 30.0)

    def test_repeated_zone_reads_do_not_advance_bar_counter(self) -> None:
        scheduler = HotZoneScheduler(build_config())
        current_time = pd.Timestamp("2026-03-13 09:05", tz="America/Chicago").to_pydatetime()

        first = scheduler.get_current_zone(current_time)
        second = scheduler.get_current_zone(current_time)

        assert first is not None
        assert second is not None
        self.assertTrue(first.is_first_bar)
        self.assertTrue(second.is_first_bar)
        self.assertEqual(scheduler._bars_in_zone, 1)


class AccountSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = build_config()
        self.config.safety.prac_only = True
        self.config.account.require_preferred_account = True
        set_config(self.config)
        self.client = TopstepClient(self.config.api)

    def test_select_account_refuses_live_only_accounts_when_prac_required(self) -> None:
        selected = self.client._select_account(
            [
                {"id": "LIVE-1", "name": "LIVE-1", "canTrade": True, "balance": 50000, "simulated": False},
            ]
        )

        self.assertIsNone(selected)

    def test_select_account_rejects_preferred_env_when_it_targets_non_practice(self) -> None:
        with patch.dict(os.environ, {"PREFERRED_ACCOUNT_ID": "LIVE-1"}, clear=False):
            selected = self.client._select_account(
                [
                    {"id": "LIVE-1", "name": "LIVE-1", "canTrade": True, "balance": 50000, "simulated": False},
                    {"id": "PRAC-1", "name": "PRAC-1", "canTrade": True, "balance": 50000, "simulated": True},
                ]
            )

        self.assertIsNone(selected)

    def test_place_order_refuses_non_practice_account_when_prac_only_enabled(self) -> None:
        self.client._access_token = "token"
        self.client._token_expires = float("inf")
        self.client._account_id = 123
        self.client._account = Account(account_id="123", name="LIVE-1", balance=50000, is_practice=False)

        order_id = self.client.place_order("ES", 1, "buy", "market")

        self.assertIsNone(order_id)

    def test_build_hub_url_injects_access_token_query_param(self) -> None:
        self.client._access_token = "jwt-token"

        hub_url = self.client._build_hub_url("https://rtc.topstepx.com/hubs/market")

        self.assertTrue(hub_url.startswith("wss://rtc.topstepx.com/hubs/market"))
        self.assertIn("access_token=jwt-token", hub_url)

    def test_decode_signalr_frames_splits_record_separator_payloads(self) -> None:
        frames = self.client._decode_signalr_frames('{"type":6}\x1e{"type":1,"target":"GatewayQuote","arguments":["CON",{"lastPrice":1.0}]}\x1e')

        self.assertEqual(len(frames), 2)
        self.assertEqual(frames[1]["target"], "GatewayQuote")


class LoggingBootstrapTests(unittest.TestCase):
    def test_configure_logging_creates_rotating_file_handler(self) -> None:
        config = build_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            config.logging.file = str(Path(temp_dir) / "trading.log")
            log_path = _configure_logging(config)
            root_logger = logging.getLogger()

            self.assertTrue(log_path.exists())
            self.assertTrue(any(getattr(handler, "baseFilename", None) == str(log_path) for handler in root_logger.handlers))

    def test_configure_logging_supports_console_color_toggle(self) -> None:
        config = build_config()
        config.logging.console_colors = False
        with tempfile.TemporaryDirectory() as temp_dir:
            config.logging.file = str(Path(temp_dir) / "trading.log")
            _configure_logging(config)
            root_logger = logging.getLogger()

            self.assertGreaterEqual(len(root_logger.handlers), 2)

    def test_startup_summary_logs_structured_lines(self) -> None:
        config = build_config()

        with self.assertLogs("src.cli.commands", level="INFO") as captured:
            _log_startup_summary(config, Path("/tmp/trading.log"), False, "Post-Open", "active")

        joined = "\n".join(captured.output)
        self.assertIn("startup_summary", joined)
        self.assertIn("startup_endpoints", joined)


class DebugServerLifecycleTests(unittest.TestCase):
    def test_stop_closes_servers_without_httpserver_shutdown(self) -> None:
        config = build_config()
        set_config(config)
        server = DebugServer(config.server)
        server._running = True

        thread = Mock()
        thread.is_alive.return_value = True
        server._thread = thread

        health_server = Mock()
        debug_server = Mock()
        server._health_server = health_server
        server._debug_server = debug_server

        server.stop()

        thread.join.assert_called_once_with(timeout=2)
        health_server.server_close.assert_called_once()
        debug_server.server_close.assert_called_once()
        self.assertIsNone(server._thread)
        self.assertIsNone(server._health_server)
        self.assertIsNone(server._debug_server)


class ObservabilityStoreTests(unittest.TestCase):
    def test_observability_store_persists_and_queries_events(self) -> None:
        config = build_config()
        config.observability.enabled = True
        with tempfile.TemporaryDirectory() as temp_dir:
            config.observability.sqlite_path = str(Path(temp_dir) / "observability.db")
            set_config(config)
            store = get_observability_store(force_recreate=True, config=config)
            store.start()
            store.record_event(
                category="system",
                event_type="test_event",
                source="tests.test_matrix_engine",
                payload={"value": 7},
                symbol="ES",
                action="test",
                reason="test_event",
            )
            store.force_flush()

            rows = store.query_events(limit=5, category="system", event_type="test_event")

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["payload"]["value"], 7)
            self.assertTrue(Path(config.observability.sqlite_path).exists())
            store.stop()

    def test_observability_store_persists_run_manifest_and_completed_trade(self) -> None:
        config = build_config()
        config.observability.enabled = True
        with tempfile.TemporaryDirectory() as temp_dir:
            config.observability.sqlite_path = str(Path(temp_dir) / "observability.db")
            set_config(config)
            store = get_observability_store(force_recreate=True, config=config)
            store.start()
            store.record_run_manifest(
                {
                    "run_id": store.get_run_id(),
                    "started_at": "2026-03-16T12:00:00+00:00",
                    "process_id": 42,
                    "data_mode": "mock",
                    "symbols": ["ES"],
                    "config_path": "/tmp/config.yaml",
                    "config_hash": "abc123",
                    "log_path": "/tmp/trading.log",
                    "sqlite_path": config.observability.sqlite_path,
                    "git_commit": "deadbeef",
                    "git_branch": "main",
                    "git_dirty": False,
                    "git_available": True,
                    "app_version": "0.1.0",
                }
            )
            trade = TradeRecord(
                entry_time=datetime.fromisoformat("2026-03-16T12:01:00+00:00"),
                exit_time=datetime.fromisoformat("2026-03-16T12:02:00+00:00"),
                direction=1,
                contracts=2,
                entry_price=100.0,
                exit_price=101.0,
                pnl=100.0,
                zone="Post-Open",
                strategy="WEIGHTED_SCORE_MATRIX",
                regime="TREND",
                event_tags=["opening_drive"],
            )
            store.record_completed_trade(trade)

            manifests = store.query_run_manifests(limit=5)
            trades = store.query_completed_trades(limit=5)

            self.assertEqual(manifests[0]["run_id"], store.get_run_id())
            self.assertEqual(manifests[0]["git_commit"], "deadbeef")
            self.assertEqual(trades[0]["run_id"], store.get_run_id())
            self.assertEqual(trades[0]["zone"], "Post-Open")
            self.assertEqual(trades[0]["event_tags"], ["opening_drive"])
            store.stop()

    def test_observability_store_backfills_completed_trades_from_events(self) -> None:
        config = build_config()
        config.observability.enabled = True
        with tempfile.TemporaryDirectory() as temp_dir:
            config.observability.sqlite_path = str(Path(temp_dir) / "observability.db")
            set_config(config)
            store = get_observability_store(force_recreate=True, config=config)
            store.start()
            trade_exit = datetime.fromisoformat("2026-03-16T12:02:00+00:00")
            store.record_event(
                category="risk",
                event_type="trade_recorded",
                source="tests.test_matrix_engine",
                payload={
                    "entry_time": "2026-03-16T12:01:00+00:00",
                    "exit_time": "2026-03-16T12:02:00+00:00",
                    "contracts": 1,
                    "entry_price": 100.0,
                    "exit_price": 101.0,
                    "pnl": 50.0,
                    "direction": 1,
                    "zone": "Post-Open",
                    "strategy": "WEIGHTED_SCORE_MATRIX",
                    "regime": "TREND",
                    "event_tags": ["opening_drive"],
                },
                event_time=trade_exit,
                zone="Post-Open",
                action="record_trade",
                reason="trade_recorded",
            )
            store.force_flush()

            result = store.backfill_completed_trades_from_events()
            trades = store.query_completed_trades(limit=5)

            self.assertEqual(result["backfilled"], 1)
            self.assertEqual(len(trades), 1)
            self.assertTrue(trades[0]["backfilled"])
            self.assertEqual(trades[0]["zone"], "Post-Open")
            store.stop()

    def test_engine_no_trade_persists_decision_event(self) -> None:
        config = build_config()
        config.observability.enabled = True
        with tempfile.TemporaryDirectory() as temp_dir:
            config.observability.sqlite_path = str(Path(temp_dir) / "observability.db")
            set_config(config)
            store = get_observability_store(force_recreate=True, config=config)
            store.start()
            get_client(force_recreate=True)
            get_executor(force_recreate=True)
            get_scheduler(force_recreate=True)
            get_risk_manager(force_recreate=True)
            engine = TradingEngine(config)
            engine.reset_runtime_state()
            engine.executor.enable_mock_mode()
            engine._bars = bars_from_prices("2026-03-13 09:00", [100.0 + (i * 0.2) for i in range(30)])
            engine._last_price = float(engine._bars["close"].iloc[-1])
            engine._latest_market_data = MarketData(
                symbol="ES",
                bid=engine._last_price - 0.25,
                ask=engine._last_price + 0.25,
                last=engine._last_price,
                volume=1000,
                timestamp=engine._bars.index[-1].tz_convert("UTC").to_pydatetime(),
            )
            engine._latest_flow_snapshot = OrderFlowSnapshot(
                ofi=12.0,
                ofi_zscore=1.2,
                quote_rate_per_minute=30.0,
                quote_rate_state=0.8,
                spread_regime=1.4,
                volume_pace=1.0,
            )
            decision = MatrixDecision(
                zone_name="Post-Open",
                action="NO_TRADE",
                reason="matrix_not_decisive",
                long_score=0.9,
                short_score=0.8,
                flat_bias=1.0,
                active_vetoes=["trend_flat"],
                feature_snapshot=FeatureSnapshot(
                    zone_name="Post-Open",
                    current_price=engine._last_price,
                    atr_value=1.0,
                    long_features={},
                    short_features={},
                    flat_features={},
                    signed_features={},
                    execution_tradeable=True,
                    active_session="RTH",
                    regime_state="range",
                    regime_reason="contained_rotation",
                    event_tags=[],
                ),
                execution_tradeable=True,
            )

            with patch.object(engine.scheduler, "get_current_zone", return_value=zone("Post-Open", engine._bars.index[-1], start_time=engine._bars.index[0])):
                with patch.object(engine.event_provider, "get_context", return_value=EventContext()):
                    with patch.object(engine.risk_manager, "should_flatten_position", return_value=(False, "ok")):
                        with patch.object(engine.matrix, "evaluate", return_value=decision):
                            engine._evaluate_current_state(allow_entries=True)

            store.force_flush()
            rows = store.query_events(limit=10, category="decision", event_type="decision_evaluated")

            self.assertTrue(any(row["action"] == "NO_TRADE" for row in rows))
            self.assertTrue(any(row["payload"]["outcome"] == "no_trade" for row in rows))
            store.stop()


class McpServerTests(unittest.TestCase):
    def test_mcp_query_events_tool_returns_structured_content(self) -> None:
        config = build_config()
        config.observability.enabled = True
        with tempfile.TemporaryDirectory() as temp_dir:
            config.observability.sqlite_path = str(Path(temp_dir) / "observability.db")
            set_config(config)
            store = get_observability_store(force_recreate=True, config=config)
            store.start()
            store.record_event(
                category="system",
                event_type="test_event",
                source="tests.test_matrix_engine",
                payload={"value": 9},
                action="test",
                reason="mcp_test",
            )
            store.force_flush()
            state = TradingState()
            state.run_id = store.get_run_id()
            status, response = handle_mcp_request(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {
                        "name": "query_events",
                        "arguments": {"limit": 5, "category": "system", "event_type": "test_event"},
                    },
                },
                lambda: state,
            )

            self.assertEqual(status, 200)
            payload = response["result"]["structuredContent"]
            self.assertEqual(len(payload["events"]), 1)
            self.assertEqual(payload["events"][0]["payload"]["value"], 9)
            store.stop()

    def test_mcp_current_run_resource_reads_manifest_and_trade_summary(self) -> None:
        config = build_config()
        config.observability.enabled = True
        with tempfile.TemporaryDirectory() as temp_dir:
            config.observability.sqlite_path = str(Path(temp_dir) / "observability.db")
            set_config(config)
            store = get_observability_store(force_recreate=True, config=config)
            store.start()
            store.record_run_manifest(
                {
                    "run_id": store.get_run_id(),
                    "started_at": "2026-03-16T12:00:00+00:00",
                    "process_id": 42,
                    "data_mode": "mock",
                    "symbols": ["ES"],
                    "config_path": "/tmp/config.yaml",
                    "config_hash": "abc123",
                    "log_path": "/tmp/trading.log",
                    "sqlite_path": config.observability.sqlite_path,
                    "app_version": "0.1.0",
                }
            )
            store.record_completed_trade(
                TradeRecord(
                    entry_time=datetime.fromisoformat("2026-03-16T12:01:00+00:00"),
                    exit_time=datetime.fromisoformat("2026-03-16T12:02:00+00:00"),
                    direction=1,
                    contracts=1,
                    entry_price=100.0,
                    exit_price=102.0,
                    pnl=100.0,
                    zone="Post-Open",
                    strategy="WEIGHTED_SCORE_MATRIX",
                    regime="TREND",
                    event_tags=["opening_drive"],
                )
            )
            state = TradingState()
            state.run_id = store.get_run_id()
            status, response = handle_mcp_request(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "resources/read",
                    "params": {"uri": "observability://current-run"},
                },
                lambda: state,
            )

            self.assertEqual(status, 200)
            content = json.loads(response["result"]["contents"][0]["text"])
            self.assertEqual(content["run"]["run_id"], store.get_run_id())
            self.assertEqual(content["performance"]["trade_count"], 1)
            self.assertEqual(content["performance"]["total_pnl"], 100.0)
            store.stop()


class RiskManagerLoggingTests(unittest.TestCase):
    def test_volatility_circuit_breaker_logs_only_on_state_transition(self) -> None:
        config = build_config()
        config.risk.vol_spike_threshold = 1.2
        set_config(config)
        manager = RiskManager(config)

        for atr in (1.0, 1.0, 1.0, 1.0):
            manager.update_volatility(atr)

        with self.assertLogs("src.engine.risk_manager", level="WARNING") as captured:
            manager.update_volatility(2.0)
            manager.update_volatility(2.1)

        self.assertEqual(len(captured.output), 1)
        self.assertIn("risk_circuit_breaker_activated", captured.output[0])


class ExecutorProtectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = build_config()
        set_config(self.config)
        self.executor = OrderExecutor(self.config)
        self.executor.enable_mock_mode()

    def test_failed_sibling_cancel_keeps_tracking_record(self) -> None:
        self.executor._protective_orders["ES"] = {"orders": ["stop", "target"], "direction": 1, "stop_price": 99.0, "take_profit": 101.0}
        with patch.object(self.executor, "cancel_order", side_effect=[False]):
            self.executor._cancel_sibling_protection("ES", "stop")

        self.assertIn("ES", self.executor._protective_orders)
        self.assertEqual(self.executor._protective_orders["ES"]["orders"], ["target"])

    def test_mock_reversal_resets_average_price_to_reversal_fill(self) -> None:
        self.executor._apply_mock_fill("buy", 1, 100.0)
        self.executor._apply_mock_fill("sell", 2, 99.5)

        self.assertEqual(self.executor.get_position(), -1)
        self.assertEqual(self.executor._mock_avg_price, 99.5)

    def test_ensure_protection_uses_tolerance_for_matching_prices(self) -> None:
        self.executor._protective_orders["ES"] = {
            "orders": ["stop", "target"],
            "direction": 1,
            "stop_price": 99.0,
            "take_profit": 101.0,
        }
        with patch.object(self.executor, "clear_protection") as clear_protection:
            orders = self.executor.ensure_protection("ES", 1, 1, 99.0 + 1e-8, 101.0 - 1e-8)

        self.assertEqual(orders, 2)
        clear_protection.assert_not_called()

    def test_ensure_protection_places_and_clears_stop_and_target(self) -> None:
        placed = self.executor.ensure_protection("ES", 1, 1, 5800.0, 6000.0)

        self.assertEqual(placed, 2)
        self.assertTrue(self.executor.is_protected("ES"))
        self.assertEqual(len(self.executor.get_active_orders(symbol="ES", is_protective=True)), 2)

        cancelled = self.executor.clear_protection("ES")

        self.assertEqual(cancelled, 2)
        self.assertFalse(self.executor.is_protected("ES"))

    def test_protection_pending_too_long_only_when_requested_without_orders(self) -> None:
        now = datetime.fromisoformat("2026-03-13T15:00:10+00:00")
        self.executor._protection_requested_at["ES"] = now - timedelta(seconds=10)

        self.assertTrue(self.executor.protection_pending_too_long("ES", now, timeout_seconds=5))

        self.executor._protective_orders["ES"] = {"orders": ["stop"], "direction": 1, "stop_price": 99.0, "take_profit": None}

        self.assertFalse(self.executor.protection_pending_too_long("ES", now, timeout_seconds=5))


class EventProviderTests(unittest.TestCase):
    def test_stale_emergency_halt_file_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            halt_path = root / "emergency_halt.flag"
            halt_path.write_text("halt", encoding="utf-8")
            stale_age_seconds = 3600
            stale_mtime = pd.Timestamp("2026-03-13T12:00:00Z").timestamp() - stale_age_seconds
            os.utime(halt_path, (stale_mtime, stale_mtime))

            config = EventProviderConfig(
                calendar_path="missing.yaml",
                emergency_halt_path="emergency_halt.flag",
                emergency_halt_max_age_minutes=5,
            )
            provider = LocalEventProvider(config, BlackoutConfig(news_times=[]), root)

            context = provider.get_context(pd.Timestamp("2026-03-13T12:00:00Z").to_pydatetime())

        self.assertFalse(context.blackout_active)
        self.assertNotEqual(context.reason, "manual_emergency_halt")


if __name__ == "__main__":
    unittest.main()

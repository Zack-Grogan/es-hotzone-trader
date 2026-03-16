"""Core engine and safety tests."""
from __future__ import annotations

from datetime import datetime, timedelta
import time
import unittest
from unittest.mock import Mock, patch

import pandas as pd
import pytz

from src.config import Config, set_config
from src.engine.risk_manager import RiskManager, RiskState
from src.market.topstep_client import TopstepClient
from src.strategies.orb_strategy import ORBStrategy
from src.strategies.vwap_mr import VWAPMeanReversionStrategy


class AccountSelectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config()
        set_config(self.config)

    def test_select_account_prefers_prac_match(self) -> None:
        client = TopstepClient(self.config.api)
        selected = client._select_account(
            [
                {"id": "LIVE-123", "name": "Live", "canTrade": True, "balance": 50000},
                {"id": "ABC-PRAC-01", "name": "Practice", "canTrade": True, "balance": 50000},
            ]
        )

        self.assertIsNotNone(selected)
        assert selected is not None
        self.assertEqual(selected.account_id, "ABC-PRAC-01")
        self.assertTrue(selected.is_practice)

    def test_place_order_rejects_non_practice_account_by_default(self) -> None:
        client = TopstepClient(self.config.api)
        client._access_token = "token"
        client._token_expires = time.time() + 3600
        client._account_id = 123
        client._account = client._account_summary(
            {"id": "LIVE-123", "name": "Live", "balance": 50000, "canTrade": True}
        )

        with patch("src.market.topstep_client.requests.post") as request_post:
            order_id = client.place_order("ES", 1, "buy", "market")

        self.assertIsNone(order_id)
        request_post.assert_not_called()


class StrategySafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config()
        set_config(self.config)

    def test_vwap_mr_sets_entry_time_on_signal(self) -> None:
        strategy = VWAPMeanReversionStrategy(self.config)
        index = pd.date_range("2026-03-13 12:00", periods=30, freq="min", tz="America/Chicago")
        df = pd.DataFrame(
            {
                "open": [100.0] * 30,
                "high": [100.5] * 30,
                "low": [95.0] * 30,
                "close": [95.0] * 30,
                "volume": [100] * 30,
            },
            index=index,
        )

        with patch("src.strategies.vwap_mr.atr") as mock_atr, patch(
            "src.strategies.vwap_mr.vwap"
        ) as mock_vwap, patch("src.strategies.vwap_mr.rsi") as mock_rsi:
            mock_atr.return_value = pd.Series([1.0] * len(df), index=df.index)
            mock_vwap.return_value = pd.Series([100.0] * len(df), index=df.index)
            mock_rsi.return_value = pd.Series([20.0] * len(df), index=df.index)

            signal = strategy.compute_signal(df, position=0, zone_info=None)

        self.assertIsNotNone(signal)
        self.assertEqual(strategy._entry_bar_time, df.index[-1])

    def test_vwap_mr_clears_entry_time_on_flat_exit(self) -> None:
        strategy = VWAPMeanReversionStrategy(self.config)
        index = pd.date_range("2026-03-13 12:00", periods=30, freq="min", tz="America/Chicago")
        df = pd.DataFrame(
            {
                "open": [100.0] * 30,
                "high": [100.5] * 30,
                "low": [99.5] * 30,
                "close": [100.0] * 30,
                "volume": [100] * 30,
            },
            index=index,
        )
        strategy._entry_bar_time = df.index[-5]

        with patch("src.strategies.vwap_mr.atr") as mock_atr, patch(
            "src.strategies.vwap_mr.vwap"
        ) as mock_vwap, patch("src.strategies.vwap_mr.rsi") as mock_rsi:
            mock_atr.return_value = pd.Series([1.0] * len(df), index=df.index)
            mock_vwap.return_value = pd.Series([99.5] * len(df), index=df.index)
            mock_rsi.return_value = pd.Series([55.0] * len(df), index=df.index)

            signal = strategy.compute_signal(df, position=1, zone_info=None)

        self.assertIsNotNone(signal)
        assert signal is not None
        self.assertEqual(signal.reason, "vwap_target")
        self.assertIsNone(strategy._entry_bar_time)

    def test_orb_range_respects_zone_start(self) -> None:
        strategy = ORBStrategy(self.config)
        strategy.range_minutes = 5
        index = pd.date_range("2026-03-13 08:50", periods=20, freq="min", tz="America/Chicago")
        df = pd.DataFrame(
            {
                "open": [100, 200] + [100] * 18,
                "high": [100, 250] + [101] * 18,
                "low": [99, 150] + [99] * 18,
                "close": [100, 200] + [100] * 18,
                "volume": [100] * 20,
            },
            index=index,
        )

        strategy._range_start_time = index[10]
        strategy._reset_range(df)

        self.assertEqual(strategy._range_high, 101)
        self.assertEqual(strategy._range_low, 99)


class RiskManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config()
        set_config(self.config)

    def test_blackout_blocks_new_trades(self) -> None:
        risk_manager = RiskManager(self.config)
        risk_manager.set_blackout(True, "news_blackout")

        allowed, reason = risk_manager.can_trade("Midday")

        self.assertFalse(allowed)
        self.assertEqual(reason, "news_blackout")

    def test_max_position_loss_forces_flatten(self) -> None:
        risk_manager = RiskManager(self.config)
        risk_manager.open_position(contracts=1, entry_price=100, direction=1, zone="Midday")

        should_flatten, reason = risk_manager.should_flatten_position(95.5)

        self.assertTrue(should_flatten)
        self.assertEqual(reason, "max_position_loss")

    def test_volatility_breaker_uses_prior_baseline(self) -> None:
        risk_manager = RiskManager(self.config)
        for atr_value in [1.0, 1.0, 1.1, 1.0]:
            risk_manager.update_volatility(atr_value)

        risk_manager.update_volatility(2.2)

        self.assertEqual(risk_manager.get_state().value, "circuit_breaker")

    def test_session_reset_uses_cme_boundary(self) -> None:
        risk_manager = RiskManager(self.config)
        pre_reset = pytz.timezone("America/Chicago").localize(datetime(2026, 3, 13, 16, 55))
        post_reset = pytz.timezone("America/Chicago").localize(datetime(2026, 3, 13, 17, 5))

        risk_manager.can_trade("Midday", current_time=pre_reset)
        risk_manager._daily_pnl = -150
        risk_manager.can_trade("Midday", current_time=pre_reset + timedelta(minutes=1))
        self.assertEqual(risk_manager._daily_pnl, -150)

        risk_manager.can_trade("Midday", current_time=post_reset)
        self.assertEqual(risk_manager._daily_pnl, 0)

    def test_trade_record_uses_engine_time(self) -> None:
        risk_manager = RiskManager(self.config)
        entry_time = pytz.timezone("America/Chicago").localize(datetime(2026, 3, 13, 12, 1))
        exit_time = pytz.timezone("America/Chicago").localize(datetime(2026, 3, 13, 12, 9))

        risk_manager.open_position(contracts=1, entry_price=100, direction=1, zone="Midday", current_time=entry_time)
        trade = risk_manager.close_position(101, current_time=exit_time)

        self.assertIsNotNone(trade)
        assert trade is not None
        self.assertEqual(trade.entry_time, entry_time)
        self.assertEqual(trade.exit_time, exit_time)

    def test_is_reduced_risk_only_for_reduced_state(self) -> None:
        risk_manager = RiskManager(self.config)
        self.assertFalse(risk_manager.is_reduced_risk())

        risk_manager.reduce_risk()
        self.assertTrue(risk_manager.is_reduced_risk())

        risk_manager.reset_risk()
        risk_manager._risk_state = RiskState.CIRCUIT_BREAKER
        self.assertFalse(risk_manager.is_reduced_risk())

    def test_should_flatten_can_use_observed_market_price(self) -> None:
        risk_manager = RiskManager(self.config)
        risk_manager.open_position(contracts=1, entry_price=100, direction=1, zone="Midday")
        risk_manager.observe_market_price(95.0)

        should_flatten, reason = risk_manager.should_flatten_position()

        self.assertTrue(should_flatten)
        self.assertEqual(reason, "max_position_loss")

    def test_zero_position_unrealized_pnl_is_zero(self) -> None:
        risk_manager = RiskManager(self.config)

        self.assertEqual(risk_manager._calculate_unrealized_pnl(101.0), 0.0)

    def test_stale_observed_market_price_is_rejected(self) -> None:
        risk_manager = RiskManager(self.config)
        entry_time = pytz.timezone("America/Chicago").localize(datetime(2026, 3, 13, 12, 0))
        stale_time = entry_time
        current_time = stale_time + timedelta(seconds=self.config.watchdog.feed_stale_seconds + 5)
        risk_manager.open_position(contracts=1, entry_price=100, direction=1, zone="Midday", current_time=entry_time)
        risk_manager.observe_market_price(95.0, stale_time)

        should_flatten, reason = risk_manager.should_flatten_position(current_time=current_time)

        self.assertFalse(should_flatten)
        self.assertEqual(reason, "stale_market_price")

    def test_daily_reset_clears_consecutive_losses(self) -> None:
        risk_manager = RiskManager(self.config)
        risk_manager._consecutive_losses = 2
        post_reset = pytz.timezone("America/Chicago").localize(datetime(2026, 3, 13, 17, 5))

        risk_manager.can_trade("Midday", current_time=post_reset)

        self.assertEqual(risk_manager._consecutive_losses, 0)


class TransportSafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config()
        set_config(self.config)

    def test_stop_market_stream_uses_threadsafe_close(self) -> None:
        client = TopstepClient(self.config.api)
        client._ws = Mock()
        client._ws.close = Mock(return_value=Mock())
        client._ws_loop = Mock()
        client._ws_loop.is_running.return_value = True
        client._ws_thread = Mock()

        future = Mock()
        with patch("src.market.topstep_client.asyncio.run_coroutine_threadsafe", return_value=future) as close_call:
            client.stop_market_stream()

        close_call.assert_called_once()
        future.result.assert_called_once()
        client._ws_thread.join.assert_called_once()


if __name__ == "__main__":
    unittest.main()

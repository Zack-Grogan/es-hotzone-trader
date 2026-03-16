"""Market microstructure context derived from top-of-book data."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from statistics import mean, pstdev
from typing import Optional

from src.config import OrderFlowConfig
from src.market import MarketData


@dataclass
class OrderFlowSnapshot:
    """Normalized order-flow snapshot used by the matrix engine."""

    ofi: float = 0.0
    ofi_zscore: float = 0.0
    quote_rate_per_minute: float = 0.0
    quote_rate_state: float = 0.0
    spread_regime: float = 0.0
    volume_pace: float = 0.0
    volume_delta: float = 0.0
    trade_side_available: bool = False


class MicrostructureTracker:
    """Track simple BBO-derived order flow and quote activity."""

    def __init__(self, config: OrderFlowConfig):
        self.config = config
        self._history: deque[tuple[datetime, float, float, float, float]] = deque()
        self._ofi_history: deque[float] = deque(maxlen=max(int(config.zscore_window), 20))
        self._volume_history: deque[float] = deque(maxlen=max(int(config.volume_window), 20))
        self._last_market: Optional[MarketData] = None

    def reset(self) -> None:
        self._history.clear()
        self._ofi_history.clear()
        self._volume_history.clear()
        self._last_market = None

    def update(self, market_data: MarketData) -> OrderFlowSnapshot:
        """Update the tracker and return the latest snapshot."""
        current_ts = market_data.timestamp
        window_start = current_ts - timedelta(seconds=max(int(self.config.window_seconds), 1))

        bid_size = market_data.bid_size or 1.0
        ask_size = market_data.ask_size or 1.0
        volume_delta = float(market_data.last_size or 0.0)
        if volume_delta <= 0.0:
            if not market_data.volume_is_cumulative:
                volume_delta = float(market_data.volume or 0.0)
            elif self._last_market is None:
                volume_delta = float(market_data.volume or 0.0)
            else:
                volume_delta = max(float(market_data.volume or 0.0) - float(self._last_market.volume or 0.0), 0.0)

        ofi = 0.0
        if self._last_market is not None:
            prev = self._last_market
            prev_bid_size = prev.bid_size or 1.0
            prev_ask_size = prev.ask_size or 1.0
            ofi = (
                (bid_size if market_data.bid >= prev.bid else 0.0)
                - (prev_bid_size if market_data.bid <= prev.bid else 0.0)
                - (ask_size if market_data.ask <= prev.ask else 0.0)
                + (prev_ask_size if market_data.ask >= prev.ask else 0.0)
            )

        spread_ticks = (market_data.spread / 0.25) if market_data.spread else 0.0
        self._history.append((current_ts, ofi, spread_ticks, volume_delta, float(market_data.last or market_data.mid or 0.0)))
        self._ofi_history.append(ofi)
        self._volume_history.append(volume_delta)
        self._last_market = market_data

        while self._history and self._history[0][0] < window_start:
            self._history.popleft()

        minutes = max(self.config.window_seconds / 60.0, 1.0)
        quote_rate = len(self._history) / minutes
        baseline_quote_rate = max(float(self.config.quote_rate_baseline), 1.0)
        quote_rate_state = max(-2.0, min(2.0, (quote_rate / baseline_quote_rate) - 1.0))

        spread_values = [item[2] for item in self._history] or [spread_ticks]
        spread_mean = mean(spread_values)
        spread_regime = max(-2.0, min(2.0, 2.0 - (spread_mean / max(self.config.stress_spread_ticks, 1.0))))

        volume_values = [value for value in self._volume_history if value > 0]
        volume_baseline = mean(volume_values) if volume_values else max(volume_delta, 1.0)
        volume_pace = max(-2.0, min(2.0, (volume_delta / max(volume_baseline, 1.0)) - 1.0))

        ofi_values = list(self._ofi_history)
        ofi_mean = mean(ofi_values) if ofi_values else 0.0
        ofi_std = pstdev(ofi_values) if len(ofi_values) >= 2 else 0.0
        ofi_zscore = (ofi - ofi_mean) / ofi_std if ofi_std else 0.0
        ofi_zscore = max(-2.0, min(2.0, ofi_zscore))

        return OrderFlowSnapshot(
            ofi=ofi,
            ofi_zscore=ofi_zscore,
            quote_rate_per_minute=quote_rate,
            quote_rate_state=quote_rate_state,
            spread_regime=spread_regime,
            volume_pace=volume_pace,
            volume_delta=volume_delta,
            trade_side_available=bool(market_data.trade_side),
        )

"""Historical replay runner using the live engine path."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterable, Optional

import pandas as pd

from src.config import Config, get_config
from src.engine.risk_manager import TradeRecord
from src.engine.scheduler import HotZoneScheduler, ZoneState
from src.engine.trading_engine import TradingEngine, get_trading_engine
from src.market import MarketData
from src.server import get_state, set_state
from src.strategies.base import SignalDirection
from src.strategies.flatten_strategy import FlattenStrategy
from src.strategies.orb_strategy import ORBStrategy
from src.strategies.vwap_mr import VWAPMeanReversionStrategy
from src.strategies.vwap_trend import VWAPTrendStrategy


@dataclass
class ReplayResult:
    """Replay summary."""

    path: str
    events: int
    segments: list[dict]
    summary: dict


class ReplayRunner:
    """Feed historical events through the same engine used live."""

    def __init__(self, config: Optional[Config] = None, engine: Optional[TradingEngine] = None):
        self.config = config or get_config()
        self.engine = engine or get_trading_engine()

    def run(self, path: str) -> ReplayResult:
        """Replay a ProjectX-style CSV or JSONL file."""
        replay_path = Path(path)
        events = list(self._load_events(replay_path))
        synthetic_quotes_detected = any(event.quote_is_synthetic for event in events)

        self.engine.reset_runtime_state(clear_history=True)
        self.engine.enable_mock_mode()

        processed = 0
        for event in events:
            self.engine.on_market_data(event)
            processed += 1

        self.engine.flush_pending_bar()

        matrix_trades = list(self.engine.risk_manager.get_trade_history())
        bars = self.engine._bars.copy()  # noqa: SLF001 - replay needs the finalized shared engine state.
        matrix_summary = self.engine.build_performance_summary()
        matrix_cost_summary = self._costed_trade_summary(matrix_trades, strategy_name="WEIGHTED_SCORE_MATRIX")

        benchmark_trades = self._run_benchmark_portfolio(bars)
        benchmark_cost_summary = self._costed_trade_summary(benchmark_trades, strategy_name="BENCHMARK_PORTFOLIO")

        walk_forward = self._walk_forward_segments(bars, matrix_trades, benchmark_trades)
        approx_dsr = self._deflated_sharpe_ratio(
            [trade.pnl for trade in matrix_trades],
            max(self.config.replay_execution.dsr_trials, max(len(walk_forward), 1)),
        )
        acceptance = self._acceptance_gates(
            matrix_cost_summary=matrix_cost_summary,
            benchmark_cost_summary=benchmark_cost_summary,
            walk_forward=walk_forward,
            synthetic_quotes_detected=synthetic_quotes_detected,
        )

        summary = {
            **matrix_summary,
            "matrix": matrix_cost_summary,
            "benchmarks": {
                "enabled": self.config.validation.benchmarks_enabled,
                "strategy_by_zone": dict(self.config.validation.benchmark_zone_strategies),
                "portfolio": benchmark_cost_summary,
            },
            "comparison": {
                "net_pnl_delta": round(matrix_cost_summary["net_pnl"] - benchmark_cost_summary["net_pnl"], 2),
                "stressed_net_pnl_delta": round(
                    matrix_cost_summary["stressed_net_pnl"] - benchmark_cost_summary["stressed_net_pnl"],
                    2,
                ),
                "trade_count_delta": matrix_cost_summary["trade_count"] - benchmark_cost_summary["trade_count"],
            },
            "cost_assumptions": {
                "commission_per_contract": self.config.replay_execution.commission_per_contract,
                "exchange_fee_per_contract": self.config.replay_execution.exchange_fee_per_contract,
                "market_slippage_ticks": self.config.replay_execution.market_slippage_ticks,
                "stress_slippage_ticks": self.config.replay_execution.stress_slippage_ticks,
                "limit_fill_penalty_ticks": self.config.replay_execution.limit_fill_penalty_ticks,
            },
            "deflated_sharpe_ratio_approx": approx_dsr,
            "deflated_sharpe_ratio_is_approximation": True,
            "synthetic_quotes_detected": synthetic_quotes_detected,
            "decision_ready": acceptance["decision_ready"],
            "decision_ready_reason": acceptance["decision_ready_reason"],
            "walk_forward": walk_forward,
            "acceptance": acceptance,
        }
        set_state(replay_summary=summary)

        return ReplayResult(path=str(replay_path), events=processed, segments=walk_forward, summary=summary)

    def _load_events(self, replay_path: Path) -> Iterable[MarketData]:
        """Yield market data snapshots from a replay file."""
        if replay_path.suffix.lower() == ".jsonl":
            with replay_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    yield self._event_to_market_data(json.loads(line))
            return

        frame = pd.read_csv(replay_path)
        for _, row in frame.iterrows():
            yield self._event_to_market_data(row.to_dict())

    def _event_to_market_data(self, raw: dict) -> MarketData:
        """Map raw replay event payload into MarketData."""
        timestamp = pd.Timestamp(raw.get("timestamp") or raw.get("time"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize("UTC")
        else:
            timestamp = timestamp.tz_convert("UTC")

        def has_value(value: object) -> bool:
            return value is not None and not pd.isna(value) and value != ""

        def read_float(*keys: str, default: float = 0.0) -> float:
            for key in keys:
                value = raw.get(key)
                if has_value(value):
                    return float(value)
            return float(default)

        def read_int(*keys: str, default: int = 0) -> int:
            for key in keys:
                value = raw.get(key)
                if has_value(value):
                    return int(float(value))
            return int(default)

        last = read_float("last", "price", "close", default=0.0)
        has_bid = has_value(raw.get("bid"))
        has_ask = has_value(raw.get("ask"))
        bid = read_float("bid", default=last - 0.25)
        ask = read_float("ask", default=last + 0.25)

        configured_mode = str(self.config.replay_execution.volume_mode or "auto").strip().lower()
        raw_mode = str(raw.get("volume_mode", raw.get("volumeMode", configured_mode)) or configured_mode).strip().lower()
        volume_field = raw.get("volume")
        size_field = raw.get("size")
        if raw_mode == "per_event":
            volume = int(float(volume_field)) if has_value(volume_field) else read_int("size", default=0)
            volume_is_cumulative = False
        elif raw_mode == "cumulative":
            volume = int(float(volume_field)) if has_value(volume_field) else read_int("size", default=0)
            volume_is_cumulative = True
        else:
            if has_value(volume_field):
                volume = int(float(volume_field))
                volume_is_cumulative = True
            else:
                volume = int(float(size_field)) if has_value(size_field) else 0
                volume_is_cumulative = False

        return MarketData(
            symbol=str(raw.get("symbol", "ES")),
            bid=bid,
            ask=ask,
            last=last,
            volume=volume,
            volume_is_cumulative=volume_is_cumulative,
            quote_is_synthetic=not (has_bid and has_ask),
            bid_size=read_float("bid_size", "bidSize", default=0.0),
            ask_size=read_float("ask_size", "askSize", default=0.0),
            last_size=read_float("last_size", "lastSize", "size", default=0.0),
            trade_side=str(raw.get("trade_side", raw.get("tradeSide", "")) or ""),
            latency_ms=read_int("latency_ms", "latencyMs", default=0),
            timestamp=timestamp.to_pydatetime(),
        )

    def _run_benchmark_portfolio(self, bars: pd.DataFrame) -> list[TradeRecord]:
        """Replay a simple per-zone benchmark portfolio over the same bars."""
        if bars.empty or not self.config.validation.benchmarks_enabled:
            return []

        scheduler = HotZoneScheduler(self.config)
        strategies = {
            "ORB": ORBStrategy(self.config),
            "VWAP_TREND": VWAPTrendStrategy(self.config),
            "VWAP_MR": VWAPMeanReversionStrategy(self.config),
            "FLATTEN_ONLY": FlattenStrategy(self.config),
        }
        for strategy in strategies.values():
            strategy.reset()

        current_position = 0
        entry_price = 0.0
        entry_time = None
        stop_loss: Optional[float] = None
        take_profit: Optional[float] = None
        active_zone = "inactive"
        active_strategy = "FLATTEN_ONLY"
        trades: list[TradeRecord] = []

        for idx in range(len(bars)):
            frame = bars.iloc[: idx + 1]
            current_time = frame.index[-1]
            current_bar = frame.iloc[-1]
            zone = scheduler.get_current_zone(current_time=current_time.to_pydatetime())

            if current_position != 0:
                exit_price, exit_reason = self._benchmark_exit_price(current_bar, current_position, stop_loss, take_profit)
                if exit_price is not None and entry_time is not None:
                    trades.append(
                        self._build_trade_record(
                            entry_time=entry_time,
                            exit_time=current_time.to_pydatetime(),
                            direction=1 if current_position > 0 else -1,
                            contracts=abs(current_position),
                            entry_price=entry_price,
                            exit_price=exit_price,
                            zone=active_zone,
                            strategy=active_strategy,
                            regime="BENCHMARK",
                            event_tags=[exit_reason],
                        )
                    )
                    current_position = 0
                    entry_price = 0.0
                    entry_time = None
                    stop_loss = None
                    take_profit = None
                    continue

            if zone is None and current_position != 0 and entry_time is not None:
                trades.append(
                    self._build_trade_record(
                        entry_time=entry_time,
                        exit_time=current_time.to_pydatetime(),
                        direction=1 if current_position > 0 else -1,
                        contracts=abs(current_position),
                        entry_price=entry_price,
                        exit_price=float(current_bar["close"]),
                        zone=active_zone,
                        strategy=active_strategy,
                        regime="BENCHMARK",
                        event_tags=["outside_zone_flatten"],
                    )
                )
                current_position = 0
                entry_price = 0.0
                entry_time = None
                stop_loss = None
                take_profit = None
                continue

            zone_name = zone.name if zone else "Close-Scalp"
            strategy_name = self.config.validation.benchmark_zone_strategies.get(zone_name, "FLATTEN_ONLY")
            strategy = strategies.get(strategy_name, strategies["FLATTEN_ONLY"])
            signal = strategy.compute_signal(frame, current_position, zone)
            if signal is None:
                continue

            price = float(signal.price or current_bar["close"])
            if signal.direction == SignalDirection.FLAT:
                if current_position != 0 and entry_time is not None:
                    trades.append(
                        self._build_trade_record(
                            entry_time=entry_time,
                            exit_time=current_time.to_pydatetime(),
                            direction=1 if current_position > 0 else -1,
                            contracts=abs(current_position),
                            entry_price=entry_price,
                            exit_price=price,
                            zone=active_zone,
                            strategy=active_strategy,
                            regime="BENCHMARK",
                            event_tags=[signal.reason or "signal_flat"],
                        )
                    )
                    current_position = 0
                    entry_price = 0.0
                    entry_time = None
                    stop_loss = None
                    take_profit = None
                continue

            if current_position == 0 and signal.direction in {SignalDirection.LONG, SignalDirection.SHORT}:
                current_position = signal.contracts if signal.direction == SignalDirection.LONG else -signal.contracts
                entry_price = price
                entry_time = current_time.to_pydatetime()
                stop_loss = signal.stop_loss
                take_profit = signal.take_profit
                active_zone = zone.name if zone else zone_name
                active_strategy = strategy_name

        if current_position != 0 and entry_time is not None and not bars.empty:
            last_time = bars.index[-1].to_pydatetime()
            last_close = float(bars.iloc[-1]["close"])
            trades.append(
                self._build_trade_record(
                    entry_time=entry_time,
                    exit_time=last_time,
                    direction=1 if current_position > 0 else -1,
                    contracts=abs(current_position),
                    entry_price=entry_price,
                    exit_price=last_close,
                    zone=active_zone,
                    strategy=active_strategy,
                    regime="BENCHMARK",
                    event_tags=["end_of_replay"],
                )
            )

        return trades

    def _benchmark_exit_price(
        self,
        current_bar: pd.Series,
        current_position: int,
        stop_loss: Optional[float],
        take_profit: Optional[float],
    ) -> tuple[Optional[float], str]:
        """Return a conservative stop/target exit for the benchmark portfolio."""
        low = float(current_bar["low"])
        high = float(current_bar["high"])
        if current_position > 0:
            if stop_loss is not None and low <= stop_loss:
                return float(stop_loss), "stop_loss"
            if take_profit is not None and high >= take_profit:
                return float(take_profit), "take_profit"
        else:
            if stop_loss is not None and high >= stop_loss:
                return float(stop_loss), "stop_loss"
            if take_profit is not None and low <= take_profit:
                return float(take_profit), "take_profit"
        return None, ""

    def _build_trade_record(
        self,
        entry_time,
        exit_time,
        direction: int,
        contracts: int,
        entry_price: float,
        exit_price: float,
        zone: str,
        strategy: str,
        regime: str,
        event_tags: list[str],
    ) -> TradeRecord:
        multiplier = 50
        pnl = (exit_price - entry_price) * multiplier * contracts if direction > 0 else (entry_price - exit_price) * multiplier * contracts
        return TradeRecord(
            entry_time=entry_time,
            exit_time=exit_time,
            direction=direction,
            contracts=contracts,
            entry_price=entry_price,
            exit_price=exit_price,
            pnl=round(pnl, 2),
            zone=zone,
            strategy=strategy,
            regime=regime,
            event_tags=event_tags,
        )

    def _costed_trade_summary(self, trades: list[TradeRecord], strategy_name: str) -> dict:
        """Summarize trades with fees and conservative slippage stress."""
        fee_per_contract = self.config.replay_execution.commission_per_contract + self.config.replay_execution.exchange_fee_per_contract
        limit_penalty_value = self.config.replay_execution.limit_fill_penalty_ticks * 12.5
        extra_stress_ticks = max(
            float(self.config.replay_execution.stress_slippage_ticks) - float(self.config.replay_execution.market_slippage_ticks),
            0.0,
        )
        stress_penalty_value = extra_stress_ticks * 12.5

        zone_stats: dict[str, dict[str, float]] = {}
        regime_stats: dict[str, dict[str, float]] = {}
        event_tag_stats: dict[str, dict[str, float]] = {}
        gross_pnl = 0.0
        total_fees = 0.0
        total_limit_penalty = 0.0
        total_stress_penalty = 0.0

        for trade in trades:
            gross = float(trade.pnl)
            contracts = max(int(trade.contracts), 1)
            fees = fee_per_contract * contracts
            limit_penalty = limit_penalty_value * contracts
            stress_penalty = stress_penalty_value * contracts
            net = gross - fees - limit_penalty
            stressed_net = net - stress_penalty

            gross_pnl += gross
            total_fees += fees
            total_limit_penalty += limit_penalty
            total_stress_penalty += stress_penalty

            for bucket, key in (
                (zone_stats, trade.zone or "Unknown"),
                (regime_stats, trade.regime or "Unknown"),
            ):
                stats = bucket.setdefault(key, {"trades": 0, "gross_pnl": 0.0, "net_pnl": 0.0, "stressed_net_pnl": 0.0})
                stats["trades"] += 1
                stats["gross_pnl"] += gross
                stats["net_pnl"] += net
                stats["stressed_net_pnl"] += stressed_net
            for tag in trade.event_tags or ["none"]:
                stats = event_tag_stats.setdefault(tag, {"trades": 0, "gross_pnl": 0.0, "net_pnl": 0.0, "stressed_net_pnl": 0.0})
                stats["trades"] += 1
                stats["gross_pnl"] += gross
                stats["net_pnl"] += net
                stats["stressed_net_pnl"] += stressed_net

        net_pnl = gross_pnl - total_fees - total_limit_penalty
        stressed_net_pnl = net_pnl - total_stress_penalty
        return {
            "strategy": strategy_name,
            "trade_count": len(trades),
            "gross_pnl": round(gross_pnl, 2),
            "fees": round(total_fees, 2),
            "limit_fill_penalty": round(total_limit_penalty, 2),
            "stress_slippage_penalty": round(total_stress_penalty, 2),
            "net_pnl": round(net_pnl, 2),
            "stressed_net_pnl": round(stressed_net_pnl, 2),
            "avg_gross_trade": round(gross_pnl / len(trades), 2) if trades else 0.0,
            "avg_net_trade": round(net_pnl / len(trades), 2) if trades else 0.0,
            "zone_stats": {key: self._rounded_stats(value) for key, value in zone_stats.items()},
            "regime_stats": {key: self._rounded_stats(value) for key, value in regime_stats.items()},
            "event_tag_stats": {key: self._rounded_stats(value) for key, value in event_tag_stats.items()},
        }

    def _walk_forward_segments(
        self,
        bars: pd.DataFrame,
        matrix_trades: list[TradeRecord],
        benchmark_trades: list[TradeRecord],
    ) -> list[dict]:
        """Build fixed chronological train/test windows with frozen weights."""
        if bars.empty:
            return []

        train_bars = max(int(self.config.validation.walk_forward_train_bars), 1)
        test_bars = max(int(self.config.validation.walk_forward_test_bars), 1)
        if len(bars) <= train_bars:
            return []

        segments: list[dict] = []
        start = 0
        segment_number = 1
        while start + train_bars < len(bars):
            train_end_idx = start + train_bars - 1
            test_end_idx = min(train_end_idx + test_bars, len(bars) - 1)
            test_start = bars.index[train_end_idx + 1]
            test_end = bars.index[test_end_idx]
            matrix_window = [trade for trade in matrix_trades if self._trade_in_window(trade, test_start, test_end)]
            benchmark_window = [trade for trade in benchmark_trades if self._trade_in_window(trade, test_start, test_end)]
            matrix_summary = self._costed_trade_summary(matrix_window, strategy_name="WEIGHTED_SCORE_MATRIX")
            benchmark_summary = self._costed_trade_summary(benchmark_window, strategy_name="BENCHMARK_PORTFOLIO")
            segments.append(
                {
                    "segment": segment_number,
                    "calibration": "weights_frozen",
                    "train_start": bars.index[start].isoformat(),
                    "train_end": bars.index[train_end_idx].isoformat(),
                    "test_start": test_start.isoformat(),
                    "test_end": test_end.isoformat(),
                    "matrix": matrix_summary,
                    "benchmark": benchmark_summary,
                    "matrix_positive": matrix_summary["net_pnl"] > 0,
                }
            )
            segment_number += 1
            start += test_bars
            if test_end_idx >= len(bars) - 1:
                break
        return segments

    def _acceptance_gates(
        self,
        matrix_cost_summary: dict,
        benchmark_cost_summary: dict,
        walk_forward: list[dict],
        synthetic_quotes_detected: bool,
    ) -> dict:
        """Apply fixed promotion gates for the current matrix version."""
        execution_state = get_state().execution or {}
        zone_stats = matrix_cost_summary.get("zone_stats", {})
        total_positive_zone_pnl = sum(max(stats.get("net_pnl", 0.0), 0.0) for stats in zone_stats.values())
        dominant_zone_share = 1.0
        if total_positive_zone_pnl > 0:
            dominant_zone_share = max(max(stats.get("net_pnl", 0.0), 0.0) for stats in zone_stats.values()) / total_positive_zone_pnl

        positive_window_ratio = (
            sum(1 for segment in walk_forward if segment["matrix_positive"]) / len(walk_forward)
            if walk_forward
            else 0.0
        )
        max_fill_drift_ticks = float(self.config.validation.max_prac_fill_drift_ticks)
        fill_drift = execution_state.get("fill_drift_ticks")
        fill_drift_ok = fill_drift is None or abs(float(fill_drift)) <= max_fill_drift_ticks
        decision_ready = not (
            synthetic_quotes_detected and self.config.validation.synthetic_quote_policy == "reject"
        )
        decision_ready_reason = "ok"
        if not decision_ready:
            decision_ready_reason = "synthetic_quotes_rejected"

        gates = {
            "decision_ready": decision_ready,
            "decision_ready_reason": decision_ready_reason,
            "benchmark_outperformance": matrix_cost_summary["net_pnl"] > benchmark_cost_summary["net_pnl"],
            "stress_survival_ok": matrix_cost_summary["stressed_net_pnl"] >= 0,
            "zone_diversity_ok": dominant_zone_share <= float(self.config.validation.max_zone_pnl_share),
            "dominant_zone_share": round(dominant_zone_share, 4),
            "walk_forward_ok": positive_window_ratio >= float(self.config.validation.min_positive_test_window_ratio),
            "positive_test_window_ratio": round(positive_window_ratio, 4),
            "prac_observability_ok": (
                int(execution_state.get("protection_failures", 0)) == 0
                and int(execution_state.get("fail_safe_count", 0)) == 0
                and fill_drift_ok
            ),
        }
        gates["promotable"] = decision_ready and all(
            gates[name]
            for name in (
                "benchmark_outperformance",
                "stress_survival_ok",
                "zone_diversity_ok",
                "walk_forward_ok",
                "prac_observability_ok",
            )
        )
        return gates

    def _trade_in_window(self, trade: TradeRecord, start: pd.Timestamp, end: pd.Timestamp) -> bool:
        exit_time = pd.Timestamp(trade.exit_time)
        if exit_time.tzinfo is None:
            exit_time = exit_time.tz_localize(start.tz)
        else:
            exit_time = exit_time.tz_convert(start.tz)
        return start <= exit_time <= end

    def _rounded_stats(self, stats: dict[str, float]) -> dict[str, float]:
        return {
            key: round(value, 2) if isinstance(value, float) else value
            for key, value in stats.items()
        }

    def _deflated_sharpe_ratio(self, pnl: list[float], trials: int) -> float:
        """Return a lightweight DSR-style score for replay reporting."""
        if len(pnl) < 2:
            return 0.0
        avg = mean(pnl)
        std = pstdev(pnl)
        if std == 0:
            return 0.0
        sharpe = avg / std * math.sqrt(len(pnl))
        trial_penalty = math.sqrt(max(math.log(max(trials, 2)), 1.0))
        return round(sharpe / trial_penalty, 4)

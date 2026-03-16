"""Weighted score matrix alpha engine."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from src.config import Config
from src.engine.event_provider import EventContext
from src.engine.market_context import OrderFlowSnapshot
from src.engine.regime import DeterministicRegimeClassifier, RegimeState
from src.engine.risk_manager import RiskState
from src.engine.scheduler import ZoneInfo, ZoneState
from src.indicators import atr, build_volume_profile, rsi, session_labels, session_vwap_bands
from src.market import MarketData


def _clip(value: float, lower: float = -2.0, upper: float = 2.0) -> float:
    return max(lower, min(upper, float(value)))


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if np.isnan(result):
        return default
    return result


@dataclass
class FeatureSnapshot:
    """Normalized directional feature set for the score matrix."""

    zone_name: str
    current_price: float
    atr_value: float
    long_features: Dict[str, float]
    short_features: Dict[str, float]
    flat_features: Dict[str, float]
    signed_features: Dict[str, float]
    diagnostics: Dict[str, float] = field(default_factory=dict)
    mean_reversion_ready_long: bool = False
    mean_reversion_ready_short: bool = False
    execution_tradeable: bool = True
    active_session: str = "RTH"
    regime_state: str = RegimeState.RANGE.value
    regime_reason: str = ""
    event_tags: list[str] = field(default_factory=list)
    capabilities: Dict[str, bool] = field(default_factory=dict)


@dataclass
class MatrixDecision:
    """Decision emitted by the weighted score matrix."""

    zone_name: str
    action: str
    reason: str
    long_score: float
    short_score: float
    flat_bias: float
    active_vetoes: list[str]
    feature_snapshot: FeatureSnapshot
    execution_tradeable: bool
    size_fraction: float = 0.0
    side: Optional[str] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    max_hold_minutes: int = 0


class DecisionMatrixEvaluator:
    """Evaluate the weighted score matrix for the current zone."""

    LONG_SHORT_FEATURES = {
        "atr_state",
        "vwap_state",
        "trend_state",
        "range_state",
        "orb_break",
        "opening_drive",
        "pullback_quality",
        "breakout_failure_absent",
        "vwap_distance",
        "extension_state",
        "wick_rejection",
        "execution_state",
        "event_state",
        "rth_vwap_distance_z",
        "eth_vwap_distance_z",
        "rth_vwap_slope",
        "eth_vwap_slope",
        "poc_distance",
        "value_area_position",
        "value_acceptance",
        "value_rejection",
        "ofi_zscore",
        "quote_rate_state",
        "spread_regime",
        "volume_pace",
    }
    FLAT_FEATURES = {"atr_state", "event_state", "execution_state", "range_state", "trend_state", "spread_regime", "regime_stress"}
    SUPPORTED_VETO_KEYS = {
        "max_atr_accel",
        "max_spread_ticks",
        "reject_orb_middle",
        "require_execution_tradeable",
        "flat_vwap_threshold",
        "flat_ema_threshold",
        "max_ema_slope",
        "max_atr_percentile",
        "min_minutes_remaining",
        "require_mean_reversion_confirmation",
        "flatten_only",
        "blocked_regimes",
    }

    def __init__(self, config: Config):
        self.config = config
        self.alpha = config.alpha
        self.strategy = config.strategy
        self.risk = config.risk
        self.validation = config.validation
        self.sessions = config.sessions
        self.volume_profile = config.volume_profile
        self.regime_classifier = DeterministicRegimeClassifier(config.regime)
        self._validate_configuration()

    def _effective_zone(self, zone: Optional[ZoneInfo], bars: pd.DataFrame) -> Optional[ZoneInfo]:
        if zone is not None:
            return zone
        if not self.strategy.trade_outside_hotzones:
            return None
        current_time = pd.Timestamp(bars.index[-1]) if not bars.empty else pd.Timestamp.utcnow()
        if current_time.tzinfo is None:
            current_time = current_time.tz_localize(self.sessions.timezone)
        return ZoneInfo(
            name="Outside",
            state=ZoneState.ACTIVE,
            start_time=(current_time - pd.Timedelta(minutes=self._zone_hold_limit("Outside"))).to_pydatetime(),
            end_time=(current_time + pd.Timedelta(minutes=self._zone_hold_limit("Outside"))).to_pydatetime(),
            minutes_remaining=float(self._zone_hold_limit("Outside")),
            is_first_bar=False,
            is_last_bar=False,
        )

    def _validate_configuration(self) -> None:
        for zone_name, weight_map in self.alpha.zone_weights.items():
            for side, features in weight_map.items():
                supported = self.FLAT_FEATURES if side == "flat" else self.LONG_SHORT_FEATURES
                unknown = set(features) - supported
                if unknown:
                    raise ValueError(f"Unsupported alpha feature(s) for {zone_name}/{side}: {sorted(unknown)}")
                if side in {"long", "short"} and len(features) > max(int(self.validation.max_features_per_side), 1):
                    raise ValueError(
                        f"Too many weighted features for {zone_name}/{side}: {len(features)} > {self.validation.max_features_per_side}"
                    )
        for zone_name, vetoes in self.alpha.zone_vetoes.items():
            unknown = set(vetoes) - self.SUPPORTED_VETO_KEYS
            if unknown:
                raise ValueError(f"Unsupported veto key(s) for {zone_name}: {sorted(unknown)}")

    def _session_slice(self, bars: pd.DataFrame, start_hour: int, start_minute: int) -> pd.DataFrame:
        if bars.empty:
            return bars
        labels = session_labels(pd.DatetimeIndex(bars.index), start_hour, start_minute)
        current_label = labels.iloc[-1]
        return bars.loc[labels == current_label]

    def extract_features(
        self,
        bars: pd.DataFrame,
        zone: Optional[ZoneInfo],
        market_data: Optional[MarketData],
        event_context: Optional[EventContext],
        flow_snapshot: Optional[OrderFlowSnapshot],
        current_position: int,
    ) -> FeatureSnapshot:
        """Extract reusable normalized features from the latest market state."""
        zone_name = zone.name if zone else "Outside"
        event_context = event_context or EventContext()
        flow_snapshot = flow_snapshot or OrderFlowSnapshot()

        if bars.empty:
            return FeatureSnapshot(
                zone_name=zone_name,
                current_price=0.0,
                atr_value=0.0,
                long_features={},
                short_features={},
                flat_features={},
                signed_features={},
                diagnostics={},
                execution_tradeable=False,
                event_tags=list(event_context.active_tags),
            )

        close = bars["close"].astype(float)
        high = bars["high"].astype(float)
        low = bars["low"].astype(float)
        current_price = _safe_float(close.iloc[-1])
        recent_open = _safe_float(bars["open"].iloc[-1])

        if len(bars) >= self.strategy.atr_length:
            atr_series = atr(high, low, close, self.strategy.atr_length)
            atr_value = _safe_float(atr_series.iloc[-1], default=max((high.iloc[-1] - low.iloc[-1]), 0.25))
            atr_window = atr_series.dropna().tail(50)
        else:
            atr_series = pd.Series([high.iloc[-1] - low.iloc[-1]], index=[bars.index[-1]], dtype=float)
            atr_value = max(_safe_float(high.iloc[-1] - low.iloc[-1], 0.25), 0.25)
            atr_window = atr_series

        atr_percentile = 0.5
        if not atr_window.empty and atr_value > 0:
            atr_percentile = float((atr_window <= atr_value).sum()) / float(len(atr_window))
        atr_non_na = atr_series.dropna()
        prev_atr = _safe_float(atr_non_na.iloc[-2], atr_value) if len(atr_non_na) >= 2 else atr_value
        atr_accel = ((atr_value - prev_atr) / prev_atr) if prev_atr else 0.0
        atr_ratio = atr_value / max(_safe_float(atr_window.mean(), atr_value), 0.25)

        rth_bars = self._session_slice(bars, self.sessions.rth_start_hour, self.sessions.rth_start_minute)
        eth_bars = self._session_slice(bars, self.sessions.eth_reset_hour, self.sessions.eth_reset_minute)
        active_session = "ETH" if zone_name == "Pre-Open" else "RTH"
        active_session_bars = eth_bars if active_session == "ETH" else rth_bars

        rth_metrics = session_vwap_bands(rth_bars, self.sessions.rth_start_hour, self.sessions.rth_start_minute, self.strategy.vwap_source)
        eth_metrics = session_vwap_bands(eth_bars, self.sessions.eth_reset_hour, self.sessions.eth_reset_minute, self.strategy.vwap_source)

        rth_vwap = _safe_float(rth_metrics.vwap.iloc[-1], current_price) if not rth_metrics.vwap.empty else current_price
        eth_vwap = _safe_float(eth_metrics.vwap.iloc[-1], current_price) if not eth_metrics.vwap.empty else current_price
        rth_prev = _safe_float(rth_metrics.vwap.iloc[-2], rth_vwap) if len(rth_metrics.vwap) >= 2 else rth_vwap
        eth_prev = _safe_float(eth_metrics.vwap.iloc[-2], eth_vwap) if len(eth_metrics.vwap) >= 2 else eth_vwap
        rth_sigma = max(_safe_float(rth_metrics.sigma.iloc[-1], atr_value / 2.0), 0.25) if not rth_metrics.sigma.empty else max(atr_value / 2.0, 0.25)
        eth_sigma = max(_safe_float(eth_metrics.sigma.iloc[-1], atr_value / 2.0), 0.25) if not eth_metrics.sigma.empty else max(atr_value / 2.0, 0.25)

        rth_vwap_distance = _clip((current_price - rth_vwap) / rth_sigma)
        eth_vwap_distance = _clip((current_price - eth_vwap) / eth_sigma)
        rth_vwap_slope = _clip((rth_vwap - rth_prev) / max(atr_value, 0.25))
        eth_vwap_slope = _clip((eth_vwap - eth_prev) / max(atr_value, 0.25))

        active_profile = build_volume_profile(
            active_session_bars.tail(240),
            tick_size=self.volume_profile.tick_size,
            value_area_pct=self.volume_profile.value_area_pct,
            source=self.volume_profile.source,
        )
        profile_width = max(active_profile.value_area_width, self.volume_profile.tick_size)
        value_area_position = _clip((current_price - active_profile.poc) / profile_width)
        poc_distance_long = _clip((current_price - active_profile.poc) / max(atr_value, 0.25))
        poc_distance_short = _clip((active_profile.poc - current_price) / max(atr_value, 0.25))

        recent_closes = active_session_bars["close"].tail(3) if not active_session_bars.empty else close.tail(3)
        value_acceptance_long = 1.5 if len(recent_closes) >= 2 and (recent_closes >= active_profile.vah).tail(2).all() else _clip(value_area_position + max(rth_vwap_slope, eth_vwap_slope))
        value_acceptance_short = 1.5 if len(recent_closes) >= 2 and (recent_closes <= active_profile.val).tail(2).all() else _clip((-value_area_position) + max(-rth_vwap_slope, -eth_vwap_slope))
        value_rejection_long = 1.2 if _safe_float(low.iloc[-1], current_price) < active_profile.val and current_price > active_profile.val else _clip(-value_area_position)
        value_rejection_short = 1.2 if _safe_float(high.iloc[-1], current_price) > active_profile.vah and current_price < active_profile.vah else _clip(value_area_position)

        ema = close.ewm(span=self.strategy.trend_ma_length, adjust=False).mean()
        ema_value = _safe_float(ema.iloc[-1], current_price)
        ema_prev = _safe_float(ema.iloc[-2], ema_value) if len(ema) >= 2 else ema_value
        ema_slope = _clip((ema_value - ema_prev) / max(atr_value, 0.25))
        price_vs_ema = _clip((current_price - ema_value) / max(atr_value, 0.25))

        recent_high_diff = high.diff().tail(5).fillna(0)
        recent_low_diff = low.diff().tail(5).fillna(0)
        structure_bias = _clip(
            (recent_high_diff.gt(0).sum() - recent_high_diff.lt(0).sum()) / 2.0
            + (recent_low_diff.gt(0).sum() - recent_low_diff.lt(0).sum()) / 2.0
        )

        range_window = bars.tail(8)
        bar_ranges = (range_window["high"] - range_window["low"]).replace(0, np.nan)
        compression_raw = 1.0 - (_safe_float(bar_ranges.mean(), atr_value) / max(atr_value * 1.5, 0.25))
        compression_score = _clip(compression_raw)
        inside_count = 0
        for idx in range(1, len(range_window)):
            if range_window["high"].iloc[idx] <= range_window["high"].iloc[idx - 1] and range_window["low"].iloc[idx] >= range_window["low"].iloc[idx - 1]:
                inside_count += 1
        inside_bar_density = inside_count / max(len(range_window) - 1, 1)
        expansion_failure = _clip((_safe_float(bar_ranges.iloc[-1], 0.0) - _safe_float(bar_ranges.mean(), 0.0)) / max(atr_value, 0.25))

        zone_bars = bars if zone is None else bars.loc[bars.index >= zone.start_time]
        if zone_bars.empty:
            zone_bars = bars
        orb_bars = zone_bars.head(self.strategy.orb_range_minutes)
        orb_high = _safe_float(orb_bars["high"].max(), current_price)
        orb_low = _safe_float(orb_bars["low"].min(), current_price)
        orb_span = max(orb_high - orb_low, 0.25)
        orb_mid = (orb_high + orb_low) / 2.0
        orb_position = _clip((current_price - orb_mid) / orb_span * 2.0)
        orb_break_long = _clip((current_price - orb_high) / max(atr_value, 0.25) + max(ema_slope, 0.0))
        orb_break_short = _clip((orb_low - current_price) / max(atr_value, 0.25) + max(-ema_slope, 0.0))
        opening_drive = _clip((current_price - _safe_float(zone_bars["open"].iloc[0], recent_open)) / max(atr_value, 0.25))

        bb_basis = close.rolling(20).mean()
        bb_std = close.rolling(20).std(ddof=0).fillna(0)
        bb_upper = _safe_float((bb_basis + (bb_std * self.strategy.mr_band_deviation)).iloc[-1], current_price)
        bb_lower = _safe_float((bb_basis - (bb_std * self.strategy.mr_band_deviation)).iloc[-1], current_price)
        rsi_series = rsi(close, self.strategy.mr_rsi_length)
        rsi_value = _safe_float(rsi_series.iloc[-1], 50.0)
        lower_penetration = _clip((bb_lower - current_price) / max(atr_value, 0.25))
        upper_penetration = _clip((current_price - bb_upper) / max(atr_value, 0.25))
        wick_lower = min(recent_open, current_price) - _safe_float(low.iloc[-1], current_price)
        wick_upper = _safe_float(high.iloc[-1], current_price) - max(recent_open, current_price)
        bar_range = max(_safe_float(high.iloc[-1], current_price) - _safe_float(low.iloc[-1], current_price), 0.25)
        wick_rejection_long = _clip((wick_lower / bar_range) * 2.0)
        wick_rejection_short = _clip((wick_upper / bar_range) * 2.0)

        pullback_long = _clip(max(price_vs_ema, 0.0) + max(rth_vwap_slope, 0.0) - max((current_price - _safe_float(high.tail(4).max(), current_price)) / max(atr_value, 0.25), -2.0))
        pullback_short = _clip(max(-price_vs_ema, 0.0) + max(-rth_vwap_slope, 0.0) - max((_safe_float(low.tail(4).min(), current_price) - current_price) / max(atr_value, 0.25), -2.0))
        breakout_failure_absent = _clip(abs(orb_position) - 0.4)

        spread = market_data.spread if market_data else 0.25
        quote_age = 0.0
        if market_data is not None and market_data.timestamp is not None:
            market_ts = pd.Timestamp(market_data.timestamp)
            market_ts = market_ts.tz_localize("UTC") if market_ts.tzinfo is None else market_ts.tz_convert("UTC")
            current_ts = pd.Timestamp(bars.index[-1])
            current_ts = current_ts.tz_localize("UTC") if current_ts.tzinfo is None else current_ts.tz_convert("UTC")
            quote_age = max((current_ts - market_ts).total_seconds(), 0.0)

        spread_proxy = _clip(2.0 - (spread / 0.25))
        freshness_proxy = _clip(2.0 - (quote_age / 2.0))
        bar_range_spread_ratio = _clip((bar_range / max(spread, 0.25)) / 4.0)
        slippage_proxy = _clip(2.0 - (spread / 0.5))
        execution_tradeable = spread <= (self.config.order_execution.max_slippage_ticks * 0.25) and quote_age <= 5.0
        if market_data is None:
            execution_tradeable = True
            freshness_proxy = 1.0
            spread_proxy = 1.0
            slippage_proxy = 1.0
            bar_range_spread_ratio = _clip(bar_range / 1.0)

        atr_state = _clip(1.2 - (atr_percentile * 2.0) - max(atr_accel, 0.0))
        trend_long = _clip(price_vs_ema + ema_slope + structure_bias / 2.0)
        trend_short = _clip((-price_vs_ema) + (-ema_slope) + (-structure_bias) / 2.0)
        vwap_long = _clip(max(rth_vwap_distance, 0.0) + max(rth_vwap_slope, 0.0))
        vwap_short = _clip(max(-rth_vwap_distance, 0.0) + max(-rth_vwap_slope, 0.0))
        range_state = _clip((compression_score + inside_bar_density * 2.0 - max(expansion_failure, 0.0)) / 2.0)
        extension_long = _clip(lower_penetration + max((50.0 - rsi_value) / 12.5, 0.0))
        extension_short = _clip(upper_penetration + max((rsi_value - 50.0) / 12.5, 0.0))
        vwap_distance_long = _clip((rth_vwap - current_price) / max(atr_value, 0.25))
        vwap_distance_short = _clip((current_price - rth_vwap) / max(atr_value, 0.25))
        event_state = -2.0 if event_context.blackout_active else (0.25 if event_context.post_event_cooling else 0.75)
        execution_state = _clip((spread_proxy + freshness_proxy + bar_range_spread_ratio + slippage_proxy) / 4.0)

        mean_reversion_ready_long = lower_penetration > 0 and rsi_value <= self.strategy.mr_rsi_oversold and wick_rejection_long > 0.25
        mean_reversion_ready_short = upper_penetration > 0 and rsi_value >= self.strategy.mr_rsi_overbought and wick_rejection_short > 0.25

        snapshot = FeatureSnapshot(
            zone_name=zone_name,
            current_price=current_price,
            atr_value=atr_value,
            long_features={
                "atr_state": max(atr_state, -2.0),
                "vwap_state": vwap_long,
                "trend_state": trend_long,
                "range_state": range_state,
                "orb_break": orb_break_long,
                "opening_drive": max(opening_drive, -2.0),
                "pullback_quality": pullback_long,
                "breakout_failure_absent": breakout_failure_absent,
                "vwap_distance": vwap_distance_long,
                "extension_state": extension_long,
                "wick_rejection": wick_rejection_long,
                "execution_state": execution_state,
                "event_state": event_state,
                "rth_vwap_distance_z": max(rth_vwap_distance, -2.0),
                "eth_vwap_distance_z": max(eth_vwap_distance, -2.0),
                "rth_vwap_slope": max(rth_vwap_slope, -2.0),
                "eth_vwap_slope": max(eth_vwap_slope, -2.0),
                "poc_distance": poc_distance_long,
                "value_area_position": max(value_area_position, -2.0),
                "value_acceptance": value_acceptance_long,
                "value_rejection": value_rejection_long,
                "ofi_zscore": max(flow_snapshot.ofi_zscore, -2.0),
                "quote_rate_state": flow_snapshot.quote_rate_state,
                "spread_regime": flow_snapshot.spread_regime or spread_proxy,
                "volume_pace": flow_snapshot.volume_pace,
            },
            short_features={
                "atr_state": max(atr_state, -2.0),
                "vwap_state": vwap_short,
                "trend_state": trend_short,
                "range_state": range_state,
                "orb_break": orb_break_short,
                "opening_drive": max(-opening_drive, -2.0),
                "pullback_quality": pullback_short,
                "breakout_failure_absent": breakout_failure_absent,
                "vwap_distance": vwap_distance_short,
                "extension_state": extension_short,
                "wick_rejection": wick_rejection_short,
                "execution_state": execution_state,
                "event_state": event_state,
                "rth_vwap_distance_z": max(-rth_vwap_distance, -2.0),
                "eth_vwap_distance_z": max(-eth_vwap_distance, -2.0),
                "rth_vwap_slope": max(-rth_vwap_slope, -2.0),
                "eth_vwap_slope": max(-eth_vwap_slope, -2.0),
                "poc_distance": poc_distance_short,
                "value_area_position": max(-value_area_position, -2.0),
                "value_acceptance": value_acceptance_short,
                "value_rejection": value_rejection_short,
                "ofi_zscore": max(-flow_snapshot.ofi_zscore, -2.0),
                "quote_rate_state": flow_snapshot.quote_rate_state,
                "spread_regime": flow_snapshot.spread_regime or spread_proxy,
                "volume_pace": flow_snapshot.volume_pace,
            },
            flat_features={
                "atr_state": -atr_state,
                "event_state": -event_state,
                "execution_state": -execution_state,
                "range_state": abs(range_state),
                "trend_state": _clip(2.0 - max(trend_long, trend_short)),
                "spread_regime": _clip(2.0 - max(flow_snapshot.spread_regime or spread_proxy, -2.0)),
                "regime_stress": 0.0,
            },
            signed_features={
                "atr_percentile": atr_percentile,
                "atr_accel": _clip(atr_accel),
                "atr_ratio": atr_ratio,
                "ema_slope": ema_slope,
                "rth_vwap_slope": rth_vwap_slope,
                "eth_vwap_slope": eth_vwap_slope,
                "rth_vwap_distance": rth_vwap_distance,
                "eth_vwap_distance": eth_vwap_distance,
                "price_vs_ema": price_vs_ema,
                "orb_position": orb_position,
                "inside_bar_density": inside_bar_density,
                "quote_age_seconds": quote_age,
                "spread_ticks": spread / 0.25,
                "minutes_remaining": zone.minutes_remaining if zone else 0.0,
                "quote_rate_per_minute": flow_snapshot.quote_rate_per_minute,
                "ofi_zscore": flow_snapshot.ofi_zscore,
                "value_area_position": value_area_position,
            },
            diagnostics={
                "atr_value": atr_value,
                "rth_vwap": rth_vwap,
                "eth_vwap": eth_vwap,
                "rth_sigma": rth_sigma,
                "eth_sigma": eth_sigma,
                "ema": ema_value,
                "rsi": rsi_value,
                "orb_high": orb_high,
                "orb_low": orb_low,
                "orb_mid": orb_mid,
                "bar_range": bar_range,
                "spread": spread,
                "poc": active_profile.poc,
                "vah": active_profile.vah,
                "val": active_profile.val,
                "current_position": current_position,
            },
            mean_reversion_ready_long=mean_reversion_ready_long,
            mean_reversion_ready_short=mean_reversion_ready_short,
            execution_tradeable=execution_tradeable,
            active_session=active_session,
            event_tags=list(event_context.active_tags),
            capabilities={"trade_side_available": flow_snapshot.trade_side_available},
        )

        regime_snapshot = self.regime_classifier.classify(
            ema_slope=ema_slope,
            atr_ratio=atr_ratio,
            spread_ticks=spread / 0.25,
            quote_rate=flow_snapshot.quote_rate_per_minute,
            ofi_zscore=flow_snapshot.ofi_zscore,
            value_area_position=value_area_position,
            event_active=event_context.blackout_active,
            post_event_cooling=event_context.post_event_cooling,
        )
        snapshot.regime_state = regime_snapshot.state.value
        snapshot.regime_reason = regime_snapshot.reason
        snapshot.flat_features["regime_stress"] = 2.0 if regime_snapshot.state == RegimeState.STRESS else 0.0
        return snapshot

    def evaluate(
        self,
        bars: pd.DataFrame,
        zone: Optional[ZoneInfo],
        market_data: Optional[MarketData],
        risk_state: RiskState,
        blackout_active: bool,
        current_position: int,
        allow_entries: bool,
        current_entry_time: Optional[pd.Timestamp] = None,
        event_context: Optional[EventContext] = None,
        flow_snapshot: Optional[OrderFlowSnapshot] = None,
    ) -> MatrixDecision:
        """Evaluate matrix scores, vetoes, and the resulting action."""
        event_context = event_context or EventContext(blackout_active=blackout_active)
        zone = self._effective_zone(zone, bars)

        if zone is None:
            snapshot = self.extract_features(bars, zone, market_data, event_context, flow_snapshot, current_position)
            return MatrixDecision(
                zone_name="Outside",
                action="FLAT" if current_position else "NO_TRADE",
                reason="outside_zone",
                long_score=0.0,
                short_score=0.0,
                flat_bias=2.0,
                active_vetoes=["outside_zone"],
                feature_snapshot=snapshot,
                execution_tradeable=False,
            )

        snapshot = self.extract_features(bars, zone, market_data, event_context, flow_snapshot, current_position)
        weights = self.alpha.zone_weights.get(zone.name, {})
        long_score = round(sum(snapshot.long_features.get(name, 0.0) * weight for name, weight in weights.get("long", {}).items()), 4)
        short_score = round(sum(snapshot.short_features.get(name, 0.0) * weight for name, weight in weights.get("short", {}).items()), 4)
        flat_bias = round(sum(snapshot.flat_features.get(name, 0.0) * weight for name, weight in weights.get("flat", {}).items()), 4)

        vetoes = self._evaluate_vetoes(zone, snapshot, event_context.blackout_active)
        if risk_state == RiskState.CIRCUIT_BREAKER:
            vetoes.append("risk_circuit_breaker")
        elif risk_state == RiskState.REDUCED:
            vetoes.append("reduced_risk")

        dominant_side = "long" if long_score >= short_score else "short"
        dominant_score = long_score if dominant_side == "long" else short_score
        opposing_score = short_score if dominant_side == "long" else long_score
        score_gap = dominant_score - opposing_score

        if zone.state.value == "flatten_only" or self.alpha.zone_vetoes.get(zone.name, {}).get("flatten_only"):
            action = "FLAT" if current_position else "NO_TRADE"
            reason = "flatten_only_zone"
        elif current_position > 0:
            held_minutes = self._held_minutes(current_entry_time, bars.index[-1])
            if snapshot.regime_state == RegimeState.STRESS.value:
                action = "FLAT"
                reason = "stress_regime"
            elif dominant_side == "short" and (short_score - long_score) >= self.alpha.reverse_score_gap:
                action = "FLAT"
                reason = "opposite_score_dominance"
            elif long_score < self.alpha.exit_decay_score:
                action = "FLAT"
                reason = "matrix_decay"
            elif held_minutes >= self._zone_hold_limit(zone.name):
                action = "FLAT"
                reason = "time_stop"
            elif zone.is_last_bar or event_context.blackout_active or "risk_circuit_breaker" in vetoes:
                action = "FLAT"
                reason = "risk_or_zone_exit"
            else:
                action = "HOLD"
                reason = "long_still_valid"
        elif current_position < 0:
            held_minutes = self._held_minutes(current_entry_time, bars.index[-1])
            if snapshot.regime_state == RegimeState.STRESS.value:
                action = "FLAT"
                reason = "stress_regime"
            elif dominant_side == "long" and (long_score - short_score) >= self.alpha.reverse_score_gap:
                action = "FLAT"
                reason = "opposite_score_dominance"
            elif short_score < self.alpha.exit_decay_score:
                action = "FLAT"
                reason = "matrix_decay"
            elif held_minutes >= self._zone_hold_limit(zone.name):
                action = "FLAT"
                reason = "time_stop"
            elif zone.is_last_bar or event_context.blackout_active or "risk_circuit_breaker" in vetoes:
                action = "FLAT"
                reason = "risk_or_zone_exit"
            else:
                action = "HOLD"
                reason = "short_still_valid"
        elif (
            allow_entries
            and dominant_score >= self.alpha.min_entry_score
            and score_gap >= self.alpha.min_score_gap
            and dominant_score >= (flat_bias + self.alpha.flat_bias_buffer)
            and not [item for item in vetoes if item != "reduced_risk"]
        ):
            action = "LONG" if dominant_side == "long" else "SHORT"
            reason = f"{zone.name.lower().replace(' ', '_')}_{dominant_side}_matrix"
        else:
            action = "NO_TRADE"
            reason = "matrix_not_decisive"

        size_fraction = 0.0
        if action in {"LONG", "SHORT"}:
            size_fraction = 1.0 if dominant_score > self.alpha.full_size_score else 0.5
            if risk_state != RiskState.NORMAL:
                size_fraction = min(size_fraction, 0.5)

        side = "buy" if action == "LONG" else ("sell" if action == "SHORT" else None)
        stop_loss, take_profit = self._risk_targets(action, snapshot, zone.name)

        return MatrixDecision(
            zone_name=zone.name,
            action=action,
            reason=reason,
            long_score=long_score,
            short_score=short_score,
            flat_bias=flat_bias,
            active_vetoes=list(dict.fromkeys(vetoes)),
            feature_snapshot=snapshot,
            execution_tradeable=snapshot.execution_tradeable,
            size_fraction=size_fraction,
            side=side,
            stop_loss=stop_loss,
            take_profit=take_profit,
            max_hold_minutes=self._zone_hold_limit(zone.name),
        )

    def _evaluate_vetoes(self, zone: ZoneInfo, snapshot: FeatureSnapshot, blackout_active: bool) -> list[str]:
        """Apply hard entry vetoes for the current zone."""
        vetoes: list[str] = []
        rules = self.alpha.zone_vetoes.get(zone.name, {})
        signed = snapshot.signed_features

        if blackout_active:
            vetoes.append("event_blackout")
        if rules.get("require_execution_tradeable") and not snapshot.execution_tradeable:
            vetoes.append("execution_degraded")

        max_spread_ticks = rules.get("max_spread_ticks")
        if max_spread_ticks is not None and signed.get("spread_ticks", 0.0) > float(max_spread_ticks):
            vetoes.append("spread_too_wide")

        max_atr_percentile = rules.get("max_atr_percentile")
        if max_atr_percentile is not None and signed.get("atr_percentile", 0.0) > float(max_atr_percentile):
            vetoes.append("atr_percentile_too_high")

        max_atr_accel = rules.get("max_atr_accel")
        if max_atr_accel is not None and signed.get("atr_accel", 0.0) > float(max_atr_accel):
            vetoes.append("atr_spike_active")

        blocked_regimes = rules.get("blocked_regimes", [])
        if snapshot.regime_state in blocked_regimes:
            vetoes.append(f"regime_{snapshot.regime_state.lower()}")

        if zone.name == "Pre-Open" and rules.get("reject_orb_middle"):
            if abs(snapshot.signed_features.get("orb_position", 0.0)) <= 0.4:
                vetoes.append("inside_orb_middle")

        if zone.name == "Post-Open":
            if abs(signed.get("rth_vwap_slope", 0.0)) <= float(rules.get("flat_vwap_threshold", 0.08)) and abs(signed.get("ema_slope", 0.0)) <= float(rules.get("flat_ema_threshold", 0.08)):
                vetoes.append("trend_flat")

        if zone.name == "Midday":
            if abs(signed.get("ema_slope", 0.0)) > float(rules.get("max_ema_slope", 0.15)):
                vetoes.append("ema_slope_outside_range")
            if zone.minutes_remaining < float(rules.get("min_minutes_remaining", 10)):
                vetoes.append("zone_too_late")
            if snapshot.long_features.get("orb_break", 0.0) > 1.0 or snapshot.short_features.get("orb_break", 0.0) > 1.0:
                vetoes.append("breakout_follow_through_active")
            if rules.get("require_mean_reversion_confirmation") and not (snapshot.mean_reversion_ready_long or snapshot.mean_reversion_ready_short):
                vetoes.append("missing_mean_reversion_confirmation")

        return vetoes

    def _risk_targets(self, action: str, snapshot: FeatureSnapshot, zone_name: str) -> tuple[Optional[float], Optional[float]]:
        if action not in {"LONG", "SHORT"}:
            return None, None
        stop_distance = max(snapshot.atr_value * self.alpha.stop_loss_atr, 0.5)
        take_profit_distance = max(snapshot.atr_value * self.alpha.take_profit_atr.get(zone_name, 1.5), stop_distance)
        if action == "LONG":
            return snapshot.current_price - stop_distance, snapshot.current_price + take_profit_distance
        return snapshot.current_price + stop_distance, snapshot.current_price - take_profit_distance

    def _held_minutes(self, entry_time: Optional[pd.Timestamp], current_time: pd.Timestamp) -> float:
        if entry_time is None:
            return 0.0
        current_ts = pd.Timestamp(current_time)
        entry_ts = pd.Timestamp(entry_time)
        if current_ts.tzinfo is None and entry_ts.tzinfo is not None:
            current_ts = current_ts.tz_localize(entry_ts.tzinfo)
        elif entry_ts.tzinfo is None and current_ts.tzinfo is not None:
            entry_ts = entry_ts.tz_localize(current_ts.tzinfo)
        if current_ts.tzinfo is not None and entry_ts.tzinfo is not None:
            current_ts = current_ts.tz_convert("UTC")
            entry_ts = entry_ts.tz_convert("UTC")
        return (current_ts - entry_ts).total_seconds() / 60.0

    def _zone_hold_limit(self, zone_name: str) -> int:
        return int(self.alpha.max_hold_minutes.get(zone_name, 20))

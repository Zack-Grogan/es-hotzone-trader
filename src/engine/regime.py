"""Deterministic intraday regime classifier."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.config import RegimeConfig


class RegimeState(Enum):
    TREND = "TREND"
    RANGE = "RANGE"
    STRESS = "STRESS"


@dataclass
class RegimeSnapshot:
    """Current regime and its gating reason."""

    state: RegimeState
    confidence: float
    reason: str


class DeterministicRegimeClassifier:
    """A lightweight 3-state gate for the matrix engine."""

    def __init__(self, config: RegimeConfig):
        self.config = config

    def classify(
        self,
        *,
        ema_slope: float,
        atr_ratio: float,
        spread_ticks: float,
        quote_rate: float,
        ofi_zscore: float,
        value_area_position: float,
        event_active: bool,
        post_event_cooling: bool,
    ) -> RegimeSnapshot:
        if event_active or post_event_cooling:
            return RegimeSnapshot(RegimeState.STRESS, 1.0, "event_cooling")
        if spread_ticks >= self.config.stress_spread_ticks:
            return RegimeSnapshot(RegimeState.STRESS, 1.0, "spread_widening")
        if atr_ratio >= self.config.stress_vol_ratio:
            return RegimeSnapshot(RegimeState.STRESS, min(2.0, atr_ratio / self.config.stress_vol_ratio), "volatility_spike")
        if quote_rate > 0 and quote_rate <= self.config.stress_quote_rate:
            return RegimeSnapshot(RegimeState.STRESS, 0.8, "quote_rate_collapse")

        slope_strength = abs(ema_slope)
        flow_strength = abs(ofi_zscore)
        if (
            slope_strength >= self.config.trend_slope_threshold
            and flow_strength >= self.config.trend_ofi_threshold
        ):
            confidence = min(2.0, (slope_strength + flow_strength) / 2.0)
            return RegimeSnapshot(RegimeState.TREND, confidence, "directional_alignment")

        if slope_strength <= self.config.range_slope_threshold and flow_strength <= self.config.trend_ofi_threshold:
            confidence = min(2.0, 1.0 + max(0.0, 0.5 - slope_strength))
            return RegimeSnapshot(RegimeState.RANGE, confidence, "contained_rotation")

        return RegimeSnapshot(RegimeState.RANGE, 0.75, "mixed_conditions")

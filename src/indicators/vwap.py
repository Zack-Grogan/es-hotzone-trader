"""VWAP and session-context indicators."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd


PriceSource = Literal["HLC3", "HL2", "CLOSE"]


@dataclass
class SessionVWAPMetrics:
    """Developing VWAP and sigma bands for a single anchored session."""

    vwap: pd.Series
    sigma: pd.Series
    upper_1: pd.Series
    lower_1: pd.Series
    upper_2: pd.Series
    lower_2: pd.Series


def _typical_price(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    source: PriceSource = "HLC3",
) -> pd.Series:
    if source == "HLC3":
        return (high + low + close) / 3.0
    if source == "HL2":
        return (high + low) / 2.0
    return close.astype(float)


def session_labels(index: pd.DatetimeIndex, start_hour: int, start_minute: int) -> pd.Series:
    """Return anchored session labels for a timezone-aware index."""
    idx = pd.DatetimeIndex(index)
    session_minutes = (start_hour * 60) + start_minute
    intraday_minutes = (idx.hour * 60) + idx.minute
    labels = pd.Series(idx.normalize(), index=idx)
    return labels.where(intraday_minutes >= session_minutes, labels - pd.Timedelta(days=1))


def vwap(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    volume: pd.Series,
    source: PriceSource = "HLC3",
) -> pd.Series:
    """Calculate cumulative VWAP across the provided series."""
    typical_price = _typical_price(high.astype(float), low.astype(float), close.astype(float), source)
    cumsum_pv = (typical_price * volume.astype(float)).cumsum()
    cumsum_v = volume.astype(float).cumsum()
    return cumsum_pv / cumsum_v.replace(0, np.nan)


def anchored_vwap(
    df: pd.DataFrame,
    anchor_time: Optional[pd.Timestamp] = None,
    price_col: str = "close",
    volume_col: str = "volume",
    source: PriceSource = "HLC3",
) -> pd.Series:
    """Calculate anchored VWAP from a specific timestamp."""
    if anchor_time is not None:
        df = df.loc[df.index >= anchor_time]
    if df.empty:
        return pd.Series(dtype=float)
    return vwap(df["high"], df["low"], df[price_col], df[volume_col], source)


def session_vwap(
    df: pd.DataFrame,
    session_start_hour: int = 9,
    session_start_minute: int = 30,
    source: PriceSource = "HLC3",
) -> pd.Series:
    """Calculate session-based VWAP using an anchored session label."""
    if df.empty:
        return pd.Series(dtype=float)
    metrics = session_vwap_bands(df, session_start_hour, session_start_minute, source)
    return metrics.vwap


def session_vwap_bands(
    df: pd.DataFrame,
    session_start_hour: int = 9,
    session_start_minute: int = 30,
    source: PriceSource = "HLC3",
) -> SessionVWAPMetrics:
    """Return developing VWAP and weighted sigma bands for each anchored session."""
    if df.empty:
        empty = pd.Series(dtype=float)
        return SessionVWAPMetrics(empty, empty, empty, empty, empty, empty)

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    volume = df["volume"].astype(float).clip(lower=0.0)
    typical_price = _typical_price(high, low, close, source)

    labels = session_labels(pd.DatetimeIndex(df.index), session_start_hour, session_start_minute)
    weighted_price = typical_price * volume
    weighted_square = (typical_price ** 2) * volume
    cum_volume = volume.groupby(labels).cumsum().replace(0, np.nan)
    cum_weighted_price = weighted_price.groupby(labels).cumsum()
    cum_weighted_square = weighted_square.groupby(labels).cumsum()

    vwap_values = cum_weighted_price / cum_volume
    variance = (cum_weighted_square / cum_volume) - (vwap_values ** 2)
    variance = variance.clip(lower=0.0).fillna(0.0)
    sigma = np.sqrt(variance)

    return SessionVWAPMetrics(
        vwap=vwap_values,
        sigma=sigma,
        upper_1=vwap_values + sigma,
        lower_1=vwap_values - sigma,
        upper_2=vwap_values + (sigma * 2.0),
        lower_2=vwap_values - (sigma * 2.0),
    )


def vwap_deviation(price: float, vwap_value: float, atr_value: float) -> float:
    """Calculate price deviation from VWAP in ATR units."""
    if atr_value == 0:
        return 0.0
    return (price - vwap_value) / atr_value

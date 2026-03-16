"""Auction-style volume profile helpers."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class VolumeProfile:
    """Developing volume profile summary."""

    poc: float
    vah: float
    val: float
    total_volume: float
    value_area_width: float


def _round_to_tick(price: float, tick_size: float) -> float:
    return round(round(price / tick_size) * tick_size, 10)


def build_volume_profile(
    df: pd.DataFrame,
    tick_size: float = 0.25,
    value_area_pct: float = 0.7,
    source: str = "close",
) -> VolumeProfile:
    """Build a simple developing volume profile from aggregated bars."""
    if df.empty:
        return VolumeProfile(poc=0.0, vah=0.0, val=0.0, total_volume=0.0, value_area_width=0.0)

    prices = df[source].astype(float) if source in df.columns else df["close"].astype(float)
    volumes = df["volume"].astype(float).clip(lower=0.0)
    bins: dict[float, float] = {}
    for price, volume in zip(prices, volumes):
        level = _round_to_tick(float(price), tick_size)
        bins[level] = bins.get(level, 0.0) + float(volume)

    if not bins:
        last_price = _round_to_tick(float(prices.iloc[-1]), tick_size)
        return VolumeProfile(poc=last_price, vah=last_price, val=last_price, total_volume=0.0, value_area_width=0.0)

    sorted_levels = sorted(bins)
    total_volume = sum(bins.values())
    poc = max(sorted_levels, key=lambda level: (bins[level], -abs(level - prices.iloc[-1])))
    target_volume = total_volume * value_area_pct

    included = {poc}
    accumulated = bins[poc]
    left = sorted_levels.index(poc) - 1
    right = sorted_levels.index(poc) + 1

    while accumulated < target_volume and (left >= 0 or right < len(sorted_levels)):
        left_level = sorted_levels[left] if left >= 0 else None
        right_level = sorted_levels[right] if right < len(sorted_levels) else None
        left_volume = bins[left_level] if left_level is not None else -1.0
        right_volume = bins[right_level] if right_level is not None else -1.0

        if right_volume > left_volume:
            included.add(right_level)
            accumulated += right_volume
            right += 1
        else:
            included.add(left_level)
            accumulated += left_volume
            left -= 1

    vah = max(included)
    val = min(included)
    return VolumeProfile(
        poc=poc,
        vah=vah,
        val=val,
        total_volume=total_volume,
        value_area_width=max(vah - val, tick_size),
    )
